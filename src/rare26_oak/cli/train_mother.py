"""CLI entry point: Train mother models on RARE26 data.

Trains ConvNext-Tiny, ConvNext-Base, and/or EfficientNetV2-S using
2-fold cross-center validation (center_1 vs center_2).

Usage:
    python -m rare26_oak.cli.train_mother --model convnext_tiny --data-dir data/rare26
    python -m rare26_oak.cli.train_mother --model all --data-dir data/rare26 --debug
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rare26_oak.config import PipelineConfig, PathConfig
from rare26_oak.utils import set_seed, get_logger, ensure_dir
from rare26_oak.data.discover import discover_rare26_images, build_cross_center_folds
from rare26_oak.data.transforms import get_mother_train_transform, get_val_transform
from rare26_oak.data.loaders import create_train_loader, create_val_loader
from rare26_oak.models.mother import create_mother_model
from rare26_oak.training.losses import FocalLoss, mixup_data, mixup_criterion
from rare26_oak.evaluation.metrics import calculate_oof_auc, calculate_ppv_at_recall

logger = get_logger(__name__)

# Model-specific overrides
MODEL_CONFIGS = {
    "convnext_tiny": {
        "img_size": 384, "batch_size": 8, "accumulation_steps": 1,
        "lr": 1e-4, "focal_gamma": 2.0,
    },
    "convnext_base": {
        "img_size": 512, "batch_size": 4, "accumulation_steps": 4,
        "lr": 5e-5, "focal_gamma": 2.0,
    },
    "efficientnet_v2_s": {
        "img_size": 384, "batch_size": 8, "accumulation_steps": 1,
        "lr": 1e-4, "focal_gamma": 3.0,
    },
}

WEIGHT_FILENAMES = {
    "convnext_tiny": "best_rare26_convnext_center_fold_{fold}_v1.pth",
    "convnext_base": "best_rare26_convbase512_fold_{fold}.pth",
    "efficientnet_v2_s": "best_rare26_effnet_fold_{fold}.pth",
}


def train_single_mother(model_name: str, config: PipelineConfig) -> None:
    """Train a single mother model with 2-fold cross-center validation."""
    mc = MODEL_CONFIGS[model_name]
    device = torch.device(config.resolve_device())
    epochs = config.get_mother_epochs()
    max_samples = config.debug_max_samples if config.debug else None

    logger.info(f"Training mother model: {model_name} (epochs={epochs}, device={device})")

    # Discover data
    df = discover_rare26_images(config.paths.rare26_data_dir, deduplicate=not config.debug)
    df = build_cross_center_folds(df)

    output_dir = ensure_dir(config.paths.mother_weights_dir)
    centers = ["center_1", "center_2"]
    oof_preds = np.zeros(len(df))

    for fold, val_center in enumerate(centers):
        train_center = "center_2" if val_center == "center_1" else "center_1"
        logger.info(f"Fold {fold + 1}/2: Train on [{train_center}] -> Validate on [{val_center}]")

        train_df = df[df["center"] == train_center].reset_index(drop=True)
        val_df = df[df["center"] == val_center].reset_index(drop=True)
        val_indices = df[df["center"] == val_center].index.values

        train_transform = get_mother_train_transform(mc["img_size"])
        val_transform = get_val_transform(mc["img_size"])

        train_loader = create_train_loader(
            train_df, train_transform, batch_size=mc["batch_size"],
            num_workers=config.mother.num_workers, max_samples=max_samples,
        )
        val_loader = create_val_loader(
            val_df, val_transform, batch_size=mc["batch_size"],
            num_workers=config.mother.num_workers, max_samples=max_samples,
        )

        model = create_mother_model(model_name, pretrained=not config.debug).to(device)
        criterion = FocalLoss(gamma=mc["focal_gamma"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=mc["lr"], weight_decay=config.mother.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = -1.0
        best_preds = None
        save_path = output_dir / WEIGHT_FILENAMES[model_name].format(fold=fold)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(device), targets.to(device)
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, alpha=config.mother.mixup_alpha
                )

                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss = loss / mc["accumulation_steps"]
                loss.backward()

                if (batch_idx + 1) % mc["accumulation_steps"] == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * mc["accumulation_steps"]

            scheduler.step()

            # Validation
            model.eval()
            all_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    probs = F.softmax(model(images.to(device)), dim=1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())

            val_targets = val_df["label"].values[:len(all_preds)]
            auc_score = calculate_oof_auc(val_targets, np.array(all_preds))

            if auc_score > best_auc:
                best_auc = auc_score
                best_preds = np.array(all_preds)
                torch.save(model.state_dict(), save_path)
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs} | New best AUC: {auc_score:.4f} -> saved {save_path.name}"
                )
            else:
                logger.info(f"  Epoch {epoch + 1}/{epochs} | AUC: {auc_score:.4f}")

        logger.info(f"Fold {fold + 1} complete. Best AUC: {best_auc:.4f}")
        if best_preds is not None and len(best_preds) == len(val_indices):
            oof_preds[val_indices] = best_preds

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # OOF evaluation
    true_labels = df["label"].values
    if oof_preds.sum() > 0:
        final_auc = calculate_oof_auc(true_labels, oof_preds)
        final_ppv = calculate_ppv_at_recall(true_labels, oof_preds)
        logger.info(f"[{model_name}] OOF ROC-AUC: {final_auc:.4f} | PPV@90%: {final_ppv:.4f}")


def main() -> int:
    """Entry point for mother model training."""
    parser = argparse.ArgumentParser(description="Train mother models on RARE26 data")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["convnext_tiny", "convnext_base", "efficientnet_v2_s", "all"],
        help="Which mother model to train",
    )
    parser.add_argument("--data-dir", type=str, default="data/rare26",
                        help="Path to RARE26 data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/mother_weights",
                        help="Directory to save trained weights")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: minimal epochs/samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        paths=PathConfig(
            rare26_data_dir=Path(args.data_dir),
            mother_weights_dir=Path(args.output_dir),
        ),
        seed=args.seed,
        debug=args.debug,
    )
    set_seed(config.seed)

    models_to_train = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_train:
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting mother training: {model_name}")
        logger.info(f"{'=' * 60}")
        train_single_mother(model_name, config)

    logger.info("All mother model training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
