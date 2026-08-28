"""CLI entry point: Train champion models with RARE26 + pseudo-labeled data.

Trains DINOv3-ConvNeXt-Base, GastroNet-ResNet50, and/or DINOv2-ViT-Base
using 2-fold cross-center validation with curriculum detox strategy.

Usage:
    python -m rare26_oak.cli.train_champion --model gastronet_resnet50 --debug
    python -m rare26_oak.cli.train_champion --model all --debug
"""
import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR

from rare26_oak.config import PipelineConfig, PathConfig
from rare26_oak.utils import set_seed, get_logger, ensure_dir
from rare26_oak.data.discover import (
    discover_rare26_images, build_cross_center_folds, discover_fake_cancer_images,
)
from rare26_oak.data.transforms import (
    get_champ_train_transform, get_champ_convnext_train_transform, get_val_transform,
)
from rare26_oak.data.dataset import Rare26Dataset
from rare26_oak.data.loaders import create_val_loader
from rare26_oak.models.champion import create_champion_model
from rare26_oak.training.losses import SurrogatePPVRankingLoss, ChampionLoss
from rare26_oak.training.ema import EMA
from rare26_oak.evaluation.metrics import calculate_prevalence_corrected_ppv

logger = get_logger(__name__)

# Model-specific settings
MODEL_SETTINGS = {
    "dinov3_convnext_base": {
        "img_size": 512, "batch_size": 8, "grad_accum": 4,
        "lr_backbone": 2e-5, "lr_head": 2e-5,
        "loss": "champion", "transform": "convnext_heavy",
        "use_ema": False, "use_swa": False,
        "warmup_epochs": 3, "patience": 8,
        "weight_prefix": "best_oak3_convnext_fold",
        "always_pseudo": True,  # pseudo always included, only fake is detoxed
    },
    "gastronet_resnet50": {
        "img_size": 512, "batch_size": 4, "grad_accum": 8,
        "lr_backbone": 5e-6, "lr_head": 2e-5,
        "loss": "surrogate_ppv", "transform": "champ_standard",
        "use_ema": True, "use_swa": True, "freeze_bn": True,
        "weight_prefix": "best_gastronet_rn50_fold",
        "always_pseudo": False,
    },
    "gastronet_vit_base": {
        "img_size": 518, "batch_size": 2, "grad_accum": 16,
        "lr_backbone": 2e-6, "lr_head": 1e-4,
        "loss": "surrogate_ppv", "transform": "champ_standard",
        "use_ema": True, "use_swa": True,
        "weight_prefix": "best_gastronet_vit_fold",
        "always_pseudo": False,
    },
}


def _get_model_kwargs(model_name: str, config: PipelineConfig) -> dict:
    """Build model constructor kwargs."""
    kwargs = {}
    if model_name == "dinov3_convnext_base":
        if config.paths.hf_convnext_path:
            kwargs["hf_model_path"] = str(config.paths.hf_convnext_path)
    elif model_name == "gastronet_resnet50":
        if config.paths.gastronet_resnet_weights:
            kwargs["gastronet_weights"] = str(config.paths.gastronet_resnet_weights)
    elif model_name == "gastronet_vit_base":
        if config.paths.dinov2_weights:
            kwargs["gastronet_weights"] = str(config.paths.dinov2_weights)
        if config.paths.dinov2_repo_path:
            kwargs["dinov2_repo_path"] = str(config.paths.dinov2_repo_path)
    return kwargs


def _build_train_loader(dfs: list, transform, batch_size: int,
                        num_workers: int, max_samples=None) -> DataLoader:
    """Build a training DataLoader from a list of DataFrames with weighted sampling."""
    non_empty = [d for d in dfs if d is not None and not d.empty]
    if non_empty:
        df_concat = pd.concat(non_empty, ignore_index=True)
    else:
        df_concat = pd.DataFrame({"filepath": [], "label": []})

    dataset = Rare26Dataset(df_concat, transform=transform, max_samples=max_samples)

    if len(dataset) == 0:
        logger.warning("_build_train_loader: dataset is empty. Returning empty loader.")
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)

    labels = dataset.df["label"].tolist()

    counts: dict = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    weights = [1.0 / counts.get(lbl, 1.0) for lbl in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )


def train_single_champion(model_name: str, config: PipelineConfig) -> None:
    """Train a single champion model with 2-fold cross-center validation."""
    ms = MODEL_SETTINGS[model_name]
    device = torch.device(config.resolve_device())
    epochs = config.get_champ_epochs()
    max_samples = config.debug_max_samples if config.debug else None
    detox_epoch = min(config.champ.detox_epoch, epochs)

    logger.info(f"Training champion: {model_name} (epochs={epochs}, device={device})")

    # Discover real data
    df = discover_rare26_images(config.paths.rare26_data_dir, deduplicate=not config.debug)
    df = build_cross_center_folds(df)

    # Load pseudo labels
    pseudo_csv = config.paths.pseudo_labels_csv
    df_pseudo = pd.DataFrame()
    if pseudo_csv.exists():
        df_pseudo = pd.read_csv(pseudo_csv)
        if "fold" not in df_pseudo.columns:
            df_pseudo["fold"] = -1
        if "filepath" not in df_pseudo.columns and "path" in df_pseudo.columns:
            df_pseudo["filepath"] = df_pseudo["path"]
        logger.info(f"Loaded {len(df_pseudo)} pseudo labels")
    else:
        logger.warning(f"No pseudo labels at {pseudo_csv}")

    # Load fake cancer data
    df_fake = pd.DataFrame()
    if config.paths.fake_neo_dir and Path(config.paths.fake_neo_dir).exists():
        df_fake = discover_fake_cancer_images(config.paths.fake_neo_dir)

    output_dir = ensure_dir(config.paths.champ_weights_dir)

    for fold in [0, 1]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FOLD {fold} -- Champion {model_name}")
        logger.info(f"{'=' * 60}")

        df_real_train = df[df.fold != fold].copy()
        df_val = df[df.fold == fold].copy()

        # Get transforms
        if ms["transform"] == "convnext_heavy":
            train_tf = get_champ_convnext_train_transform(ms["img_size"])
        else:
            train_tf = get_champ_train_transform(ms["img_size"])
        val_tf = get_val_transform(ms["img_size"])

        val_loader = create_val_loader(
            df_val, val_tf, batch_size=ms["batch_size"],
            num_workers=config.champ.num_workers, max_samples=max_samples,
        )

        # Create model
        model_kwargs = _get_model_kwargs(model_name, config)
        try:
            model = create_champion_model(model_name, **model_kwargs).to(device)
        except Exception as e:
            if config.debug:
                logger.warning(f"Could not create {model_name} (debug mode): {e}")
                logger.info("Skipping this model in debug mode.")
                continue
            raise

        # EMA
        ema = EMA(model, decay=config.champ.ema_decay) if ms.get("use_ema") else None

        # Optimizer with differential LRs
        if ms["lr_backbone"] != ms["lr_head"]:
            optimizer = optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if "backbone" in n],
                 "lr": ms["lr_backbone"]},
                {"params": [p for n, p in model.named_parameters() if "head" in n],
                 "lr": ms["lr_head"]},
            ], weight_decay=config.champ.weight_decay)
        else:
            optimizer = optim.AdamW(
                model.parameters(), lr=ms["lr_backbone"],
                weight_decay=config.champ.weight_decay,
            )

        # Scheduler
        warmup = ms.get("warmup_epochs", 0)
        if warmup > 0:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda ep: (
                    (ep + 1) / (warmup + 1)
                    if ep < warmup
                    else max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
                ),
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # SWA
        swa_model = AveragedModel(model) if ms.get("use_swa") else None
        swa_scheduler = SWALR(optimizer, swa_lr=ms["lr_backbone"]) if ms.get("use_swa") else None

        # Loss
        if ms["loss"] == "champion":
            class_counts = df_real_train["label"].value_counts().to_dict()
            cw = torch.tensor(
                [1.0 / class_counts.get(0, 1.0), 1.0 / class_counts.get(1, 1.0)],
                dtype=torch.float32,
            ).to(device)
            criterion = ChampionLoss(
                class_weights=cw, recall_threshold=0.90, surrogate_lambda=50.0
            ).to(device)
        else:
            criterion = SurrogatePPVRankingLoss()

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        best_ppv = -1.0
        patience_counter = 0

        save_path = output_dir / f"{ms['weight_prefix']}_{fold}.pth"

        for epoch in range(1, epochs + 1):
            # Early stopping
            if patience_counter >= ms.get("patience", 999):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Curriculum composition
            in_detox = epoch > detox_epoch
            dfs_train = [df_real_train]

            if ms.get("always_pseudo"):
                # ConvNext: always include pseudo, detox only fake
                if not df_pseudo.empty:
                    dfs_train.append(df_pseudo)
                if not in_detox and not df_fake.empty:
                    dfs_train.append(df_fake)
            else:
                # ResNet/ViT: include pseudo+fake in stage1, real-only in stage2
                if not in_detox:
                    if not df_pseudo.empty:
                        dfs_train.append(df_pseudo)
                    if not df_fake.empty:
                        dfs_train.append(df_fake)

            train_loader = _build_train_loader(
                dfs_train, train_tf, batch_size=ms["batch_size"],
                num_workers=config.champ.num_workers, max_samples=max_samples,
            )

            stage = "Detox (Real Only)" if in_detox else "Real+Pseudo+Fake"
            logger.info(f"Epoch {epoch:02d}/{epochs} | {stage}")

            model.train()
            if ms.get("freeze_bn"):
                model.backbone.eval()
            optimizer.zero_grad()

            for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="[Train]", leave=False)):
                imgs, labels = imgs.to(device), labels.to(device)

                if use_amp and scaler:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(imgs)
                        if ms["loss"] == "champion":
                            loss = criterion(outputs, labels, epoch=epoch) / ms["grad_accum"]
                        else:
                            loss = criterion(outputs, labels) / ms["grad_accum"]
                    scaler.scale(loss).backward()

                    if (i + 1) % ms["grad_accum"] == 0 or (i + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=config.champ.grad_clip_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if ema:
                            ema.update(model)
                else:
                    outputs = model(imgs)
                    if ms["loss"] == "champion":
                        loss = criterion(outputs, labels, epoch=epoch) / ms["grad_accum"]
                    else:
                        loss = criterion(outputs, labels) / ms["grad_accum"]
                    loss.backward()

                    if (i + 1) % ms["grad_accum"] == 0 or (i + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=config.champ.grad_clip_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        if ema:
                            ema.update(model)

            # SWA update
            if swa_model and epoch > config.champ.swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # Validation
            if ema:
                ema.apply(model)
            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="[Val]", leave=False):
                    imgs = imgs.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            p = torch.softmax(model(imgs), dim=1)[:, 1]
                    else:
                        p = torch.softmax(model(imgs), dim=1)[:, 1]
                    all_probs.extend(p.cpu().float().numpy())
                    all_labels.extend(labels.numpy())

            ppv = calculate_prevalence_corrected_ppv(
                np.array(all_labels), np.array(all_probs)
            )
            logger.info(f"  Val PPV@90R (1:101): {ppv:.4f}")

            if ppv > best_ppv:
                best_ppv = ppv
                torch.save(model.state_dict(), save_path)
                logger.info(f"  NEW BEST! -> {save_path.name} (PPV={ppv:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if ema:
                ema.restore(model)

        # Save SWA model
        if swa_model:
            swa_path = output_dir / f"swa_{ms['weight_prefix']}_{fold}.pth"
            torch.save(swa_model.module.state_dict(), swa_path)
            logger.info(f"SWA model saved: {swa_path.name}")

        logger.info(f"Fold {fold} complete. Best PPV: {best_ppv:.4f}")

        del model, optimizer, scheduler
        if swa_model:
            del swa_model
        torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    """Entry point for champion model training."""
    parser = argparse.ArgumentParser(description="Train champion models")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["dinov3_convnext_base", "gastronet_resnet50", "gastronet_vit_base", "all"],
        help="Which champion model to train",
    )
    parser.add_argument("--data-dir", type=str, default="data/rare26")
    parser.add_argument("--pseudo-csv", type=str, default="outputs/pseudo_labels/pseudo_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/champ_weights")
    parser.add_argument("--gastronet-weights", type=str, default=None,
                        help="Path to GastroNet ResNet50 SSL weights")
    parser.add_argument("--dinov2-weights", type=str, default=None,
                        help="Path to DINOv2 ViT weights")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: minimal epochs/samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        paths=PathConfig(
            rare26_data_dir=Path(args.data_dir),
            pseudo_labels_csv=Path(args.pseudo_csv),
            champ_weights_dir=Path(args.output_dir),
            gastronet_resnet_weights=Path(args.gastronet_weights) if args.gastronet_weights else None,
            dinov2_weights=Path(args.dinov2_weights) if args.dinov2_weights else None,
        ),
        seed=args.seed,
        debug=args.debug,
    )
    set_seed(config.seed)

    models_to_train = list(MODEL_SETTINGS.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_train:
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting champion training: {model_name}")
        logger.info(f"{'=' * 60}")
        train_single_champion(model_name, config)

    logger.info("All champion model training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
