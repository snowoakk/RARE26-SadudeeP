"""CLI entry point: Generate pseudo labels from mother model ensemble.

Loads trained mother models (ConvNext-Base, ConvNext-Tiny, EfficientNetV2-S)
and runs inference on unlabeled GastroNet images to create pseudo_labels.csv.

Usage:
    python -m rare26_oak.cli.generate_pseudo_labels --gastronet-dir data/gastronet --debug
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rare26_oak.config import PipelineConfig, PathConfig
from rare26_oak.utils import set_seed, get_logger, ensure_dir
from rare26_oak.data.discover import discover_gastronet_images
from rare26_oak.data.transforms import get_pseudo_label_transform
from rare26_oak.data.loaders import create_unlabeled_loader
from rare26_oak.models.mother import create_mother_model

logger = get_logger(__name__)


def generate_pseudo_labels(config: PipelineConfig) -> Path:
    """Generate pseudo labels using mother model ensemble.

    Returns path to the generated pseudo_labels.csv.
    """
    device = torch.device(config.resolve_device())
    max_samples = config.debug_max_samples if config.debug else None
    pc = config.pseudo_label

    # Discover unlabeled data
    df_unlabeled = discover_gastronet_images(config.paths.gastronet_data_dir)
    if len(df_unlabeled) == 0:
        logger.warning("No unlabeled images found. Creating empty pseudo_labels.csv.")
        output_dir = ensure_dir(config.paths.pseudo_labels_dir)
        output_path = output_dir / "pseudo_labels.csv"
        pd.DataFrame(columns=["path", "label", "filepath", "fold"]).to_csv(output_path, index=False)
        return output_path

    logger.info(f"Found {len(df_unlabeled)} unlabeled images")

    # Create data loaders at different resolutions
    tf_384 = get_pseudo_label_transform(384)
    tf_512 = get_pseudo_label_transform(512)

    loader_384 = create_unlabeled_loader(
        df_unlabeled, tf_384, batch_size=pc.batch_size_384,
        num_workers=pc.num_workers, max_samples=max_samples,
    )
    loader_512 = create_unlabeled_loader(
        df_unlabeled, tf_512, batch_size=pc.batch_size_512,
        num_workers=pc.num_workers, max_samples=max_samples,
    )

    weights_dir = config.paths.mother_weights_dir

    # Load teacher models (fold 0 weights)
    logger.info("Loading ConvNext-Tiny teacher...")
    m_tiny = create_mother_model("convnext_tiny", pretrained=False)
    tiny_path = weights_dir / "best_rare26_convnext_center_fold_0_v1.pth"
    if tiny_path.exists():
        m_tiny.load_trained_weights(str(tiny_path), device=str(device))
    elif config.debug:
        logger.warning(f"Mother weights not found (debug mode): {tiny_path}")
    else:
        raise FileNotFoundError(f"Mother weights not found: {tiny_path}")
    m_tiny.to(device).eval()

    logger.info("Loading EfficientNetV2-S teacher...")
    m_eff = create_mother_model("efficientnet_v2_s", pretrained=False)
    eff_path = weights_dir / "best_rare26_effnet_fold_0.pth"
    if eff_path.exists():
        m_eff.load_trained_weights(str(eff_path), device=str(device))
    elif config.debug:
        logger.warning(f"Mother weights not found (debug mode): {eff_path}")
    else:
        raise FileNotFoundError(f"Mother weights not found: {eff_path}")
    m_eff.to(device).eval()

    logger.info("Loading ConvNext-Base teacher...")
    m_base = create_mother_model("convnext_base", pretrained=False)
    base_path = weights_dir / "best_rare26_convbase512_fold_0.pth"
    if base_path.exists():
        m_base.load_trained_weights(str(base_path), device=str(device))
    elif config.debug:
        logger.warning(f"Mother weights not found (debug mode): {base_path}")
    else:
        raise FileNotFoundError(f"Mother weights not found: {base_path}")
    m_base.to(device).eval()

    # Run inference on 384px models
    image_paths = []
    p_tiny, p_eff = [], []
    logger.info("Running 384px inference (Tiny + EffNet)...")
    with torch.no_grad():
        for imgs, pths in tqdm(loader_384, desc="384px Inference"):
            imgs = imgs.to(device)
            p_tiny.extend(F.softmax(m_tiny(imgs), dim=1)[:, 1].cpu().numpy())
            p_eff.extend(F.softmax(m_eff(imgs), dim=1)[:, 1].cpu().numpy())
            image_paths.extend(pths)

    # Run inference on 512px model
    p_base = []
    logger.info("Running 512px inference (Base)...")
    with torch.no_grad():
        for imgs, _ in tqdm(loader_512, desc="512px Inference"):
            p_base.extend(F.softmax(m_base(imgs.to(device)), dim=1)[:, 1].cpu().numpy())

    del m_tiny, m_eff, m_base
    torch.cuda.empty_cache()

    # Weighted ensemble
    n = len(image_paths)
    ensemble_probs = (
        pc.ensemble_weight_base * np.array(p_base[:n])
        + pc.ensemble_weight_tiny * np.array(p_tiny[:n])
        + pc.ensemble_weight_eff * np.array(p_eff[:n])
    )

    # Filter by confidence thresholds
    pseudo_data = []
    for path, prob in zip(image_paths, ensemble_probs):
        if prob < pc.threshold_negative:
            pseudo_data.append({"path": path, "label": 0})
        elif prob > pc.threshold_positive:
            pseudo_data.append({"path": path, "label": 1})

    df_pseudo = pd.DataFrame(pseudo_data)

    # Save
    output_dir = ensure_dir(config.paths.pseudo_labels_dir)
    output_path = output_dir / "pseudo_labels.csv"

    if len(df_pseudo) > 0:
        df_pseudo["filepath"] = df_pseudo["path"]
        df_pseudo["fold"] = -1
        df_pseudo.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_pseudo)} pseudo-labeled images to {output_path}")
        logger.info(f"  Class 0 (ndbe): {(df_pseudo['label'] == 0).sum()}")
        logger.info(f"  Class 1 (neo):  {(df_pseudo['label'] == 1).sum()}")
    else:
        pd.DataFrame(columns=["path", "label", "filepath", "fold"]).to_csv(
            output_path, index=False
        )
        logger.warning("No images passed confidence thresholds!")

    return output_path


def main() -> int:
    """Entry point for pseudo-label generation."""
    parser = argparse.ArgumentParser(description="Generate pseudo labels from mother models")
    parser.add_argument("--gastronet-dir", type=str, default="data/gastronet",
                        help="Path to unlabeled GastroNet images")
    parser.add_argument("--mother-weights-dir", type=str, default="outputs/mother_weights",
                        help="Directory containing trained mother weights")
    parser.add_argument("--output-dir", type=str, default="outputs/pseudo_labels",
                        help="Directory to save pseudo_labels.csv")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: minimal samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        paths=PathConfig(
            gastronet_data_dir=Path(args.gastronet_dir),
            mother_weights_dir=Path(args.mother_weights_dir),
            pseudo_labels_dir=Path(args.output_dir),
        ),
        seed=args.seed,
        debug=args.debug,
    )
    set_seed(config.seed)

    output_path = generate_pseudo_labels(config)
    logger.info(f"Pseudo label generation complete: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
