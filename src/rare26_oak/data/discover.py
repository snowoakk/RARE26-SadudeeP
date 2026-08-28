"""Data discovery: scan directories and build metadata DataFrames."""
from pathlib import Path

import pandas as pd

from rare26_oak.utils import get_md5, get_logger

logger = get_logger(__name__)


def discover_rare26_images(data_dir: Path, deduplicate: bool = True) -> pd.DataFrame:
    """Discover RARE26 images arranged as data_dir/center_X/{ndbe,neo}/*.{png,jpg}.

    Returns DataFrame with columns: [path, center, label, image_hash(optional)].
    label: 0=ndbe, 1=neo.
    """
    data_dir = Path(data_dir)
    all_files = (
        list(data_dir.glob("center_*/*/*.png"))
        + list(data_dir.glob("center_*/*/*.jpg"))
    )

    if not all_files:
        logger.warning(f"No images found in {data_dir}")
        return pd.DataFrame(columns=["path", "center", "label"])

    records = []
    for f in all_files:
        parts = f.parts
        center = None
        label = None
        for part in parts:
            if part.startswith("center_"):
                center = part
            if part == "ndbe":
                label = 0
            elif part == "neo":
                label = 1
        if center is not None and label is not None:
            records.append({"path": str(f), "center": center, "label": label})

    df = pd.DataFrame(records)
    logger.info(f"Discovered {len(df)} images in {data_dir}")

    if deduplicate and len(df) > 0:
        df["image_hash"] = df["path"].apply(lambda p: get_md5(Path(p)))
        before = len(df)
        df = df.drop_duplicates(subset=["image_hash"], keep="first").reset_index(drop=True)
        logger.info(f"Deduplication: {before} -> {len(df)} images")

    return df


def discover_gastronet_images(data_dir: Path) -> pd.DataFrame:
    """Discover unlabeled GastroNet images for pseudo-labeling.

    Returns DataFrame with column: [path].
    """
    data_dir = Path(data_dir)
    all_files = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))

    if not all_files:
        logger.warning(f"No images found in {data_dir}")
        return pd.DataFrame(columns=["path"])

    df = pd.DataFrame({"path": [str(f) for f in all_files]})
    logger.info(f"Discovered {len(df)} GastroNet images in {data_dir}")
    return df


def discover_fake_cancer_images(fake_dir: Path) -> pd.DataFrame:
    """Discover synthetic/fake cancer images.

    Returns DataFrame with columns: [filepath, label, fold] where label=1 and fold=-1.
    """
    fake_dir = Path(fake_dir)
    all_files = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))

    if not all_files:
        logger.warning(f"No fake cancer images found in {fake_dir}")
        return pd.DataFrame(columns=["filepath", "label", "fold"])

    df = pd.DataFrame([{"filepath": str(p), "label": 1, "fold": -1} for p in all_files])
    logger.info(f"Discovered {len(df)} fake cancer images")
    return df


def build_cross_center_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Assign fold indices based on center.

    fold 0 = center_1 images (validated when training on center_2).
    fold 1 = center_2 images (validated when training on center_1).

    Expects 'center' column. Renames 'path' to 'filepath' if needed.
    """
    df = df.copy()
    if "path" in df.columns and "filepath" not in df.columns:
        df = df.rename(columns={"path": "filepath"})

    df["fold"] = df["center"].apply(
        lambda c: 0 if "center_1" in c or c == "center1" else 1
    )
    return df
