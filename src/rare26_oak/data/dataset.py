"""PyTorch Dataset classes for RARE26 image classification."""
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Rare26Dataset(Dataset):
    """Dataset for labeled RARE26 images.

    Expects a DataFrame with columns:
    - 'filepath' or 'path': path to image file
    - 'label': integer class label (0=ndbe, 1=neo)
    """

    PATH_COL = "filepath"
    PATH_COL_ALT = "path"

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None,
                 max_samples: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        if self.PATH_COL not in self.df.columns and self.PATH_COL_ALT in self.df.columns:
            self.df = self.df.rename(columns={self.PATH_COL_ALT: self.PATH_COL})
        self.transform = transform

        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        try:
            img = Image.open(row[self.PATH_COL]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = int(row["label"])
        return img, label


class UnlabeledDataset(Dataset):
    """Dataset for unlabeled images (used in pseudo-label generation).

    Expects a DataFrame with column 'path'.
    Returns (transformed_image, file_path_string).
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None,
                 max_samples: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.df.iloc[idx]["path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, path
