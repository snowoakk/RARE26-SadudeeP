"""DataLoader factories with weighted sampling for class imbalance."""
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

from rare26_oak.data.dataset import Rare26Dataset, UnlabeledDataset


def create_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to handle class imbalance."""
    class_counts: dict = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    weights = [1.0 / class_counts[lbl] for lbl in labels]
    # Guard: num_samples must be >= 1 (labels can be empty in debug/small splits)
    return WeightedRandomSampler(
        weights=weights, num_samples=max(1, len(labels)), replacement=True,
    )


def create_train_loader(
    df: pd.DataFrame, transform, batch_size: int = 8,
    num_workers: int = 2, max_samples: Optional[int] = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create a training DataLoader with weighted sampling."""
    import logging
    dataset = Rare26Dataset(df, transform=transform, max_samples=max_samples)

    if len(dataset) == 0:
        logging.getLogger(__name__).warning(
            "create_train_loader: dataset is empty after applying max_samples. "
            "Returning a loader with no batches."
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)

    labels = dataset.df["label"].tolist()
    sampler = create_weighted_sampler(labels)

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
    )


def create_val_loader(
    df: pd.DataFrame, transform, batch_size: int = 8,
    num_workers: int = 2, max_samples: Optional[int] = None,
) -> DataLoader:
    """Create a validation DataLoader (no shuffling, no sampling)."""
    dataset = Rare26Dataset(df, transform=transform, max_samples=max_samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def create_unlabeled_loader(
    df: pd.DataFrame, transform, batch_size: int = 32,
    num_workers: int = 2, max_samples: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for unlabeled images (pseudo-label generation)."""
    dataset = UnlabeledDataset(df, transform=transform, max_samples=max_samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
