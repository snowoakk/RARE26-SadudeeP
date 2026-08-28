"""Evaluation metrics for the RARE26 challenge."""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def calculate_prevalence_corrected_ppv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    prior_target: float = 1.0 / 101.0,
    min_recall: float = 0.90,
) -> float:
    """Calculate prevalence-corrected PPV at a given recall threshold.

    The RARE26 challenge metric: PPV corrected to the real-world prevalence
    of 1:101 (approx 0.0099) at >= 90% recall.

    Formula:
        PPV_corrected = (R * prev) / (R * prev + (1 - S) * (1 - prev))
    where R = recall (sensitivity), S = specificity.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.0

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = n_pos - tp
    tn = n_neg - fp

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    numerator = recall * prior_target
    denominator = recall * prior_target + (1 - specificity) * (1 - prior_target)
    ppv_corrected = np.where(denominator > 0, numerator / denominator, 0.0)

    valid = recall >= min_recall
    if not valid.any():
        return 0.0
    return float(ppv_corrected[valid].max())


def calculate_oof_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate out-of-fold ROC-AUC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def calculate_ppv_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> float:
    """Calculate raw (non-prevalence-corrected) PPV at target recall."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(precision[valid_idx[-1]])
