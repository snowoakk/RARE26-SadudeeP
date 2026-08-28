"""Loss functions for RARE26 training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply MixUp augmentation on a batch.

    Returns (mixed_x, y_a, y_b, lambda).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute loss for MixUp (linear combination of losses)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Used by mother models (ConvNext-Tiny, EfficientNetV2, ConvNext-Base).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SurrogatePPVRankingLoss(nn.Module):
    """Surrogate PPV Ranking Loss for champion models.

    Combines CrossEntropy with a pairwise margin ranking loss that
    forces positive (neo) logits above negative (ndbe) logits.
    Used by ResNet50 and ViT champion models.
    """

    def __init__(self, ce_weight: float = 1.0, rank_weight: float = 2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = logits[pos_mask, 1]
            neg_scores = logits[neg_mask, 1]
            diff = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
            rank_loss = torch.log1p(torch.exp(diff)).mean()
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)

        return self.ce_weight * ce_loss + self.rank_weight * rank_loss


class ChampionLoss(nn.Module):
    """Champion Loss for DINOv3-ConvNeXt champion model.

    Combines weighted CrossEntropy (with label smoothing) and a differentiable
    surrogate PPV loss with curriculum blending schedule.
    """

    def __init__(self, class_weights: torch.Tensor = None,
                 recall_threshold: float = 0.90,
                 surrogate_lambda: float = 50.0):
        super().__init__()
        self.recall_threshold = recall_threshold
        self.surrogate_lambda = surrogate_lambda
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch: int = 1) -> torch.Tensor:
        targets_long = targets.long()

        # Weighted CrossEntropy with label smoothing
        ce_loss = F.cross_entropy(
            logits, targets_long,
            weight=self.class_weights,
            label_smoothing=0.05,
        )

        # Differentiable surrogate PPV loss
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets_float = targets.float()

        tp = (probs * targets_float).sum()
        fp = (probs * (1.0 - targets_float)).sum()
        fn = ((1.0 - probs) * targets_float).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        surrogate = -precision + self.surrogate_lambda * F.relu(
            self.recall_threshold - recall
        )

        # Curriculum blending: more CE early, more surrogate later
        if epoch <= 5:
            alpha = 0.15
        elif epoch <= 10:
            alpha = 0.35
        else:
            alpha = 0.50

        return (1.0 - alpha) * ce_loss + alpha * surrogate
