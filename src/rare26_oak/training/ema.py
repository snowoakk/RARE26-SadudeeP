"""Exponential Moving Average (EMA) for model weights."""
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model weights updated as:
    shadow = decay * shadow + (1 - decay) * params

    Used during champion model training for stable validation evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights."""
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k] -= (1.0 - self.decay) * (self.shadow[k] - v.detach())

    def apply(self, model: nn.Module) -> None:
        """Replace model weights with shadow weights (backup original)."""
        self.backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore model weights from backup."""
        model.load_state_dict(self.backup)
