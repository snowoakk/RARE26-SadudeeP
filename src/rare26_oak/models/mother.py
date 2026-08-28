"""Mother model architectures: ConvNext-Tiny, ConvNext-Base, EfficientNetV2-S.

These are trained on RARE26 data only using cross-center validation.
Their weights are used as teacher models for pseudo-label generation.
"""
import torch
import torch.nn as nn
from torchvision import models

from rare26_oak.utils import get_logger

logger = get_logger(__name__)


class MotherConvNextTiny(nn.Module):
    """ConvNext-Tiny with 2-class head. Input: 384x384."""

    MODEL_NAME = "convnext_tiny"
    IMG_SIZE = 384

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def load_trained_weights(self, path: str, device: str = "cpu") -> None:
        """Load trained weights from a checkpoint file."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"Loaded {self.MODEL_NAME} weights from {path}")


class MotherConvNextBase(nn.Module):
    """ConvNext-Base with 2-class head. Input: 512x512."""

    MODEL_NAME = "convnext_base"
    IMG_SIZE = 512

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_base(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def load_trained_weights(self, path: str, device: str = "cpu") -> None:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"Loaded {self.MODEL_NAME} weights from {path}")


class MotherEfficientNetV2(nn.Module):
    """EfficientNetV2-S with 2-class head. Input: 384x384."""

    MODEL_NAME = "efficientnet_v2_s"
    IMG_SIZE = 384

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def load_trained_weights(self, path: str, device: str = "cpu") -> None:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"Loaded {self.MODEL_NAME} weights from {path}")


def create_mother_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """Factory: create a mother model by name.

    Args:
        model_name: One of 'convnext_tiny', 'convnext_base', 'efficientnet_v2_s'
    """
    REGISTRY = {
        "convnext_tiny": MotherConvNextTiny,
        "convnext_base": MotherConvNextBase,
        "efficientnet_v2_s": MotherEfficientNetV2,
    }
    if model_name not in REGISTRY:
        raise ValueError(
            f"Unknown mother model '{model_name}'. Choose from: {list(REGISTRY.keys())}"
        )

    logger.info(f"Creating mother model: {model_name} (pretrained={pretrained})")
    return REGISTRY[model_name](pretrained=pretrained)
