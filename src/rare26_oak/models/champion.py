"""Champion model architectures: DINOv3-ConvNeXt, GastroNet-ResNet50, DINOv2-ViT-Base.

These are trained on RARE26 + pseudo-labeled GastroNet data with curriculum detox.
Their weights are used in the final ensemble for inference.
"""
import os
import math
from typing import Optional

import torch
import torch.nn as nn
import timm

from rare26_oak.utils import get_logger

logger = get_logger(__name__)


def _clean_ssl_state_dict(state_dict: dict) -> dict:
    """Strip common SSL/DINO prefixes from state dict keys."""
    if isinstance(state_dict, dict):
        for key in ["student", "teacher", "state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break

    clean = {}
    for k, v in state_dict.items():
        clean_k = (
            k.replace("module.", "")
            .replace("backbone.", "")
            .replace("teacher.", "")
            .replace("student.", "")
        )
        clean[clean_k] = v
    return clean


def _interpolate_pos_embed(pos_embed: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Interpolate ViT positional embeddings for resolution change."""
    if pos_embed.shape == target_shape:
        return pos_embed

    logger.info(f"Interpolating pos_embed from {pos_embed.shape} to {target_shape}")
    cls_token = pos_embed[:, 0:1, :]
    patch_tokens = pos_embed[:, 1:, :]
    dim = pos_embed.shape[-1]
    orig_size = int(math.sqrt(patch_tokens.shape[1]))
    target_size = int(math.sqrt(target_shape[1] - 1))

    patch_tokens = patch_tokens.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    patch_tokens = torch.nn.functional.interpolate(
        patch_tokens, size=(target_size, target_size), mode="bicubic", align_corners=False
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat((cls_token, patch_tokens), dim=1)


class ChampConvNextBase(nn.Module):
    """DINOv3-ConvNeXt-Base for champion training.

    Uses HuggingFace AutoModel for the backbone with a custom nonlinear head.
    Input: 512x512.
    """

    MODEL_NAME = "dinov3_convnext_base"
    IMG_SIZE = 512

    def __init__(self, hf_model_path: Optional[str] = None):
        super().__init__()
        from transformers import AutoModel

        model_id = hf_model_path or "facebook/dinov3-convnext-base-pretrain-lvd1689m"
        load_kwargs = {}
        if hf_model_path and os.path.isdir(hf_model_path):
            load_kwargs["local_files_only"] = True

        self.backbone = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, **load_kwargs
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        features = (
            out.pooler_output
            if hasattr(out, "pooler_output")
            else out.last_hidden_state[:, 0, :]
        )
        return self.head(features)


class ChampResNet50(nn.Module):
    """GastroNet-ResNet50 for champion training.

    Uses timm ResNet50 backbone with GastroNet SSL pre-trained weights
    and a custom LayerNorm + GELU + Dropout head.
    Input: 512x512.
    """

    MODEL_NAME = "gastronet_resnet50"
    IMG_SIZE = 512

    def __init__(self, gastronet_weights: Optional[str] = None):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)

        if gastronet_weights:
            self._load_gastronet_weights(gastronet_weights)

        self.backbone.requires_grad_(True)

        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def _load_gastronet_weights(self, path: str) -> None:
        logger.info(f"Loading GastroNet SSL weights from {os.path.basename(path)}")
        checkpoint = torch.load(path, map_location="cpu")
        clean_dict = _clean_ssl_state_dict(checkpoint)
        msg = self.backbone.load_state_dict(clean_dict, strict=False)
        logger.info(f"GastroNet weight load: {len(msg.missing_keys)} missing keys")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ChampViTBaseDINOv2(nn.Module):
    """DINOv2 ViT-Base for champion training.

    Uses PyTorch Hub dinov2_vitb14 backbone with GastroNet pre-trained weights
    and positional embedding interpolation for 518x518 input.
    Input: 518x518.
    """

    MODEL_NAME = "gastronet_vit_base"
    IMG_SIZE = 518

    def __init__(self, gastronet_weights: Optional[str] = None,
                 dinov2_repo_path: Optional[str] = None):
        super().__init__()
        if dinov2_repo_path and os.path.isdir(dinov2_repo_path):
            self.backbone = torch.hub.load(
                str(dinov2_repo_path), "dinov2_vitb14",
                source="local", pretrained=False,
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
            )

        if gastronet_weights:
            self._load_gastronet_weights(gastronet_weights)

        self.backbone.requires_grad_(True)

        self.head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def _load_gastronet_weights(self, path: str) -> None:
        logger.info(f"Loading GastroNet ViT weights from {os.path.basename(path)}")
        checkpoint = torch.load(path, map_location="cpu")
        clean_dict = _clean_ssl_state_dict(checkpoint)

        if "pos_embed" in clean_dict and hasattr(self.backbone, "pos_embed"):
            clean_dict["pos_embed"] = _interpolate_pos_embed(
                clean_dict["pos_embed"], self.backbone.pos_embed.shape
            )

        msg = self.backbone.load_state_dict(clean_dict, strict=False)
        logger.info(f"GastroNet ViT weight load: {len(msg.missing_keys)} missing keys")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def create_champion_model(model_name: str, **kwargs) -> nn.Module:
    """Factory: create a champion model by name.

    Args:
        model_name: One of 'dinov3_convnext_base', 'gastronet_resnet50', 'gastronet_vit_base'
    """
    REGISTRY = {
        "dinov3_convnext_base": ChampConvNextBase,
        "gastronet_resnet50": ChampResNet50,
        "gastronet_vit_base": ChampViTBaseDINOv2,
    }
    if model_name not in REGISTRY:
        raise ValueError(
            f"Unknown champion model '{model_name}'. Choose from: {list(REGISTRY.keys())}"
        )

    logger.info(f"Creating champion model: {model_name}")
    return REGISTRY[model_name](**kwargs)
