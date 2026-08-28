"""Centralized configuration for the RARE26-Oak training pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """All filesystem paths used by the pipeline."""
    # Data paths
    rare26_data_dir: Path = field(default_factory=lambda: Path("data/rare26"))
    gastronet_data_dir: Path = field(default_factory=lambda: Path("data/gastronet"))
    pseudo_labels_csv: Path = field(default_factory=lambda: Path("outputs/pseudo_labels/pseudo_labels.csv"))
    fake_neo_dir: Optional[Path] = None

    # Pre-trained weights
    gastronet_resnet_weights: Optional[Path] = None
    dinov2_weights: Optional[Path] = None

    # HuggingFace local models (for offline inference)
    hf_convnext_path: Optional[Path] = None
    dinov2_repo_path: Optional[Path] = None

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    mother_weights_dir: Path = field(default_factory=lambda: Path("outputs/mother_weights"))
    champ_weights_dir: Path = field(default_factory=lambda: Path("outputs/champ_weights"))
    pseudo_labels_dir: Path = field(default_factory=lambda: Path("outputs/pseudo_labels"))


@dataclass
class MotherTrainingConfig:
    """Config for mother model training (ConvNext-Tiny, ConvNext-Base, EfficientNetV2)."""
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    img_size: int = 384
    mixup_alpha: float = 0.2
    focal_gamma: float = 2.0
    num_workers: int = 2
    accumulation_steps: int = 1


@dataclass
class ChampTrainingConfig:
    """Config for champion model training."""
    epochs: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 8
    img_size: int = 512
    lr_backbone: float = 5e-6
    lr_head: float = 2e-5
    weight_decay: float = 1e-2
    detox_epoch: int = 8
    swa_start: int = 14
    ema_decay: float = 0.999
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    warmup_epochs: int = 3
    patience: int = 8


@dataclass
class PseudoLabelConfig:
    """Config for pseudo-label generation."""
    batch_size_384: int = 32
    batch_size_512: int = 16
    threshold_negative: float = 0.15
    threshold_positive: float = 0.85
    ensemble_weight_base: float = 0.50
    ensemble_weight_tiny: float = 0.30
    ensemble_weight_eff: float = 0.20
    num_workers: int = 2


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    mother: MotherTrainingConfig = field(default_factory=MotherTrainingConfig)
    champ: ChampTrainingConfig = field(default_factory=ChampTrainingConfig)
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)

    # Global settings
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    debug: bool = False  # When True, use minimal iterations for pipeline testing

    # Debug overrides
    debug_epochs: int = 1
    debug_max_samples: int = 16

    def resolve_device(self) -> str:
        """Resolve 'auto' device to cuda/cpu based on availability."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def get_mother_epochs(self) -> int:
        """Return epoch count (1 in debug mode)."""
        return self.debug_epochs if self.debug else self.mother.epochs

    def get_champ_epochs(self) -> int:
        """Return epoch count (1 in debug mode)."""
        return self.debug_epochs if self.debug else self.champ.epochs
