"""Image transforms for training, validation, and inference."""
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_mother_train_transform(img_size: int = 384) -> transforms.Compose:
    """Training transforms for mother models (ConvNext-Tiny, EfficientNetV2).

    Uses resize-to-larger then random crop, flips, rotation, color jitter.
    """
    crop_pad = max(16, img_size // 24)
    return transforms.Compose([
        transforms.Resize((img_size + crop_pad, img_size + crop_pad)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_champ_train_transform(img_size: int = 512) -> transforms.Compose:
    """Training transforms for champion models (ResNet50, ViT).

    Heavy spatial augmentation (rotation 180, affine).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_champ_convnext_train_transform(img_size: int = 512) -> transforms.Compose:
    """Training transforms for DINOv3-ConvNeXt champion (most aggressive).

    Includes GaussianBlur and RandomErasing.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.12)
        ], p=0.6),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
    ])


def get_val_transform(img_size: int = 384) -> transforms.Compose:
    """Validation/inference transforms (resize + normalize only)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_pseudo_label_transform(img_size: int) -> transforms.Compose:
    """Transforms for pseudo-label inference (same as val)."""
    return get_val_transform(img_size)
