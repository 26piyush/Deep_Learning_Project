"""Stage 1 — Preprocessing Pipeline for Crop Disease Detection.
Handles corrupted image validation, RGB conversion, augmentations, and normalization.
"""

from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageFile
import torch
from torchvision import transforms

from src.config import NORM_MEAN, NORM_STD, VIT_IMAGE_SIZE

# ImageNet statistics compatibility constants
IMAGENET_MEAN = NORM_MEAN
IMAGENET_STD = NORM_STD

# Enable tolerant image loading for truncated images in field datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True


def validate_image(image_path: Path) -> Tuple[bool, str]:
    """Check if an image file is present, readable, and uncorrupted.
    
    Returns:
        (is_valid, error_message or "")
    """
    path = Path(image_path)
    if not path.exists():
        return False, f"File does not exist: {path}"
    if path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {path}"

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid image dimensions: {w}x{h}"
        return True, ""
    except Exception as e:
        return False, f"Corrupted image ({type(e).__name__}): {e}"


def convert_to_rgb(image) -> Image.Image:
    """Convert any PIL Image or NumPy array to 3-channel RGB PIL Image."""
    import numpy as np
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            return Image.fromarray(image[:, :, :3]).convert("RGB")
        else:
            return Image.fromarray(image).convert("RGB")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported type: {type(image)}")
    return image.convert("RGB")


def validate_and_open_image(image_path: Path) -> Optional[Image.Image]:
    """Validate image file integrity and convert to RGB format.
    
    Returns:
        PIL.Image in RGB format if valid, None if corrupt/unreadable.
    """
    image_path = Path(image_path)
    if not image_path.exists() or not image_path.is_file():
        return None

    try:
        with Image.open(image_path) as img:
            img.verify()  # Check for corruption without full decode
        
        # Re-open after verify to get image data
        with Image.open(image_path) as img:
            rgb_image = img.convert("RGB")
            # Load pixel data into memory so file can safely close
            rgb_image.load()
            return rgb_image
    except Exception:
        return None


def get_vit_transforms(
    image_size: int = VIT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = tuple(NORM_MEAN),
    std: Tuple[float, float, float] = tuple(NORM_STD),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build PyTorch transformation pipelines for ViT.
    
    Returns:
        (train_transforms, val_transforms)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # ~256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def get_severity_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = tuple(NORM_MEAN),
    std: Tuple[float, float, float] = tuple(NORM_STD),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build PyTorch transformation pipelines for Severity model."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform
