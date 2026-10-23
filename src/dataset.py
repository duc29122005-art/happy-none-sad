"""
dataset.py - PyTorch Dataset and DataLoader factory for emotion classification.

Directory structure expected:
    dataset/
    ├── train/
    │   ├── happy/   ← face crop images
    │   ├── sad/
    │   └── none/
    ├── val/
    │   ├── happy/
    │   ├── sad/
    │   └── none/
    └── test/
        ├── happy/
        ├── sad/
        └── none/

The class index order is determined alphabetically by torchvision's
ImageFolder (happy=0, none=1, sad=2), which matches CLASS_NAMES in config.py.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Heavy augmentation for the training split.

    Includes random flips, colour jitter, affine transforms and
    normalisation with ImageNet mean/std (required by EfficientNet).
    """
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """
    Minimal, deterministic transform for val / test / inference.

    No random augmentation — only resize and normalise.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    root: Path,
    split: str,
    image_size: int = 224,
) -> datasets.ImageFolder:
    """
    Load an ImageFolder dataset for the given split.

    Args:
        root:       Root dataset directory (contains train / val / test).
        split:      One of 'train', 'val', 'test'.
        image_size: Spatial size for the transform.

    Returns:
        An ImageFolder dataset with the appropriate transform applied.

    Raises:
        FileNotFoundError: If the split directory does not exist.
    """
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    transform = (
        build_train_transform(image_size)
        if split == "train"
        else build_eval_transform(image_size)
    )

    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    logger.info(
        f"[{split}] Loaded {len(dataset)} samples | "
        f"classes={dataset.classes}"
    )
    return dataset


def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class-imbalanced datasets.

    Each sample gets a weight inversely proportional to its class frequency,
    so minority classes are upsampled during training.

    Args:
        dataset: An ImageFolder dataset whose targets we inspect.

    Returns:
        A WeightedRandomSampler covering all samples in the dataset.
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / (class_counts + 1e-6)   # avoid divide-by-zero
    sample_weights = class_weights[targets]

    logger.info(f"Class counts: {dict(zip(dataset.classes, class_counts.tolist()))}")
    logger.info(f"Class weights: {dict(zip(dataset.classes, class_weights.tolist()))}")

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_dir: Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the dataset directory.

    Args:
        dataset_dir:          Root dataset directory.
        image_size:           Spatial size for transforms.
        batch_size:           Mini-batch size for all splits.
        num_workers:          Worker processes for data loading.
        use_weighted_sampler: If True, oversample minority classes in train.
        pin_memory:           If True, pin CPU memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = load_dataset(dataset_dir, "train", image_size)
    val_ds   = load_dataset(dataset_dir, "val",   image_size)
    test_ds  = load_dataset(dataset_dir, "test",  image_size)

    # Optional: weighted sampler to handle imbalanced classes
    train_sampler: Optional[WeightedRandomSampler] = None
    shuffle_train = True
    if use_weighted_sampler and len(train_ds) > 0:
        train_sampler = build_weighted_sampler(train_ds)
        shuffle_train = False  # sampler is mutually exclusive with shuffle

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,       # avoid partial batches that confuse BatchNorm
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        f"DataLoaders ready — "
        f"train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches | "
        f"test={len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Inference preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_face_for_inference(
    face_bgr: "np.ndarray",
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert a raw BGR face crop (from OpenCV) to a normalised model input.

    This function mirrors the eval transform but operates directly on a
    numpy array without requiring PIL, making it faster in the video loop.

    Args:
        face_bgr:   Face crop in BGR uint8 (H × W × 3).
        image_size: Target spatial size.

    Returns:
        Float tensor of shape (1, 3, image_size, image_size), range [0, 1]
        normalised with ImageNet statistics.
    """
    import cv2
    import numpy as np

    # BGR → RGB, resize
    rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # HWC → CHW, [0,255] → [0,1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)   # (1, 3, H, W)