"""
emotion_model.py - EfficientNet-B0 emotion classifier (happy / none / sad).

Architecture:
    - Backbone : EfficientNet-B0 loaded via timm (ImageNet pretrained)
    - Head     : Dropout → Linear(num_features → num_classes)
    - Loss     : CrossEntropyLoss  (defined in train_model.py)
    - Output   : logits of shape (B, num_classes)

Usage:
    model = EmotionModel(num_classes=3)
    logits = model(images)              # images: (B, 3, 224, 224) float32
    probs  = torch.softmax(logits, -1)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmotionModel(nn.Module):
    """
    Lightweight emotion classifier built on an EfficientNet-B0 backbone.

    The original classifier head is replaced with a task-specific head that
    outputs logits for (happy, none, sad).

    Args:
        num_classes:   Number of emotion categories.
        model_name:    timm model identifier (default: 'efficientnet_b0').
        pretrained:    Whether to load ImageNet pretrained weights.
        drop_rate:     Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 3,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Load backbone via timm ──────────────────────────────────────
        # num_classes=0 removes the original classification head so we
        # receive raw feature embeddings from the global average pool.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # Strip the original head
            drop_rate=drop_rate,
        )

        # Infer the backbone's output feature dimension
        num_features: int = self.backbone.num_features

        # ── Custom classification head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_features, num_classes),
        )

        logger.info(
            f"EmotionModel created | backbone={model_name} | "
            f"features={num_features} | classes={num_classes} | "
            f"pretrained={pretrained}"
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)       # (B, num_features)
        logits   = self.classifier(features)  # (B, num_classes)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: EmotionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    save_path: Path,
) -> None:
    """
    Save model + optimizer state to a checkpoint file.

    Args:
        model:     The emotion model instance.
        optimizer: The optimizer whose state we want to persist.
        epoch:     Current training epoch (0-indexed).
        val_loss:  Validation loss at this epoch.
        val_acc:   Validation accuracy at this epoch.
        save_path: Full .pth file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    logger.info(f"Checkpoint saved → {save_path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: EmotionModel,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """
    Load a checkpoint into the model (and optionally the optimizer).

    Args:
        model:           The emotion model instance.
        checkpoint_path: Path to the .pth checkpoint file.
        optimizer:       If provided, restore its state as well.
        device:          Target device string ('cuda' or 'cpu').

    Returns:
        The epoch number stored in the checkpoint (useful for resuming).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch    = ckpt.get("epoch", 0)
    val_acc  = ckpt.get("val_acc", 0.0)
    val_loss = ckpt.get("val_loss", float("inf"))

    logger.info(
        f"Checkpoint loaded ← {checkpoint_path} "
        f"(epoch={epoch}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f})"
    )
    return epoch


def export_torchscript(
    model: EmotionModel,
    export_path: Path,
    image_size: int = 224,
    device: str = "cpu",
) -> None:
    """
    Export the model to TorchScript format for deployment.

    Args:
        model:       Trained EmotionModel instance.
        export_path: Output .pt file path.
        image_size:  Spatial size of the expected input (default 224).
        device:      Device to run the trace on.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(export_path))

    logger.info(f"TorchScript model exported → {export_path}")