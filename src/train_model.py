"""
train_model.py - Training loop for the EfficientNet emotion classifier.

Features:
    • CrossEntropyLoss  with optional label smoothing
    • Adam optimiser with weight decay
    • CosineAnnealingLR scheduler for smooth LR decay
    • Mixed precision training (torch.cuda.amp) when CUDA is available
    • Best model checkpoint saved by validation accuracy
    • Early stopping to prevent overfitting
    • Epoch-level metrics: loss, accuracy, per-class accuracy
    • GPU memory cleared between epochs to avoid OOM

Usage:
    python src/train_model.py
    python src/train_model.py --epochs 50 --lr 5e-5 --batch_size 64
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── Make src importable when run directly ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATASET_DIR,
    CHECKPOINTS_DIR,
    EXPORTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    NUM_CLASSES,
    CLASS_NAMES,
    EMOTION_MODEL_NAME,
)
from src.emotion_model import EmotionModel, save_checkpoint, export_torchscript
from src.dataset import build_dataloaders
from src.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Return top-1 accuracy for a single batch."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    is_train: bool,
) -> Tuple[float, float]:
    """
    Run one full epoch (train or eval).

    Args:
        model:    The emotion model.
        loader:   DataLoader for this split.
        criterion:Loss function.
        optimizer:Optimiser (only used when is_train=True).
        scaler:   AMP gradient scaler (only active on CUDA).
        device:   Target device string.
        is_train: If True, perform backprop and parameter updates.

    Returns:
        Tuple of (mean_loss, mean_accuracy) for the epoch.
    """
    model.train(is_train)

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Forward (AMP) ─────────────────────────────────────────────────
            with autocast(enabled=(device == "cuda")):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, labels)
            n_batches  += 1

    if n_batches == 0:
        return 0.0, 0.0

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Per-class evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    class_names: list,
    device: str,
) -> Dict[str, float]:
    """
    Compute per-class accuracy on an evaluation loader.

    Returns:
        Dict mapping class_name → accuracy (0–1).
    """
    model.eval()
    n_classes  = len(class_names)
    correct    = [0] * n_classes
    totals     = [0] * n_classes

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device == "cuda")):
            preds = model(images).argmax(dim=1)

        for cls_idx in range(n_classes):
            mask = labels == cls_idx
            correct[cls_idx] += (preds[mask] == cls_idx).sum().item()
            totals[cls_idx]  += mask.sum().item()

    return {
        class_names[i]: (correct[i] / totals[i] if totals[i] > 0 else 0.0)
        for i in range(n_classes)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def train(
    dataset_dir: Path,
    checkpoints_dir: Path,
    exports_dir: Path,
    device: str,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    label_smoothing: float,
    early_stop_patience: int,
) -> None:
    """
    Full training run.

    Args:
        dataset_dir:         Root dataset directory (train/val/test inside).
        checkpoints_dir:     Directory to save .pth checkpoints.
        exports_dir:         Directory to save exported models.
        device:              'cuda' or 'cpu'.
        image_size:          Spatial size for model input.
        batch_size:          Mini-batch size.
        num_epochs:          Maximum training epochs.
        lr:                  Initial learning rate for Adam.
        weight_decay:        L2 regularisation factor.
        num_workers:         DataLoader worker count.
        label_smoothing:     CrossEntropy label smoothing (0 = none).
        early_stop_patience: Stop training if val_acc does not improve
                             for this many consecutive epochs.
    """
    ensure_dirs(checkpoints_dir, exports_dir)
    logger.info(f"Training on device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = EmotionModel(
        num_classes=NUM_CLASSES,
        model_name=EMOTION_MODEL_NAME,
        pretrained=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: total={total_params:,} | trainable={trainable:,}")

    # ── Loss, Optimiser, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 1e-3
    )
    scaler = GradScaler(enabled=(device == "cuda"))

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc      = 0.0
    early_stop_count  = 0
    best_ckpt_path    = checkpoints_dir / "best_model.pth"

    logger.info(f"\n{'═' * 65}")
    logger.info(f"Starting training for up to {num_epochs} epochs")
    logger.info(f"{'═' * 65}")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, is_train=True
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, is_train=False
        )

        scheduler.step()
        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch [{epoch + 1:03d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, best_ckpt_path
            )
            logger.info(f"  ★ New best val_acc={best_val_acc:.4f} — checkpoint saved")
        else:
            early_stop_count += 1
            logger.info(
                f"  No improvement ({early_stop_count}/{early_stop_patience})"
            )

        # ── Early stopping ─────────────────────────────────────────────────────
        if early_stop_count >= early_stop_patience:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no improvement for {early_stop_patience} epochs)."
            )
            break

        # ── Free GPU cache between epochs ─────────────────────────────────────
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Final evaluation on test set ─────────────────────────────────────────
    logger.info(f"\n{'═' * 65}")
    logger.info("Loading best checkpoint for test evaluation …")

    from src.emotion_model import load_checkpoint
    load_checkpoint(model, best_ckpt_path, device=device)

    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer, scaler, device, is_train=False
    )
    logger.info(f"Test — loss={test_loss:.4f} | acc={test_acc:.4f}")

    per_class = evaluate_per_class(model, test_loader, CLASS_NAMES, device)
    for cls_name, cls_acc in per_class.items():
        logger.info(f"  {cls_name:>8s} accuracy: {cls_acc:.4f}")

    # ── Export TorchScript ────────────────────────────────────────────────────
    ts_path = exports_dir / "emotion_model.pt"
    export_torchscript(model, ts_path, image_size=image_size, device=device)
    logger.info(f"TorchScript model exported → {ts_path}")
    logger.info(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet emotion classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",     type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoints_dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--exports_dir",     type=Path, default=EXPORTS_DIR)
    parser.add_argument("--device",          type=str,  default=DEVICE)
    parser.add_argument("--image_size",      type=int,  default=IMAGE_SIZE)
    parser.add_argument("--batch_size",      type=int,  default=BATCH_SIZE)
    parser.add_argument("--epochs",          type=int,  default=NUM_EPOCHS)
    parser.add_argument("--lr",              type=float,default=LEARNING_RATE)
    parser.add_argument("--weight_decay",    type=float,default=WEIGHT_DECAY)
    parser.add_argument("--num_workers",     type=int,  default=NUM_WORKERS)
    parser.add_argument("--label_smoothing", type=float,default=0.1,
                        help="CrossEntropy label smoothing factor.")
    parser.add_argument("--early_stop",      type=int,  default=7,
                        help="Early stopping patience in epochs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_dir          = args.dataset_dir,
        checkpoints_dir      = args.checkpoints_dir,
        exports_dir          = args.exports_dir,
        device               = args.device,
        image_size           = args.image_size,
        batch_size           = args.batch_size,
        num_epochs           = args.epochs,
        lr                   = args.lr,
        weight_decay         = args.weight_decay,
        num_workers          = args.num_workers,
        label_smoothing      = args.label_smoothing,
        early_stop_patience  = args.early_stop,
    )