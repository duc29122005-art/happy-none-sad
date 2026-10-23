"""
main.py - Top-level CLI entry point for the emotion AI pipeline.

Subcommands:
    build   → src/build_dataset.py  (extract frames + faces from raw videos)
    train   → src/train_model.py    (train EfficientNet emotion classifier)
    infer   → src/inference_video.py (run full pipeline on a video)
    export  → export best checkpoint to TorchScript

Usage:
    python main.py build --capture_every_n_frames 30
    python main.py train --epochs 30 --batch_size 32
    python main.py --input data/raw_videos/test.mp4 --output data/outputs/test_out.mp4
    python main.py infer --input data/raw_videos/test.mp4
    python main.py export
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure src package is importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    RAW_VIDEOS_DIR,
    FRAMES_DIR,
    FACES_DIR,
    DATASET_DIR,
    TRAIN_DIR,
    CHECKPOINTS_DIR,
    EXPORTS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    CAPTURE_EVERY_N_FRAMES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    MIN_FACE_SIZE,
    BLUR_THRESHOLD,
    EMOTION_SMOOTH_WINDOW,
    DETR_SKIP_FRAMES,
)
from src.utils import ensure_dirs
from src.build_dataset import (
    DEFAULT_EMOTION_LABEL_MODEL,
    DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
)
from src.multiview_face.cli import add_multiview_parser, run_from_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def find_default_input_source(videos_dir: Path = RAW_VIDEOS_DIR) -> str:
    """Return the first video in data/raw_videos for default inference."""
    video_files = sorted(
        path for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        logger.error(
            "No input video provided and no videos found in %s. "
            "Pass --input path/to/video.mp4 or put a video in data/raw_videos.",
            videos_dir,
        )
        sys.exit(1)
    return str(video_files[0])


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> None:
    """Build face dataset from raw videos."""
    from src.build_dataset import build_dataset
    build_dataset(
        videos_dir=args.videos_dir,
        frames_dir=args.frames_dir,
        faces_dir=args.faces_dir,
        train_dir=args.train_dir,
        capture_every_n_frames=(
            args.fps if args.fps is not None else args.capture_every_n_frames
        ),
        min_face_size=args.min_face_size,
        blur_threshold=args.blur_threshold,
        target_size=args.image_size,
        emotion_model=args.emotion_model,
        label_confidence_threshold=args.label_confidence_threshold,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train the emotion classifier."""
    from src.train_model import train
    train(
        dataset_dir         = args.dataset_dir,
        checkpoints_dir     = args.checkpoints_dir,
        exports_dir         = args.exports_dir,
        device              = args.device,
        image_size          = args.image_size,
        batch_size          = args.batch_size,
        num_epochs          = args.epochs,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        num_workers         = args.num_workers,
        label_smoothing     = args.label_smoothing,
        early_stop_patience = args.early_stop,
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """Run real-time emotion inference on a video."""
    from src.inference_video import run_inference
    input_source = args.input or find_default_input_source()
    output_path = args.output or (OUTPUTS_DIR / "output.mp4")
    run_inference(
        input_source     = input_source,
        output_path      = output_path,
        checkpoint_path  = args.checkpoint,
        device           = args.device,
        image_size       = args.image_size,
        smooth_window    = args.smooth_window,
        detr_skip        = args.detr_skip,
        display          = not args.no_display,
        use_segmentation = not args.no_segmentation,
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export the best checkpoint to TorchScript."""
    from src.emotion_model import EmotionModel, load_checkpoint, export_torchscript
    from src.config import NUM_CLASSES, EMOTION_MODEL_NAME

    ckpt_path   = args.checkpoint
    export_path = args.exports_dir / "emotion_model.pt"

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = EmotionModel(
        num_classes=NUM_CLASSES,
        model_name=EMOTION_MODEL_NAME,
        pretrained=False,
    ).to(args.device)

    load_checkpoint(model, ckpt_path, device=args.device)
    export_torchscript(model, export_path, image_size=args.image_size, device=args.device)
    logger.info(f"TorchScript model exported → {export_path}")


def cmd_multiview(args: argparse.Namespace) -> None:
    """Run stereo face frontalization and synthetic multi-angle export."""
    stats = run_from_args(args)
    logger.info(f"Multi-view export complete: {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Emotion AI Pipeline — defaults to inference when no subcommand is provided."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Default inference input video path or webcam index.")
    parser.add_argument("--output", "-o", type=Path, default=OUTPUTS_DIR / "output.mp4",
                        help="Default inference output MP4 path.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best_model.pth",
                        help="Default inference checkpoint path.")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"],
                        help="Default inference device.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Default inference face crop size.")
    parser.add_argument("--smooth_window", type=int, default=EMOTION_SMOOTH_WINDOW,
                        help="Default inference smoothing window.")
    parser.add_argument("--detr_skip", type=int, default=DETR_SKIP_FRAMES,
                        help="Default inference DETR skip interval.")
    parser.add_argument("--no_display", action="store_true",
                        help="Disable live preview window for default inference.")
    parser.add_argument("--no_segmentation", action="store_true",
                        help="Disable DETR segmentation for default inference.")

    subparsers = parser.add_subparsers(dest="command")

    # ── build ────────────────────────────────────────────────────────────────
    p_build = subparsers.add_parser(
        "build",
        help="Extract frames from raw videos and detect/save face crops.",
    )
    p_build.add_argument("--videos_dir",     type=Path,  default=RAW_VIDEOS_DIR)
    p_build.add_argument("--frames_dir",     type=Path,  default=FRAMES_DIR)
    p_build.add_argument("--faces_dir",      type=Path,  default=FACES_DIR)
    p_build.add_argument("--train_dir",      type=Path,  default=TRAIN_DIR)
    p_build.add_argument("--capture_every_n_frames", type=int, default=CAPTURE_EVERY_N_FRAMES)
    p_build.add_argument("--fps", type=int, default=None,
                         help="Deprecated alias for --capture_every_n_frames.")
    p_build.add_argument("--min_face_size",  type=int,   default=MIN_FACE_SIZE)
    p_build.add_argument("--blur_threshold", type=float, default=BLUR_THRESHOLD)
    p_build.add_argument("--image_size",     type=int,   default=IMAGE_SIZE)
    p_build.add_argument("--emotion_model", type=str, default=DEFAULT_EMOTION_LABEL_MODEL)
    p_build.add_argument("--label_confidence_threshold", type=float,
                         default=DEFAULT_LABEL_CONFIDENCE_THRESHOLD)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser(
        "train",
        help="Train the EfficientNet emotion classifier.",
    )
    p_train.add_argument("--dataset_dir",     type=Path,  default=DATASET_DIR)
    p_train.add_argument("--checkpoints_dir", type=Path,  default=CHECKPOINTS_DIR)
    p_train.add_argument("--exports_dir",     type=Path,  default=EXPORTS_DIR)
    p_train.add_argument("--device",          type=str,   default=DEVICE)
    p_train.add_argument("--image_size",      type=int,   default=IMAGE_SIZE)
    p_train.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    p_train.add_argument("--epochs",          type=int,   default=NUM_EPOCHS)
    p_train.add_argument("--lr",              type=float, default=LEARNING_RATE)
    p_train.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    p_train.add_argument("--num_workers",     type=int,   default=NUM_WORKERS)
    p_train.add_argument("--label_smoothing", type=float, default=0.1)
    p_train.add_argument("--early_stop",      type=int,   default=7)

    # ── infer ────────────────────────────────────────────────────────────────
    p_infer = subparsers.add_parser(
        "infer",
        help="Run real-time emotion inference on a video file or webcam.",
    )
    p_infer.add_argument("--input", "-i",     type=str,   default=None,
                         help="Video file path or webcam index (e.g. 0).")
    p_infer.add_argument("--output", "-o",    type=Path,
                         default=OUTPUTS_DIR / "output.mp4")
    p_infer.add_argument("--checkpoint",      type=Path,
                         default=CHECKPOINTS_DIR / "best_model.pth")
    p_infer.add_argument("--device",          type=str,   default=DEVICE)
    p_infer.add_argument("--image_size",      type=int,   default=IMAGE_SIZE)
    p_infer.add_argument("--smooth_window",   type=int,   default=EMOTION_SMOOTH_WINDOW)
    p_infer.add_argument("--detr_skip",       type=int,   default=DETR_SKIP_FRAMES)
    p_infer.add_argument("--no_display",      action="store_true",
                         help="Disable live preview window.")
    p_infer.add_argument("--no_segmentation", action="store_true",
                         help="Disable DETR segmentation (faster).")

    # ── export ───────────────────────────────────────────────────────────────
    p_export = subparsers.add_parser(
        "export",
        help="Export trained checkpoint to TorchScript format.",
    )
    p_export.add_argument("--checkpoint",  type=Path, default=CHECKPOINTS_DIR / "best_model.pth")
    p_export.add_argument("--exports_dir", type=Path, default=EXPORTS_DIR)
    p_export.add_argument("--device",      type=str,  default=DEVICE)
    p_export.add_argument("--image_size",  type=int,  default=IMAGE_SIZE)

    # multiview
    add_multiview_parser(subparsers)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure all output directories exist on startup
    ensure_dirs(
        RAW_VIDEOS_DIR,
        FRAMES_DIR,
        FACES_DIR,
        DATASET_DIR / "train",
        DATASET_DIR / "val",
        DATASET_DIR / "test",
        OUTPUTS_DIR,
        CHECKPOINTS_DIR,
        EXPORTS_DIR,
    )

    parser = build_parser()
    args   = parser.parse_args()
    if args.command is None:
        args.command = "infer"

    command_map = {
        "build":  cmd_build,
        "train":  cmd_train,
        "infer":  cmd_infer,
        "export": cmd_export,
        "multiview": cmd_multiview,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Running command: {args.command}")
    handler(args)


if __name__ == "__main__":
    main()
