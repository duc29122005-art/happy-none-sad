"""
inference_video.py - Real-time emotion inference pipeline on video.

Full pipeline per frame:
    1. Read frame from video / webcam.
    2. Every DETR_SKIP_FRAMES: run DETR panoptic segmentation to get person mask.
    3. Run InsightFace to detect faces.
    4. For each detected face:
        a. Crop + preprocess the face.
        b. Run EmotionModel → predicted class + confidence.
        c. Update the per-face sliding-window emotion history.
        d. Resolve dominant smoothed emotion.
        e. Draw bounding box + emotion label on frame.
    5. Overlay person mask on frame.
    6. Write annotated frame to output video.
    7. Display live preview (optional).

Usage:
    python src/inference_video.py --input data/raw_videos/my_video.mp4
    python src/inference_video.py --input 0            # webcam
    python src/inference_video.py --input video.mp4 --no_display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

# ── Ensure src is importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    CLASS_NAMES,
    NUM_CLASSES,
    INSIGHTFACE_MODEL,
    ONNX_PROVIDERS,
    DETR_MODEL_NAME,
    EMOTION_SMOOTH_WINDOW,
    DETR_SKIP_FRAMES,
    EMOTION_COLORS,
    MIN_FACE_SIZE,
    BLUR_THRESHOLD,
)
from src.emotion_model import EmotionModel, load_checkpoint
from src.segmentation_model import SegmentationModel
from src.dataset import preprocess_face_for_inference
from src.utils import (
    ensure_dir,
    valid_face_size,
    is_blurry,
    crop_face_from_bbox,
    update_emotion_history,
    get_dominant_emotion,
    draw_label_box,
    overlay_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_emotion_model(
    checkpoint_path: Path,
    device: str,
) -> EmotionModel:
    """Load trained EmotionModel from checkpoint."""
    model = EmotionModel(
        num_classes=NUM_CLASSES,
        model_name="efficientnet_b0",
        pretrained=False,      # weights come from checkpoint
    ).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    logger.info(f"EmotionModel loaded from {checkpoint_path}")
    return model


def load_face_detector(providers: List[str]):
    """Initialise InsightFace FaceAnalysis."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.error("insightface is not installed. Run: pip install insightface")
        sys.exit(1)

    detector = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    ctx_id = 0 if "CUDA" in providers[0] else -1
    detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
    logger.info(f"InsightFace ready | providers={providers}")
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# Emotion inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_emotion(
    model: EmotionModel,
    face_bgr: np.ndarray,
    device: str,
    image_size: int,
) -> Tuple[str, float]:
    """
    Classify the emotion in a face crop.

    Args:
        model:      Trained EmotionModel.
        face_bgr:   Face image in BGR uint8.
        device:     'cuda' or 'cpu'.
        image_size: Expected model input size.

    Returns:
        Tuple of (emotion_label, confidence).
    """
    tensor = preprocess_face_for_inference(face_bgr, image_size).to(device)

    with autocast(enabled=(device == "cuda")):
        logits = model(tensor)                    # (1, num_classes)

    probs = torch.softmax(logits, dim=1).squeeze()  # (num_classes,)
    cls_idx = int(probs.argmax().item())
    conf    = float(probs[cls_idx].item())
    label   = CLASS_NAMES[cls_idx]

    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# Video writer helper
# ─────────────────────────────────────────────────────────────────────────────

def make_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter with H.264 codec."""
    ensure_dir(output_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")
    return writer


# ─────────────────────────────────────────────────────────────────────────────
# HUD rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Render an FPS counter in the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_frame_counter(
    frame: np.ndarray,
    current: int,
    total: Optional[int],
) -> np.ndarray:
    """Render current / total frame counter."""
    text = f"Frame: {current}" if total is None else f"Frame: {current}/{total}"
    h = frame.shape[0]
    cv2.putText(
        frame,
        text,
        (12, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    input_source: str,
    output_path: Path,
    checkpoint_path: Path,
    device: str,
    image_size: int,
    smooth_window: int,
    detr_skip: int,
    display: bool,
    use_segmentation: bool,
) -> None:
    """
    Run the full real-time emotion inference pipeline on a video.

    Args:
        input_source:     Video file path or webcam index (e.g. '0').
        output_path:      Annotated output MP4 path.
        checkpoint_path:  Path to the emotion model .pth checkpoint.
        device:           'cuda' or 'cpu'.
        image_size:       Emotion model input size.
        smooth_window:    Sliding-window size for emotion smoothing.
        detr_skip:        Run DETR every N frames (higher = faster).
        display:          Show live preview window.
        use_segmentation: Whether to run DETR segmentation at all.
    """
    # ── Open video source ─────────────────────────────────────────────────────
    source = int(input_source) if input_source.isdigit() else input_source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {input_source}")
        sys.exit(1)

    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames= total_frames if total_frames > 0 else None

    logger.info(
        f"Input : {input_source} | {width}×{height} | "
        f"{video_fps:.1f} fps | frames={total_frames or '∞'}"
    )

    writer = make_video_writer(output_path, width, height, video_fps)
    logger.info(f"Output: {output_path}")

    # ── Load models ───────────────────────────────────────────────────────────
    emotion_model = load_emotion_model(checkpoint_path, device)
    face_detector = load_face_detector(ONNX_PROVIDERS)

    seg_model: Optional[SegmentationModel] = None
    if use_segmentation:
        seg_model = SegmentationModel(
            model_name=DETR_MODEL_NAME,
            device=device,
        )

    # ── State: per-face sliding-window emotion history ─────────────────────────
    # Key: face index (by detection order); resets each frame.
    # We use a defaultdict so new faces automatically get an empty deque.
    emotion_histories: Dict[int, deque] = defaultdict(
        lambda: deque(maxlen=smooth_window)
    )

    # Current person mask (reused across DETR-skip frames)
    person_mask: Optional[np.ndarray] = None

    # Timing
    frame_idx   = 0
    fps_counter = 0
    t_fps_start = time.time()
    display_fps = 0.0

    logger.info("Pipeline started — press Q to quit.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        annotated = frame.copy()

        # ── DETR segmentation (every detr_skip frames) ─────────────────────
        if use_segmentation and seg_model is not None:
            if frame_idx % detr_skip == 0:
                try:
                    person_mask = seg_model.get_person_mask(frame)
                except Exception as e:
                    logger.debug(f"DETR error on frame {frame_idx}: {e}")
                    person_mask = None

            # Overlay previously computed mask
            if person_mask is not None:
                annotated = seg_model.overlay_on_frame(annotated, person_mask)

        # ── Face detection ──────────────────────────────────────────────────
        try:
            faces = face_detector.get(frame)
        except Exception as e:
            logger.debug(f"InsightFace error on frame {frame_idx}: {e}")
            faces = []

        for face_idx, face in enumerate(faces):
            bbox = face.bbox  # float32 [x1, y1, x2, y2]

            # ── Crop face ────────────────────────────────────────────────────
            face_crop = crop_face_from_bbox(frame, bbox, padding=0.10)
            if face_crop is None:
                continue

            # ── Quality filters ──────────────────────────────────────────────
            if not valid_face_size(face_crop, MIN_FACE_SIZE):
                continue
            if is_blurry(face_crop, BLUR_THRESHOLD):
                continue

            # ── Emotion classification ───────────────────────────────────────
            try:
                raw_emotion, conf = classify_emotion(
                    emotion_model, face_crop, device, image_size
                )
            except Exception as e:
                logger.debug(f"Emotion classification error: {e}")
                raw_emotion, conf = "none", 0.0

            # ── Smooth emotion via sliding window ────────────────────────────
            history = emotion_histories[face_idx]
            update_emotion_history(history, raw_emotion)
            smoothed_emotion = get_dominant_emotion(history, CLASS_NAMES)

            # ── Draw annotation ──────────────────────────────────────────────
            color = EMOTION_COLORS.get(smoothed_emotion, (200, 200, 200))
            draw_label_box(
                annotated,
                bbox,
                label=smoothed_emotion,
                color=color,
                conf=conf,
            )

            # Also draw face detection confidence from InsightFace
            det_score = getattr(face, "det_score", None)
            if det_score is not None:
                x1, y1 = int(bbox[0]), int(bbox[1])
                cv2.putText(
                    annotated,
                    f"det:{det_score:.2f}",
                    (x1, max(0, y1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        # ── HUD overlays ────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - t_fps_start
        if elapsed >= 1.0:
            display_fps   = fps_counter / elapsed
            fps_counter   = 0
            t_fps_start   = time.time()

        draw_fps(annotated, display_fps)
        draw_frame_counter(annotated, frame_idx, total_frames)

        # ── Write & display ─────────────────────────────────────────────────
        writer.write(annotated)

        if display:
            cv2.imshow("Emotion Inference", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or ESC
                logger.info("User quit.")
                break

        frame_idx += 1

        # Progress log every 100 frames
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx} frames | FPS={display_fps:.1f}")

    # ── Teardown ────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    logger.info(f"\nDone. Output saved → {output_path}")
    logger.info(f"Total frames processed: {frame_idx}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time emotion inference pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file, or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUTS_DIR / "output.mp4",
        help="Path for the annotated output MP4.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_model.pth",
        help="Path to trained emotion model checkpoint (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Face crop size for emotion model.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=EMOTION_SMOOTH_WINDOW,
        help="Sliding-window size for emotion smoothing.",
    )
    parser.add_argument(
        "--detr_skip",
        type=int,
        default=DETR_SKIP_FRAMES,
        help="Run DETR segmentation every N frames.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable live preview window (headless mode).",
    )
    parser.add_argument(
        "--no_segmentation",
        action="store_true",
        help="Skip DETR segmentation entirely (faster).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_source     = args.input,
        output_path      = args.output,
        checkpoint_path  = args.checkpoint,
        device           = args.device,
        image_size       = args.image_size,
        smooth_window    = args.smooth_window,
        detr_skip        = args.detr_skip,
        display          = not args.no_display,
        use_segmentation = not args.no_segmentation,
    )