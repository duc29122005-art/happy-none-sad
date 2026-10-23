"""
utils.py - Shared helper functions used across the entire pipeline.

Functions:
    ensure_dir          - Create directories safely
    valid_face_size     - Filter faces that are too small
    is_blurry           - Laplacian-variance blur detector
    save_face           - Crop + save a detected face
    update_emotion_history  - Sliding-window emotion smoother
    get_dominant_emotion    - Resolve the smoothed emotion label
    draw_label_box      - Draw bounding box + emotion text on frame
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, List, Optional

import logging

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Args:
        path: Target directory path.

    Returns:
        The same path (useful for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(*paths: Path) -> None:
    """Create multiple directories in one call."""
    for p in paths:
        ensure_dir(p)


# ─────────────────────────────────────────────
# Face validation helpers
# ─────────────────────────────────────────────

def valid_face_size(
    face_img: np.ndarray,
    min_size: int,
) -> bool:
    """
    Return True if the face crop is at least min_size × min_size pixels.

    Args:
        face_img: Cropped face image (H × W × C).
        min_size: Minimum acceptable dimension in pixels.

    Returns:
        True if both height and width >= min_size.
    """
    if face_img is None or face_img.size == 0:
        return False
    h, w = face_img.shape[:2]
    return h >= min_size and w >= min_size


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """
    Detect blurry images using the variance of the Laplacian operator.

    A low variance means the image lacks sharp edges → blurry.

    Args:
        image:     Input image (BGR or grayscale).
        threshold: Images with Laplacian variance below this are blurry.

    Returns:
        True if the image is considered blurry.
    """
    if image is None or image.size == 0:
        return True

    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


# ─────────────────────────────────────────────
# Face I/O helpers
# ─────────────────────────────────────────────

def crop_face_from_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Crop a face region from a frame using a bounding box.

    Args:
        frame:   Full video frame (H × W × C).
        bbox:    Array [x1, y1, x2, y2] in pixel coordinates.
        padding: Fractional padding to add around the face (default 10 %).

    Returns:
        Cropped face image, or None if the crop is invalid.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    # Apply padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def save_face(
    face_img: np.ndarray,
    save_path: Path,
    target_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Save a face crop to disk, optionally resizing it first.

    Args:
        face_img:    Cropped face image.
        save_path:   Full file path (including filename + extension).
        target_size: (width, height) to resize before saving, or None.

    Returns:
        True on success, False on failure.
    """
    try:
        if target_size is not None:
            face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        ensure_dir(save_path.parent)
        success = cv2.imwrite(str(save_path), face_img)
        if not success:
            logger.warning(f"cv2.imwrite failed for {save_path}")
        return success
    except Exception as e:
        logger.error(f"Failed to save face to {save_path}: {e}")
        return False


# ─────────────────────────────────────────────
# Emotion smoothing helpers
# ─────────────────────────────────────────────

def update_emotion_history(
    history: deque,
    emotion: str,
) -> deque:
    """
    Append a new emotion prediction to the sliding-window history.

    The deque has a fixed maxlen defined at creation time, so old entries
    are automatically discarded.

    Args:
        history: A collections.deque with maxlen set.
        emotion: The raw predicted emotion label for this frame.

    Returns:
        The updated deque (same object, mutated in-place).
    """
    history.append(emotion)
    return history


def get_dominant_emotion(
    history: deque,
    class_names: List[str],
) -> str:
    """
    Return the most frequent emotion in the history window (majority vote).

    Falls back to 'none' if the history is empty.

    Args:
        history:     Sliding-window deque of recent emotion labels.
        class_names: All valid emotion class names.

    Returns:
        The smoothed (dominant) emotion label.
    """
    if not history:
        return "none"

    counts: Dict[str, int] = {c: 0 for c in class_names}
    for e in history:
        if e in counts:
            counts[e] += 1

    return max(counts, key=counts.get)  # type: ignore[arg-type]


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def draw_label_box(
    frame: np.ndarray,
    bbox: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    conf: float = 1.0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box and an emotion label on the frame.

    Args:
        frame:     Frame to draw on (mutated in-place).
        bbox:      [x1, y1, x2, y2] bounding box.
        label:     Emotion label string.
        color:     BGR color for the box and text background.
        conf:      Confidence score (0–1) appended to the label.
        thickness: Box line thickness in pixels.

    Returns:
        The frame with the annotation rendered.
    """
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build label string
    text = f"{label} {conf:.2f}" if conf < 1.0 else label

    # Text background pill
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

    # Clamp text box inside frame
    tx1 = x1
    ty1 = max(0, y1 - th - baseline - 6)
    tx2 = x1 + tw + 8
    ty2 = y1

    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        frame,
        text,
        (tx1 + 4, ty2 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA,
    )

    return frame


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 180, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Blend a binary mask overlay onto a frame.

    Args:
        frame: Original BGR frame.
        mask:  Binary mask (H × W), uint8, values 0 or 255.
        color: BGR tint color for the mask region.
        alpha: Opacity of the overlay (0 = transparent, 1 = opaque).

    Returns:
        Composited frame with the mask overlay applied.
    """
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)