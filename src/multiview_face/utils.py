from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def crop_with_padding(
    image: np.ndarray,
    bbox: np.ndarray,
    padding: float = 0.12,
) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    px = int(round(bw * padding))
    py = int(round(bh * padding))
    ix1 = max(0, int(math.floor(x1)) - px)
    iy1 = max(0, int(math.floor(y1)) - py)
    ix2 = min(w, int(math.ceil(x2)) + px)
    iy2 = min(h, int(math.ceil(y2)) + py)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return image[iy1:iy2, ix1:ix2].copy()


def laplacian_variance(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return np.array(
        [
            np.clip(x1, 0, width - 1),
            np.clip(y1, 0, height - 1),
            np.clip(x2, 0, width - 1),
            np.clip(y2, 0, height - 1),
        ],
        dtype=np.float32,
    )


def resize_letterbox(image: np.ndarray, size: int, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def stable_embedding_from_crop(crop: Optional[np.ndarray], bins: int = 16) -> Optional[np.ndarray]:
    """Fallback identity descriptor when a face-recognition embedding is unavailable."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins // 2, bins // 2], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    if np.linalg.norm(hist) > 1e-8:
        hist /= np.linalg.norm(hist)
    return hist


def rotation_matrix_from_euler(yaw: float, pitch: float, roll: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ rx @ ry).astype(np.float32)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(depth)
    if not mask.any():
        return np.zeros(depth.shape, dtype=np.uint8)
    dmin = float(depth[mask].min())
    dmax = float(depth[mask].max())
    if dmax - dmin < 1e-6:
        return np.zeros(depth.shape, dtype=np.uint8)
    out = np.zeros(depth.shape, dtype=np.float32)
    out[mask] = (depth[mask] - dmin) / (dmax - dmin)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def median(values: Iterable[float], default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if np.isfinite(v))
    if not vals:
        return default
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])
