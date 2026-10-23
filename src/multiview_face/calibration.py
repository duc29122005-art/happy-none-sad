from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .geometry import skew_symmetric
from .types import StereoCameraModel
from .utils import write_json

logger = logging.getLogger(__name__)


def chessboard_object_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    """Return metric 3D chessboard points on the z=0 plane."""
    cols, rows = board_size
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= float(square_size)
    return grid


def find_chessboard_corners(image: np.ndarray, board_size: tuple[int, int]) -> np.ndarray | None:
    """Find subpixel chessboard corners in one calibration image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = None
    ok = False

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )

    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ok or corners is None:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def calibrate_stereo_from_chessboards(
    left_images: Sequence[str | Path],
    right_images: Sequence[str | Path],
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> StereoCameraModel:
    """Estimate K1, K2, distortion, R, T, E, and F from synchronized chessboard images.

    board_size is the number of inner chessboard corners as (columns, rows).
    square_size is any real metric unit, for example meters or millimeters.
    """
    if len(left_images) != len(right_images):
        raise ValueError("left_images and right_images must contain the same number of files")

    object_template = chessboard_object_points(board_size, square_size)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for left_path, right_path in zip(left_images, right_images):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            logger.warning("Skipping unreadable calibration pair: %s | %s", left_path, right_path)
            continue
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"Calibration pair has different image sizes: {left_path} | {right_path}")

        image_size = (left.shape[1], left.shape[0])
        left_corners = find_chessboard_corners(left, board_size)
        right_corners = find_chessboard_corners(right, board_size)
        if left_corners is None or right_corners is None:
            logger.warning("Skipping pair without chessboard corners: %s | %s", left_path, right_path)
            continue

        object_points.append(object_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)

    if image_size is None or len(object_points) < 8:
        raise ValueError("Need at least 8 valid stereo calibration pairs; 15-30 is recommended")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    _, k_left, d_left, _, _ = cv2.calibrateCamera(object_points, left_points, image_size, None, None)
    _, k_right, d_right, _, _ = cv2.calibrateCamera(object_points, right_points, image_size, None, None)
    rms, k_left, d_left, k_right, d_right, r, t, e, f = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    logger.info("Stereo calibration RMS reprojection error: %.4f", rms)
    if e is None:
        e = skew_symmetric(t) @ r
    return StereoCameraModel(
        K_left=k_left.astype(np.float32),
        K_right=k_right.astype(np.float32),
        R=r.astype(np.float32),
        T=t.astype(np.float32).reshape(3, 1),
        E=e.astype(np.float32),
        F=f.astype(np.float32),
        image_size=image_size,
        source=f"opencv_stereo_calibrate_rms_{rms:.4f}",
        D_left=d_left.astype(np.float32),
        D_right=d_right.astype(np.float32),
    )


def save_camera_model(model: StereoCameraModel, path: str | Path) -> None:
    """Write a calibration JSON accepted by main.py multiview --calibration_file."""
    write_json(
        Path(path),
        {
            "K_left": model.K_left,
            "K_right": model.K_right,
            "D_left": model.D_left,
            "D_right": model.D_right,
            "R": model.R,
            "T": model.T,
            "E": model.E,
            "F": model.F,
            "image_size": model.image_size,
            "source": model.source,
        },
    )
