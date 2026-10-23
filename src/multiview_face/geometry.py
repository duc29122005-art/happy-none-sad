from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .types import StereoCameraModel

logger = logging.getLogger(__name__)


def approximate_intrinsics(width: int, height: int) -> np.ndarray:
    focal = 0.9 * max(width, height)
    return np.array([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float32).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)


def projection_matrices(model: StereoCameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Return P1=K1[I|0], P2=K2[R|T] for calibrated stereo triangulation."""
    p1 = model.K_left @ np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])
    p2 = model.K_right @ np.hstack([model.R.astype(np.float32), model.T.astype(np.float32).reshape(3, 1)])
    return p1.astype(np.float32), p2.astype(np.float32)


def reprojection_errors(points_3d: np.ndarray, points_2d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project 3D points back to image space and return per-point pixel errors."""
    points = np.asarray(points_3d, dtype=np.float32)
    observed = np.asarray(points_2d, dtype=np.float32)[:, :2]
    homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    projected = (projection @ homog.T).T
    valid = np.abs(projected[:, 2]) > 1e-8
    errors = np.full(points.shape[0], np.inf, dtype=np.float32)
    xy = projected[valid, :2] / projected[valid, 2:3]
    errors[valid] = np.linalg.norm(xy - observed[valid], axis=1)
    return errors


def load_camera_model(path: Path) -> StereoCameraModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = StereoCameraModel(
        K_left=np.asarray(payload["K_left"], dtype=np.float32),
        K_right=np.asarray(payload["K_right"], dtype=np.float32),
        R=np.asarray(payload["R"], dtype=np.float32),
        T=np.asarray(payload["T"], dtype=np.float32).reshape(3, 1),
        E=np.asarray(payload["E"], dtype=np.float32) if payload.get("E") is not None else None,
        F=np.asarray(payload["F"], dtype=np.float32) if payload.get("F") is not None else None,
        image_size=tuple(payload["image_size"]) if payload.get("image_size") else None,
        source=str(path),
        D_left=np.asarray(payload["D_left"], dtype=np.float32) if payload.get("D_left") is not None else None,
        D_right=np.asarray(payload["D_right"], dtype=np.float32) if payload.get("D_right") is not None else None,
    )
    return model


def default_stereo_model(width: int, height: int) -> StereoCameraModel:
    k = approximate_intrinsics(width, height)
    r = np.eye(3, dtype=np.float32)
    t = np.array([[0.08], [0.0], [0.0]], dtype=np.float32)
    tx = skew_symmetric(t)
    e = tx @ r
    f = np.linalg.inv(k).T @ e @ np.linalg.inv(k)
    return StereoCameraModel(k.copy(), k.copy(), r, t, e, f, (width, height), "default_equal_height")


def estimate_fundamental_from_landmarks(left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
    if left is None or right is None:
        return None
    n = min(left.shape[0], right.shape[0])
    if n < 8:
        return None
    pts1 = left[:n, :2].astype(np.float32)
    pts2 = right[:n, :2].astype(np.float32)
    f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.5, 0.995)
    if f is None or f.shape != (3, 3):
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    logger.debug("Estimated fundamental matrix from %d/%d landmark inliers", inliers, n)
    return f.astype(np.float32)


def update_fundamental(model: StereoCameraModel, left: np.ndarray, right: np.ndarray) -> StereoCameraModel:
    f = estimate_fundamental_from_landmarks(left, right)
    if f is not None:
        model.F = f
        model.source = "landmark_fundamental"
    return model


def _undistort_landmark_pixels(points: np.ndarray, k: np.ndarray, d: Optional[np.ndarray]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)[:, :2]
    if d is None:
        return pts
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), k, d, P=k).reshape(-1, 2).astype(np.float32)


def triangulate_landmarks(
    left: np.ndarray,
    right: np.ndarray,
    model: StereoCameraModel,
) -> Optional[np.ndarray]:
    n = min(left.shape[0], right.shape[0])
    if n < 4:
        return None
    p1, p2 = projection_matrices(model)
    left_pts = _undistort_landmark_pixels(left[:n], model.K_left, model.D_left)
    right_pts = _undistort_landmark_pixels(right[:n], model.K_right, model.D_right)
    pts4 = cv2.triangulatePoints(p1, p2, left_pts.T.astype(np.float32), right_pts.T.astype(np.float32))
    w = pts4[3:4]
    valid_w = np.abs(w) > 1e-8
    if int(valid_w.sum()) < 4:
        return None
    pts3 = np.empty((n, 3), dtype=np.float32)
    pts3[:] = np.nan
    pts3[valid_w.reshape(-1)] = (pts4[:3, valid_w.reshape(-1)] / w[:, valid_w.reshape(-1)]).T
    if not np.isfinite(pts3).all():
        return None
    left_err = reprojection_errors(pts3, left_pts, p1)
    right_err = reprojection_errors(pts3, right_pts, p2)
    median_error = float(np.median(np.concatenate([left_err, right_err])))
    if not np.isfinite(median_error) or median_error > 12.0:
        logger.debug("Rejecting triangulation with median reprojection error %.2f px", median_error)
        return None
    return pts3.astype(np.float32)


def estimate_head_pose(landmarks: Optional[np.ndarray], width: int, height: int) -> dict[str, float]:
    if landmarks is None or landmarks.shape[0] < 468:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    image_points = np.array(
        [
            landmarks[1, :2],
            landmarks[152, :2],
            landmarks[33, :2],
            landmarks[263, :2],
            landmarks[61, :2],
            landmarks[291, :2],
        ],
        dtype=np.float32,
    )
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1],
        ],
        dtype=np.float32,
    )
    camera = approximate_intrinsics(width, height)
    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(-rot[2, 0], sy)
        roll = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        pitch = np.arctan2(-rot[1, 2], rot[1, 1])
        yaw = np.arctan2(-rot[2, 0], sy)
        roll = 0.0
    return {"yaw": float(np.rad2deg(yaw)), "pitch": float(np.rad2deg(pitch)), "roll": float(np.rad2deg(roll))}
