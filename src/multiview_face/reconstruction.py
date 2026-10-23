from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .geometry import estimate_head_pose, triangulate_landmarks
from .landmarks import expression_coefficients
from .types import FaceMesh3D, StereoCameraModel, StereoFacePair

logger = logging.getLogger(__name__)


def build_triangles(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return np.empty((0, 3), dtype=np.int32)
    try:
        from scipy.spatial import Delaunay  # type: ignore

        tri = Delaunay(pts[:, :2]).simplices.astype(np.int32)
        return tri
    except Exception:
        center = int(np.argmin(np.linalg.norm(pts[:, :2] - pts[:, :2].mean(axis=0), axis=1)))
        order = np.argsort(np.arctan2(pts[:, 1] - pts[center, 1], pts[:, 0] - pts[center, 0]))
        faces = [[center, int(order[i]), int(order[(i + 1) % len(order)])] for i in range(len(order))]
        return np.asarray(faces, dtype=np.int32)


class ExpressionAwareReconstructor:
    """
    Builds an expression-preserving 3D mesh from stereo FaceMesh landmarks.

    When calibrated stereo is available, landmarks are triangulated. Otherwise
    the system keeps the expression geometry from the dense 2D landmarks and
    estimates a stable pseudo-depth from MediaPipe z plus cross-view disparity.
    """

    def reconstruct(self, pair: StereoFacePair, camera: StereoCameraModel) -> Optional[FaceMesh3D]:
        left = pair.left.dense_landmarks
        right = pair.right.dense_landmarks
        if left is None or right is None:
            return None

        n = min(left.shape[0], right.shape[0])
        left = left[:n]
        right = right[:n]
        vertices = triangulate_landmarks(left, right, camera)
        backend = "stereo_triangulated"

        if vertices is None or vertices.shape[0] != n or np.nanmax(np.abs(vertices)) > 1000:
            vertices = self._pseudo_depth_vertices(left, right)
            backend = "expression_pseudo_depth"

        vertices = self._canonicalize(vertices)
        source_left = self._landmarks_to_crop(left, pair.left.crop)
        source_right = self._landmarks_to_crop(right, pair.right.crop)
        faces = build_triangles(source_left[:, :2])
        depth = self._sparse_depth_map(left, vertices[:, 2], pair.left.crop)
        pose = pair.left.head_pose or (
            estimate_head_pose(left, int(max(pair.left.bbox[2], 1)), int(max(pair.left.bbox[3], 1)))
            if pair.left.crop is not None
            else {}
        )
        expr = expression_coefficients(left)
        identity = pair.left.embedding if pair.left.embedding is not None else pair.right.embedding

        return FaceMesh3D(
            vertices=vertices.astype(np.float32),
            faces=faces,
            source_landmarks={"cam1": source_left.astype(np.float32), "cam2": source_right.astype(np.float32)},
            source_images={"cam1": pair.left.crop, "cam2": pair.right.crop},
            expression_coefficients=expr,
            head_pose=pose,
            identity_embedding=identity,
            depth_map=depth,
            backend=backend,
        )

    def _landmarks_to_crop(self, landmarks: np.ndarray, crop: Optional[np.ndarray]) -> np.ndarray:
        if crop is None or crop.size == 0:
            return landmarks.astype(np.float32)
        h, w = crop.shape[:2]
        pts = landmarks.astype(np.float32).copy()
        x1, y1 = np.percentile(pts[:, :2], 1, axis=0)
        x2, y2 = np.percentile(pts[:, :2], 99, axis=0)
        pts[:, 0] = (pts[:, 0] - x1) / max(1.0, x2 - x1) * (w - 1)
        pts[:, 1] = (pts[:, 1] - y1) / max(1.0, y2 - y1) * (h - 1)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def _pseudo_depth_vertices(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        xy = left[:, :2].astype(np.float32)
        center = xy.mean(axis=0)
        scale = max(float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))), 1.0)
        disparity = (left[:, 0] - right[:, 0]).astype(np.float32)
        disparity = (disparity - np.median(disparity)) / max(float(np.std(disparity)), 1.0)
        mp_z = left[:, 2].astype(np.float32)
        mp_z = (mp_z - np.median(mp_z)) / max(float(np.std(mp_z)), 1.0)
        z = 0.65 * mp_z + 0.35 * disparity
        verts = np.column_stack([(xy[:, 0] - center[0]) / scale, -(xy[:, 1] - center[1]) / scale, z / 8.0])
        return verts.astype(np.float32)

    def _canonicalize(self, vertices: np.ndarray) -> np.ndarray:
        verts = vertices.astype(np.float32).copy()
        verts -= np.mean(verts, axis=0, keepdims=True)
        span = np.max(np.linalg.norm(verts[:, :2], axis=1))
        if span > 1e-6:
            verts /= span
        return verts

    def _sparse_depth_map(self, landmarks: np.ndarray, z: np.ndarray, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        x1, y1, x2, y2 = np.min(landmarks[:, 0]), np.min(landmarks[:, 1]), np.max(landmarks[:, 0]), np.max(landmarks[:, 1])
        local = landmarks[:, :2].copy()
        local[:, 0] = (local[:, 0] - x1) / max(1.0, x2 - x1) * (w - 1)
        local[:, 1] = (local[:, 1] - y1) / max(1.0, y2 - y1) * (h - 1)
        depth = np.full((h, w), np.nan, dtype=np.float32)
        for point, value in zip(local.astype(int), z):
            x = int(np.clip(point[0], 0, w - 1))
            y = int(np.clip(point[1], 0, h - 1))
            cv2.circle(depth, (x, y), 3, float(value), -1)
        mask = np.isfinite(depth).astype(np.uint8)
        if mask.sum() == 0:
            return None
        depth[~np.isfinite(depth)] = 0.0
        return cv2.blur(depth, (15, 15))
