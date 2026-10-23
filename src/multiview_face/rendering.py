from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .config import MultiViewFaceConfig
from .types import FaceMesh3D
from .utils import normalize_depth, rotation_matrix_from_euler

logger = logging.getLogger(__name__)


class FaceRenderer:
    """Expression-preserving 3D mesh renderer with OpenCV texture warping."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.backend = "opencv_mesh_renderer"

    def render(
        self,
        mesh: FaceMesh3D,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.output_image_size
        vertices = self._rotate(mesh.vertices, yaw, pitch, roll)
        projected = self._project(vertices, size)
        canvas = np.zeros((size, size, 3), dtype=np.float32)
        weight = np.zeros((size, size, 1), dtype=np.float32)
        depth = np.full((size, size), np.inf, dtype=np.float32)

        if mesh.faces.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8), np.zeros((size, size), dtype=np.uint8)

        tri_depth = vertices[mesh.faces, 2].mean(axis=1)
        for tri_index in np.argsort(tri_depth):
            indices = mesh.faces[int(tri_index)]
            dst_tri = projected[indices].astype(np.float32)
            if abs(cv2.contourArea(dst_tri)) < 1.0:
                continue
            z_value = float(tri_depth[int(tri_index)])
            for cam_id, source in mesh.source_images.items():
                if source is None or source.size == 0:
                    continue
                src_lm = mesh.source_landmarks.get(cam_id)
                if src_lm is None or src_lm.shape[0] <= int(indices.max()):
                    continue
                src_tri = src_lm[indices, :2].astype(np.float32)
                if abs(cv2.contourArea(src_tri)) < 1.0:
                    continue
                cam_weight = self._camera_weight(cam_id, yaw)
                self._accumulate_triangle(source, src_tri, dst_tri, canvas, weight, depth, z_value, cam_weight)

        rendered = canvas / np.maximum(weight, 1e-6)
        rendered = np.clip(rendered, 0, 255).astype(np.uint8)
        hole_mask = (weight[..., 0] <= 1e-6).astype(np.uint8) * 255
        if hole_mask.any():
            rendered = cv2.inpaint(rendered, hole_mask, 3, cv2.INPAINT_TELEA)
        rendered = self._soft_face_crop(rendered)
        return rendered, normalize_depth(depth)

    def _rotate(self, vertices: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
        rot = rotation_matrix_from_euler(yaw, pitch, roll)
        return (vertices @ rot.T).astype(np.float32)

    def _project(self, vertices: np.ndarray, size: int) -> np.ndarray:
        scale = size * 0.42
        xy = np.empty((vertices.shape[0], 2), dtype=np.float32)
        xy[:, 0] = size * 0.5 + vertices[:, 0] * scale
        xy[:, 1] = size * 0.5 - vertices[:, 1] * scale
        return xy

    def _camera_weight(self, cam_id: str, yaw: float) -> float:
        if cam_id == "cam1":
            return float(np.clip(0.55 - yaw / 160.0, 0.20, 0.85))
        return float(np.clip(0.45 + yaw / 160.0, 0.20, 0.85))

    def _accumulate_triangle(
        self,
        source: np.ndarray,
        src_tri: np.ndarray,
        dst_tri: np.ndarray,
        canvas: np.ndarray,
        weight: np.ndarray,
        depth: np.ndarray,
        z_value: float,
        cam_weight: float,
    ) -> None:
        size = canvas.shape[0]
        matrix = cv2.getAffineTransform(src_tri, dst_tri)
        warped = cv2.warpAffine(
            source,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_tri).astype(np.int32), 255)
        visible = (mask > 0) & (z_value <= depth + 0.03)
        if not visible.any():
            return
        alpha = visible.astype(np.float32)[..., None] * cam_weight
        canvas += warped * alpha
        weight += alpha
        depth[visible] = np.minimum(depth[visible], z_value)

    def _soft_face_crop(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 2).astype(np.uint8) * 255
        if mask.sum() == 0:
            return image
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        background = np.zeros_like(image)
        return np.clip(image.astype(np.float32) * mask[..., None] + background * (1.0 - mask[..., None]), 0, 255).astype(np.uint8)


class FaceRestorer:
    """Conservative restoration adapter; avoids changing expression geometry."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.backend = "opencv_unsharp"
        self.model = None
        if not enabled:
            self.backend = "disabled"

    def restore(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
        return cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
