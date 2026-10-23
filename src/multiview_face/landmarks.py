from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .config import MultiViewFaceConfig
from .types import FaceObservation

logger = logging.getLogger(__name__)


class DenseLandmarkExtractor:
    """MediaPipe FaceMesh adapter returning 468 landmarks in frame pixels."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.backend = "none"
        self.face_mesh = None
        if config.landmark_backend in {"auto", "mediapipe"}:
            if self._init_classic_mediapipe(config):
                return
            if self._init_tasks_mediapipe(config):
                return
            if config.landmark_backend == "mediapipe":
                raise RuntimeError("MediaPipe FaceMesh requested but no supported FaceMesh backend is available")
            logger.warning("MediaPipe FaceMesh unavailable; sparse detector landmarks will be used when allowed")

    def _init_classic_mediapipe(self, config: MultiViewFaceConfig) -> bool:
        try:
            import mediapipe as mp  # type: ignore

            face_mesh_module = getattr(getattr(mp, "solutions", None), "face_mesh", None)
            if face_mesh_module is None:
                return False
            self.face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self.backend = "mediapipe_classic"
            logger.info("Dense landmark backend: MediaPipe classic FaceMesh")
            return True
        except Exception as exc:
            logger.debug("MediaPipe classic FaceMesh unavailable: %s", exc)
            return False

    def _init_tasks_mediapipe(self, config: MultiViewFaceConfig) -> bool:
        if config.mediapipe_task_model is None or not config.mediapipe_task_model.exists():
            return False
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(config.mediapipe_task_model)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat.SRGB
            self.backend = "mediapipe_tasks"
            logger.info("Dense landmark backend: MediaPipe Tasks FaceLandmarker")
            return True
        except Exception as exc:
            logger.warning("MediaPipe Tasks FaceLandmarker unavailable: %s", exc)
            return False

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()

    def extract(self, frame: np.ndarray, obs: FaceObservation) -> Optional[np.ndarray]:
        if self.backend == "mediapipe_classic" and self.face_mesh is not None:
            landmarks = self._extract_mediapipe(frame, obs)
            obs.dense_landmarks = landmarks
            return landmarks
        if self.backend == "mediapipe_tasks" and self.face_mesh is not None:
            landmarks = self._extract_mediapipe_tasks(frame, obs)
            obs.dense_landmarks = landmarks
            return landmarks
        landmarks = self._fallback_from_sparse(obs)
        obs.dense_landmarks = landmarks
        return landmarks

    def _extract_mediapipe(self, frame: np.ndarray, obs: FaceObservation) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = obs.bbox.astype(int)
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.18)
        pad_y = int((y2 - y1) * 0.18)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        ch, cw = crop.shape[:2]
        points = np.array(
            [[p.x * cw + x1, p.y * ch + y1, p.z * max(cw, ch)] for p in lm],
            dtype=np.float32,
        )
        return points

    def _extract_mediapipe_tasks(self, frame: np.ndarray, obs: FaceObservation) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = obs.bbox.astype(int)
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.18)
        pad_y = int((y2 - y1) * 0.18)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
        result = self.face_mesh.detect(mp_image)
        if not result.face_landmarks:
            return None
        ch, cw = crop.shape[:2]
        points = np.array(
            [[p.x * cw + x1, p.y * ch + y1, p.z * max(cw, ch)] for p in result.face_landmarks[0]],
            dtype=np.float32,
        )
        return points

    def _fallback_from_sparse(self, obs: FaceObservation) -> Optional[np.ndarray]:
        if obs.landmarks_2d is None:
            return None
        points = np.asarray(obs.landmarks_2d, dtype=np.float32)
        if points.shape[0] < 5:
            return None
        z = np.zeros((points.shape[0], 1), dtype=np.float32)
        return np.hstack([points, z])


def expression_coefficients(landmarks: Optional[np.ndarray]) -> dict[str, float]:
    """Compute compact expression proxies without changing expression state."""
    if landmarks is None or landmarks.shape[0] < 300:
        return {"mouth_open": 0.0, "smile": 0.0, "brow_raise": 0.0, "eye_open": 0.0}

    pts = landmarks[:, :2].astype(np.float32)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    face_width = max(dist(234, 454), 1.0)
    mouth_width = max(dist(61, 291), 1.0)
    mouth_open = dist(13, 14) / mouth_width
    smile = (dist(61, 291) / face_width) - 0.34
    left_eye = dist(159, 145) / max(dist(33, 133), 1.0)
    right_eye = dist(386, 374) / max(dist(362, 263), 1.0)
    brow_left = (pts[159, 1] - pts[105, 1]) / face_width
    brow_right = (pts[386, 1] - pts[334, 1]) / face_width
    brow_raise = max(0.0, (brow_left + brow_right) * 0.5)
    return {
        "mouth_open": float(np.clip(mouth_open, 0.0, 1.0)),
        "smile": float(np.clip(smile * 3.0, -1.0, 1.0)),
        "brow_raise": float(np.clip(brow_raise * 4.0, 0.0, 1.0)),
        "eye_open": float(np.clip((left_eye + right_eye) * 0.5, 0.0, 1.0)),
    }
