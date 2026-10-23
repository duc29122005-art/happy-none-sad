from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from .config import MultiViewFaceConfig
from .types import FaceObservation, VideoFrame
from .utils import crop_with_padding, stable_embedding_from_crop

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector adapter with InsightFace preferred and OpenCV fallback."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.backend = "opencv_haar"
        self.app = None
        self.haar = None
        if config.detector_backend in {"auto", "insightface"}:
            try:
                from insightface.app import FaceAnalysis  # type: ignore

                self.app = FaceAnalysis(
                    name=config.insightface_model,
                    providers=list(config.onnx_providers),
                )
                ctx_id = 0 if config.device == "cuda" else -1
                self.app.prepare(ctx_id=ctx_id, det_size=config.det_size)
                self.backend = "insightface"
                logger.info("Face detector backend: InsightFace")
                return
            except Exception as exc:
                if config.detector_backend == "insightface":
                    raise RuntimeError("InsightFace detector requested but failed to initialize") from exc
                logger.warning("InsightFace unavailable; falling back to OpenCV Haar: %s", exc)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        if self.haar.empty():
            raise RuntimeError(f"Cannot load OpenCV Haar cascade: {cascade_path}")
        logger.info("Face detector backend: OpenCV Haar")

    def detect(self, frame: VideoFrame) -> List[FaceObservation]:
        if self.backend == "insightface" and self.app is not None:
            return self._detect_insightface(frame)
        return self._detect_haar(frame)

    def _detect_insightface(self, frame: VideoFrame) -> List[FaceObservation]:
        observations: list[FaceObservation] = []
        try:
            faces = self.app.get(frame.image)
        except Exception as exc:
            logger.debug("InsightFace detection failed on frame %d: %s", frame.metadata.frame_index, exc)
            faces = []

        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.config.detection_threshold:
                continue
            bbox = np.asarray(face.bbox, dtype=np.float32)
            crop = crop_with_padding(frame.image, bbox)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is not None:
                embedding = np.asarray(embedding, dtype=np.float32)
            else:
                embedding = stable_embedding_from_crop(crop)

            landmarks = None
            for attr in ("landmark_2d_106", "landmark_3d_68", "kps"):
                value = getattr(face, attr, None)
                if value is not None:
                    landmarks = np.asarray(value, dtype=np.float32)
                    break

            pose = {}
            if getattr(face, "pose", None) is not None:
                yaw, pitch, roll = np.asarray(face.pose).reshape(-1)[:3]
                pose = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}

            observations.append(
                FaceObservation(
                    camera_id=frame.metadata.camera_id,
                    frame_index=frame.metadata.frame_index,
                    timestamp_seconds=frame.metadata.sync_time_seconds,
                    bbox=bbox,
                    confidence=score,
                    landmarks_2d=landmarks,
                    embedding=embedding,
                    crop=crop,
                    head_pose=pose,
                )
            )
        return observations

    def _detect_haar(self, frame: VideoFrame) -> List[FaceObservation]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
        observations: list[FaceObservation] = []
        for x, y, w, h in faces:
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            crop = crop_with_padding(frame.image, bbox)
            observations.append(
                FaceObservation(
                    camera_id=frame.metadata.camera_id,
                    frame_index=frame.metadata.frame_index,
                    timestamp_seconds=frame.metadata.sync_time_seconds,
                    bbox=bbox,
                    confidence=0.75,
                    crop=crop,
                    embedding=stable_embedding_from_crop(crop),
                )
            )
        return observations
