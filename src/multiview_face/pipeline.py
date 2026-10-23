from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MultiViewFaceConfig
from .detection import FaceDetector
from .emotion import EmotionAnalyzer, EmotionConsistencyValidator
from .exporter import DatasetExporter
from .geometry import default_stereo_model, load_camera_model, update_fundamental
from .identity import CrossCameraMatcher
from .landmarks import DenseLandmarkExtractor
from .quality import FaceQualityFilter
from .reconstruction import ExpressionAwareReconstructor
from .rendering import FaceRenderer, FaceRestorer
from .sync import FrameSynchronizer
from .tracking import IdentityTracker
from .types import FaceObservation, StereoCameraModel, StereoFacePair

logger = logging.getLogger(__name__)


class MultiViewFacePipeline:
    """End-to-end stereo face frontalization and synthetic data pipeline."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.synchronizer = FrameSynchronizer(config)
        self.detector = FaceDetector(config)
        self.landmarks = DenseLandmarkExtractor(config)
        self.quality = FaceQualityFilter(config)
        self.matcher = CrossCameraMatcher(config)
        self.tracker = IdentityTracker(config)
        self.reconstructor = ExpressionAwareReconstructor()
        self.renderer = FaceRenderer(config)
        self.restorer = FaceRestorer(config.restore_faces)
        self.emotions = EmotionAnalyzer(config.emotion_checkpoint, config.device)
        self.validator = EmotionConsistencyValidator(config.emotion_tolerance, config.strict_emotion_validation)
        self.exporter = DatasetExporter(config)
        self.camera: Optional[StereoCameraModel] = None

    def close(self) -> None:
        self.landmarks.close()

    def run(self, cam1_path: str | Path, cam2_path: str | Path) -> dict[str, int]:
        stats = {
            "frame_pairs": 0,
            "detections": 0,
            "stereo_matches": 0,
            "frontal_exports": 0,
            "synthetic_exports": 0,
            "rejected_quality": 0,
            "rejected_landmarks": 0,
            "rejected_emotion": 0,
            "rejected_reconstruction": 0,
        }

        try:
            for stereo_pair in self.synchronizer.iter_pairs(cam1_path, cam2_path):
                stats["frame_pairs"] += 1
                if self.camera is None:
                    width = min(stereo_pair.left.metadata.width, stereo_pair.right.metadata.width)
                    height = min(stereo_pair.left.metadata.height, stereo_pair.right.metadata.height)
                    self.camera = self._load_camera(width, height)

                left_faces = self._detect_and_prepare(stereo_pair.left, stats)
                right_faces = self._detect_and_prepare(stereo_pair.right, stats)
                stats["detections"] += len(left_faces) + len(right_faces)
                if not left_faces or not right_faces:
                    continue

                matches = self.matcher.match(left_faces, right_faces, self.camera)
                stats["stereo_matches"] += len(matches)
                for face_pair in matches:
                    self._process_face_pair(face_pair, stats)

                if stats["frame_pairs"] % 25 == 0:
                    logger.info(
                        "Processed %d stereo pairs | frontal=%d synthetic=%d",
                        stats["frame_pairs"],
                        stats["frontal_exports"],
                        stats["synthetic_exports"],
                    )
        finally:
            self.close()

        logger.info("Multi-view pipeline finished: %s", stats)
        return stats

    def _load_camera(self, width: int, height: int) -> StereoCameraModel:
        if self.config.calibration_file and self.config.calibration_file.exists():
            logger.info("Loading stereo calibration: %s", self.config.calibration_file)
            return load_camera_model(self.config.calibration_file)
        logger.warning("No calibration file supplied; using equal-height approximate stereo geometry")
        return default_stereo_model(width, height)

    def _detect_and_prepare(self, frame, stats: dict[str, int]) -> list[FaceObservation]:
        detections = self.detector.detect(frame)
        prepared: list[FaceObservation] = []
        for obs in detections:
            dense = self.landmarks.extract(frame.image, obs)
            if self.config.require_dense_landmarks and (dense is None or dense.shape[0] < 100):
                stats["rejected_landmarks"] += 1
                continue
            if not self.quality.keep(obs):
                stats["rejected_quality"] += 1
                continue
            prepared.append(obs)
        return prepared

    def _process_face_pair(self, pair: StereoFacePair, stats: dict[str, int]) -> None:
        if self.camera is not None and pair.left.dense_landmarks is not None and pair.right.dense_landmarks is not None:
            self.camera = update_fundamental(self.camera, pair.left.dense_landmarks, pair.right.dense_landmarks)

        mesh = self.reconstructor.reconstruct(pair, self.camera)
        if mesh is None:
            stats["rejected_reconstruction"] += 1
            return

        person_id = self.tracker.assign(pair)
        original_image = pair.left.crop if pair.left.crop is not None else next(iter(mesh.source_images.values()))
        if original_image is None or original_image.size == 0:
            original_image = np.zeros((self.config.output_image_size, self.config.output_image_size, 3), dtype=np.uint8)
        original_emotion = self.emotions.analyze(original_image, mesh.expression_coefficients)
        pair.left.emotion = original_emotion.label
        pair.left.emotion_confidence = original_emotion.confidence
        pair.right.emotion = original_emotion.label
        pair.right.emotion_confidence = original_emotion.confidence

        frontal, frontal_depth = self.renderer.render(mesh, yaw=0.0, pitch=0.0, roll=0.0)
        frontal = self.restorer.restore(frontal)
        frontal_emotion = self.emotions.analyze(frontal, mesh.expression_coefficients)
        if not self.validator.accept(original_emotion, frontal_emotion):
            stats["rejected_emotion"] += 1
            logger.debug("Rejected frontal emotion drift for person %s: %s -> %s", person_id, original_emotion, frontal_emotion)
            return

        self.exporter.export_sample(
            pair=pair,
            mesh=mesh,
            image=frontal,
            depth=frontal_depth,
            emotion=original_emotion.label,
            view_name="frontal",
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            synthetic=False,
            source_emotion=original_emotion.label,
        )
        stats["frontal_exports"] += 1

        for yaw, pitch, roll, view_name in self.config.synthetic_pose_grid():
            synthetic, synthetic_depth = self.renderer.render(mesh, yaw=yaw, pitch=pitch, roll=roll)
            synthetic = self.restorer.restore(synthetic)
            synthetic_emotion = self.emotions.analyze(synthetic, mesh.expression_coefficients)
            if not self.validator.accept(original_emotion, synthetic_emotion):
                stats["rejected_emotion"] += 1
                continue
            self.exporter.export_sample(
                pair=pair,
                mesh=mesh,
                image=synthetic,
                depth=synthetic_depth,
                emotion=original_emotion.label,
                view_name=view_name,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                synthetic=True,
                source_emotion=original_emotion.label,
            )
            stats["synthetic_exports"] += 1
