from __future__ import annotations

import numpy as np

from .config import MultiViewFaceConfig
from .types import FaceObservation
from .utils import laplacian_variance


class FaceQualityFilter:
    """Rejects low-resolution, blurry, and low-confidence faces."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config

    def score(self, obs: FaceObservation) -> float:
        w, h = obs.size
        min_size = min(w, h)
        size_score = float(np.clip((min_size - self.config.min_face_size) / max(1.0, self.config.min_face_size), 0.0, 1.0))
        blur = laplacian_variance(obs.crop) if obs.crop is not None else 0.0
        blur_score = float(np.clip(blur / max(1.0, self.config.blur_threshold * 2.0), 0.0, 1.0))
        det_score = float(np.clip(obs.confidence, 0.0, 1.0))
        dense_score = 1.0 if obs.dense_landmarks is not None else 0.70
        total = 0.30 * size_score + 0.30 * blur_score + 0.25 * det_score + 0.15 * dense_score
        obs.quality.update(
            {
                "size_score": size_score,
                "blur_variance": blur,
                "blur_score": blur_score,
                "det_score": det_score,
                "dense_landmark_score": dense_score,
                "quality_score": total,
            }
        )
        return total

    def keep(self, obs: FaceObservation) -> bool:
        w, h = obs.size
        if min(w, h) < self.config.min_face_size:
            self.score(obs)
            return False
        if obs.crop is None or obs.crop.size == 0:
            self.score(obs)
            return False
        if laplacian_variance(obs.crop) < self.config.blur_threshold:
            self.score(obs)
            return False
        return self.score(obs) >= self.config.min_quality_score
