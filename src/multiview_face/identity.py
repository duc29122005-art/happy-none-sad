from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import MultiViewFaceConfig
from .types import FaceObservation, StereoCameraModel, StereoFacePair
from .utils import cosine_similarity

logger = logging.getLogger(__name__)


def _point_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    a, b, c = line.astype(float)
    denom = np.sqrt(a * a + b * b)
    if denom < 1e-8:
        return float("inf")
    x, y = point.astype(float)
    return float(abs(a * x + b * y + c) / denom)


class CrossCameraMatcher:
    """Greedy identity matcher with embedding and stereo geometry cues."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config

    def geometry_score(
        self,
        left: FaceObservation,
        right: FaceObservation,
        camera: Optional[StereoCameraModel],
    ) -> float:
        if camera is not None and camera.F is not None:
            p1 = np.array([left.center[0], left.center[1], 1.0], dtype=np.float32)
            line = camera.F @ p1
            err = _point_line_distance(right.center, line)
            return float(np.clip(1.0 - err / self.config.epipolar_error_threshold, 0.0, 1.0))

        ly = left.center[1] / max(1.0, float(left.bbox[3] + left.bbox[1]))
        ry = right.center[1] / max(1.0, float(right.bbox[3] + right.bbox[1]))
        y_penalty = abs(ly - ry)
        return float(np.clip(1.0 - y_penalty / max(1e-6, self.config.y_alignment_weight), 0.0, 1.0))

    def match(
        self,
        left_faces: List[FaceObservation],
        right_faces: List[FaceObservation],
        camera: Optional[StereoCameraModel],
    ) -> List[StereoFacePair]:
        candidates: list[tuple[float, int, int, float, float]] = []
        for i, left in enumerate(left_faces):
            for j, right in enumerate(right_faces):
                identity = cosine_similarity(left.embedding, right.embedding)
                geom = self.geometry_score(left, right, camera)
                score = 0.72 * max(identity, 0.0) + 0.28 * geom
                if identity >= self.config.identity_similarity_threshold or geom >= 0.72:
                    candidates.append((score, i, j, identity, geom))

        candidates.sort(reverse=True, key=lambda x: x[0])
        used_left: set[int] = set()
        used_right: set[int] = set()
        matches: list[StereoFacePair] = []
        for score, i, j, identity, geom in candidates:
            if i in used_left or j in used_right:
                continue
            used_left.add(i)
            used_right.add(j)
            matches.append(StereoFacePair(left_faces[i], right_faces[j], identity, geom, score))
        return matches
