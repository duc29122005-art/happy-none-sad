from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameMetadata:
    camera_id: str
    frame_index: int
    pts_seconds: float
    fps: float
    width: int
    height: int
    timestamp_text: Optional[str] = None
    timestamp_seconds: Optional[float] = None

    @property
    def sync_time_seconds(self) -> float:
        return self.timestamp_seconds if self.timestamp_seconds is not None else self.pts_seconds


@dataclass
class VideoFrame:
    image: np.ndarray
    metadata: FrameMetadata


@dataclass
class StereoFramePair:
    left: VideoFrame
    right: VideoFrame
    sync_error_seconds: float


@dataclass
class FaceObservation:
    camera_id: str
    frame_index: int
    timestamp_seconds: float
    bbox: np.ndarray
    confidence: float
    landmarks_2d: Optional[np.ndarray] = None
    dense_landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None
    head_pose: Dict[str, float] = field(default_factory=dict)
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    quality: Dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox.astype(float)
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox.astype(float)
        return max(0.0, x2 - x1), max(0.0, y2 - y1)


@dataclass
class StereoFacePair:
    left: FaceObservation
    right: FaceObservation
    identity_score: float
    geometry_score: float
    match_score: float
    person_id: Optional[str] = None


@dataclass
class StereoCameraModel:
    K_left: np.ndarray
    K_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    image_size: Optional[tuple[int, int]] = None
    source: str = "estimated"
    D_left: Optional[np.ndarray] = None
    D_right: Optional[np.ndarray] = None


@dataclass
class FaceMesh3D:
    vertices: np.ndarray
    faces: np.ndarray
    source_landmarks: Dict[str, np.ndarray]
    source_images: Dict[str, np.ndarray]
    expression_coefficients: Dict[str, float]
    head_pose: Dict[str, float]
    identity_embedding: Optional[np.ndarray]
    depth_map: Optional[np.ndarray] = None
    backend: str = "opencv_landmark_mesh"


@dataclass
class EmotionResult:
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExportedSample:
    image_path: Path
    metadata_path: Path
    mesh_path: Optional[Path]
    depth_path: Optional[Path]
    metadata: Dict[str, Any]
