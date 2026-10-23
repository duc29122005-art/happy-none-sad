from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

from src.config import CHECKPOINTS_DIR, DATA_DIR, INSIGHTFACE_MODEL, ONNX_PROVIDERS


@dataclass
class MultiViewFaceConfig:
    """Runtime configuration for the full multi-view face pipeline."""

    output_dir: Path = DATA_DIR / "multiview_dataset"
    calibration_file: Optional[Path] = None
    emotion_checkpoint: Path = CHECKPOINTS_DIR / "best_model.pth"

    detector_backend: str = "auto"
    landmark_backend: str = "auto"
    renderer_backend: str = "opencv"
    mediapipe_task_model: Optional[Path] = None
    insightface_model: str = INSIGHTFACE_MODEL
    onnx_providers: Sequence[str] = field(default_factory=lambda: list(ONNX_PROVIDERS))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    frame_stride: int = 5
    max_pairs: Optional[int] = None
    batch_size: int = 8

    use_timestamp_ocr: bool = True
    timestamp_sample_count: int = 24
    timestamp_roi: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.22)
    sync_max_drift_seconds: float = 0.120

    det_size: Tuple[int, int] = (640, 640)
    detection_threshold: float = 0.50
    identity_similarity_threshold: float = 0.32
    epipolar_error_threshold: float = 22.0
    y_alignment_weight: float = 0.18

    min_face_size: int = 64
    blur_threshold: float = 80.0
    min_quality_score: float = 0.48
    require_dense_landmarks: bool = True

    output_image_size: int = 512
    restore_faces: bool = True
    emotion_tolerance: float = 0.35
    strict_emotion_validation: bool = False

    synthetic_yaws: Sequence[float] = field(default_factory=lambda: (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0))
    synthetic_pitches: Sequence[float] = field(default_factory=lambda: (-15.0, -10.0, 0.0, 10.0, 15.0))
    synthetic_rolls: Sequence[float] = field(default_factory=lambda: (0.0,))
    include_frontal_synthetic: bool = False

    max_track_age: int = 20
    track_similarity_threshold: float = 0.34

    save_debug: bool = False

    def synthetic_pose_grid(self) -> list[tuple[float, float, float, str]]:
        """Return virtual camera poses with stable dataset folder names."""
        poses: list[tuple[float, float, float, str]] = []
        for yaw in self.synthetic_yaws:
            for pitch in self.synthetic_pitches:
                for roll in self.synthetic_rolls:
                    if yaw == 0.0 and pitch == 0.0 and roll == 0.0 and not self.include_frontal_synthetic:
                        continue
                    yaw_name = "yaw_center_0" if yaw == 0.0 else f"yaw_{'left' if yaw < 0 else 'right'}_{abs(int(yaw))}"
                    pitch_name = "pitch_center_0" if pitch == 0.0 else f"pitch_{'down' if pitch < 0 else 'up'}_{abs(int(pitch))}"
                    roll_name = "" if roll == 0.0 else f"_roll_{'left' if roll < 0 else 'right'}_{abs(int(roll))}"
                    poses.append((yaw, pitch, roll, f"{yaw_name}_{pitch_name}{roll_name}"))
        return poses
