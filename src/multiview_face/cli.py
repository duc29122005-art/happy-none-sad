from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import MultiViewFaceConfig
from .pipeline import MultiViewFacePipeline


def parse_float_list(value: str) -> tuple[float, ...]:
    if not value:
        return tuple()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def add_multiview_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "multiview",
        help="Run stereo face frontalization and synthetic multi-angle dataset generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cam1", type=Path, required=True, help="Path to camera 1 video.")
    parser.add_argument("--cam2", type=Path, required=True, help="Path to camera 2 video.")
    parser.add_argument("--output_dir", type=Path, default=MultiViewFaceConfig.output_dir)
    parser.add_argument("--calibration_file", type=Path, default=None)
    parser.add_argument("--emotion_checkpoint", type=Path, default=MultiViewFaceConfig.emotion_checkpoint)
    parser.add_argument("--device", type=str, default=MultiViewFaceConfig().device, choices=["cuda", "cpu"])
    parser.add_argument("--frame_stride", type=int, default=MultiViewFaceConfig.frame_stride)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--output_image_size", type=int, default=MultiViewFaceConfig.output_image_size)
    parser.add_argument("--detector_backend", type=str, default="auto", choices=["auto", "insightface", "opencv"])
    parser.add_argument("--landmark_backend", type=str, default="auto", choices=["auto", "mediapipe"])
    parser.add_argument(
        "--mediapipe_task_model",
        type=Path,
        default=None,
        help="Optional face_landmarker.task path for MediaPipe Tasks 468-landmark extraction.",
    )
    parser.add_argument("--no_timestamp_ocr", action="store_true")
    parser.add_argument("--no_restoration", action="store_true")
    parser.add_argument("--strict_emotion_validation", action="store_true")
    parser.add_argument("--allow_sparse_landmarks", action="store_true")
    parser.add_argument("--synthetic_yaws", type=parse_float_list, default=parse_float_list("-30,-20,-10,0,10,20,30"))
    parser.add_argument("--synthetic_pitches", type=parse_float_list, default=parse_float_list("-15,-10,0,10,15"))
    parser.add_argument("--synthetic_rolls", type=parse_float_list, default=parse_float_list("0"))
    return parser


def config_from_args(args: argparse.Namespace) -> MultiViewFaceConfig:
    return MultiViewFaceConfig(
        output_dir=args.output_dir,
        calibration_file=args.calibration_file,
        emotion_checkpoint=args.emotion_checkpoint,
        detector_backend=args.detector_backend,
        landmark_backend=args.landmark_backend,
        mediapipe_task_model=args.mediapipe_task_model,
        device=args.device,
        frame_stride=args.frame_stride,
        max_pairs=args.max_pairs,
        output_image_size=args.output_image_size,
        use_timestamp_ocr=not args.no_timestamp_ocr,
        restore_faces=not args.no_restoration,
        strict_emotion_validation=args.strict_emotion_validation,
        require_dense_landmarks=not args.allow_sparse_landmarks,
        synthetic_yaws=args.synthetic_yaws,
        synthetic_pitches=args.synthetic_pitches,
        synthetic_rolls=args.synthetic_rolls,
    )


def run_from_args(args: argparse.Namespace) -> dict[str, int]:
    config = config_from_args(args)
    pipeline = MultiViewFacePipeline(config)
    return pipeline.run(args.cam1, args.cam2)
