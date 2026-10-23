from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import MultiViewFaceConfig
from .types import ExportedSample, FaceMesh3D, StereoFacePair
from .utils import ensure_dir, normalize_depth, write_json


def write_obj(path: Path, mesh: FaceMesh3D) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for vertex in mesh.vertices:
            fh.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        for face in mesh.faces:
            a, b, c = (int(x) + 1 for x in face)
            fh.write(f"f {a} {b} {c}\n")


class DatasetExporter:
    """Writes frontal, mesh, depth, synthetic images, and JSON metadata."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        ensure_dir(config.output_dir)

    def export_sample(
        self,
        pair: StereoFacePair,
        mesh: FaceMesh3D,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        emotion: str,
        view_name: str,
        yaw: float,
        pitch: float,
        roll: float,
        synthetic: bool,
        source_emotion: str,
    ) -> ExportedSample:
        person_id = pair.person_id or "unknown"
        base = self.config.output_dir / f"person_{person_id}" / emotion / view_name
        ensure_dir(base)
        stem = f"f{pair.left.frame_index:07d}_c{pair.right.frame_index:07d}_{view_name}"
        image_path = base / f"{stem}.jpg"
        metadata_path = base / f"{stem}.json"
        mesh_path = base / f"{stem}.obj"
        depth_path = base / f"{stem}_depth.png"

        cv2.imwrite(str(image_path), image)
        write_obj(mesh_path, mesh)
        if depth is None:
            depth = mesh.depth_map
        if depth is not None:
            cv2.imwrite(str(depth_path), normalize_depth(depth))

        metadata: dict[str, Any] = {
            "person_id": person_id,
            "timestamp": {
                "cam1_seconds": pair.left.timestamp_seconds,
                "cam2_seconds": pair.right.timestamp_seconds,
            },
            "frame_index": {
                "cam1": pair.left.frame_index,
                "cam2": pair.right.frame_index,
            },
            "emotion": emotion,
            "source_emotion": source_emotion,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "synthetic": synthetic,
            "view": view_name,
            "camera_source": ["cam1", "cam2"],
            "identity_score": pair.identity_score,
            "geometry_score": pair.geometry_score,
            "match_score": pair.match_score,
            "head_pose_original": mesh.head_pose,
            "expression_coefficients": mesh.expression_coefficients,
            "mesh_backend": mesh.backend,
            "quality": {
                "cam1": pair.left.quality,
                "cam2": pair.right.quality,
            },
            "files": {
                "image": image_path.name,
                "mesh": mesh_path.name,
                "depth": depth_path.name if depth is not None else None,
            },
        }
        write_json(metadata_path, metadata)
        return ExportedSample(image_path, metadata_path, mesh_path, depth_path if depth is not None else None, metadata)
