from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2

from .types import FrameMetadata, VideoFrame


class VideoReader:
    """Small OpenCV wrapper that keeps frame metadata attached to images."""

    def __init__(self, path: str | Path, camera_id: str) -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count: Optional[int] = count if count > 0 else None

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _metadata(self, frame_index: int) -> FrameMetadata:
        pts = frame_index / self.fps if self.fps > 0 else 0.0
        return FrameMetadata(
            camera_id=self.camera_id,
            frame_index=frame_index,
            pts_seconds=pts,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    def read_at(self, frame_index: int) -> Optional[VideoFrame]:
        if frame_index < 0:
            return None
        if self.frame_count is not None and frame_index >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return VideoFrame(frame, self._metadata(frame_index))

    def iter_frames(
        self,
        stride: int = 1,
        max_frames: Optional[int] = None,
        start_index: int = 0,
    ) -> Iterator[VideoFrame]:
        stride = max(1, int(stride))
        frame_index = max(0, int(start_index))
        yielded = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_index < start_index:
                continue
            if (current_index - start_index) % stride == 0:
                yield VideoFrame(frame, self._metadata(current_index))
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
