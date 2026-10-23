from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .config import MultiViewFaceConfig
from .types import StereoFramePair, VideoFrame
from .utils import median
from .video import VideoReader

logger = logging.getLogger(__name__)


def parse_timestamp_seconds(text: str) -> Optional[float]:
    """Parse common overlay timestamp formats into seconds within the day."""
    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    match = re.search(r"(\d{1,2})[:\-](\d{2})[:\-](\d{2})(?:[.\:](\d{1,3}))?", cleaned)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int((match.group(4) or "0").ljust(3, "0")[:3])
        return float(hours * 3600 + minutes * 60 + seconds) + millis / 1000.0

    compact = re.search(r"(\d{8})[_\-\s]?(\d{2})(\d{2})(\d{2})(\d{3})?", cleaned)
    if compact:
        hours = int(compact.group(2))
        minutes = int(compact.group(3))
        seconds = int(compact.group(4))
        millis = int(compact.group(5) or "0")
        return float(hours * 3600 + minutes * 60 + seconds) + millis / 1000.0
    return None


@dataclass
class OCRResult:
    text: str
    seconds: Optional[float]


class TimestampOCR:
    """Optional OCR adapter for camera timestamp overlays."""

    def __init__(self, roi: tuple[float, float, float, float]) -> None:
        self.roi = roi
        self.backend = "none"
        self.reader = None

        try:
            import easyocr  # type: ignore

            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self.backend = "easyocr"
            logger.info("Timestamp OCR backend: EasyOCR")
            return
        except Exception:
            pass

        try:
            import pytesseract  # type: ignore

            self.reader = pytesseract
            self.backend = "pytesseract"
            logger.info("Timestamp OCR backend: pytesseract")
        except Exception:
            logger.info("Timestamp OCR disabled; no OCR backend is installed")

    def available(self) -> bool:
        return self.backend != "none" and self.reader is not None

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        ix1 = int(np.clip(x1, 0.0, 1.0) * w)
        iy1 = int(np.clip(y1, 0.0, 1.0) * h)
        ix2 = int(np.clip(x2, 0.0, 1.0) * w)
        iy2 = int(np.clip(y2, 0.0, 1.0) * h)
        if ix2 <= ix1 or iy2 <= iy1:
            return frame
        return frame[iy1:iy2, ix1:ix2]

    def read(self, frame: np.ndarray) -> OCRResult:
        if not self.available():
            return OCRResult("", None)
        crop = self._crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = ""
        try:
            if self.backend == "easyocr":
                result = self.reader.readtext(gray, detail=0, paragraph=True)
                text = " ".join(str(x) for x in result)
            elif self.backend == "pytesseract":
                text = str(self.reader.image_to_string(gray, config="--psm 7"))
        except Exception as exc:
            logger.debug("Timestamp OCR failed: %s", exc)
        return OCRResult(text, parse_timestamp_seconds(text))


class FrameSynchronizer:
    """Build synchronized stereo pairs from two fixed-camera videos."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.ocr = TimestampOCR(config.timestamp_roi) if config.use_timestamp_ocr else None

    def _annotate_ocr(self, frame: VideoFrame) -> VideoFrame:
        if self.ocr is None or not self.ocr.available():
            return frame
        result = self.ocr.read(frame.image)
        frame.metadata.timestamp_text = result.text
        frame.metadata.timestamp_seconds = result.seconds
        return frame

    def estimate_offset_seconds(self, left: VideoReader, right: VideoReader) -> float:
        """Return the right-video PTS offset relative to the left video.

        OCR timestamps are wall-clock-like seconds within a day. To seek in a
        video, we convert those clock readings into per-video PTS offsets:

            clock = pts + camera_start_clock

        The returned value is camera_start_right - camera_start_left. A left
        frame at pts L therefore maps to right pts L - returned_offset.
        """
        if self.ocr is None or not self.ocr.available():
            return 0.0
        if left.frame_count is None or right.frame_count is None:
            return 0.0

        sample_count = max(1, self.config.timestamp_sample_count)
        limit = min(left.frame_count, right.frame_count)
        if limit <= 1:
            return 0.0
        indices = np.linspace(0, limit - 1, num=min(sample_count, limit), dtype=int)
        offsets: list[float] = []

        for idx in indices:
            lf = left.read_at(int(idx))
            rf = right.read_at(int(idx))
            if lf is None or rf is None:
                continue
            lf = self._annotate_ocr(lf)
            rf = self._annotate_ocr(rf)
            lt = lf.metadata.timestamp_seconds
            rt = rf.metadata.timestamp_seconds
            if lt is not None and rt is not None:
                offsets.append((rt - rf.metadata.pts_seconds) - (lt - lf.metadata.pts_seconds))

        offset = median(offsets, default=0.0)
        if offsets:
            logger.info("Estimated PTS offset cam2-cam1: %.3fs from %d OCR samples", offset, len(offsets))
        return offset

    def iter_pairs(self, cam1_path: str | Path, cam2_path: str | Path) -> Iterator[StereoFramePair]:
        with VideoReader(cam1_path, "cam1") as left, VideoReader(cam2_path, "cam2") as right:
            offset = self.estimate_offset_seconds(left, right)
            yielded = 0
            for lf in left.iter_frames(stride=self.config.frame_stride, max_frames=self.config.max_pairs):
                lf = self._annotate_ocr(lf)
                target_right_time = lf.metadata.pts_seconds - offset
                right_index = int(round(target_right_time * right.fps))
                rf = right.read_at(right_index)
                if rf is None:
                    continue
                rf = self._annotate_ocr(rf)
                sync_error = abs(rf.metadata.pts_seconds - target_right_time)
                if sync_error > self.config.sync_max_drift_seconds:
                    logger.debug("Skipping pair at frame %d, sync error %.3fs", lf.metadata.frame_index, sync_error)
                    continue
                yield StereoFramePair(lf, rf, sync_error)
                yielded += 1
                if self.config.max_pairs is not None and yielded >= self.config.max_pairs:
                    break
