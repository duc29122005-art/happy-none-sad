"""
segmentation_model.py - Person segmentation using DETR Panoptic.

Model  : facebook/detr-resnet-50-panoptic
Task   : Extract a binary person mask from any video frame.
Output : uint8 mask (H × W), values 0 or 255 — 255 where a person is detected.

Design decisions:
  • The model is loaded ONCE and cached inside SegmentationModel, so
    repeated calls to get_person_mask() are cheap (no re-loading).
  • DETR_SKIP_FRAMES in config.py controls how often to re-run the heavy
    segmentation pass; between runs the previous mask is reused.
  • The model runs on the same DEVICE as everything else (CUDA or CPU).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DetrForSegmentation,
)

logger = logging.getLogger(__name__)

# COCO panoptic "person" category id
_COCO_PERSON_ID = 1


class SegmentationModel:
    """
    Wrapper around DETR panoptic segmentation that extracts person masks.

    Args:
        model_name: HuggingFace model identifier.
        device:     'cuda' or 'cpu'.
        threshold:  Minimum DETR segment score to accept (0–1).

    Example::

        seg = SegmentationModel("facebook/detr-resnet-50-panoptic", device="cuda")
        mask = seg.get_person_mask(frame_bgr)   # np.ndarray H×W uint8
        overlay = seg.overlay_on_frame(frame_bgr, mask)
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50-panoptic",
        device: str = "cpu",
        threshold: float = 0.85,
    ) -> None:
        self.device    = device
        self.threshold = threshold
        self._load_model(model_name)

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self, model_name: str) -> None:
        """Download / cache and initialise the DETR model."""
        logger.info(f"Loading DETR segmentation model: {model_name} on {self.device}")

        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DetrForSegmentation.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        logger.info("DETR model loaded and ready.")

    # ── Core inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def get_person_mask(
        self,
        frame_bgr: np.ndarray,
        target_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Run panoptic segmentation and return a person binary mask.

        Args:
            frame_bgr:   BGR frame from OpenCV (H × W × 3, uint8).
            target_size: If provided, the output mask is resized to (W, H).
                         Defaults to the input frame's size.

        Returns:
            Binary mask (H × W, uint8) — 255 where a person is detected,
            0 everywhere else.
        """
        h, w = frame_bgr.shape[:2]
        out_size = target_size if target_size is not None else (w, h)

        # ── Preprocess ───────────────────────────────────────────────────────
        # DETR expects RGB PIL images
        rgb_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        inputs = self.feature_extractor(images=rgb_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Forward pass ─────────────────────────────────────────────────────
        outputs = self.model(**inputs)

        # ── Post-process panoptic segmentation ───────────────────────────────
        # Returns a list of dicts, one per image in the batch
        results = self.feature_extractor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.threshold,
            target_sizes=[(h, w)],   # resize seg map to original frame size
        )
        result = results[0]

        # ── Build person mask ─────────────────────────────────────────────────
        # result["segmentation"] is a (H × W) int tensor where each pixel
        # holds the segment id (0 = background).
        segmentation = result["segmentation"].cpu().numpy()  # (H, W) int32/int64
        segments_info = result["segments_info"]              # list of dicts

        # Collect segment ids that correspond to the "person" category
        person_mask = np.zeros((h, w), dtype=np.uint8)
        for seg in segments_info:
            if seg.get("label_id") == _COCO_PERSON_ID:
                person_mask[segmentation == seg["id"]] = 255

        # ── Morphological cleanup ─────────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN,  kernel)

        # ── Resize to requested output size ───────────────────────────────────
        if (out_size[0], out_size[1]) != (w, h):
            person_mask = cv2.resize(
                person_mask,
                out_size,
                interpolation=cv2.INTER_NEAREST,
            )

        return person_mask

    # ── Visualisation ────────────────────────────────────────────────────────

    def overlay_on_frame(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 180, 255),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Blend the person mask as a semi-transparent tint on the frame.

        Args:
            frame_bgr: Original BGR frame.
            mask:      Binary person mask (H × W, uint8, 0 or 255).
            color:     BGR tint colour for masked pixels.
            alpha:     Mask opacity (0 = invisible, 1 = solid).

        Returns:
            New BGR frame with the mask overlay applied.
        """
        overlay = frame_bgr.copy()
        overlay[mask == 255] = color
        return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)

    # ── Utility ──────────────────────────────────────────────────────────────

    def has_person(self, mask: np.ndarray, min_pixels: int = 500) -> bool:
        """
        Return True if the mask contains at least min_pixels person pixels.

        Args:
            mask:       Binary person mask.
            min_pixels: Minimum non-zero pixel count to declare "person present".

        Returns:
            True if a person is considered to be in frame.
        """
        return int(np.sum(mask > 0)) >= min_pixels


# ─────────────────────────────────────────────────────────────────────────────
# Module-level lazy singleton (optional convenience)
# ─────────────────────────────────────────────────────────────────────────────

_default_seg_model: Optional[SegmentationModel] = None


def get_segmentation_model(
    model_name: str = "facebook/detr-resnet-50-panoptic",
    device: str = "cpu",
) -> SegmentationModel:
    """
    Return (and lazily create) the module-level SegmentationModel singleton.

    Useful when multiple parts of the code need the same model instance
    without passing it around explicitly.
    """
    global _default_seg_model
    if _default_seg_model is None:
        _default_seg_model = SegmentationModel(
            model_name=model_name,
            device=device,
        )
    return _default_seg_model
