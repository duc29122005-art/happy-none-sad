from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.config import CLASS_NAMES, EMOTION_MODEL_NAME, NUM_CLASSES
from src.emotion_model import EmotionModel, load_checkpoint

from .types import EmotionResult

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Emotion adapter for original, frontalized, and synthetic consistency checks."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu", image_size: int = 224) -> None:
        self.device = device
        self.image_size = image_size
        self.model: Optional[EmotionModel] = None
        self.labels = ["happy" if x == "happy" else "sad" if x == "sad" else "neutral" for x in CLASS_NAMES]
        if checkpoint_path.exists():
            try:
                model = EmotionModel(NUM_CLASSES, EMOTION_MODEL_NAME, pretrained=False).to(device)
                load_checkpoint(model, checkpoint_path, device=device)
                model.eval()
                self.model = model
                logger.info("Emotion analyzer loaded checkpoint: %s", checkpoint_path)
            except Exception as exc:
                logger.warning("Emotion checkpoint could not be loaded; using geometry heuristic: %s", exc)
        else:
            logger.warning("Emotion checkpoint not found; using geometry heuristic: %s", checkpoint_path)

    @torch.no_grad()
    def analyze(
        self,
        image_bgr: np.ndarray,
        expression: Optional[dict[str, float]] = None,
    ) -> EmotionResult:
        if self.model is None:
            return self._heuristic(expression)
        image = cv2.resize(image_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        label = self.labels[idx]
        probabilities = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return EmotionResult(label, float(probs[idx]), probabilities)

    def _heuristic(self, expression: Optional[dict[str, float]]) -> EmotionResult:
        expr = expression or {}
        smile = float(expr.get("smile", 0.0))
        mouth_open = float(expr.get("mouth_open", 0.0))
        brow = float(expr.get("brow_raise", 0.0))
        if smile > 0.20:
            return EmotionResult("happy", min(0.95, 0.60 + smile), {"happy": min(0.95, 0.60 + smile)})
        if mouth_open > 0.28 and brow > 0.15:
            return EmotionResult("surprised", min(0.90, 0.55 + mouth_open), {"surprised": min(0.90, 0.55 + mouth_open)})
        if smile < -0.20:
            return EmotionResult("sad", min(0.85, 0.55 + abs(smile)), {"sad": min(0.85, 0.55 + abs(smile))})
        return EmotionResult("neutral", 0.62, {"neutral": 0.62})


class EmotionConsistencyValidator:
    def __init__(self, tolerance: float, strict: bool) -> None:
        self.tolerance = tolerance
        self.strict = strict

    def accept(self, original: EmotionResult, generated: EmotionResult) -> bool:
        if original.label == generated.label:
            return True
        if not self.strict and (original.confidence < 0.55 or generated.confidence < 0.55):
            return True
        original_prob = original.probabilities.get(original.label, original.confidence)
        generated_prob = generated.probabilities.get(original.label, 0.0)
        return (original_prob - generated_prob) <= self.tolerance
