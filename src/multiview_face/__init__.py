"""
Multi-view face frontalization and synthetic dataset generation package.

The package is intentionally adapter based: production backends such as
InsightFace, MediaPipe, PyTorch3D, GFPGAN, and OCR engines are used when they
are present, while deterministic OpenCV/NumPy fallbacks keep the pipeline
executable in constrained environments.
"""

from .config import MultiViewFaceConfig
from .pipeline import MultiViewFacePipeline

__all__ = ["MultiViewFaceConfig", "MultiViewFacePipeline"]
