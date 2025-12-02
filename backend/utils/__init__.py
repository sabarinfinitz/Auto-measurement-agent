"""
Utilities package for TailorAI v2.

This package contains utility functions for:
- Image preprocessing
- Geometric calculations
- Post-processing operations
"""

__version__ = "2.0.0"

from . import geometry
from . import preprocess
from . import postprocess

__all__ = ["geometry", "preprocess", "postprocess"]