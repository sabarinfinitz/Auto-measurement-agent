"""
Models package for TailorAI v2.

This package contains the core machine learning models and algorithms:
- Pose detection using MediaPipe
- Camera calibration
- Measurement extraction
- SMPL model integration
- Virtual try-on functionality
"""

__version__ = "2.0.0"
__author__ = "TailorAI Team"

from .pose import extract_pose_from_image
from .calibration import CameraCalibration
from .measurements import TailorMeasurementExtractor

__all__ = [
    "extract_pose_from_image",
    "CameraCalibration", 
    "TailorMeasurementExtractor"
]