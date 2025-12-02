"""Lightweight measurement estimation using MediaPipe pose landmarks.

This module provides a minimal `estimate_measurements` implementation that
accepts either image bytes (and runs pose detection) or already-extracted
landmarks (dict) and returns simple measurements. It's not a replacement for
SMPL/SMPLify-X but gives usable values for frontend/demo purposes.
"""
from typing import Dict, Optional
import math

from models import pose as pose_model


def _dist(a: Dict, b: Dict) -> float:
    """Euclidean distance between two landmark points in pixels (expects x_px,y_px)."""
    dx = a["x_px"] - b["x_px"]
    dy = a["y_px"] - b["y_px"]
    return math.hypot(dx, dy)


def estimate_measurements(image_bytes: bytes, known_height_cm: Optional[float] = None) -> Dict:
    """Estimate simple body measurements from image bytes.

    Args:
        image_bytes: image bytes to process
        known_height_cm: if provided, scale pixel distances to centimeters

    Returns:
        dict with measurements in cm (when known_height_cm provided) or pixels
    """
    landmarks = pose_model.extract_pose_from_image(image_bytes)

    lms = landmarks.get("landmarks", [])
    if not lms:
        return {"error": "no_pose_detected"}

    # Helper to get landmark by MediaPipe index
    def get(i):
        if 0 <= i < len(lms):
            return lms[i]
        return None

    # MediaPipe indices: 0 nose, 11 left_shoulder, 12 right_shoulder, 23 left_hip, 24 right_hip, 15 left_wrist, 16 right_wrist
    left_sh = get(11)
    right_sh = get(12)
    left_hip = get(23)
    right_hip = get(24)
    left_ankle = get(27)
    right_ankle = get(28)
    nose = get(0)

    # Approximate widths
    shoulders_px = None
    hips_px = None
    height_px = None

    if left_sh and right_sh:
        shoulders_px = _dist(left_sh, right_sh)
    if left_hip and right_hip:
        hips_px = _dist(left_hip, right_hip)
    # height: distance from nose to midpoint of ankles if available
    if nose and left_ankle and right_ankle:
        ankle_mid = {"x_px": int((left_ankle["x_px"] + right_ankle["x_px"]) / 2), "y_px": int((left_ankle["y_px"] + right_ankle["y_px"]) / 2)}
        height_px = _dist(nose, ankle_mid)

    scale = None
    if known_height_cm and height_px:
        scale = known_height_cm / height_px

    def to_cm(px):
        if px is None:
            return None
        return float(px * scale) if scale else px

    measurements = {
        "shoulder_width": to_cm(shoulders_px),
        "hip_width": to_cm(hips_px),
        "estimated_height": to_cm(height_px),
        "units": "cm" if scale else ("px"),
    }

    # Provide raw pixels as fallback
    measurements["raw"] = {"shoulder_px": shoulders_px, "hip_px": hips_px, "height_px": height_px}
    if scale:
        measurements["note"] = "scaled using known_height_cm"
    else:
        measurements["note"] = "provide known_height_cm to get cm units"

    return measurements
