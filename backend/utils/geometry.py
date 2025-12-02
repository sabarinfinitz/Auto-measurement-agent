"""Geometry utilities for calculating distances, circumferences, and angles from pose landmarks."""

import math
from typing import Dict, List, Tuple, Optional


class GeometryCalculator:
    """Helper class for geometric calculations on pose landmarks."""

    @staticmethod
    def distance_2d(point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two 2D points.

        Args:
            point1: Dict with 'x_px' and 'y_px' keys
            point2: Dict with 'x_px' and 'y_px' keys

        Returns:
            Distance in pixels
        """
        if not point1 or not point2:
            return 0.0

        dx = point1.get("x_px", 0) - point2.get("x_px", 0)
        dy = point1.get("y_px", 0) - point2.get("y_px", 0)
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def distance_3d(point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two 3D points.

        Args:
            point1: Dict with 'x', 'y', 'z' keys (normalized 0-1)
            point2: Dict with 'x', 'y', 'z' keys (normalized 0-1)

        Returns:
            Distance in normalized units
        """
        if not point1 or not point2:
            return 0.0

        # Support both normalized 'x','y' coordinates and pixel-based 'x_px','y_px'.
        x1 = point1.get("x", point1.get("x_px", 0))
        y1 = point1.get("y", point1.get("y_px", 0))
        z1 = point1.get("z", 0)

        x2 = point2.get("x", point2.get("x_px", 0))
        y2 = point2.get("y", point2.get("y_px", 0))
        z2 = point2.get("z", 0)

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def midpoint(point1: Dict, point2: Dict) -> Dict:
        """Calculate midpoint between two points.

        Args:
            point1: Dict with 'x_px' and 'y_px' keys
            point2: Dict with 'x_px' and 'y_px' keys

        Returns:
            Midpoint dict with 'x_px' and 'y_px'
        """
        if not point1 or not point2:
            return {"x_px": 0, "y_px": 0}

        return {
            "x_px": (point1.get("x_px", 0) + point2.get("x_px", 0)) / 2,
            "y_px": (point1.get("y_px", 0) + point2.get("y_px", 0)) / 2,
        }

    @staticmethod
    def angle_between(point1: Dict, vertex: Dict, point2: Dict) -> float:
        """Calculate angle at vertex between point1-vertex-point2.

        Args:
            point1: First point
            vertex: Vertex point
            point2: Second point

        Returns:
            Angle in degrees
        """
        if not (point1 and vertex and point2):
            return 0.0

        # Vectors from vertex to point1 and point2
        v1 = (
            point1.get("x_px", 0) - vertex.get("x_px", 0),
            point1.get("y_px", 0) - vertex.get("y_px", 0),
        )
        v2 = (
            point2.get("x_px", 0) - vertex.get("x_px", 0),
            point2.get("y_px", 0) - vertex.get("y_px", 0),
        )

        # Calculate angle using dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle_radians = math.acos(cos_angle)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    @staticmethod
    def estimate_circumference(width: float, depth: float, multiplier: float = 1.2) -> float:
        """Estimate circumference from width and depth measurements.

        Uses ellipse approximation: circumference ≈ π × (width + depth) / 2 × multiplier

        Args:
            width: Lateral distance in pixels/cm
            depth: Anterior-posterior distance in pixels/cm
            multiplier: Empirical adjustment factor (default 1.2)

        Returns:
            Estimated circumference in same units
        """
        if width <= 0 or depth <= 0:
            return 0.0

        # Ramanujan approximation for ellipse perimeter
        a = width / 2
        b = depth / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        circumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
        return circumference * multiplier

    @staticmethod
    def visibility_check(landmarks: List[Dict], indices: List[int], min_visibility: float = 0.5) -> bool:
        """Check if landmarks at given indices have sufficient visibility.

        Args:
            landmarks: List of landmarks
            indices: Landmark indices to check
            min_visibility: Minimum visibility threshold (0-1)

        Returns:
            True if all landmarks are visible
        """
        for idx in indices:
            if idx >= len(landmarks):
                return False
            landmark = landmarks[idx]
            if landmark.get("visibility", 0) < min_visibility:
                return False
        return True

    @staticmethod
    def average_visibility(landmarks: List[Dict], indices: List[int]) -> float:
        """Calculate average visibility for given landmark indices.

        Args:
            landmarks: List of landmarks
            indices: Landmark indices to check

        Returns:
            Average visibility (0-1)
        """
        if not indices:
            return 0.0

        visibilities = []
        for idx in indices:
            if idx < len(landmarks):
                visibilities.append(landmarks[idx].get("visibility", 0))

        if not visibilities:
            return 0.0

        return sum(visibilities) / len(visibilities)


class MeasurementQuality:
    """Assess quality of measurements for reliability."""

    @staticmethod
    def assess_capture_quality(
        landmarks: List[Dict],
        image_brightness: float = None,
        angles_captured: List[str] = None,
    ) -> Dict:
        """Assess overall capture quality.

        Args:
            landmarks: Detected pose landmarks
            image_brightness: Average brightness (0-255) or None
            angles_captured: List of captured angles (e.g., ["front", "side"])

        Returns:
            Quality assessment dict
        """
        if not landmarks or len(landmarks) < 33:
            return {"quality": "poor", "score": 0.0, "reason": "Insufficient landmarks"}

        # Calculate average visibility
        avg_visibility = sum(l.get("visibility", 0) for l in landmarks) / len(landmarks)

        # Assess brightness
        brightness_ok = True
        if image_brightness is not None:
            brightness_ok = 50 < image_brightness < 200

        # Assess angles
        angles_ok = angles_captured and len(angles_captured) >= 2

        # Overall score
        score = avg_visibility * 0.7
        if brightness_ok:
            score += 0.2
        if angles_ok:
            score += 0.1

        quality = "excellent" if score > 0.85 else "good" if score > 0.70 else "fair" if score > 0.50 else "poor"

        return {
            "quality": quality,
            "score": score,
            "avg_visibility": avg_visibility,
            "lighting_ok": brightness_ok,
            "angles_ok": angles_ok,
            "recommendations": MeasurementQuality._get_recommendations(
                avg_visibility, brightness_ok, angles_ok
            ),
        }

    @staticmethod
    def _get_recommendations(
        avg_visibility: float,
        brightness_ok: bool,
        angles_ok: bool,
    ) -> List[str]:
        """Generate recommendations for improvement.

        Args:
            avg_visibility: Average pose visibility
            brightness_ok: Whether lighting is adequate
            angles_ok: Whether all angles captured

        Returns:
            List of recommendations
        """
        recommendations = []

        if avg_visibility < 0.7:
            recommendations.append("Improve pose detection. Try better lighting or adjust camera angle.")
        if not brightness_ok:
            recommendations.append("Adjust lighting. Image is too dark or too bright.")
        if not angles_ok:
            recommendations.append("Capture from all three angles: front, side, and back.")

        if not recommendations:
            recommendations.append("Quality is good. Ready to process.")

        return recommendations

    @staticmethod
    def consistency_check(
        measurement1: float,
        measurement2: float,
        tolerance_percent: float = 5.0,
    ) -> Dict:
        """Check consistency between repeated measurements.

        Args:
            measurement1: First measurement value
            measurement2: Second measurement value
            tolerance_percent: Acceptable variance percentage

        Returns:
            Consistency check result
        """
        if measurement1 == 0 or measurement2 == 0:
            return {"consistent": False, "variance_percent": 0, "reason": "Invalid measurement"}

        variance_percent = abs(measurement1 - measurement2) / max(measurement1, measurement2) * 100
        is_consistent = variance_percent <= tolerance_percent

        return {
            "consistent": is_consistent,
            "variance_percent": variance_percent,
            "tolerance_percent": tolerance_percent,
            "reason": "Measurements are consistent" if is_consistent else f"Variance {variance_percent:.1f}% exceeds {tolerance_percent}%",
        }
