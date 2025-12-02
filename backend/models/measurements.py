"""Professional tailor shop measurement extraction from 33 MediaPipe landmarks.

Extracts 20+ measurements:
- Upper body: shoulders, chest, sleeves, neck
- Torso: back length, waist, torso length
- Lower body: hips, inseam, thigh, knee
- Full body: height, arm span, posture
"""

from typing import Dict, List, Optional, Tuple
import math
from utils.geometry import GeometryCalculator, MeasurementQuality


class TailorMeasurementExtractor:
    """Extract professional tailor shop measurements from pose landmarks."""

    # MediaPipe landmark indices
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_MOUTH_CORNER = 9
    RIGHT_MOUTH_CORNER = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __init__(self, calibration_data: Optional[Dict] = None):
        """Initialize extractor with optional calibration data.

        Args:
            calibration_data: Dict with 'cm_per_pixel' for unit conversion
        """
        # Keep both a simple calibration mapping and an explicit calibration_data attribute
        self.calibration = calibration_data or {"cm_per_pixel": 0.26}
        self.calibration_data = calibration_data or {"cm_per_pixel": 0.26}
        self.geom = GeometryCalculator()

    def _px_to_cm(self, pixels: float) -> float:
        """Convert pixels to centimeters using calibration."""
        return pixels * self.calibration.get("cm_per_pixel", 0.26)

    def _get_landmark(self, landmarks: List[Dict], idx: int) -> Optional[Dict]:
        """Safely get landmark by index."""
        if idx < len(landmarks):
            return landmarks[idx]
        return None

    def extract_all_measurements(
        self,
        landmarks_front: List[Dict],
        landmarks_side: Optional[List[Dict]] = None,
        landmarks_back: Optional[List[Dict]] = None,
    ) -> Dict:
        """Extract all 20+ measurements from poses.

        Args:
            landmarks_front: Front view pose (33 landmarks)
            landmarks_side: Side view pose (optional)
            landmarks_back: Back view pose (optional)

        Returns:
            Dict with all measurements in cm
        """
        measurements = {
            "upper_body": {},
            "torso": {},
            "lower_body": {},
            "full_body": {},
            "metadata": {
                "has_front": bool(landmarks_front),
                "has_side": bool(landmarks_side),
                "has_back": bool(landmarks_back),
                "extraction_quality": {},
            },
        }

        if landmarks_front:
            measurements["upper_body"].update(self._extract_upper_body(landmarks_front))
            measurements["torso"].update(self._extract_torso_front(landmarks_front))
            measurements["lower_body"].update(self._extract_lower_body_front(landmarks_front))
            measurements["full_body"].update(self._extract_full_body_front(landmarks_front))

        if landmarks_side:
            measurements["torso"].update(self._extract_torso_side(landmarks_side))
            measurements["full_body"].update(self._extract_full_body_side(landmarks_side))

        if landmarks_back:
            measurements["upper_body"].update(self._extract_upper_body_back(landmarks_back))
            measurements["torso"].update(self._extract_torso_back(landmarks_back))

        # Calculate quality metrics
        measurements["metadata"]["extraction_quality"] = self._assess_measurement_quality(
            landmarks_front, landmarks_side, landmarks_back
        )

        return measurements

    def _extract_upper_body(self, landmarks: List[Dict]) -> Dict:
        """Extract upper body measurements from front view."""
        measurements = {}

        # Shoulder width
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        r_shoulder = self._get_landmark(landmarks, self.RIGHT_SHOULDER)
        if l_shoulder and r_shoulder:
            shoulder_px = self.geom.distance_2d(l_shoulder, r_shoulder)
            measurements["shoulder_width_cm"] = self._px_to_cm(shoulder_px)
            measurements["shoulder_width_px"] = shoulder_px

        # Chest width (at shoulder level, lateral extent)
        if l_shoulder and r_shoulder:
            measurements["chest_width_cm"] = measurements.get("shoulder_width_cm", 0)

        # Chest circumference estimation (needs side view for depth)
        # Will be overridden by side view data if available
        if "chest_width_cm" in measurements:
            measurements["chest_circumference_cm"] = measurements["chest_width_cm"] * 2.5  # Empirical factor

        # Neck circumference (rough estimate from head width)
        l_eye = self._get_landmark(landmarks, self.LEFT_EYE)
        r_eye = self._get_landmark(landmarks, self.RIGHT_EYE)
        if l_eye and r_eye:
            head_width_px = self.geom.distance_2d(l_eye, r_eye) * 4  # Eye width to head width approximation
            neck_px = head_width_px * 0.65  # Neck is roughly 65% of head width
            measurements["neck_circumference_cm"] = self._px_to_cm(neck_px * math.pi)

        # Sleeve length (shoulder to wrist)
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        l_wrist = self._get_landmark(landmarks, self.LEFT_WRIST)
        if l_shoulder and l_wrist:
            sleeve_px = self.geom.distance_2d(l_shoulder, l_wrist)
            measurements["sleeve_length_cm"] = self._px_to_cm(sleeve_px)

        # Armpit to wrist (another sleeve reference)
        l_shoulder_y = l_shoulder.get("y_px", 0)
        armpit_y = l_shoulder_y + (self.geom.distance_2d(l_shoulder, self._get_landmark(landmarks, self.LEFT_HIP)) / 3)
        measurements["half_sleeve_cm"] = measurements.get("sleeve_length_cm", 0) * 0.5

        return measurements

    def _extract_upper_body_back(self, landmarks: List[Dict]) -> Dict:
        """Extract additional upper body measurements from back view."""
        measurements = {}

        # Back shoulder width (should match front but can differ due to posture)
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        r_shoulder = self._get_landmark(landmarks, self.RIGHT_SHOULDER)
        if l_shoulder and r_shoulder:
            back_shoulder_px = self.geom.distance_2d(l_shoulder, r_shoulder)
            measurements["back_shoulder_width_cm"] = self._px_to_cm(back_shoulder_px)

        return measurements

    def _extract_torso_front(self, landmarks: List[Dict]) -> Dict:
        """Extract torso measurements from front view."""
        measurements = {}

        # Torso length (shoulder to hip)
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        r_shoulder = self._get_landmark(landmarks, self.RIGHT_SHOULDER)
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)

        if l_shoulder and l_hip:
            shoulder_mid = self.geom.midpoint(
                l_shoulder,
                r_shoulder if r_shoulder else l_shoulder
            )
            torso_px = self.geom.distance_2d(shoulder_mid, l_hip)
            measurements["torso_length_cm"] = self._px_to_cm(torso_px)

        # Waist width (estimate from hip position, at waist height)
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)
        r_hip = self._get_landmark(landmarks, self.RIGHT_HIP)
        if l_hip and r_hip:
            hip_px = self.geom.distance_2d(l_hip, r_hip)
            measurements["waist_width_cm"] = self._px_to_cm(hip_px * 0.9)  # Waist is usually 90% of hip width
            measurements["waist_circumference_cm"] = self._px_to_cm(hip_px * 0.9) * 2.8  # Empirical circumference factor

        # Hip width and circumference
        if l_hip and r_hip:
            hip_px = self.geom.distance_2d(l_hip, r_hip)
            measurements["hip_width_cm"] = self._px_to_cm(hip_px)
            measurements["hip_circumference_cm"] = self._px_to_cm(hip_px) * 3.0  # Empirical factor

        return measurements

    def _extract_torso_side(self, landmarks: List[Dict]) -> Dict:
        """Extract torso depth measurements from side view."""
        measurements = {}

        # Chest depth (front to back)
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        nose = self._get_landmark(landmarks, self.NOSE)

        if l_shoulder and nose:
            chest_depth_px = abs(nose.get("x_px", 0) - l_shoulder.get("x_px", 0))
            measurements["chest_depth_cm"] = self._px_to_cm(chest_depth_px)

            # Recalculate chest circumference with depth
            chest_width = measurements.get("chest_width_cm", 0)
            if chest_width and measurements["chest_depth_cm"]:
                measurements["chest_circumference_cm"] = self.geom.estimate_circumference(
                    measurements["chest_depth_cm"],
                    chest_width,
                )

        # Waist depth
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)
        if l_hip and l_shoulder:
            hip_x_offset = abs(nose.get("x_px", 0) - l_hip.get("x_px", 0)) if nose else 0
            measurements["waist_depth_cm"] = self._px_to_cm(hip_x_offset)

        # Back length (from top of shoulder blade to hip)
        measurements["back_length_cm"] = measurements.get("torso_length_cm", 0)

        return measurements

    def _extract_torso_back(self, landmarks: List[Dict]) -> Dict:
        """Extract back-specific torso measurements."""
        measurements = {}

        # Back shoulder blade distance
        l_shoulder = self._get_landmark(landmarks, self.LEFT_SHOULDER)
        r_shoulder = self._get_landmark(landmarks, self.RIGHT_SHOULDER)

        if l_shoulder and r_shoulder:
            back_shoulder_px = self.geom.distance_2d(l_shoulder, r_shoulder)
            measurements["back_width_cm"] = self._px_to_cm(back_shoulder_px)

        return measurements

    def _extract_lower_body_front(self, landmarks: List[Dict]) -> Dict:
        """Extract lower body measurements from front view."""
        measurements = {}

        # Inseam length (hip to ankle)
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)
        l_ankle = self._get_landmark(landmarks, self.LEFT_ANKLE)

        if l_hip and l_ankle:
            inseam_px = self.geom.distance_2d(l_hip, l_ankle)
            measurements["inseam_length_cm"] = self._px_to_cm(inseam_px)

        # Thigh circumference (estimate from hip)
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)
        r_hip = self._get_landmark(landmarks, self.RIGHT_HIP)

        if l_hip and r_hip:
            hip_px = self.geom.distance_2d(l_hip, r_hip)
            # Thigh width is typically 75-80% of hip width
            thigh_px = hip_px * 0.75
            measurements["thigh_circumference_cm"] = self._px_to_cm(thigh_px) * 3.2

        # Knee circumference (estimate)
        l_knee = self._get_landmark(landmarks, self.LEFT_KNEE)
        r_knee = self._get_landmark(landmarks, self.RIGHT_KNEE)

        if l_knee and r_knee:
            knee_px = self.geom.distance_2d(l_knee, r_knee)
            measurements["knee_circumference_cm"] = self._px_to_cm(knee_px) * 3.0

        return measurements

    def _extract_full_body_front(self, landmarks: List[Dict]) -> Dict:
        """Extract full body measurements from front view."""
        measurements = {}

        # Total height (nose to ankle)
        nose = self._get_landmark(landmarks, self.NOSE)
        l_ankle = self._get_landmark(landmarks, self.LEFT_ANKLE)
        r_ankle = self._get_landmark(landmarks, self.RIGHT_ANKLE)

        if nose and (l_ankle or r_ankle):
            ankle = l_ankle if l_ankle else r_ankle
            height_px = self.geom.distance_2d(nose, ankle)
            measurements["height_cm"] = self._px_to_cm(height_px)

        return measurements

    def _extract_full_body_side(self, landmarks: List[Dict]) -> Dict:
        """Extract full body measurements from side view."""
        measurements = {}

        # Posture assessment
        nose = self._get_landmark(landmarks, self.NOSE)
        l_ankle = self._get_landmark(landmarks, self.LEFT_ANKLE)
        r_ankle = self._get_landmark(landmarks, self.RIGHT_ANKLE)
        l_hip = self._get_landmark(landmarks, self.LEFT_HIP)

        if nose and l_hip and (l_ankle or r_ankle):
            # Calculate forward lean
            ankle = l_ankle if l_ankle else r_ankle
            hip_to_ankle_x = ankle.get("x_px", 0) - l_hip.get("x_px", 0)
            hip_to_nose_x = nose.get("x_px", 0) - l_hip.get("x_px", 0)

            if hip_to_ankle_x != 0:
                lean_angle = math.degrees(math.atan(hip_to_nose_x / abs(hip_to_ankle_x)))
                measurements["posture_lean_angle_deg"] = lean_angle
                measurements["posture_score"] = "good" if abs(lean_angle) < 10 else "fair" if abs(lean_angle) < 20 else "poor"

        return measurements

    def _assess_measurement_quality(
        self,
        landmarks_front: Optional[List[Dict]],
        landmarks_side: Optional[List[Dict]],
        landmarks_back: Optional[List[Dict]],
    ) -> Dict:
        """Assess quality of extracted measurements."""
        quality = {}

        if landmarks_front:
            # Use GeometryCalculator helper to compute average visibility
            quality["front_visibility"] = GeometryCalculator.average_visibility(
                landmarks_front,
                [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_ANKLE, self.RIGHT_ANKLE]
            )

        if landmarks_side:
            quality["side_visibility"] = GeometryCalculator.average_visibility(landmarks_side, list(range(33)))

        if landmarks_back:
            quality["back_visibility"] = GeometryCalculator.average_visibility(landmarks_back, list(range(33)))

        quality["overall"] = sum(quality.values()) / len(quality) if quality else 0

        return quality
