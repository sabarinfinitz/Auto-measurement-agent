"""MediaPipe pose extraction utilities.

Provides a simple function `extract_pose_from_image(image_bytes)` that returns
pose landmarks formatted as JSON-serializable dict. This is intended as a
small, self-contained helper for the TailorAI starter project.
"""
from io import BytesIO
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def _read_image_bytes_to_bgr(image_bytes: bytes):
    """Decode image bytes to an OpenCV BGR image (numpy.ndarray)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def extract_pose_from_image(image_bytes: bytes) -> dict:
    """Run MediaPipe pose on the input image bytes and return landmarks.

    Returns a dict:
    {
      'image_width': int,
      'image_height': int,
      'landmarks': [ { 'index': i, 'x': ..., 'y': ..., 'z': ..., 'visibility': ... , 'x_px': ..., 'y_px': ... }, ... ]
    }
    """
    img = _read_image_bytes_to_bgr(image_bytes)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return {"image_width": w, "image_height": h, "landmarks": []}

    landmarks = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        x = lm.x
        y = lm.y
        z = lm.z
        vis = getattr(lm, 'visibility', None)
        landmarks.append({
            "index": i,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "visibility": float(vis) if vis is not None else None,
            "x_px": int(x * w),
            "y_px": int(y * h),
        })

    return {"image_width": w, "image_height": h, "landmarks": landmarks}
