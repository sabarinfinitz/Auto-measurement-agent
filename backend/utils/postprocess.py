"""Helpers to format model outputs for clients."""
from typing import List, Dict


def format_landmarks_for_client(landmarks: List[Dict]) -> Dict:
    """Ensure landmarks list is JSON-serializable and summarise some info."""
    count = len(landmarks)
    return {"count": count, "landmarks": landmarks}
