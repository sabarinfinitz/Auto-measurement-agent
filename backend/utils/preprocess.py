"""Image preprocessing helpers."""
from io import BytesIO
from typing import Tuple
import numpy as np
import cv2
from PIL import Image


def bytes_to_cv2(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes to cv2 image")
    return img


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert('RGB')


def save_bytes_to_file(image_bytes: bytes, path: str) -> None:
    with open(path, 'wb') as f:
        f.write(image_bytes)
