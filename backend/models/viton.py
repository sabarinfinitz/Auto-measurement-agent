"""VITON placeholder module.

This module provides a simple placeholder for a virtual try-on pipeline.
It does not run a real VITON-HD model — instead it creates a simple output
image by copying / compositing inputs. Replace with real model integration
when ready.
"""
from io import BytesIO
import os
import uuid
from PIL import Image


def apply_tryon(person_image_bytes: bytes, garment_image_bytes: bytes | None = None, output_dir: str = '.') -> str:
    """Create a placeholder try-on image and save to output_dir.

    Returns the absolute path to the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    person_img = Image.open(BytesIO(person_image_bytes)).convert('RGBA')

    if garment_image_bytes:
        try:
            garment_img = Image.open(BytesIO(garment_image_bytes)).convert('RGBA')
            # Resize garment to ~40% of person width and paste near upper-center
            gw = int(person_img.width * 0.4)
            gh = int(garment_img.height * (gw / garment_img.width))
            garment_img = garment_img.resize((gw, gh), Image.LANCZOS)
            paste_x = (person_img.width - gw) // 2
            paste_y = int(person_img.height * 0.2)
            base = person_img.copy()
            base.alpha_composite(garment_img, dest=(paste_x, paste_y))
            out = base.convert('RGB')
        except Exception:
            out = person_img.convert('RGB')
    else:
        out = person_img.convert('RGB')

    filename = f"tryon_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(output_dir, filename)
    out.save(out_path, format='JPEG', quality=85)
    return out_path
