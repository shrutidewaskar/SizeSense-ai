"""Utility helpers for image loading and conversion."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg", "webp"}


def load_image(uploaded_file) -> Optional[Image.Image]:
    """Load an uploaded Streamlit image file into a PIL image."""
    if uploaded_file is None:
        return None

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        return None


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to OpenCV BGR format."""
    rgb_image = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL image."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def resize_pil_image(image: Image.Image, max_width: int = 1200) -> Image.Image:
    """Resize an image for consistent display without distorting aspect ratio."""
    width, height = image.size
    if width <= max_width:
        return image

    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_bytes(image: Image.Image, format_name: str = "PNG") -> bytes:
    """Serialize a PIL image to bytes for download or caching."""
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()
