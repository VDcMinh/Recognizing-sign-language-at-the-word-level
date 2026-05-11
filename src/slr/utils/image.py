"""Image helpers used by region and visualization pipelines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a BGR numpy array."""

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def ensure_uint8_bgr_or_rgb(image: np.ndarray) -> np.ndarray:
    """Return an image array as contiguous ``uint8`` data."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(clipped)


def resize_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    letterbox_value: int = 0,
) -> np.ndarray:
    """Resize with aspect ratio preserved, then center-pad to the target size."""

    image_uint8 = ensure_uint8_bgr_or_rgb(image)
    height, width = image_uint8.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image must have a positive width and height.")

    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image_uint8,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (target_height, target_width, 3),
        fill_value=int(letterbox_value),
        dtype=np.uint8,
    )
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def save_image(path: str | Path, image: np.ndarray, jpg_quality: int = 95) -> None:
    """Write an image to disk, creating parent directories when needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = ensure_uint8_bgr_or_rgb(image)

    suffix = target.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    if not cv2.imwrite(str(target), image_uint8, params):
        raise OSError(f"Could not write image: {target}")
