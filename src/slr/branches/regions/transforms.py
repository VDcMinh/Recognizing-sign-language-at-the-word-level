"""Transforms for face and hand crop sequences."""

from __future__ import annotations

import numpy as np


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Scale uint8 images to the ``[0, 1]`` range."""

    return image.astype(np.float32) / 255.0


def normalize_region_clip_uint8(clip: np.ndarray) -> np.ndarray:
    """Scale a region clip tensor from ``uint8`` to ``float32`` in ``[0, 1]``."""

    return clip.astype(np.float32) / 255.0
