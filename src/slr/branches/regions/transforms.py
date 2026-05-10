"""Transforms for face and hand crop sequences."""

from __future__ import annotations

import numpy as np


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Scale uint8 images to the ``[0, 1]`` range."""

    return image.astype(np.float32) / 255.0
