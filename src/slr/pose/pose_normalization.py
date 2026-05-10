"""Normalization helpers for graph-based pose models."""

from __future__ import annotations

import numpy as np


def center_keypoints(keypoints: np.ndarray, center_index: int) -> np.ndarray:
    """Center a keypoint sequence around a chosen landmark index."""

    centered = keypoints.copy()
    origin = centered[:, center_index : center_index + 1, :2]
    centered[:, :, :2] = centered[:, :, :2] - origin
    return centered
