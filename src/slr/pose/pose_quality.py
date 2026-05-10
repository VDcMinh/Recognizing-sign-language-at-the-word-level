"""Pose quality heuristics and future audit hooks."""

from __future__ import annotations

import numpy as np


def confidence_coverage(keypoints: np.ndarray, threshold: float = 0.2) -> float:
    """Compute the fraction of pose points whose confidence exceeds ``threshold``."""

    if keypoints.size == 0:
        return 0.0
    scores = keypoints[..., 2]
    return float((scores >= threshold).mean())
