"""Sequence transforms for skeleton branch inputs."""

from __future__ import annotations

import numpy as np


def pad_or_trim(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad with zeros or trim to a fixed number of frames."""

    current_frames = sequence.shape[0]
    if current_frames == target_frames:
        return sequence
    if current_frames > target_frames:
        return sequence[:target_frames]

    pad_shape = (target_frames - current_frames, *sequence.shape[1:])
    padding = np.zeros(pad_shape, dtype=sequence.dtype)
    return np.concatenate([sequence, padding], axis=0)
