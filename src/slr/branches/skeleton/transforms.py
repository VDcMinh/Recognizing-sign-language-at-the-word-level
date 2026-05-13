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


def fix_sequence_length(
    keypoints: np.ndarray,
    target_num_frames: int = 150,
    short_strategy: str = "repeat",
    long_strategy: str = "head",
) -> np.ndarray:
    """Convert one variable-length pose sequence into a deterministic fixed length."""

    sequence = np.asarray(keypoints, dtype=np.float32)
    num_frames = int(sequence.shape[0])
    if num_frames == 0:
        raise ValueError("empty_pose_sequence")
    if num_frames == target_num_frames:
        return sequence

    if num_frames > target_num_frames:
        if long_strategy != "head":
            raise ValueError(f"Unsupported long sequence strategy: {long_strategy}")
        return sequence[:target_num_frames]

    if short_strategy != "repeat":
        raise ValueError(f"Unsupported short sequence strategy: {short_strategy}")

    repeats = int(np.ceil(target_num_frames / num_frames))
    tiled = np.tile(sequence, (repeats, 1, 1))
    return tiled[:target_num_frames]


def to_graph_tensor_ctvm(
    keypoints: np.ndarray,
    num_persons: int = 1,
) -> np.ndarray:
    """Convert ``(T, V, C)`` normalized keypoints into ``(C, T, V, M)``."""

    if num_persons != 1:
        raise ValueError("Only num_persons=1 is supported for this preprocessing step.")

    sequence = np.asarray(keypoints, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[2] != 3:
        raise ValueError(f"Expected normalized keypoints with shape (T, V, 3), got {sequence.shape}.")

    tensor = np.transpose(sequence, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=-1)
    return np.asarray(tensor, dtype=np.float32)
