"""Utilities for selecting reduced keypoint subsets from shared pose outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from slr.pose.pose_schema import (
    WHOLEBODY_133_LAYOUT,
    get_keypoint_indices,
    get_keypoint_names,
)


def select_keypoints(keypoints: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Slice a pose array down to a reduced keypoint set."""

    selected = keypoints[:, list(indices), ...]
    return np.asarray(selected, dtype=np.float32)


def select_keypoints_by_name(keypoints: np.ndarray, keypoint_set: str) -> np.ndarray:
    """Slice a pose array using one named reduced keypoint set."""

    return select_keypoints(keypoints, get_keypoint_indices(keypoint_set))


def build_selected_keypoints_npz_payload(
    keypoints: np.ndarray,
    keypoint_set: str,
    sample_id: str,
    video_id: str,
    gloss: str,
    class_id: int,
    split: str,
    source_layout: str = WHOLEBODY_133_LAYOUT,
) -> dict[str, Any]:
    """Build a stable payload for one selected-keypoints ``.npz`` file."""

    indices = get_keypoint_indices(keypoint_set)
    names = get_keypoint_names(indices)
    return {
        "keypoints": np.asarray(keypoints, dtype=np.float32),
        "selected_indices": np.asarray(indices, dtype=np.int32),
        "selected_names": np.asarray(names),
        "keypoint_set": np.asarray(keypoint_set),
        "source_layout": np.asarray(source_layout),
        "sample_id": np.asarray(sample_id),
        "video_id": np.asarray(video_id),
        "gloss": np.asarray(gloss),
        "class_id": np.asarray(int(class_id), dtype=np.int32),
        "split": np.asarray(split),
        "num_frames_original": np.asarray(int(keypoints.shape[0]), dtype=np.int32),
    }
