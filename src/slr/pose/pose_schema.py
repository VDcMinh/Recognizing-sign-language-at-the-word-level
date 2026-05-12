"""Pose schema constants for shared RTMW-l whole-body representations."""

from __future__ import annotations

import numpy as np


RTMW_L_BACKEND = "rtmw_l"
WHOLEBODY_133 = "wholebody_133"
WHOLEBODY_133_LAYOUT = WHOLEBODY_133
WHOLEBODY_133_NUM_KEYPOINTS = 133

WHOLEBODY_133_REGION_INDICES = {
    "body": tuple(range(0, 17)),
    "foot": tuple(range(17, 23)),
    "face": tuple(range(23, 91)),
    "left_hand": tuple(range(91, 112)),
    "right_hand": tuple(range(112, 133)),
}

# Placeholders kept for future branch-specific reductions.
SELECTED_27 = ()
SELECTED_31 = ()
SELECTED_49 = ()

KEYPOINT_SET_SIZES = {
    "selected_27": 27,
    "selected_31": 31,
    "selected_49": 49,
}

KEYPOINT_SET_INDICES = {
    "selected_27": SELECTED_27,
    "selected_31": SELECTED_31,
    "selected_49": SELECTED_49,
}


def get_keypoint_region_indices(region_name: str) -> tuple[int, ...]:
    """Return the keypoint indices for a named whole-body region."""

    try:
        return WHOLEBODY_133_REGION_INDICES[region_name]
    except KeyError as exc:  # pragma: no cover - simple lookup guard
        raise KeyError(f"Unknown keypoint region: {region_name}") from exc


def validate_keypoints_shape(
    keypoints: np.ndarray,
    expected_v: int = WHOLEBODY_133_NUM_KEYPOINTS,
) -> None:
    """Raise when the keypoint tensor does not have shape ``(T, V, 3)``."""

    if keypoints.ndim != 3:
        raise ValueError(
            f"Expected keypoints with 3 dimensions (T, V, C), got {keypoints.shape}."
        )
    if keypoints.shape[1] != expected_v:
        raise ValueError(
            f"Expected {expected_v} keypoints, got shape {keypoints.shape}."
        )
    if keypoints.shape[2] != 3:
        raise ValueError(
            f"Expected keypoint channels (x, y, score), got shape {keypoints.shape}."
        )
