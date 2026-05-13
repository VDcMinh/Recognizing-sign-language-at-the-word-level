"""Pose schema constants for shared RTMW-l whole-body representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


RTMW_L_BACKEND = "rtmw_l"
WHOLEBODY_133 = "wholebody_133"
WHOLEBODY_133_LAYOUT = WHOLEBODY_133
WHOLEBODY_133_NUM_KEYPOINTS = 133

WHOLEBODY_133_BODY_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
WHOLEBODY_133_FOOT_NAMES = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
WHOLEBODY_133_FACE_NAMES = tuple(f"face_{index}" for index in range(68))
WHOLEBODY_133_LEFT_HAND_NAMES = tuple(f"left_hand_{index}" for index in range(21))
WHOLEBODY_133_RIGHT_HAND_NAMES = tuple(f"right_hand_{index}" for index in range(21))
WHOLEBODY_133_KEYPOINT_NAMES = (
    WHOLEBODY_133_BODY_NAMES
    + WHOLEBODY_133_FOOT_NAMES
    + WHOLEBODY_133_FACE_NAMES
    + WHOLEBODY_133_LEFT_HAND_NAMES
    + WHOLEBODY_133_RIGHT_HAND_NAMES
)

WHOLEBODY_133_REGION_INDICES = {
    "body": tuple(range(0, 17)),
    "foot": tuple(range(17, 23)),
    "face": tuple(range(23, 91)),
    "left_hand": tuple(range(91, 112)),
    "right_hand": tuple(range(112, 133)),
}

BODY_7 = (
    0,   # nose
    5,   # left_shoulder
    6,   # right_shoulder
    7,   # left_elbow
    8,   # right_elbow
    9,   # left_wrist
    10,  # right_wrist
)
HAND_10_LOCAL = (
    2, 4,    # thumb mid/tip
    6, 8,    # index mid/tip
    10, 12,  # middle mid/tip
    14, 16,  # ring mid/tip
    18, 20,  # pinky mid/tip
)
LEFT_HAND_10 = tuple(91 + index for index in HAND_10_LOCAL)
RIGHT_HAND_10 = tuple(112 + index for index in HAND_10_LOCAL)

SELECTED_27 = BODY_7 + LEFT_HAND_10 + RIGHT_HAND_10
SELECTED_31: tuple[int, ...] = ()
SELECTED_49: tuple[int, ...] = ()

SELECTED_27_NOTE = (
    "Default SAM-SLR-style selected_27 mapping: "
    "7 upper-body nodes (nose, shoulders, elbows, wrists) plus "
    "10 left-hand nodes and 10 right-hand nodes chosen as finger mid/tip landmarks. "
    "This mapping should be verified with visualization before treating it as canonical."
)

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

KEYPOINT_SET_NOTES = {
    "selected_27": SELECTED_27_NOTE,
    "selected_31": "selected_31 mapping is not implemented yet.",
    "selected_49": "selected_49 mapping is not implemented yet.",
}


def validate_keypoint_indices(
    indices: Sequence[int],
    num_source_keypoints: int = WHOLEBODY_133_NUM_KEYPOINTS,
) -> tuple[int, ...]:
    """Validate one reduced keypoint mapping against the source layout."""

    if not indices:
        raise ValueError("Keypoint indices must not be empty.")

    validated = tuple(int(index) for index in indices)
    invalid = [index for index in validated if index < 0 or index >= num_source_keypoints]
    if invalid:
        raise ValueError(
            f"Keypoint indices out of range for {num_source_keypoints} source keypoints: {invalid}"
        )
    if len(set(validated)) != len(validated):
        raise ValueError(f"Keypoint indices contain duplicates: {validated}")
    return validated


validate_keypoint_indices(SELECTED_27)
assert len(SELECTED_27) == 27


def get_keypoint_region_indices(region_name: str) -> tuple[int, ...]:
    """Return the keypoint indices for a named whole-body region."""

    try:
        return WHOLEBODY_133_REGION_INDICES[region_name]
    except KeyError as exc:  # pragma: no cover - simple lookup guard
        raise KeyError(f"Unknown keypoint region: {region_name}") from exc


def get_keypoint_indices(keypoint_set: str) -> tuple[int, ...]:
    """Return validated source indices for a named reduced keypoint set."""

    try:
        indices = KEYPOINT_SET_INDICES[keypoint_set]
    except KeyError as exc:
        raise KeyError(f"Unknown keypoint set: {keypoint_set}") from exc
    if not indices:
        raise NotImplementedError(f"Keypoint mapping for '{keypoint_set}' is still a TODO.")
    return validate_keypoint_indices(indices)


def get_keypoint_names(indices: Sequence[int]) -> tuple[str, ...]:
    """Resolve source-layout keypoint names for one reduced mapping."""

    validated = validate_keypoint_indices(indices)
    return tuple(WHOLEBODY_133_KEYPOINT_NAMES[index] for index in validated)


def get_keypoint_set_names(keypoint_set: str) -> tuple[str, ...]:
    """Resolve source-layout keypoint names for a named reduced mapping."""

    return get_keypoint_names(get_keypoint_indices(keypoint_set))


def get_keypoint_set_note(keypoint_set: str) -> str:
    """Return the documentation note for a reduced keypoint mapping."""

    return KEYPOINT_SET_NOTES.get(keypoint_set, "")


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
