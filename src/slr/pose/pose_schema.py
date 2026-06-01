"""Pose schema constants for shared RTMW-l whole-body representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


RTMW_L_BACKEND = "rtmw_l"
WHOLEBODY_133 = "wholebody_133"
WHOLEBODY_133_LAYOUT = WHOLEBODY_133
WHOLEBODY_133_NUM_KEYPOINTS = 133

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4

BODY_17 = tuple(range(0, 17))
FOOT_6 = tuple(range(17, 23))
FACE_68 = tuple(range(23, 91))
LEFT_HAND_21 = tuple(range(91, 112))
RIGHT_HAND_21 = tuple(range(112, 133))
BODY_FACE_ANCHORS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

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
    "body": BODY_17,
    "foot": FOOT_6,
    "face": FACE_68,
    "left_hand": LEFT_HAND_21,
    "right_hand": RIGHT_HAND_21,
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
MOUTH_4 = (
    71,  # mouth_left_corner = 23 + 48
    77,  # mouth_right_corner = 23 + 54
    74,  # upper_lip = 23 + 51
    80,  # lower_lip = 23 + 57
)

SELECTED_27 = BODY_7 + LEFT_HAND_10 + RIGHT_HAND_10
SELECTED_31 = SELECTED_27 + MOUTH_4
SELECTED_49: tuple[int, ...] = ()

SELECTED_27_NOTE = (
    "Default SAM-SLR-style selected_27 mapping: "
    "7 upper-body nodes (nose, shoulders, elbows, wrists) plus "
    "10 left-hand nodes and 10 right-hand nodes chosen as finger mid/tip landmarks. "
    "This mapping should be verified with visualization before treating it as canonical."
)
SELECTED_31_NOTE = (
    "Default selected_31 mapping: selected_27 plus 4 mouth landmarks "
    "(mouth_left_corner, mouth_right_corner, upper_lip, lower_lip) using the "
    "default COCO-WholeBody face-landmark indexing. This mapping should be "
    "verified with visualization before treating it as canonical."
)
KEYPOINT_NAME_OVERRIDES = {
    71: "mouth_left_corner",
    77: "mouth_right_corner",
    74: "upper_lip",
    80: "lower_lip",
}
KEYPOINT_SET_COMPONENT_INDICES = {
    "selected_27": {},
    "selected_31": {
        "mouth": MOUTH_4,
    },
    "selected_49": {},
}

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
    "selected_31": SELECTED_31_NOTE,
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
validate_keypoint_indices(SELECTED_31)
assert len(SELECTED_27) == 27
assert len(SELECTED_31) == 31


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
    return tuple(
        KEYPOINT_NAME_OVERRIDES.get(index, WHOLEBODY_133_KEYPOINT_NAMES[index])
        for index in validated
    )


def get_keypoint_set_names(keypoint_set: str) -> tuple[str, ...]:
    """Resolve source-layout keypoint names for a named reduced mapping."""

    return get_keypoint_names(get_keypoint_indices(keypoint_set))


def get_keypoint_set_note(keypoint_set: str) -> str:
    """Return the documentation note for a reduced keypoint mapping."""

    return KEYPOINT_SET_NOTES.get(keypoint_set, "")


def get_keypoint_component_indices(
    keypoint_set: str,
    component_name: str,
) -> tuple[int, ...]:
    """Return auxiliary component indices, such as mouth landmarks, for one set."""

    try:
        components = KEYPOINT_SET_COMPONENT_INDICES[keypoint_set]
    except KeyError as exc:
        raise KeyError(f"Unknown keypoint set: {keypoint_set}") from exc
    indices = components.get(component_name, ())
    return validate_keypoint_indices(indices) if indices else ()


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
