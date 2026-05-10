"""Pose schema constants for shared RTMW-l whole-body representations."""

from __future__ import annotations

RTMW_L_BACKEND = "rtmw_l"
WHOLEBODY_133 = "wholebody_133"
WHOLEBODY_133_NUM_KEYPOINTS = 133

# TODO: replace placeholder tuples with finalized landmark indices once the
# exact mapping from RTMW-l / MMPose output to project keypoint subsets is fixed.
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
