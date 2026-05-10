"""Utilities for selecting reduced keypoint subsets from shared pose outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from slr.pose.pose_schema import KEYPOINT_SET_INDICES


def select_keypoints(keypoints: np.ndarray, keypoint_set: str) -> np.ndarray:
    """Select a named keypoint subset from a ``(T, 133, C)`` pose array.

    Parameters
    ----------
    keypoints:
        Pose array whose second dimension corresponds to the 133 RTMW-l
        whole-body landmarks.
    keypoint_set:
        Named subset such as ``selected_27``, ``selected_31``, or
        ``selected_49``.

    Returns
    -------
    np.ndarray
        Pose array sliced to the requested subset.

    Raises
    ------
    KeyError
        If the named keypoint set is unknown.
    NotImplementedError
        If the subset mapping exists conceptually but the exact landmark indices
        are still pending finalization.
    """

    if keypoint_set not in KEYPOINT_SET_INDICES:
        raise KeyError(f"Unknown keypoint set: {keypoint_set}")

    indices: Any = KEYPOINT_SET_INDICES[keypoint_set]
    if not indices:
        raise NotImplementedError(
            f"Keypoint mapping for '{keypoint_set}' is still a TODO."
        )

    return keypoints[:, list(indices), ...]
