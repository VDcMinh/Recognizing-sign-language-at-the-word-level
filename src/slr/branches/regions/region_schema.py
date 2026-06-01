"""Schema constants for region branch assets."""

from __future__ import annotations

from slr.pose.pose_schema import (
    BODY_FACE_ANCHORS,
    FACE_68,
    LEFT_HAND_21,
    RIGHT_HAND_21,
)

REGION_NAMES = ("left_hand", "right_hand", "face")
REGION_TO_INDEX = {name: index for index, name in enumerate(REGION_NAMES)}
NUM_REGIONS = len(REGION_NAMES)
NUM_CHANNELS = 3
DEFAULT_CLIP_LEN = 64
DEFAULT_CROP_SIZE = 112
DEFAULT_IMAGE_DTYPE = "uint8"
TENSOR_FORMAT = "R,C,T,H,W"
DEFAULT_TENSOR_SHAPE = (
    NUM_REGIONS,
    NUM_CHANNELS,
    DEFAULT_CLIP_LEN,
    DEFAULT_CROP_SIZE,
    DEFAULT_CROP_SIZE,
)

__all__ = [
    "BODY_FACE_ANCHORS",
    "DEFAULT_CLIP_LEN",
    "DEFAULT_CROP_SIZE",
    "DEFAULT_IMAGE_DTYPE",
    "DEFAULT_TENSOR_SHAPE",
    "FACE_68",
    "LEFT_HAND_21",
    "NUM_CHANNELS",
    "NUM_REGIONS",
    "REGION_NAMES",
    "REGION_TO_INDEX",
    "RIGHT_HAND_21",
    "TENSOR_FORMAT",
]
