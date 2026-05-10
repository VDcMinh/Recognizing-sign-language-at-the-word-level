"""Shared manifest column names used across preprocessing stages."""

from __future__ import annotations

INDEX_COLUMNS = [
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "subset",
    "split",
    "raw_video_path",
    "has_video",
    "frame_start",
    "frame_end",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "signer_id",
    "fps",
    "width",
    "height",
    "num_frames",
]

STANDARDIZED_COLUMNS = [
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "standardized_video_path",
    "frames_dir",
    "num_frames",
    "output_size",
    "crop_bbox",
    "status",
    "error_message",
]

POSE_NPZ_FIELDS = [
    "keypoints",
    "image_size",
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
]

SKELETON_NPZ_FIELDS = [
    "keypoints",
    "keypoint_set",
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
]
