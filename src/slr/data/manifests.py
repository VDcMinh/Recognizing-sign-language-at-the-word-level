"""Shared manifest schemas used across preprocessing stages."""

from __future__ import annotations

MASTER_INSTANCE_COLUMNS = [
    "instance_uid",
    "sample_id",
    "instance_id",
    "video_id",
    "gloss",
    "gloss_id",
    "class_id",
    "metadata_source",
    "source",
    "url",
    "variation_id",
    "split_source",
    "subset_membership",
    "raw_video_filename",
    "raw_video_path",
    "is_present_locally",
    "local_file_size_bytes",
    "start_frame",
    "end_frame",
    "bbox",
    "fps",
    "signer_id",
    "notes",
]

SUBSET_MANIFEST_COLUMNS = [
    "instance_uid",
    "sample_id",
    "instance_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "video_path",
    "is_present_locally",
    "start_frame",
    "end_frame",
    "master_start_frame",
    "master_end_frame",
    "bbox",
    "fps",
    "signer_id",
    "source",
    "url",
    "notes",
]

CLASS_MAP_COLUMNS = [
    "class_id",
    "gloss_id",
    "gloss",
    "class_list_gloss",
    "master_gloss",
    "gloss_match",
    "master_total",
    "master_available",
    "master_missing",
    "nslt100_total",
    "nslt300_total",
    "nslt1000_total",
    "nslt2000_total",
    "notes",
]

VIDEO_TO_SPLIT_COLUMNS = [
    "video_id",
    "master_split",
    "nslt100_split",
    "nslt300_split",
    "nslt1000_split",
    "nslt2000_split",
    "is_in_master",
    "is_in_any_nslt",
    "is_present_locally",
    "notes",
]

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
