# Recognizing Sign Language at the Word Level

Research project scaffold for Word-Level Sign Language Recognition (WSLR) on the WLASL dataset.

## Goal

This repository is organized around a clean, stage-based pipeline for building and evaluating WLASL word-level recognition systems, with the current implementation priority on the skeleton branch.

## Dataset

- Dataset: WLASL
- Raw data root: `data/datasets/WLASL/raw/`
- Raw data is treated as read-only source of truth and must not be modified in-place.

## Branches

1. Skeleton branch: current primary branch, based on RTMW-l / MMPose whole-body pose, selected keypoint subsets, and graph-based sequence models such as ST-GCN++ and CTR-GCN.
2. Region branch: sequence modeling over cropped face, left-hand, and right-hand regions.
3. Hand sequence + Pose Flow branch: hand image sequences combined with keypoint motion / pose flow representations.

## Pipeline

```text
raw
-> index
-> standardized
-> pose/rtmw_l/wholebody_133
-> branch_inputs
-> training/evaluation
```

## Roadmap

1. Build index from raw metadata.
2. Standardize video clips.
3. Extract RTMW-l wholebody 133 pose.
4. Build skeleton `selected_31` inputs.
5. Train an ST-GCN++ baseline.
6. Add label smoothing / Language Label Smoothing.
7. Later implement the region branch.
8. Later implement the hand pose flow branch.

## Expected Data Schemas

### Index manifest columns

- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `subset`
- `split`
- `raw_video_path`
- `has_video`
- `frame_start`
- `frame_end`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `signer_id`
- `fps`
- `width`
- `height`
- `num_frames`

### Standardized manifest columns

- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`
- `standardized_video_path`
- `frames_dir`
- `num_frames`
- `output_size`
- `crop_bbox`
- `status`
- `error_message`

### Shared pose `.npz`

- `keypoints`: shape `(T, 133, 3)`
- `image_size`
- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`

### Skeleton selected `.npz`

- `keypoints`: shape `(T, 31, 3)` for `selected_31`
- `keypoint_set`
- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`

### Graph tensor layout

- Shape: `C x T x V x M`
- `C = 2` or `3`
- `T = 150`
- `V = 31`
- `M = 1`

## Environment Notes

`requirements.txt` intentionally keeps the base environment light for dataset indexing and preprocessing utilities. Install PyTorch, MMPose, MMEngine, and related CUDA-specific dependencies separately for the target machine and GPU stack.
