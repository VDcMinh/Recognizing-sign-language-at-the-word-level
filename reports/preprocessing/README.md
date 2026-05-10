# Preprocessing Pipeline

This folder stores documentation and outputs for preprocessing stages.

## Layers

- `raw/`: immutable source-of-truth dataset layer containing original WLASL metadata, videos, and docs.
- `index/`: generated metadata tables, splits, class maps, and audit reports built from `raw/`.
- `standardized/`: normalized video assets and optional extracted frames after crop / resize / letterbox.
- `pose/`: shared RTMW-l whole-body pose outputs and quality reports used by all branches.
- `branch_inputs/`: branch-specific derived inputs for skeleton, region, and hand-poseflow training pipelines.

Each stage should read from the previous layer and write only to its own output layer.
