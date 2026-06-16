# Model Registry

This directory stores model metadata only. Checkpoints stay outside `model_registry/` and are referenced through `artifacts.*`.

## Files

- `registry.yaml`: summary index for UI or backend discovery
- `models/*/model.yaml`: per-model metadata records

## Current entries

- `skeleton_nslt1000_sel31_v1`: ready
- `regions_nslt1000_face_hands_v1`: ready
- `gated_fusion_nslt1000_v1`: incomplete

## Notes

- Registry validation checks duplicate IDs, registry-file existence, branch validity, importable `model.class_path`, and declared local artifact paths.
- `gated_fusion_nslt1000_v1` is marked `incomplete` because this repo snapshot includes backbone checkpoints and configs, but not a verified local fusion checkpoint.
