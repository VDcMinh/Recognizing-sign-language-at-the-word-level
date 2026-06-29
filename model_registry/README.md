# Model Registry

This directory stores model metadata only. Checkpoints stay outside `model_registry/` and are referenced through `artifacts.*`.

## Files

- `registry.yaml`: summary index for UI or backend discovery
- `registry_serving.yaml`: scaffolded UI/serving index with fixed artifact slots for `nslt100`, `nslt300`, and `nslt1000`
- `models/*/model.yaml`: per-model metadata records

## Current entries

- `skeleton_nslt1000_sel31_v1`: ready
- `regions_nslt1000_face_hands_v1`: ready
- `gated_fusion_nslt1000_v1`: incomplete

## Serving Scaffold

Use `registry_serving.yaml` when you want a stable place to drop curated model artifacts for a future UI.

- The artifact slots live under `artifacts/serving/<branch>/<subset>/`
- Replace the placeholder `config_resolved.yaml`, `metrics.json`, `train_log.csv`, and `class_map.json` with real files when ready
- Add `best.pt` later without changing any registry path
- The scaffold registry keeps every entry at `status: incomplete` by default so it remains safe until you decide to promote a model

## Notes

- Registry validation checks duplicate IDs, registry-file existence, branch validity, importable `model.class_path`, and declared local artifact paths.
- `gated_fusion_nslt1000_v1` is marked `incomplete` because this repo snapshot includes backbone checkpoints and configs, but not a verified local fusion checkpoint.
