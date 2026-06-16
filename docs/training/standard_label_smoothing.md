# Standard Label Smoothing

Standard label smoothing variants are grouped next to their baseline training configs.

## Examples

- Skeleton:
  `configs/train/skeleton/nslt1000/selected_31/stgcnpp_standardls_eps01.yaml`
- Regions:
  `configs/train/regions/nslt1000/full/region_resnet18_gru_standardls_eps01.yaml`

## Usage

Pass the smoothing variant config directly to the normal training script:

```bash
python scripts/train/train_skeleton.py --config configs/train/skeleton/nslt1000/selected_31/stgcnpp_standardls_eps01.yaml
python scripts/train/train_regions.py --config configs/train/regions/nslt1000/full/region_resnet18_gru_standardls_eps01.yaml
```

No special script is required; the behavior is encoded in the chosen config.
