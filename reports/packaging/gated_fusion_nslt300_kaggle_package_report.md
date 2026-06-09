# Gated Fusion NSLT300 Kaggle Package Report

## 1. Muc tieu
Tao package Kaggle-ready `wlasl-nslt300-gated-fusion-ready` de train va evaluate Gated Feature Fusion tren NSLT300.

## 2. Boi canh
Package duoc xay dung bang cach tham khao flow NSLT100 trong repo, nhung toan bo checkpoint/config/data duoc giu dung cho NSLT300.
- requirement report: `F:/DMV/Recognizing-sign-language-at-the-word-level/reports/packaging/gated_fusion_nslt300_requirement_check_report.md`
- old package folder deleted with --clean: `yes`
- old zip deleted with --clean: `yes`

## 3. Ket luan READY hay NOT READY
Conclusion: READY

## 4. Cac file NSLT100 da tham khao
- F:/DMV/Recognizing-sign-language-at-the-word-level/scripts/package_gated_fusion_nslt100_kaggle_dataset.py
- F:/DMV/Recognizing-sign-language-at-the-word-level/scripts/check_gated_fusion_setup.py
- F:/DMV/Recognizing-sign-language-at-the-word-level/configs/train/gated_feature_fusion_nslt100.yaml
- F:/DMV/Recognizing-sign-language-at-the-word-level/configs/fusion/nslt100_skeleton_regions_late_fusion.yaml
- F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt100-gated-fusion-ready
- F:/DMV/Recognizing-sign-language-at-the-word-level/reports/packaging/gated_fusion_nslt100_kaggle_package_report.md
- F:/DMV/Recognizing-sign-language-at-the-word-level/artifacts/fusion/nslt100

## 5. Requirement check
- status: `READY`
- gating config path: `F:/DMV/Recognizing-sign-language-at-the-word-level/configs/train/gated_feature_fusion_nslt300.yaml`
- created or refreshed by requirement check: `no`
- pairing minimum coverage: `0.950`

## 6. Skeleton branch inputs
- source root used: `F:/DMV/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/graph_tensors/selected_31/nslt300`
- canonical tensor root: `F:/DMV/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors/nslt300`
- fallback tensor root: `F:/DMV/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/graph_tensors/selected_31/nslt300`
- counts: train=1897, val=446, test=317
- sample shape: `(3, 150, 31, 1)`
- warning: Canonical skeleton tensor root is missing; using graph_tensors/selected_31/nslt300 as the available source.

## 7. Regions branch inputs
- source root used: `F:/DMV/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/branch_inputs/regions/rtmw_l/tensors/nslt300`
- counts: train=1897, val=446, test=317
- sample shape: `(3, 3, 64, 112, 112)`
- copied reports: reports/nslt300_region_crop_quality_report.md, reports/nslt300_region_low_quality_samples.csv

## 8. Skeleton checkpoint
- selected checkpoint path: `F:/DMV/Recognizing-sign-language-at-the-word-level/artifacts/fusion/nslt300/checkpoints/skeleton/best.pt`
- packaged checkpoint path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/checkpoints/skeleton/best.pt`

## 9. Regions checkpoint
- selected checkpoint path: `F:/DMV/Recognizing-sign-language-at-the-word-level/artifacts/fusion/nslt300/checkpoints/regions/best.pt`
- packaged checkpoint path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/checkpoints/regions/best.pt`

## 10. Skeleton config
- selected config path: `F:/DMV/Recognizing-sign-language-at-the-word-level/artifacts/fusion/nslt300/configs/skeleton/config_resolved.yaml`
- packaged config path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/configs/skeleton_config_resolved.yaml`

## 11. Regions config
- selected config path: `F:/DMV/Recognizing-sign-language-at-the-word-level/artifacts/fusion/nslt300/configs/regions/config_resolved.yaml`
- packaged config path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/configs/regions_config_resolved.yaml`

## 12. Pairing check
- train matched_count: `1897`
- val matched_count: `446`
- test matched_count: `317`
- train label_mismatch: `0`
- val label_mismatch: `0`
- test label_mismatch: `0`
- train gloss_mismatch: `0`
- val gloss_mismatch: `0`
- test gloss_mismatch: `0`

## 13. Gated Fusion NSLT300 config da tao
- repo train config: `F:/DMV/Recognizing-sign-language-at-the-word-level/configs/train/gated_feature_fusion_nslt300.yaml`
- Kaggle config in package: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/configs/gated_feature_fusion_nslt300_kaggle.yaml`

## 14. Package structure neu READY
- package folder path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready`
- includes `README.md`, `metadata.json`, `configs/`, `checkpoints/`, `branch_inputs/`, `verify/`
- package file count: `5337`

## 15. Metadata
- metadata path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/metadata.json`
- metadata subset: `nslt300`
- metadata num_classes: `300`

## 16. Verify package
- verify result: `PASS`
- verify summary path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/verify/verify_summary.json`
- verify script path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready/verify/verify_package.py`

## 17. Zip output neu co
- zip path: `F:/DMV/Recognizing-sign-language-at-the-word-level/packaging_outputs/wlasl-nslt300-gated-fusion-ready.zip`
- zip size: `7.47 GB`

## 18. Cach upload Kaggle
Upload the package folder or zip to a private Kaggle Dataset, then attach that dataset to your notebook.

## 19. Cach train/evaluate tren Kaggle
- train: `python scripts/train_gated_fusion.py --config /kaggle/input/wlasl-nslt300-gated-fusion-ready/wlasl-nslt300-gated-fusion-ready/configs/gated_feature_fusion_nslt300_kaggle.yaml`
- evaluate val: `python scripts/evaluate_gated_fusion.py --config /kaggle/input/wlasl-nslt300-gated-fusion-ready/wlasl-nslt300-gated-fusion-ready/configs/gated_feature_fusion_nslt300_kaggle.yaml --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt300-sel31-ce-regions/best.pt --split val`
- evaluate test: `python scripts/evaluate_gated_fusion.py --config /kaggle/input/wlasl-nslt300-gated-fusion-ready/wlasl-nslt300-gated-fusion-ready/configs/gated_feature_fusion_nslt300_kaggle.yaml --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt300-sel31-ce-regions/best.pt --split test`

## 20. Nhung gi khong dong goi
- raw videos
- W&B logs
- intermediate checkpoints
- old outputs
- notebook cache
- .git
- __pycache__

## 21. Luu y quan trong
- `/kaggle/input` is read-only; write outputs to `/kaggle/working`.
- Skeleton tensors were sourced from `graph_tensors/selected_31/nslt300` and repackaged into the expected `branch_inputs/skeleton/rtmw_l/tensors/nslt300/` layout.
- This package is NSLT300-only; do not swap in NSLT100 checkpoints or configs.

## 22. Ket luan
Package verify passed, so the folder and zip are ready to upload to Kaggle for NSLT300 gated feature fusion.
