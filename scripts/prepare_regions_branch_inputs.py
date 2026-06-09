"""Prepare WLASL region branch inputs with test-run friendly overrides."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import yaml

from slr.branches.regions.build_crops import ALLOWED_SPLITS, run as build_regions_run
from slr.branches.regions.region_schema import REGION_NAMES
from slr.utils.io import read_yaml


DEFAULT_BASE_CONFIG = Path("configs/preprocessing/region_crops_nslt1000.yaml")
DEFAULT_DATASET_ROOT = Path("data/datasets/WLASL")
DEFAULT_FULL_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "branch_inputs/regions/rtmw_l"
DEFAULT_TEST_RUN_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "branch_inputs/regions/rtmw_l_test_run"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare WLASL regions branch inputs with nslt1000 test-run support."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base preprocessing config to override.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="nslt1000",
        help="Subset to build, for example nslt1000.",
    )
    parser.add_argument(
        "--active-regions",
        type=str,
        default="left_hand,right_hand,face",
        help="Comma-separated region order. This workflow requires left_hand,right_hand,face.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root. Test-run defaults to the dedicated test-run folder.",
    )
    parser.add_argument(
        "--frames",
        "--num-frames",
        dest="frames",
        type=int,
        default=64,
        help="Number of frames per clip.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Square crop size in pixels.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Limit processing and default outputs to the isolated test-run root.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional per-split sample limit.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously generated files in the selected output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and plan without writing manifests or tensors.",
    )
    return parser


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _validate_active_regions(value: str) -> list[str]:
    active_regions = _parse_csv_list(value)
    if active_regions != list(REGION_NAMES):
        raise ValueError(
            "This extraction workflow requires active_regions=left_hand,right_hand,face "
            f"in that exact order, got {active_regions!r}."
        )
    return active_regions


def _validate_splits(value: str) -> list[str]:
    splits = _parse_csv_list(value)
    invalid = [split for split in splits if split not in ALLOWED_SPLITS]
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}. Expected values from {ALLOWED_SPLITS}.")
    if not splits:
        raise ValueError("At least one split must be selected.")
    return splits


def _resolve_output_root(output_root: Path | None, test_run: bool) -> Path:
    if output_root is not None:
        return output_root
    return DEFAULT_TEST_RUN_OUTPUT_ROOT if test_run else DEFAULT_FULL_OUTPUT_ROOT


def _resolve_limit_per_split(limit_per_split: int | None, test_run: bool) -> int | None:
    if limit_per_split is not None:
        return int(limit_per_split)
    if test_run:
        return 3
    return None


def _ensure_test_run_does_not_touch_full_output(output_root: Path, test_run: bool) -> None:
    if test_run and output_root.resolve() == DEFAULT_FULL_OUTPUT_ROOT.resolve():
        raise ValueError(
            "Refusing to write test-run output into the full regions root. "
            "Pass a dedicated --output-root or omit it to use the default test-run root."
        )


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_inputs(dataset_root: Path, subset: str, splits: list[str]) -> dict[str, Any]:
    """Validate the expected standardized and pose inputs for one subset."""

    standardized_manifest_root = dataset_root / "standardized/manifests"
    standardized_frames_root = dataset_root / "standardized/frames" / subset
    pose_manifest_root = dataset_root / "pose/rtmw_l/manifests"
    pose_subset_manifest_root = dataset_root / f"pose/rtmw_l/{subset}/manifests"
    pose_layout_root = dataset_root / "pose/rtmw_l/wholebody_133" / subset

    missing: list[str] = []
    resolved_pose_manifests: dict[str, str] = {}
    for split in splits:
        standardized_manifest = standardized_manifest_root / f"{subset}_{split}.csv"
        standardized_frames_dir = standardized_frames_root / split
        pose_manifest = _first_existing_path(
            [
                pose_manifest_root / f"{subset}_{split}.csv",
                pose_subset_manifest_root / f"{subset}_{split}.csv",
            ]
        )
        pose_dir = pose_layout_root / split

        if not standardized_manifest.exists():
            missing.append(str(standardized_manifest))
        if not standardized_frames_dir.exists():
            missing.append(str(standardized_frames_dir))
        if pose_manifest is None:
            missing.append(str(pose_manifest_root / f"{subset}_{split}.csv"))
        else:
            resolved_pose_manifests[split] = str(pose_manifest)
        if not pose_dir.exists():
            missing.append(str(pose_dir))

    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required nslt input paths:\n{details}")

    return {
        "dataset_root": str(dataset_root),
        "subset": subset,
        "splits": list(splits),
        "standardized_manifest_root": str(standardized_manifest_root),
        "standardized_frames_root": str(standardized_frames_root),
        "pose_manifest_root": str(pose_manifest_root),
        "pose_layout_root": str(pose_layout_root),
        "resolved_pose_manifests": resolved_pose_manifests,
    }


def build_overridden_config(
    *,
    base_config_path: Path,
    subset: str,
    splits: list[str],
    output_root: Path,
    frames: int,
    image_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Load one base config and apply CLI overrides."""

    config = read_yaml(base_config_path)
    dataset_cfg = dict(config.get("dataset", {}))
    input_cfg = dict(config.get("input", {}))
    output_cfg = dict(config.get("output", {}))
    outputs_cfg = dict(config.get("outputs", {}))
    options_cfg = dict(config.get("options", {}))
    regions_cfg = dict(config.get("regions", {}))

    dataset_cfg["name"] = str(dataset_cfg.get("name", "WLASL"))
    dataset_cfg["root"] = str(dataset_cfg.get("root", DEFAULT_DATASET_ROOT))
    dataset_cfg["subset"] = subset

    input_cfg["splits"] = list(splits)
    input_cfg["pose_backend"] = str(input_cfg.get("pose_backend", "rtmw_l"))
    input_cfg["pose_layout"] = str(input_cfg.get("pose_layout", "wholebody_133"))
    input_cfg["standardized_frames_root"] = str(
        input_cfg.get("standardized_frames_root", DEFAULT_DATASET_ROOT / "standardized/frames")
    )
    input_cfg["standardized_manifests_root"] = str(
        input_cfg.get("standardized_manifests_root", DEFAULT_DATASET_ROOT / "standardized/manifests")
    )
    input_cfg["pose_backend_root"] = str(
        input_cfg.get("pose_backend_root", DEFAULT_DATASET_ROOT / "pose/rtmw_l")
    )
    input_cfg["pose_root"] = str(DEFAULT_DATASET_ROOT / f"pose/rtmw_l/{subset}/wholebody_133")
    input_cfg["pose_manifest_root"] = str(DEFAULT_DATASET_ROOT / f"pose/rtmw_l/{subset}/manifests")

    output_cfg["root"] = str(output_root)
    output_cfg["regions"] = list(REGION_NAMES)

    config["dataset"] = dataset_cfg
    config["input"] = input_cfg
    config["output"] = output_cfg
    config["outputs"] = outputs_cfg
    config["options"] = options_cfg
    config["regions"] = regions_cfg
    config["clip_len"] = int(frames)
    config["crop_size"] = int(image_size)
    config["options"]["overwrite"] = bool(overwrite)
    config["regions"]["order"] = list(REGION_NAMES)
    return config


def write_temp_config(config: dict[str, Any]) -> Path:
    """Write the generated config to a temporary YAML file."""

    temp_dir = Path(tempfile.mkdtemp(prefix="regions_prepare_"))
    config_path = temp_dir / "generated_region_crops.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    active_regions = _validate_active_regions(args.active_regions)
    splits = _validate_splits(args.splits)
    output_root = _resolve_output_root(args.output_root, args.test_run)
    limit_per_split = _resolve_limit_per_split(args.limit_per_split, args.test_run)
    _ensure_test_run_does_not_touch_full_output(output_root, args.test_run)

    dataset_root = DEFAULT_DATASET_ROOT
    validate_inputs(dataset_root, args.subset, splits)
    generated_config = build_overridden_config(
        base_config_path=args.config,
        subset=args.subset,
        splits=splits,
        output_root=output_root,
        frames=args.frames,
        image_size=args.image_size,
        overwrite=args.overwrite or args.test_run,
    )
    generated_config_path = write_temp_config(generated_config)

    print("== Regions Preparation ==")
    print(f"subset: {args.subset}")
    print(f"active_regions: {active_regions}")
    print(f"splits: {splits}")
    print(f"frames: {args.frames}")
    print(f"image_size: {args.image_size}")
    print(f"test_run: {bool(args.test_run)}")
    print(f"limit_per_split: {limit_per_split}")
    print(f"output_root: {output_root}")
    print(f"generated_config: {generated_config_path}")
    print()

    try:
        return build_regions_run(
            config_path=generated_config_path,
            subset=None,
            limit=limit_per_split,
            dry_run=bool(args.dry_run),
        )
    finally:
        generated_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
