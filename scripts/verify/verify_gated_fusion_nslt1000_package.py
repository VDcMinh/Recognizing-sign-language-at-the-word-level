"""Verify a packaged NSLT1000 gated-fusion dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from slr.branches.fusion.build import build_gated_feature_fusion_from_config
from slr.branches.fusion.package_support import (
    ACTIVE_REGIONS,
    NUM_CLASSES,
    REGIONS_EXPECTED_SHAPE,
    SAMPLE_SPLITS,
    SKELETON_EXPECTED_SHAPE,
    SPLIT_COUNTS,
    SUBSET,
    build_alignment,
    build_kaggle_root,
    build_runtime_fusion_config,
    inspect_regions_checkpoint,
    inspect_regions_config,
    inspect_skeleton_checkpoint,
    inspect_skeleton_config,
    load_regions_manifest_set,
    load_skeleton_manifest_set,
    load_tensor_shape,
    read_manifest,
    read_yaml,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the package verifier CLI."""

    parser = argparse.ArgumentParser(
        description="Verify a packaged WLASL NSLT1000 gated-fusion dataset."
    )
    parser.add_argument("--package-root", type=Path, required=True, help="Package root to verify.")
    parser.add_argument(
        "--check-all-tensors",
        action="store_true",
        help="Load every tensor file instead of checking only a small sample per split.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    return parser


def resolve_path(path_value: Path, *, package_root: Path) -> Path:
    """Resolve one possibly-relative path against package_root."""

    return path_value if path_value.is_absolute() else (package_root / path_value).resolve()


def ensure_package_tree(package_root: Path) -> list[str]:
    """Check the required package tree."""

    required_paths = (
        package_root / "README.md",
        package_root / "metadata.json",
        package_root / "checkpoints" / "skeleton" / "best.pt",
        package_root / "checkpoints" / "regions" / "best.pt",
        package_root / "configs" / "gated_feature_fusion_nslt1000_kaggle.yaml",
        package_root / "configs" / "skeleton_config_resolved.yaml",
        package_root / "configs" / "regions_config_resolved.yaml",
        package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests",
        package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests",
        package_root / "verify" / "verify_package.py",
    )
    errors: list[str] = []
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required package path: {path.as_posix()}")
    return errors


def inspect_kaggle_config(path: Path, *, package_name: str) -> dict[str, Any]:
    """Validate the packaged Kaggle config path policy."""

    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
    regions_cfg = dict(dataset_cfg.get("regions", {}))
    skeleton_branch_cfg = dict(config.get("skeleton_branch", {}))
    regions_branch_cfg = dict(config.get("regions_branch", {}))
    kaggle_root = build_kaggle_root(package_name)

    disallowed_substrings = ("F:\\", "F:/", "/kaggle/working/regions_nslt1000_runtime")
    path_fields = {
        "skeleton.data_root": skeleton_cfg.get("data_root"),
        "regions.data_root": regions_cfg.get("data_root"),
        "skeleton_branch.config_path": skeleton_branch_cfg.get("config_path"),
        "regions_branch.config_path": regions_branch_cfg.get("config_path"),
        "skeleton_branch.checkpoint_path": skeleton_branch_cfg.get("checkpoint_path"),
        "regions_branch.checkpoint_path": regions_branch_cfg.get("checkpoint_path"),
    }
    for split in SAMPLE_SPLITS:
        path_fields[f"skeleton.manifests.{split}"] = skeleton_cfg.get("manifests", {}).get(split)
        path_fields[f"regions.manifests.{split}"] = regions_cfg.get("manifests", {}).get(split)

    for label, value in path_fields.items():
        text = str(value or "")
        if not text:
            raise ValueError(f"Packaged Kaggle config path is missing: {label}")
        if any(token in text for token in disallowed_substrings):
            raise ValueError(f"Packaged Kaggle config path contains a disallowed root: {label}={text}")
        if text.startswith("/kaggle/input/") and not text.startswith(kaggle_root):
            raise ValueError(f"Packaged Kaggle config path does not stay inside the package root: {label}={text}")

    if dataset_cfg.get("subset") != SUBSET:
        raise ValueError(f"Packaged Kaggle config subset must be {SUBSET}")
    if int(dataset_cfg.get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError(f"Packaged Kaggle config num_classes must be {NUM_CLASSES}")
    return {
        "package_name": package_name,
        "kaggle_root": kaggle_root,
        "subset": dataset_cfg.get("subset"),
        "num_classes": dataset_cfg.get("num_classes"),
    }


def verify_metadata(path: Path, *, package_name: str) -> dict[str, Any]:
    """Validate metadata.json."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("package_name") != package_name:
        raise ValueError("metadata.package_name is incorrect")
    if payload.get("subset") != SUBSET:
        raise ValueError("metadata.subset is incorrect")
    if int(payload.get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError("metadata.num_classes is incorrect")
    return {
        "package_name": payload.get("package_name"),
        "subset": payload.get("subset"),
        "num_classes": payload.get("num_classes"),
    }


def verify_tensor_paths_and_shapes(
    *,
    package_root: Path,
    branch_name: str,
    manifest_root: Path,
    tensor_key: str,
    expected_shape: tuple[int, ...],
    check_all_tensors: bool,
) -> dict[str, Any]:
    """Validate packaged tensor paths and sample some or all tensor shapes."""

    errors: list[str] = []
    checked_count = 0
    rows_loaded = 0
    branch_root = package_root / "branch_inputs" / branch_name / "rtmw_l"

    for split in SAMPLE_SPLITS:
        manifest_name = (
            f"nslt1000_selected_31_{split}.csv"
            if branch_name == "skeleton"
            else f"nslt1000_{split}.csv"
        )
        _, rows = read_manifest(manifest_root / manifest_name)
        rows_loaded += len(rows)
        rows_to_load = rows if check_all_tensors else rows[: min(2, len(rows))]
        for row in rows:
            path_text = str(row.get(tensor_key, "")).strip()
            if not path_text:
                errors.append(f"{branch_name} manifest has empty {tensor_key} for split={split}")
                continue
            tensor_path = Path(path_text)
            if tensor_path.is_absolute():
                errors.append(f"{branch_name} tensor path must be relative: {path_text}")
                continue
            full_path = (branch_root / tensor_path).resolve(strict=False)
            try:
                full_path.relative_to(branch_root.resolve(strict=False))
            except ValueError:
                errors.append(f"{branch_name} tensor path escapes branch root: {path_text}")
                continue
            if "F:\\" in path_text or "F:/" in path_text or "/kaggle/working" in path_text:
                errors.append(f"{branch_name} tensor path contains a disallowed root: {path_text}")

        for row in rows_to_load:
            path_text = str(row.get(tensor_key, "")).strip()
            full_path = (branch_root / path_text).resolve(strict=False)
            if not full_path.exists():
                errors.append(f"Packaged tensor is missing: {full_path.as_posix()}")
                continue
            try:
                shape = load_tensor_shape(full_path)
            except Exception as exc:
                errors.append(f"Could not load tensor shape {full_path.as_posix()}: {exc}")
                continue
            if shape != list(expected_shape):
                errors.append(
                    f"Tensor shape mismatch for {full_path.as_posix()}: expected {list(expected_shape)}, got {shape}"
                )
                continue
            checked_count += 1

    return {
        "rows_loaded": rows_loaded,
        "tensors_checked": checked_count,
        "check_all_tensors": bool(check_all_tensors),
        "errors": errors,
    }


def run_package_smoke(package_root: Path) -> dict[str, Any]:
    """Run a local smoke test using packaged checkpoints and configs."""

    package_config = package_root / "configs" / "gated_feature_fusion_nslt1000_kaggle.yaml"
    runtime_config = build_runtime_fusion_config(
        fusion_config_path=package_config,
        skeleton_checkpoint=package_root / "checkpoints" / "skeleton" / "best.pt",
        regions_checkpoint=package_root / "checkpoints" / "regions" / "best.pt",
        skeleton_config=package_root / "configs" / "skeleton_config_resolved.yaml",
        regions_config=package_root / "configs" / "regions_config_resolved.yaml",
    )
    runtime_config["dataset"]["skeleton"]["data_root"] = (
        package_root / "branch_inputs" / "skeleton" / "rtmw_l"
    ).as_posix()
    runtime_config["dataset"]["regions"]["data_root"] = (
        package_root / "branch_inputs" / "regions" / "rtmw_l"
    ).as_posix()
    runtime_config["dataset"]["skeleton"]["manifests"] = {
        split: (package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests" / f"nslt1000_selected_31_{split}.csv").as_posix()
        for split in SAMPLE_SPLITS
    }
    runtime_config["dataset"]["regions"]["manifests"] = {
        split: (package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests" / f"nslt1000_{split}.csv").as_posix()
        for split in SAMPLE_SPLITS
    }

    model, info = build_gated_feature_fusion_from_config(runtime_config, device="cpu")
    model.eval()
    skeleton_tensor = torch.randn(1, *SKELETON_EXPECTED_SHAPE, dtype=torch.float32)
    regions_tensor = torch.randn(1, *REGIONS_EXPECTED_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        logits, features = model(
            skeleton_tensor,
            regions_tensor,
            return_features=True,
        )
    labels = torch.zeros((1,), dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if list(logits.shape) != [1, NUM_CLASSES]:
        raise ValueError(f"Smoke logits shape must be [1, {NUM_CLASSES}], got {list(logits.shape)}")
    return {
        "ok": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu().item()),
        "skeleton_feature_shape": list(features["skeleton_feature"].shape),
        "region_feature_shape": list(features["region_feature"].shape),
        "gate_shape": list(features["gate"].shape),
        "fused_shape": list(features["fused"].shape),
        "model_info": info,
    }


def main() -> int:
    """Run the package verifier."""

    args = build_parser().parse_args()
    package_root = resolve_path(args.package_root, package_root=Path.cwd())
    output_json = (
        resolve_path(args.output_json, package_root=package_root)
        if args.output_json is not None
        else None
    )
    errors = ensure_package_tree(package_root)

    skeleton_checkpoint_path = package_root / "checkpoints" / "skeleton" / "best.pt"
    regions_checkpoint_path = package_root / "checkpoints" / "regions" / "best.pt"
    skeleton_config_path = package_root / "configs" / "skeleton_config_resolved.yaml"
    regions_config_path = package_root / "configs" / "regions_config_resolved.yaml"
    package_config_path = package_root / "configs" / "gated_feature_fusion_nslt1000_kaggle.yaml"
    metadata_path = package_root / "metadata.json"
    readme_path = package_root / "README.md"
    skeleton_manifest_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests"
    regions_manifest_root = package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests"

    skeleton_checkpoint_info = {}
    regions_checkpoint_info = {}
    skeleton_config_info = {}
    regions_config_info = {}
    kaggle_config_info = {}
    metadata_info = {}
    smoke_info: dict[str, Any] = {"ok": False}

    try:
        skeleton_checkpoint_info = inspect_skeleton_checkpoint(skeleton_checkpoint_path)
    except Exception as exc:
        errors.append(f"skeleton checkpoint: {exc}")
    try:
        regions_checkpoint_info = inspect_regions_checkpoint(regions_checkpoint_path)
    except Exception as exc:
        errors.append(f"regions checkpoint: {exc}")
    try:
        skeleton_config_info = inspect_skeleton_config(skeleton_config_path)
    except Exception as exc:
        errors.append(f"skeleton config: {exc}")
    try:
        regions_config_info = inspect_regions_config(regions_config_path)
    except Exception as exc:
        errors.append(f"regions config: {exc}")
    try:
        kaggle_config_info = inspect_kaggle_config(package_config_path, package_name=package_root.name)
    except Exception as exc:
        errors.append(f"packaged fusion config: {exc}")
    try:
        metadata_info = verify_metadata(metadata_path, package_name=package_root.name)
    except Exception as exc:
        errors.append(f"metadata: {exc}")
    if not readme_path.exists():
        errors.append("README.md is missing")

    alignment_summary = {"splits": {}, "total_matched": 0}
    try:
        skeleton_manifests = load_skeleton_manifest_set(skeleton_manifest_root)
        regions_manifests = load_regions_manifest_set(regions_manifest_root)
        alignment_summary = build_alignment(skeleton_manifests, regions_manifests)
    except Exception as exc:
        errors.append(f"manifest alignment: {exc}")

    for split in SAMPLE_SPLITS:
        split_summary = alignment_summary.get("splits", {}).get(split)
        if not split_summary:
            continue
        if split_summary["matched_count"] != SPLIT_COUNTS[split]:
            errors.append(f"alignment count mismatch for {split}")
        if split_summary["class_mismatch_count"]:
            errors.append(f"class mismatches detected for {split}")
        if split_summary["gloss_mismatch_count"]:
            errors.append(f"gloss mismatches detected for {split}")
        if split_summary["split_mismatch_count"]:
            errors.append(f"split mismatches detected for {split}")
        if split_summary["skeleton_audit"]["duplicate_count"]:
            errors.append(f"skeleton duplicate IDs detected for {split}")
        if split_summary["regions_audit"]["duplicate_count"]:
            errors.append(f"regions duplicate IDs detected for {split}")
        if split_summary["skeleton_audit"]["collision_count"]:
            errors.append(f"skeleton canonical collisions detected for {split}")
        if split_summary["regions_audit"]["collision_count"]:
            errors.append(f"regions canonical collisions detected for {split}")

    skeleton_tensor_summary = verify_tensor_paths_and_shapes(
        package_root=package_root,
        branch_name="skeleton",
        manifest_root=skeleton_manifest_root,
        tensor_key="graph_tensor_path",
        expected_shape=SKELETON_EXPECTED_SHAPE,
        check_all_tensors=bool(args.check_all_tensors),
    )
    regions_tensor_summary = verify_tensor_paths_and_shapes(
        package_root=package_root,
        branch_name="regions",
        manifest_root=regions_manifest_root,
        tensor_key="tensor_path",
        expected_shape=REGIONS_EXPECTED_SHAPE,
        check_all_tensors=bool(args.check_all_tensors),
    )
    errors.extend(skeleton_tensor_summary["errors"])
    errors.extend(regions_tensor_summary["errors"])

    try:
        smoke_info = run_package_smoke(package_root)
    except Exception as exc:
        smoke_info = {"ok": False, "error": str(exc)}
        errors.append(f"fusion smoke: {exc}")

    summary = {
        "status": "pass" if not errors else "fail",
        "package_root": package_root.as_posix(),
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "active_regions": list(ACTIVE_REGIONS),
        "check_all_tensors": bool(args.check_all_tensors),
        "skeleton_checkpoint": skeleton_checkpoint_info,
        "regions_checkpoint": regions_checkpoint_info,
        "skeleton_config": skeleton_config_info,
        "regions_config": regions_config_info,
        "kaggle_config": kaggle_config_info,
        "metadata": metadata_info,
        "alignment": {
            "total_matched": alignment_summary.get("total_matched"),
            "splits": {
                split: {
                    key: value
                    for key, value in alignment_summary.get("splits", {}).get(split, {}).items()
                    if key != "items"
                }
                for split in SAMPLE_SPLITS
            },
        },
        "tensor_checks": {
            "skeleton": skeleton_tensor_summary,
            "regions": regions_tensor_summary,
        },
        "smoke": smoke_info,
        "errors": errors,
    }

    if output_json is not None:
        write_json(output_json, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        print("VERIFY FAIL")
        return 1
    print("VERIFY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
