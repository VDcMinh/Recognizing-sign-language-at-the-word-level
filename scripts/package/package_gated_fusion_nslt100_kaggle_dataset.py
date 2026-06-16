"""Rebuild a Kaggle-ready NSLT100 gated-fusion dataset package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/package/package_gated_fusion_nslt100_kaggle_dataset.py ..."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/package/package_gated_fusion_nslt100_kaggle_dataset.py ..."
    ) from exc


SUBSET = "nslt100"
PACKAGE_NAME_DEFAULT = "wlasl-nslt100-gated-fusion-ready"
KAGGLE_ROOT_TEMPLATE = "/kaggle/input/{package_name}/{package_name}"
SKELETON_EXPECTED_SHAPE = [3, 150, 31, 1]
REGIONS_EXPECTED_SHAPE = [3, 3, 64, 112, 112]
ACTIVE_REGIONS = ["left_hand", "right_hand", "face"]
REPORT_PATH_DEFAULT = Path("reports/packaging/gated_fusion_nslt100_kaggle_package_report.md")


class PackagingError(RuntimeError):
    """Raised when the package cannot be built safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a Kaggle-ready package for WLASL NSLT100 gated feature fusion."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("packaging_outputs"),
        help="Directory where the package folder and zip will be created.",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=PACKAGE_NAME_DEFAULT,
        help="Name of the package folder and zip basename.",
    )
    parser.add_argument(
        "--source-branch-inputs",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs"),
        help="Root directory containing skeleton/ and regions/ branch inputs.",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=Path("configs/train/fusion/gated_feature/nslt100/gated_feature_fusion_ce.yaml"),
        help="Base gated-fusion train config.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH_DEFAULT,
        help="Markdown report path written into the repo.",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=3,
        help="How many tensors per split the verify script should load.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing package folder and zip before rebuilding.",
    )
    parser.add_argument(
        "--zip",
        dest="create_zip",
        action="store_true",
        help="Create a zip archive after verify passes.",
    )
    parser.add_argument(
        "--no-zip",
        dest="create_zip",
        action="store_false",
        help="Skip zip creation.",
    )
    parser.set_defaults(create_zip=False)
    return parser


def resolve_path(value: Path, *, root: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise PackagingError(f"Missing required {label}: {path.as_posix()}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise PackagingError(f"Expected a YAML mapping at {path.as_posix()}, got {type(payload)!r}.")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PackagingError(f"Manifest has no header: {path.as_posix()}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def normalize_sample_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def list_existing_candidates(project_root: Path, patterns: list[str]) -> list[Path]:
    results: list[Path] = []
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.exists() and path.is_file():
                results.append(path.resolve())
    return dedupe_paths(results)


def get_state_dict(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if all(hasattr(value, "shape") for value in payload.values()):
        return payload
    return None


def verify_skeleton_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise PackagingError(f"Skeleton checkpoint does not expose a readable state dict: {path.as_posix()}")

    a_tensor = state_dict.get("A")
    data_bn_weight = state_dict.get("data_bn.weight")
    a_shape = tuple(int(item) for item in getattr(a_tensor, "shape", []))
    bn_shape = tuple(int(item) for item in getattr(data_bn_weight, "shape", []))

    if a_shape == (3, 27, 27) or bn_shape == (81,):
        raise PackagingError(
            f"Skeleton checkpoint looks like selected_27, not selected_31: {path.as_posix()}"
        )

    selected_31_confirmed = a_shape == (3, 31, 31) or bn_shape == (93,)
    if not selected_31_confirmed:
        payload_keypoint_set = str(payload.get("keypoint_set", "")).strip()
        payload_num_nodes = int(payload.get("num_nodes", -1)) if "num_nodes" in payload else -1
        if payload_keypoint_set == "selected_27" or payload_num_nodes == 27:
            raise PackagingError(
                f"Skeleton checkpoint metadata points to selected_27, not selected_31: {path.as_posix()}"
            )
        if payload_keypoint_set == "selected_31" or payload_num_nodes == 31:
            selected_31_confirmed = True

    if not selected_31_confirmed:
        raise PackagingError(
            "Could not confirm that the skeleton checkpoint is selected_31. "
            f"path={path.as_posix()} A_shape={a_shape or '<missing>'} data_bn_shape={bn_shape or '<missing>'}"
        )

    return {
        "selected_31_confirmed": True,
        "A_shape": list(a_shape) if a_shape else None,
        "data_bn_weight_shape": list(bn_shape) if bn_shape else None,
    }


def verify_regions_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise PackagingError(f"Regions checkpoint does not expose a readable state dict: {path.as_posix()}")

    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if isinstance(config, dict):
        model_cfg = dict(config.get("model", {}))
        dataset_cfg = dict(config.get("dataset", {}))
        model_name = str(model_cfg.get("name", "")).strip().lower()
        region_order = list(dataset_cfg.get("region_order", dataset_cfg.get("active_regions", [])))
        if model_name and "region_resnet18_gru" not in model_name:
            raise PackagingError(f"Regions checkpoint is not RegionResNet18GRU: {path.as_posix()}")
        if region_order:
            normalized = [str(item).strip() for item in region_order]
            if normalized != ACTIVE_REGIONS:
                raise PackagingError(
                    "Regions checkpoint config is not all-regions "
                    f"(expected {ACTIVE_REGIONS}, got {normalized}) in {path.as_posix()}"
                )

    return {"loadable": True}


def verify_skeleton_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    graph_cfg = dict(config.get("graph", {}))
    model_cfg = dict(config.get("model", {}))

    keypoint_set = str(dataset_cfg.get("keypoint_set", "")).strip()
    expected_shape = list(dataset_cfg.get("expected_shape", []))
    layout = str(graph_cfg.get("layout", "")).strip()
    num_nodes = int(model_cfg.get("num_nodes", -1)) if "num_nodes" in model_cfg else -1
    num_classes = int(model_cfg.get("num_classes", -1)) if "num_classes" in model_cfg else -1

    if keypoint_set == "selected_27" or layout == "selected_27" or num_nodes == 27:
        raise PackagingError(f"Skeleton config is selected_27, not selected_31: {path.as_posix()}")
    if keypoint_set != "selected_31":
        raise PackagingError(f"Skeleton config keypoint_set must be selected_31: {path.as_posix()}")
    if expected_shape != SKELETON_EXPECTED_SHAPE:
        raise PackagingError(
            f"Skeleton config expected_shape must be {SKELETON_EXPECTED_SHAPE}: {path.as_posix()}"
        )
    if layout != "selected_31":
        raise PackagingError(f"Skeleton config graph.layout must be selected_31: {path.as_posix()}")
    if num_nodes != 31:
        raise PackagingError(f"Skeleton config model.num_nodes must be 31: {path.as_posix()}")
    if num_classes != 100:
        raise PackagingError(f"Skeleton config model.num_classes must be 100: {path.as_posix()}")

    return {
        "selected_31_confirmed": True,
        "expected_shape": expected_shape,
        "graph_layout": layout,
        "num_nodes": num_nodes,
        "num_classes": num_classes,
    }


def verify_regions_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    model_cfg = dict(config.get("model", {}))

    expected_shape = list(dataset_cfg.get("expected_shape", []))
    region_order = list(dataset_cfg.get("region_order", dataset_cfg.get("active_regions", [])))
    if not region_order and "num_regions" in model_cfg and int(model_cfg["num_regions"]) == 3:
        region_order = list(ACTIVE_REGIONS)
    normalized_regions = [str(item).strip() for item in region_order]
    model_name = str(model_cfg.get("name", "")).strip().lower()
    encoder_name = str(model_cfg.get("encoder_name", "")).strip().lower()
    num_classes = int(model_cfg.get("num_classes", -1)) if "num_classes" in model_cfg else -1

    if expected_shape != REGIONS_EXPECTED_SHAPE:
        raise PackagingError(
            f"Regions config expected_shape must be {REGIONS_EXPECTED_SHAPE}: {path.as_posix()}"
        )
    if normalized_regions and normalized_regions != ACTIVE_REGIONS:
        raise PackagingError(
            "Regions config is not all-regions "
            f"(expected {ACTIVE_REGIONS}, got {normalized_regions}) in {path.as_posix()}"
        )
    if model_name and "region_resnet18_gru" not in model_name:
        raise PackagingError(f"Regions config model.name must be RegionResNet18GRU: {path.as_posix()}")
    if encoder_name and encoder_name != "resnet18":
        raise PackagingError(f"Regions config encoder_name must be resnet18: {path.as_posix()}")
    if num_classes != 100:
        raise PackagingError(f"Regions config model.num_classes must be 100: {path.as_posix()}")

    return {
        "all_regions_confirmed": True,
        "expected_shape": expected_shape,
        "active_regions": normalized_regions or list(ACTIVE_REGIONS),
        "model_name": model_name or "region_resnet18_gru",
        "num_classes": num_classes,
    }


def choose_candidate(
    *,
    label: str,
    candidates: list[Path],
    verifier,
) -> tuple[Path, dict[str, Any]]:
    if not candidates:
        raise PackagingError(f"No candidates found for {label}.")

    errors: list[str] = []
    for candidate in candidates:
        try:
            result = verifier(candidate)
            return candidate, result
        except Exception as exc:  # pragma: no cover - surfaced in final error message
            errors.append(f"{candidate.as_posix()}: {exc}")
    joined = "\n".join(errors)
    raise PackagingError(f"Could not choose a valid {label}. Checked:\n{joined}")


def build_repo_audit(project_root: Path, branch_inputs_root: Path) -> dict[str, Any]:
    skeleton_checkpoint_candidates = list_existing_candidates(
        project_root,
        [
            "checkpoints/models/skeleton/best.pt",
            "artifacts/fusion/nslt100/checkpoints/skeleton/best.pt",
            "artifacts/fusion/nslt100/**/skeleton*/best.pt",
            "outputs/**/skeleton*/best.pt",
        ],
    )
    regions_checkpoint_candidates = list_existing_candidates(
        project_root,
        [
            "checkpoints/models/regions/best.pt",
            "artifacts/fusion/nslt100/checkpoints/regions/best.pt",
            "artifacts/fusion/nslt100/**/regions*/best.pt",
            "outputs/**/regions*/best.pt",
        ],
    )
    skeleton_config_candidates = list_existing_candidates(
        project_root,
        [
            "artifacts/fusion/nslt100/configs/skeleton/config_resolved.yaml",
            "artifacts/fusion/nslt100/configs/skeleton_config_resolved.yaml",
            "artifacts/fusion/nslt100/configs/**/skeleton*/config_resolved.yaml",
            "checkpoints/models/skeleton/config_resolved.yaml",
        ],
    )
    regions_config_candidates = list_existing_candidates(
        project_root,
        [
            "artifacts/fusion/nslt100/configs/regions/config_resolved.yaml",
            "artifacts/fusion/nslt100/configs/regions_config_resolved.yaml",
            "artifacts/fusion/nslt100/configs/**/regions*/config_resolved.yaml",
            "checkpoints/models/regions/config_resolved.yaml",
        ],
    )

    skeleton_branch_root = (branch_inputs_root / "skeleton" / "rtmw_l").resolve()
    regions_branch_root = (branch_inputs_root / "regions" / "rtmw_l").resolve()
    skeleton_manifest_dir = skeleton_branch_root / "manifests"
    regions_manifest_dir = regions_branch_root / "manifests"

    skeleton_manifest_files = [
        skeleton_manifest_dir / "nslt100_selected_31_train.csv",
        skeleton_manifest_dir / "nslt100_selected_31_val.csv",
        skeleton_manifest_dir / "nslt100_selected_31_test.csv",
    ]
    regions_manifest_files = [
        regions_manifest_dir / "nslt100_train.csv",
        regions_manifest_dir / "nslt100_val.csv",
        regions_manifest_dir / "nslt100_test.csv",
    ]
    for path in skeleton_manifest_files + regions_manifest_files:
        ensure_exists(path, "branch-input manifest")

    skeleton_tensor_root_candidates = [
        skeleton_branch_root / "tensors" / SUBSET,
        skeleton_branch_root / "graph_tensors" / "selected_31" / SUBSET,
    ]
    skeleton_tensor_root = next((path for path in skeleton_tensor_root_candidates if path.exists()), None)
    if skeleton_tensor_root is None:
        raise PackagingError(
            "Could not find skeleton tensor source root. Checked: "
            + ", ".join(path.as_posix() for path in skeleton_tensor_root_candidates)
        )
    regions_tensor_root = regions_branch_root / "tensors" / SUBSET
    ensure_exists(regions_tensor_root, "regions tensor root")

    selected_skeleton_checkpoint, skeleton_checkpoint_verify = choose_candidate(
        label="skeleton checkpoint",
        candidates=skeleton_checkpoint_candidates,
        verifier=verify_skeleton_checkpoint,
    )
    selected_regions_checkpoint, regions_checkpoint_verify = choose_candidate(
        label="regions checkpoint",
        candidates=regions_checkpoint_candidates,
        verifier=verify_regions_checkpoint,
    )
    selected_skeleton_config, skeleton_config_verify = choose_candidate(
        label="skeleton config",
        candidates=skeleton_config_candidates,
        verifier=verify_skeleton_config,
    )
    selected_regions_config, regions_config_verify = choose_candidate(
        label="regions config",
        candidates=regions_config_candidates,
        verifier=verify_regions_config,
    )

    return {
        "skeleton_checkpoint_candidates": skeleton_checkpoint_candidates,
        "regions_checkpoint_candidates": regions_checkpoint_candidates,
        "skeleton_config_candidates": skeleton_config_candidates,
        "regions_config_candidates": regions_config_candidates,
        "skeleton_manifest_files": skeleton_manifest_files,
        "regions_manifest_files": regions_manifest_files,
        "skeleton_tensor_root": skeleton_tensor_root,
        "regions_tensor_root": regions_tensor_root,
        "skeleton_branch_root": skeleton_branch_root,
        "regions_branch_root": regions_branch_root,
        "selected_skeleton_checkpoint": selected_skeleton_checkpoint,
        "selected_regions_checkpoint": selected_regions_checkpoint,
        "selected_skeleton_config": selected_skeleton_config,
        "selected_regions_config": selected_regions_config,
        "skeleton_checkpoint_verify": skeleton_checkpoint_verify,
        "regions_checkpoint_verify": regions_checkpoint_verify,
        "skeleton_config_verify": skeleton_config_verify,
        "regions_config_verify": regions_config_verify,
    }


def print_repo_audit(audit: dict[str, Any]) -> None:
    print("Repo audit:")
    for label, key in (
        ("Skeleton checkpoint candidates", "skeleton_checkpoint_candidates"),
        ("Regions checkpoint candidates", "regions_checkpoint_candidates"),
        ("Skeleton config candidates", "skeleton_config_candidates"),
        ("Regions config candidates", "regions_config_candidates"),
    ):
        print(f"{label}:")
        for path in audit[key]:
            print(f"  - {path.as_posix()}")
    print(f"Selected skeleton checkpoint: {audit['selected_skeleton_checkpoint'].as_posix()}")
    print(f"Selected regions checkpoint: {audit['selected_regions_checkpoint'].as_posix()}")
    print(f"Selected skeleton config: {audit['selected_skeleton_config'].as_posix()}")
    print(f"Selected regions config: {audit['selected_regions_config'].as_posix()}")
    print(
        "Skeleton selected_31 verify result: "
        + json.dumps(audit["skeleton_checkpoint_verify"], ensure_ascii=False)
    )
    print(
        "Regions config verify result: "
        + json.dumps(audit["regions_config_verify"], ensure_ascii=False)
    )


def build_kaggle_root(package_name: str) -> str:
    return KAGGLE_ROOT_TEMPLATE.format(package_name=package_name)


def resolve_skeleton_tensor_path(row: dict[str, str], tensor_root: Path, split: str) -> Path:
    raw_path = Path(row.get("graph_tensor_path", ""))
    if str(raw_path) and raw_path.exists():
        return raw_path
    sample_id = str(row.get("sample_id", "")).strip()
    candidate = tensor_root / split / f"{sample_id}.npz"
    if candidate.exists():
        return candidate
    raise PackagingError(
        f"Missing skeleton tensor for sample_id={sample_id} split={split}: {candidate.as_posix()}"
    )


def resolve_regions_tensor_path(row: dict[str, str], tensor_root: Path, split: str) -> Path:
    raw_path = Path(row.get("tensor_path", ""))
    if str(raw_path) and raw_path.exists():
        return raw_path
    sample_id = str(row.get("sample_id", "")).strip()
    candidate = tensor_root / split / f"{sample_id}.npz"
    if candidate.exists():
        return candidate
    raise PackagingError(
        f"Missing regions tensor for sample_id={sample_id} split={split}: {candidate.as_posix()}"
    )


def copy_skeleton_branch(
    *,
    manifest_dir: Path,
    tensor_root: Path,
    package_root: Path,
) -> dict[str, int]:
    destination_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l"
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        manifest_name = f"{SUBSET}_selected_31_{split}.csv"
        fieldnames, rows = read_manifest(manifest_dir / manifest_name)
        rewritten_rows: list[dict[str, str]] = []

        for row in rows:
            source_tensor = resolve_skeleton_tensor_path(row, tensor_root, split)
            destination_tensor = destination_root / "tensors" / SUBSET / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["graph_tensor_path"] = f"tensors/{SUBSET}/{split}/{source_tensor.name}"
            for key in ("pose_path", "selected_path", "normalized_path", "error_message"):
                if key in updated:
                    updated[key] = ""
            rewritten_rows.append(updated)

        write_manifest(destination_root / "manifests" / manifest_name, fieldnames, rewritten_rows)
        counts[split] = len(rewritten_rows)
    return counts


def copy_regions_branch(
    *,
    manifest_dir: Path,
    tensor_root: Path,
    reports_root: Path,
    package_root: Path,
) -> tuple[dict[str, int], list[str]]:
    destination_root = package_root / "branch_inputs" / "regions" / "rtmw_l"
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        manifest_name = f"{SUBSET}_{split}.csv"
        fieldnames, rows = read_manifest(manifest_dir / manifest_name)
        rewritten_rows: list[dict[str, str]] = []

        for row in rows:
            source_tensor = resolve_regions_tensor_path(row, tensor_root, split)
            destination_tensor = destination_root / "tensors" / SUBSET / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["tensor_path"] = f"tensors/{SUBSET}/{split}/{source_tensor.name}"
            for key in ("crop_root", "preview_path", "error_message"):
                if key in updated:
                    updated[key] = ""
            rewritten_rows.append(updated)

        write_manifest(destination_root / "manifests" / manifest_name, fieldnames, rewritten_rows)
        counts[split] = len(rewritten_rows)

    copied_reports: list[str] = []
    for name in ("nslt100_region_crop_quality_report.md", "nslt100_region_low_quality_samples.csv"):
        source_path = reports_root / name
        if source_path.exists():
            copy_file(source_path, destination_root / "reports" / name)
            copied_reports.append(f"reports/{name}")
    return counts, copied_reports


def create_kaggle_config(
    *,
    base_config: dict[str, Any],
    package_name: str,
) -> dict[str, Any]:
    kaggle_root = build_kaggle_root(package_name)
    config = json.loads(json.dumps(base_config))

    config.setdefault("experiment", {})
    config["experiment"]["output_root"] = "/kaggle/working/outputs/fusion"

    config.setdefault("dataset", {})
    config["dataset"]["subset"] = SUBSET
    config["dataset"]["num_classes"] = 100
    config["dataset"]["skeleton"] = {
        "data_root": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l",
        "keypoint_set": "selected_31",
        "expected_shape": list(SKELETON_EXPECTED_SHAPE),
        "manifests": {
            "train": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt100_selected_31_train.csv",
            "val": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt100_selected_31_val.csv",
            "test": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt100_selected_31_test.csv",
        },
        "return_metadata": True,
        "strict_shape_check": True,
    }
    config["dataset"]["regions"] = {
        "data_root": f"{kaggle_root}/branch_inputs/regions/rtmw_l",
        "expected_shape": list(REGIONS_EXPECTED_SHAPE),
        "region_order": list(ACTIVE_REGIONS),
        "active_regions": list(ACTIVE_REGIONS),
        "manifests": {
            "train": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt100_train.csv",
            "val": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt100_val.csv",
            "test": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt100_test.csv",
        },
        "normalize": {"type": "imagenet"},
        "return_metadata": True,
        "strict_shape_check": True,
    }

    config["skeleton_branch"]["config_path"] = f"{kaggle_root}/configs/skeleton_config_resolved.yaml"
    config["skeleton_branch"]["checkpoint_path"] = f"{kaggle_root}/checkpoints/skeleton/best.pt"
    config["regions_branch"]["config_path"] = f"{kaggle_root}/configs/regions_config_resolved.yaml"
    config["regions_branch"]["checkpoint_path"] = f"{kaggle_root}/checkpoints/regions/best.pt"
    return config


def create_metadata(
    *,
    package_name: str,
    skeleton_counts: dict[str, int],
    regions_counts: dict[str, int],
    kaggle_config: dict[str, Any],
) -> dict[str, Any]:
    fusion_model = dict(kaggle_config.get("fusion_model", {}))
    return {
        "package_name": package_name,
        "subset": SUBSET,
        "purpose": "Kaggle-ready package for Gated Feature Fusion",
        "skeleton": {
            "checkpoint": "checkpoints/skeleton/best.pt",
            "config": "configs/skeleton_config_resolved.yaml",
            "keypoint_set": "selected_31",
            "expected_shape": list(SKELETON_EXPECTED_SHAPE),
            "counts": skeleton_counts,
        },
        "regions": {
            "checkpoint": "checkpoints/regions/best.pt",
            "config": "configs/regions_config_resolved.yaml",
            "active_regions": list(ACTIVE_REGIONS),
            "expected_shape": list(REGIONS_EXPECTED_SHAPE),
            "counts": regions_counts,
        },
        "fusion": {
            "config": "configs/gated_feature_fusion_nslt100_kaggle.yaml",
            "model": "GatedFeatureFusion",
            "hidden_dim": int(fusion_model.get("hidden_dim", 256)),
            "freeze_skeleton": bool(fusion_model.get("freeze_skeleton", True)),
            "freeze_regions": bool(fusion_model.get("freeze_regions", True)),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_readme(package_name: str) -> str:
    kaggle_root = build_kaggle_root(package_name)
    return textwrap.dedent(
        f"""
        # WLASL NSLT100 Gated Fusion Ready Dataset

        ## 1. Purpose
        This package is a Kaggle-ready bundle for training and evaluating Gated Feature Fusion on WLASL NSLT100.

        ## 2. What is included
        Package này bao gồm:
        - Skeleton selected_31 branch inputs
        - Regions all-regions branch inputs
        - Skeleton best.pt
        - Regions best.pt
        - Skeleton config_resolved.yaml
        - Regions config_resolved.yaml
        - Gated Fusion Kaggle config

        ## 3. Folder structure
        ```text
        {package_name}/
        |-- README.md
        |-- metadata.json
        |-- configs/
        |   |-- gated_feature_fusion_nslt100_kaggle.yaml
        |   |-- skeleton_config_resolved.yaml
        |   `-- regions_config_resolved.yaml
        |-- checkpoints/
        |   |-- skeleton/best.pt
        |   `-- regions/best.pt
        |-- branch_inputs/
        |   |-- skeleton/rtmw_l/...
        |   `-- regions/rtmw_l/...
        `-- verify/
            |-- verify_package.py
            `-- verify_summary.json
        ```

        ## 4. How to add this dataset to Kaggle
        Add the dataset to your Kaggle Notebook. The generated config assumes the dataset is mounted at:
        `{kaggle_root}`

        If the Kaggle dataset slug is different, edit the mounted paths in the config or copy the config into `/kaggle/working` and update it there.

        ## 5. How to train Gated Fusion on Kaggle
        ```bash
        python scripts/train/train_gated_fusion.py \\
          --config {kaggle_root}/configs/gated_feature_fusion_nslt100_kaggle.yaml
        ```

        ## 6. How to evaluate
        ```bash
        python scripts/evaluate/evaluate_gated_fusion.py \\
          --config {kaggle_root}/configs/gated_feature_fusion_nslt100_kaggle.yaml \\
          --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt100-sel31-ce-regions/best.pt \\
          --split test
        ```

        ## 7. Notes
        - `/kaggle/input` is read-only.
        - Training outputs must go to `/kaggle/working/outputs/fusion`.
        - This package does not include raw videos, W&B logs, notebook caches, or intermediate checkpoints.
        """
    ).strip() + "\n"


def create_verify_script() -> str:
    return textwrap.dedent(
        """
        \"\"\"Verify a Kaggle-ready NSLT100 gated-fusion package.\"\"\"

        from __future__ import annotations

        import argparse
        import csv
        import json
        from pathlib import Path
        from typing import Any

        import numpy as np
        import torch
        import yaml


        SKELETON_EXPECTED_SHAPE = (3, 150, 31, 1)
        REGIONS_EXPECTED_SHAPE = (3, 3, 64, 112, 112)
        ACTIVE_REGIONS = ["left_hand", "right_hand", "face"]
        EXPECTED_COUNTS = {"train": 748, "val": 165, "test": 100}


        def build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(description="Verify a packaged gated-fusion Kaggle bundle.")
            parser.add_argument("--package-root", type=Path, required=True)
            parser.add_argument("--sample-checks", type=int, default=3)
            return parser


        def read_manifest(path: Path) -> list[dict[str, str]]:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Manifest has no header: {path.as_posix()}")
                return [{key: value or "" for key, value in row.items()} for row in reader]


        def read_yaml(path: Path) -> dict[str, Any]:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if not isinstance(payload, dict):
                raise TypeError(f"Expected YAML mapping at {path.as_posix()}.")
            return payload


        def normalize_sample_id(value: Any) -> str:
            text = str(value or "").strip()
            if text.isdigit():
                return str(int(text))
            return text


        def resolve_package_path(branch_root: Path, value: str) -> Path:
            raw = Path(value)
            return raw if raw.is_absolute() else (branch_root / raw)


        def get_state_dict(payload: Any) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None
            for key in ("model_state_dict", "state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if all(hasattr(value, "shape") for value in payload.values()):
                return payload
            return None


        def verify_tensor_split(
            *,
            branch_name: str,
            branch_root: Path,
            manifest_name: str,
            tensor_key: str,
            expected_shape: tuple[int, ...],
            sample_checks: int,
        ) -> dict[str, Any]:
            split = manifest_name.replace("nslt100_selected_31_", "").replace("nslt100_", "").replace(".csv", "")
            manifest_path = branch_root / "manifests" / manifest_name
            tensor_dir = branch_root / "tensors" / "nslt100" / split
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing {branch_name} manifest: {manifest_path.as_posix()}")
            if not tensor_dir.exists():
                raise FileNotFoundError(f"Missing {branch_name} tensor directory: {tensor_dir.as_posix()}")

            rows = read_manifest(manifest_path)
            tensor_files = sorted(path for path in tensor_dir.iterdir() if path.is_file() and path.suffix == ".npz")
            if len(rows) != len(tensor_files):
                raise ValueError(
                    f"{branch_name} manifest/tensor count mismatch for split={split}: "
                    f"rows={len(rows)} tensors={len(tensor_files)}"
                )
            if len(rows) != EXPECTED_COUNTS[split]:
                raise ValueError(
                    f"{branch_name} split={split} count mismatch: expected={EXPECTED_COUNTS[split]} got={len(rows)}"
                )

            checked: list[str] = []
            for row in rows[:sample_checks]:
                tensor_path = resolve_package_path(branch_root, row.get(tensor_key, ""))
                if not tensor_path.exists():
                    raise FileNotFoundError(
                        f"Missing {branch_name} tensor referenced by manifest: {tensor_path.as_posix()}"
                    )
                with np.load(tensor_path, allow_pickle=False) as payload:
                    if "data" in payload:
                        array = payload["data"]
                    elif "tensor" in payload:
                        array = payload["tensor"]
                    else:
                        raise KeyError(f"Tensor file lacks 'data' or 'tensor': {tensor_path.as_posix()}")
                    if tuple(int(item) for item in array.shape) != expected_shape:
                        raise ValueError(
                            f"{branch_name} tensor shape mismatch for {tensor_path.name}: "
                            f"expected={expected_shape} got={tuple(array.shape)}"
                        )
                checked.append(str(row.get("sample_id", "")))

            return {
                "count": len(rows),
                "sample_checks": checked,
            }


        def verify_pairing(package_root: Path) -> dict[str, Any]:
            skeleton_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests"
            regions_root = package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests"
            summary: dict[str, Any] = {}

            for split, expected_count in EXPECTED_COUNTS.items():
                skeleton_rows = read_manifest(skeleton_root / f"nslt100_selected_31_{split}.csv")
                regions_rows = read_manifest(regions_root / f"nslt100_{split}.csv")
                skeleton_map = {normalize_sample_id(row.get("sample_id", "")): row for row in skeleton_rows}
                regions_map = {normalize_sample_id(row.get("sample_id", "")): row for row in regions_rows}
                matched_ids = sorted(set(skeleton_map) & set(regions_map))
                missing_in_skeleton = sorted(set(regions_map) - set(skeleton_map))
                missing_in_regions = sorted(set(skeleton_map) - set(regions_map))
                mismatches = [
                    sample_id
                    for sample_id in matched_ids
                    if str(skeleton_map[sample_id].get("class_id", "")).strip()
                    != str(regions_map[sample_id].get("class_id", "")).strip()
                ]
                if missing_in_skeleton or missing_in_regions:
                    raise ValueError(
                        f"Pairing mismatch in split={split}: "
                        f"missing_in_skeleton={len(missing_in_skeleton)} missing_in_regions={len(missing_in_regions)}"
                    )
                if mismatches:
                    raise ValueError(
                        f"Label mismatches in split={split}: count={len(mismatches)} examples={mismatches[:5]}"
                    )
                if len(matched_ids) != expected_count:
                    raise ValueError(
                        f"Matched pair count mismatch in split={split}: expected={expected_count} got={len(matched_ids)}"
                    )
                summary[split] = {
                    "matched": len(matched_ids),
                    "label_mismatch": 0,
                }
            return summary


        def verify_skeleton_checkpoint(path: Path) -> dict[str, Any]:
            payload = torch.load(path, map_location="cpu")
            state_dict = get_state_dict(payload)
            if state_dict is None:
                raise ValueError(f"Skeleton checkpoint lacks a readable state dict: {path.as_posix()}")
            a_shape = tuple(int(item) for item in getattr(state_dict.get("A"), "shape", []))
            bn_shape = tuple(int(item) for item in getattr(state_dict.get("data_bn.weight"), "shape", []))
            if a_shape == (3, 27, 27) or bn_shape == (81,):
                raise ValueError(f"Skeleton checkpoint looks like selected_27: {path.as_posix()}")
            if a_shape != (3, 31, 31) and bn_shape != (93,):
                raise ValueError(
                    "Could not confirm selected_31 from skeleton checkpoint. "
                    f"path={path.as_posix()} A_shape={a_shape} data_bn_weight_shape={bn_shape}"
                )
            return {
                "selected_31_confirmed": True,
                "A_shape": list(a_shape) if a_shape else None,
                "data_bn_weight_shape": list(bn_shape) if bn_shape else None,
            }


        def verify_skeleton_config(path: Path) -> dict[str, Any]:
            config = read_yaml(path)
            dataset_cfg = dict(config.get("dataset", {}))
            graph_cfg = dict(config.get("graph", {}))
            model_cfg = dict(config.get("model", {}))
            if str(dataset_cfg.get("keypoint_set", "")).strip() != "selected_31":
                raise ValueError(f"Skeleton config keypoint_set must be selected_31: {path.as_posix()}")
            if list(dataset_cfg.get("expected_shape", [])) != list(SKELETON_EXPECTED_SHAPE):
                raise ValueError(f"Skeleton config expected_shape mismatch: {path.as_posix()}")
            if str(graph_cfg.get("layout", "")).strip() != "selected_31":
                raise ValueError(f"Skeleton config graph.layout must be selected_31: {path.as_posix()}")
            if int(model_cfg.get("num_nodes", -1)) != 31:
                raise ValueError(f"Skeleton config model.num_nodes must be 31: {path.as_posix()}")
            return {"selected_31_confirmed": True}


        def verify_regions_config(path: Path) -> dict[str, Any]:
            config = read_yaml(path)
            dataset_cfg = dict(config.get("dataset", {}))
            model_cfg = dict(config.get("model", {}))
            if list(dataset_cfg.get("expected_shape", [])) != list(REGIONS_EXPECTED_SHAPE):
                raise ValueError(f"Regions config expected_shape mismatch: {path.as_posix()}")
            region_order = list(dataset_cfg.get("region_order", dataset_cfg.get("active_regions", [])))
            normalized = [str(item).strip() for item in region_order]
            if normalized and normalized != ACTIVE_REGIONS:
                raise ValueError(f"Regions config must be all-regions: {path.as_posix()}")
            if "region_resnet18_gru" not in str(model_cfg.get("name", "")).strip().lower():
                raise ValueError(f"Regions config model.name must be RegionResNet18GRU: {path.as_posix()}")
            return {"all_regions_confirmed": True}


        def verify_kaggle_config(path: Path, package_name: str) -> dict[str, Any]:
            config = read_yaml(path)
            kaggle_root = f"/kaggle/input/{package_name}/{package_name}"
            skeleton_cfg = dict(config.get("skeleton_branch", {}))
            regions_cfg = dict(config.get("regions_branch", {}))
            experiment_cfg = dict(config.get("experiment", {}))
            if skeleton_cfg.get("config_path") != f"{kaggle_root}/configs/skeleton_config_resolved.yaml":
                raise ValueError("Kaggle config skeleton_branch.config_path is incorrect.")
            if regions_cfg.get("config_path") != f"{kaggle_root}/configs/regions_config_resolved.yaml":
                raise ValueError("Kaggle config regions_branch.config_path is incorrect.")
            if skeleton_cfg.get("checkpoint_path") != f"{kaggle_root}/checkpoints/skeleton/best.pt":
                raise ValueError("Kaggle config skeleton checkpoint path is incorrect.")
            if regions_cfg.get("checkpoint_path") != f"{kaggle_root}/checkpoints/regions/best.pt":
                raise ValueError("Kaggle config regions checkpoint path is incorrect.")
            if experiment_cfg.get("output_root") != "/kaggle/working/outputs/fusion":
                raise ValueError("Kaggle config experiment.output_root is incorrect.")
            return {"paths_confirmed": True}


        def main() -> int:
            args = build_parser().parse_args()
            package_root = args.package_root.resolve()
            package_name = package_root.name

            skeleton_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l"
            regions_root = package_root / "branch_inputs" / "regions" / "rtmw_l"
            skeleton_summary = {}
            regions_summary = {}
            for split in ("train", "val", "test"):
                skeleton_summary[split] = verify_tensor_split(
                    branch_name="skeleton",
                    branch_root=skeleton_root,
                    manifest_name=f"nslt100_selected_31_{split}.csv",
                    tensor_key="graph_tensor_path",
                    expected_shape=SKELETON_EXPECTED_SHAPE,
                    sample_checks=int(args.sample_checks),
                )
                regions_summary[split] = verify_tensor_split(
                    branch_name="regions",
                    branch_root=regions_root,
                    manifest_name=f"nslt100_{split}.csv",
                    tensor_key="tensor_path",
                    expected_shape=REGIONS_EXPECTED_SHAPE,
                    sample_checks=int(args.sample_checks),
                )

            skeleton_checkpoint_path = package_root / "checkpoints" / "skeleton" / "best.pt"
            regions_checkpoint_path = package_root / "checkpoints" / "regions" / "best.pt"
            skeleton_config_path = package_root / "configs" / "skeleton_config_resolved.yaml"
            regions_config_path = package_root / "configs" / "regions_config_resolved.yaml"
            kaggle_config_path = package_root / "configs" / "gated_feature_fusion_nslt100_kaggle.yaml"

            for required in (
                skeleton_checkpoint_path,
                regions_checkpoint_path,
                skeleton_config_path,
                regions_config_path,
                kaggle_config_path,
            ):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required packaged file: {required.as_posix()}")

            pairing = verify_pairing(package_root)
            summary = {
                "status": "pass",
                "package_root": package_root.as_posix(),
                "skeleton": {
                    "splits": skeleton_summary,
                    "checkpoint": verify_skeleton_checkpoint(skeleton_checkpoint_path),
                    "config": verify_skeleton_config(skeleton_config_path),
                },
                "regions": {
                    "splits": regions_summary,
                    "checkpoint": {"loadable": True} if torch.load(regions_checkpoint_path, map_location="cpu") is not None else None,
                    "config": verify_regions_config(regions_config_path),
                },
                "kaggle_config": verify_kaggle_config(kaggle_config_path, package_name),
                "pairing": pairing,
            }
            summary_path = package_root / "verify" / "verify_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("VERIFY PASS")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip() + "\n"


def write_verify_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(create_verify_script(), encoding="utf-8")


def summarize_directory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def zip_directory(source_dir: Path, zip_path: Path, *, prefix: str) -> int:
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for item in sorted(source_dir.rglob("*")):
            if item.is_dir():
                continue
            archive.write(item, arcname=f"{prefix}/{item.relative_to(source_dir).as_posix()}")
    return zip_path.stat().st_size


def run_verify(package_root: Path, sample_checks: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str((package_root / "verify" / "verify_package.py").resolve()),
        "--package-root",
        str(package_root.resolve()),
        "--sample-checks",
        str(sample_checks),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(package_root.resolve()),
    )
    if completed.returncode != 0:
        raise PackagingError(
            "Package verify failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    summary_path = package_root / "verify" / "verify_summary.json"
    ensure_exists(summary_path, "verify summary")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_report(
    *,
    package_name: str,
    package_root: Path,
    zip_path: Path | None,
    zip_size_bytes: int | None,
    deleted_package_dir: bool,
    deleted_zip: bool,
    audit: dict[str, Any],
    skeleton_counts: dict[str, int],
    regions_counts: dict[str, int],
    copied_region_reports: list[str],
    verify_summary: dict[str, Any],
) -> str:
    checkpoint_skeleton = audit["selected_skeleton_checkpoint"].as_posix()
    checkpoint_regions = audit["selected_regions_checkpoint"].as_posix()
    config_skeleton = audit["selected_skeleton_config"].as_posix()
    config_regions = audit["selected_regions_config"].as_posix()
    zip_path_text = zip_path.as_posix() if zip_path is not None else "not created"
    zip_size_text = format_bytes(zip_size_bytes or 0) if zip_size_bytes is not None else "not created"
    package_summary = summarize_directory(package_root)

    def format_candidates(title: str, paths: list[Path]) -> str:
        lines = [f"### {title}"]
        for path in paths:
            lines.append(f"- {path.as_posix()}")
        return "\n".join(lines)

    candidate_sections = "\n\n".join(
        [
            format_candidates("Skeleton checkpoint candidates", audit["skeleton_checkpoint_candidates"]),
            format_candidates("Regions checkpoint candidates", audit["regions_checkpoint_candidates"]),
            format_candidates("Skeleton config candidates", audit["skeleton_config_candidates"]),
            format_candidates("Regions config candidates", audit["regions_config_candidates"]),
        ]
    )
    reports_text = ", ".join(copied_region_reports) if copied_region_reports else "None"

    sections = [
        "# Gated Fusion NSLT100 Kaggle Package Report",
        "",
        "## 1. Mục tiêu",
        "Tạo lại package Kaggle đầy đủ để train và evaluate Gated Feature Fusion trên NSLT100.",
        "",
        "## 2. Lý do cần tạo lại package",
        "Package cũ thiếu `config_resolved.yaml` riêng của từng backbone nên chưa đủ điều kiện chạy độc lập trên Kaggle.",
        "",
        "## 3. Package cũ đã xóa những gì",
        f"- Deleted old package folder: {'yes' if deleted_package_dir else 'no'}",
        f"- Deleted old package zip: {'yes' if deleted_zip else 'no'}",
        "",
        "## 4. Rà soát repo và file được chọn",
        candidate_sections,
        "",
        "File được chọn:",
        f"- skeleton checkpoint: `{checkpoint_skeleton}`",
        f"- regions checkpoint: `{checkpoint_regions}`",
        f"- skeleton config: `{config_skeleton}`",
        f"- regions config: `{config_regions}`",
        "- skeleton selected_31 verify pass: `yes`",
        "- regions config verify pass: `yes`",
        "",
        "## 5. Cấu trúc package mới",
        f"- package root: `{package_root.as_posix()}`",
        "- includes `README.md`, `metadata.json`, `configs/`, `checkpoints/`, `branch_inputs/`, `verify/`",
        "",
        "## 6. Data đã đóng gói",
        f"- Skeleton selected_31 shape: `{tuple(SKELETON_EXPECTED_SHAPE)}`",
        f"- Skeleton counts: train={skeleton_counts['train']}, val={skeleton_counts['val']}, test={skeleton_counts['test']}",
        f"- Regions all-regions shape: `{tuple(REGIONS_EXPECTED_SHAPE)}`",
        f"- Regions counts: train={regions_counts['train']}, val={regions_counts['val']}, test={regions_counts['test']}",
        f"- Copied region reports: {reports_text}",
        "",
        "## 7. Checkpoints đã đóng gói",
        "- `checkpoints/skeleton/best.pt`",
        "- `checkpoints/regions/best.pt`",
        "",
        "## 8. Configs đã đóng gói",
        "- `configs/skeleton_config_resolved.yaml`",
        "- `configs/regions_config_resolved.yaml`",
        "- `configs/gated_feature_fusion_nslt100_kaggle.yaml`",
        "",
        "## 9. Kaggle-ready config",
        f"- Uses `/kaggle/input/{package_name}/{package_name}` as the dataset mount root",
        "- Uses `/kaggle/working/outputs/fusion` as `experiment.output_root`",
        "- `config_path` is non-empty for both branches",
        "",
        "## 10. Metadata",
        "- `metadata.json` written with explicit skeleton, regions, and fusion sections",
        "",
        "## 11. Verify package",
        f"- Verify result: `{str(verify_summary.get('status', '')).upper()}`",
        f"- Pairing train matched: `{verify_summary['pairing']['train']['matched']}`",
        f"- Pairing val matched: `{verify_summary['pairing']['val']['matched']}`",
        f"- Pairing test matched: `{verify_summary['pairing']['test']['matched']}`",
        "",
        "## 12. Cách upload lên Kaggle Dataset",
        "Upload folder package hoặc zip file vào một Kaggle Dataset private rồi add dataset đó vào notebook.",
        "",
        "## 13. Cách train trên Kaggle",
        "```bash",
        "python scripts/train/train_gated_fusion.py \\",
        f"  --config /kaggle/input/{package_name}/{package_name}/configs/gated_feature_fusion_nslt100_kaggle.yaml",
        "```",
        "",
        "## 14. Cách evaluate trên Kaggle",
        "```bash",
        "python scripts/evaluate/evaluate_gated_fusion.py \\",
        f"  --config /kaggle/input/{package_name}/{package_name}/configs/gated_feature_fusion_nslt100_kaggle.yaml \\",
        "  --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt100-sel31-ce-regions/best.pt \\",
        "  --split test",
        "```",
        "",
        "## 15. Những gì không đóng gói",
        "- Raw videos",
        "- W&B logs",
        "- Old training outputs",
        "- Intermediate checkpoints",
        "- Notebook cache",
        "",
        "## 16. Lưu ý quan trọng",
        "- `/kaggle/input` là read-only",
        f"- Nếu slug khác `{package_name}` thì phải sửa lại path trong config",
        f"- Zip path: `{zip_path_text}`",
        f"- Zip size: `{zip_size_text}`",
        f"- Package file count: `{package_summary['file_count']}`",
        "",
        "## 17. Kết luận",
        "Package mới đã bao gồm đầy đủ branch configs riêng, checkpoints, branch inputs, Kaggle config, README, metadata, verify script và verify summary.",
    ]
    return "\n".join(sections).strip() + "\n"


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()

    output_root = resolve_path(args.output_root, root=project_root)
    branch_inputs_root = resolve_path(args.source_branch_inputs, root=project_root)
    fusion_config_path = resolve_path(args.fusion_config, root=project_root)
    report_path = resolve_path(args.report_path, root=project_root)
    package_name = str(args.package_name).strip()
    if not package_name:
        raise PackagingError("package_name must not be empty.")

    package_root = output_root / package_name
    zip_path = output_root / f"{package_name}.zip"
    output_root.mkdir(parents=True, exist_ok=True)

    audit = build_repo_audit(project_root, branch_inputs_root)
    print_repo_audit(audit)

    deleted_package_dir = False
    deleted_zip = False
    if args.clean:
        deleted_package_dir = remove_path(package_root)
        deleted_zip = remove_path(zip_path)
    elif package_root.exists() or zip_path.exists():
        raise PackagingError(
            "Package output already exists. Re-run with --clean to delete the old package folder and zip first."
        )

    ensure_exists(fusion_config_path, "gated fusion config")
    base_fusion_config = read_yaml(fusion_config_path)

    package_root.mkdir(parents=True, exist_ok=False)

    skeleton_counts = copy_skeleton_branch(
        manifest_dir=audit["skeleton_branch_root"] / "manifests",
        tensor_root=audit["skeleton_tensor_root"],
        package_root=package_root,
    )
    regions_counts, copied_region_reports = copy_regions_branch(
        manifest_dir=audit["regions_branch_root"] / "manifests",
        tensor_root=audit["regions_tensor_root"],
        reports_root=audit["regions_branch_root"] / "reports",
        package_root=package_root,
    )

    copy_file(audit["selected_skeleton_checkpoint"], package_root / "checkpoints" / "skeleton" / "best.pt")
    copy_file(audit["selected_regions_checkpoint"], package_root / "checkpoints" / "regions" / "best.pt")
    copy_file(audit["selected_skeleton_config"], package_root / "configs" / "skeleton_config_resolved.yaml")
    copy_file(audit["selected_regions_config"], package_root / "configs" / "regions_config_resolved.yaml")

    kaggle_config = create_kaggle_config(base_config=base_fusion_config, package_name=package_name)
    write_yaml(package_root / "configs" / "gated_feature_fusion_nslt100_kaggle.yaml", kaggle_config)
    write_json(
        package_root / "metadata.json",
        create_metadata(
            package_name=package_name,
            skeleton_counts=skeleton_counts,
            regions_counts=regions_counts,
            kaggle_config=kaggle_config,
        ),
    )
    (package_root / "README.md").write_text(create_readme(package_name), encoding="utf-8")
    write_verify_script(package_root / "verify" / "verify_package.py")

    verify_summary = run_verify(package_root, sample_checks=max(1, int(args.sample_checks)))

    zip_size_bytes: int | None = None
    if args.create_zip:
        zip_size_bytes = zip_directory(package_root, zip_path, prefix=package_name)

    report_text = create_report(
        package_name=package_name,
        package_root=package_root,
        zip_path=zip_path if args.create_zip else None,
        zip_size_bytes=zip_size_bytes,
        deleted_package_dir=deleted_package_dir,
        deleted_zip=deleted_zip,
        audit=audit,
        skeleton_counts=skeleton_counts,
        regions_counts=regions_counts,
        copied_region_reports=copied_region_reports,
        verify_summary=verify_summary,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    package_summary = summarize_directory(package_root)
    print(f"selected skeleton checkpoint: {audit['selected_skeleton_checkpoint'].as_posix()}")
    print(f"selected regions checkpoint: {audit['selected_regions_checkpoint'].as_posix()}")
    print(f"selected skeleton config: {audit['selected_skeleton_config'].as_posix()}")
    print(f"selected regions config: {audit['selected_regions_config'].as_posix()}")
    print(f"skeleton selected_31 verify result: {json.dumps(audit['skeleton_checkpoint_verify'], ensure_ascii=False)}")
    print(f"regions config verify result: {json.dumps(audit['regions_config_verify'], ensure_ascii=False)}")
    print(f"package path: {package_root.as_posix()}")
    print(f"zip path: {zip_path.as_posix() if args.create_zip else 'not created'}")
    print(f"zip size: {format_bytes(zip_size_bytes or 0) if args.create_zip else 'not created'}")
    print(f"total file count: {package_summary['file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
