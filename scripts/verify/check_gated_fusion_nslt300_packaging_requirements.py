"""Audit whether the repo is ready to package NSLT300 gated fusion for Kaggle."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "NumPy is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/verify/check_gated_fusion_nslt300_packaging_requirements.py"
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/verify/check_gated_fusion_nslt300_packaging_requirements.py"
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/verify/check_gated_fusion_nslt300_packaging_requirements.py"
    ) from exc


SUBSET = "nslt300"
NUM_CLASSES = 300
SKELETON_EXPECTED_SHAPE = [3, 150, 31, 1]
REGIONS_EXPECTED_SHAPE = [3, 3, 64, 112, 112]
ACTIVE_REGIONS = ["left_hand", "right_hand", "face"]
PAIRING_MIN_COVERAGE = 0.95
REPORT_PATH_DEFAULT = Path("reports/packaging/gated_fusion_nslt300_requirement_check_report.md")
GATED_CONFIG_PATH = Path("configs/train/fusion/gated_feature/nslt300/gated_feature_fusion_ce.yaml")
GATED_CONFIG_TEMPLATE = Path("configs/train/fusion/gated_feature/nslt100/gated_feature_fusion_ce.yaml")


class CheckError(RuntimeError):
    """Raised when a candidate fails validation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the repo is ready to package NSLT300 gated fusion for Kaggle."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root to scan.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH_DEFAULT,
        help="Markdown report path to write.",
    )
    parser.add_argument(
        "--pairing-min-coverage",
        type=float,
        default=PAIRING_MIN_COVERAGE,
        help="Minimum matched coverage required per split for pairing readiness.",
    )
    return parser


def resolve_path(value: Path, *, root: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise CheckError(f"Expected YAML mapping at {path.as_posix()}, got {type(payload)!r}.")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CheckError(f"Manifest has no header: {path.as_posix()}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows


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


def normalize_sample_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def parse_shape_literal(value: str) -> list[int] | None:
    if not str(value).strip():
        return None
    try:
        payload = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if isinstance(payload, (list, tuple)):
        try:
            return [int(item) for item in payload]
        except (TypeError, ValueError):
            return None
    return None


def load_tensor_shape(path: Path) -> list[int]:
    if not path.exists():
        raise CheckError(f"Tensor file does not exist: {path.as_posix()}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                array = payload[key]
                if hasattr(array, "shape"):
                    return [int(item) for item in array.shape]
        raise CheckError(f"No array found inside tensor file: {path.as_posix()}")
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return [int(item) for item in array.shape]
    raise CheckError(f"Unsupported tensor suffix for shape check: {path.as_posix()}")


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


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> tuple[int | None, list[str]]:
    preferred_keys = [
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "classifier.2.weight",
        "head.1.weight",
    ]
    evidence: list[str] = []
    for key in preferred_keys:
        tensor = state_dict.get(key)
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        value = int(shape[0])
        evidence.append(f"{key}={list(shape)}")
        return value, evidence

    candidates: list[tuple[str, int, list[int]]] = []
    for key, tensor in state_dict.items():
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        out_features = int(shape[0])
        if out_features in (100, 300):
            candidates.append((key, out_features, [int(item) for item in shape]))
    for key, out_features, shape in candidates:
        evidence.append(f"{key}={shape}")
    if candidates:
        return candidates[0][1], evidence
    return None, evidence


def candidate_result(path: Path, *, ok: bool, details: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path.as_posix(), "ok": ok}
    if details is not None:
        result["details"] = details
    if error is not None:
        result["error"] = error
    return result


def evaluate_candidates(
    *,
    label: str,
    candidates: list[Path],
    verifier,
) -> tuple[Path | None, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    selected: Path | None = None
    for path in candidates:
        try:
            details = verifier(path)
            results.append(candidate_result(path, ok=True, details=details))
            if selected is None:
                selected = path
        except Exception as exc:
            results.append(candidate_result(path, ok=False, error=str(exc)))
    if not candidates:
        results.append({"path": "<none>", "ok": False, "error": f"No candidates found for {label}."})
    return selected, results


def verify_skeleton_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise CheckError(f"Checkpoint does not expose a readable state dict: {path.as_posix()}")

    a_shape = list(getattr(state_dict.get("A"), "shape", []))
    bn_shape = list(getattr(state_dict.get("data_bn.weight"), "shape", []))
    if a_shape == [3, 27, 27] or bn_shape == [81]:
        raise CheckError("Checkpoint is selected_27, not selected_31.")
    if a_shape not in ([], [3, 31, 31]) and bn_shape not in ([], [93]):
        raise CheckError(
            "Could not confirm selected_31 from checkpoint tensors "
            f"(A={a_shape or '<missing>'}, data_bn.weight={bn_shape or '<missing>'})."
        )

    num_classes, evidence = infer_num_classes_from_state_dict(state_dict)
    if num_classes != NUM_CLASSES:
        raise CheckError(
            f"Checkpoint classifier output is not {NUM_CLASSES} classes "
            f"(detected={num_classes}, evidence={evidence or ['<none>']})."
        )

    payload_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    dataset_cfg = payload_cfg.get("dataset", {}) if isinstance(payload_cfg, dict) else {}
    model_cfg = payload_cfg.get("model", {}) if isinstance(payload_cfg, dict) else {}
    subset = str(dataset_cfg.get("subset", "")).strip()
    model_num_classes = model_cfg.get("num_classes")
    if subset and subset != SUBSET:
        raise CheckError(f"Checkpoint metadata points to subset={subset}, not {SUBSET}.")
    if model_num_classes is not None and int(model_num_classes) != NUM_CLASSES:
        raise CheckError(f"Checkpoint metadata points to num_classes={model_num_classes}, not {NUM_CLASSES}.")

    return {
        "selected_31_confirmed": True,
        "A_shape": a_shape or None,
        "data_bn_weight_shape": bn_shape or None,
        "num_classes": num_classes,
        "classifier_evidence": evidence,
        "subset": subset or None,
        "model_name": str(model_cfg.get("name", "")).strip() or None,
    }


def verify_regions_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise CheckError(f"Checkpoint does not expose a readable state dict: {path.as_posix()}")

    num_classes, evidence = infer_num_classes_from_state_dict(state_dict)
    if num_classes != NUM_CLASSES:
        raise CheckError(
            f"Checkpoint classifier output is not {NUM_CLASSES} classes "
            f"(detected={num_classes}, evidence={evidence or ['<none>']})."
        )

    payload_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    dataset_cfg = payload_cfg.get("dataset", {}) if isinstance(payload_cfg, dict) else {}
    model_cfg = payload_cfg.get("model", {}) if isinstance(payload_cfg, dict) else {}
    subset = str(dataset_cfg.get("subset", "")).strip()
    model_name = str(model_cfg.get("name", "")).strip().lower()
    active_regions = list(dataset_cfg.get("active_regions", dataset_cfg.get("region_order", [])))
    normalized_regions = [str(item).strip() for item in active_regions]

    if subset and subset != SUBSET:
        raise CheckError(f"Checkpoint metadata points to subset={subset}, not {SUBSET}.")
    if model_name and model_name != "region_resnet18_gru":
        raise CheckError(f"Checkpoint model is {model_name}, not region_resnet18_gru.")
    if normalized_regions and normalized_regions != ACTIVE_REGIONS:
        raise CheckError(
            f"Checkpoint active_regions mismatch: expected {ACTIVE_REGIONS}, got {normalized_regions}."
        )
    if "region_resnet18_gru" not in model_name:
        raise CheckError("Checkpoint is not RegionResNet18GRU-compatible.")

    return {
        "loadable": True,
        "num_classes": num_classes,
        "classifier_evidence": evidence,
        "subset": subset or None,
        "model_name": model_name or None,
        "active_regions": normalized_regions or None,
    }


def verify_skeleton_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    graph_cfg = dict(config.get("graph", {}))
    model_cfg = dict(config.get("model", {}))

    subset = str(dataset_cfg.get("subset", "")).strip()
    keypoint_set = str(dataset_cfg.get("keypoint_set", "")).strip()
    expected_shape = list(dataset_cfg.get("expected_shape", []))
    layout = str(graph_cfg.get("layout", "")).strip()
    num_nodes = int(model_cfg.get("num_nodes", -1))
    num_classes = int(model_cfg.get("num_classes", -1))

    if subset != SUBSET:
        raise CheckError(f"dataset.subset must be {SUBSET}, got {subset or '<missing>'}.")
    if keypoint_set != "selected_31":
        raise CheckError(f"dataset.keypoint_set must be selected_31, got {keypoint_set or '<missing>'}.")
    if expected_shape != SKELETON_EXPECTED_SHAPE:
        raise CheckError(
            f"dataset.expected_shape must be {SKELETON_EXPECTED_SHAPE}, got {expected_shape or '<missing>'}."
        )
    if layout != "selected_31":
        raise CheckError(f"graph.layout must be selected_31, got {layout or '<missing>'}.")
    if num_nodes != 31:
        raise CheckError(f"model.num_nodes must be 31, got {num_nodes}.")
    if num_classes != NUM_CLASSES:
        raise CheckError(f"model.num_classes must be {NUM_CLASSES}, got {num_classes}.")

    return {
        "subset": subset,
        "keypoint_set": keypoint_set,
        "expected_shape": expected_shape,
        "graph_layout": layout,
        "num_nodes": num_nodes,
        "num_classes": num_classes,
    }


def verify_regions_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    model_cfg = dict(config.get("model", {}))

    subset = str(dataset_cfg.get("subset", "")).strip()
    expected_shape = list(dataset_cfg.get("expected_shape", []))
    active_regions = list(dataset_cfg.get("active_regions", dataset_cfg.get("region_order", [])))
    normalized_regions = [str(item).strip() for item in active_regions]
    model_name = str(model_cfg.get("name", "")).strip().lower()
    num_classes = int(model_cfg.get("num_classes", -1))

    if subset != SUBSET:
        raise CheckError(f"dataset.subset must be {SUBSET}, got {subset or '<missing>'}.")
    if expected_shape != REGIONS_EXPECTED_SHAPE:
        raise CheckError(
            f"dataset.expected_shape must be {REGIONS_EXPECTED_SHAPE}, got {expected_shape or '<missing>'}."
        )
    if normalized_regions != ACTIVE_REGIONS:
        raise CheckError(
            f"dataset.active_regions/region_order must be {ACTIVE_REGIONS}, got {normalized_regions or '<missing>'}."
        )
    if model_name != "region_resnet18_gru":
        raise CheckError(f"model.name must be region_resnet18_gru, got {model_name or '<missing>'}.")
    if num_classes != NUM_CLASSES:
        raise CheckError(f"model.num_classes must be {NUM_CLASSES}, got {num_classes}.")

    return {
        "subset": subset,
        "expected_shape": expected_shape,
        "active_regions": normalized_regions,
        "model_name": model_name,
        "num_classes": num_classes,
    }


def verify_gated_fusion_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    experiment_cfg = dict(config.get("experiment", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
    regions_cfg = dict(dataset_cfg.get("regions", {}))
    skeleton_branch_cfg = dict(config.get("skeleton_branch", {}))
    regions_branch_cfg = dict(config.get("regions_branch", {}))
    skeleton_branch_model_cfg = dict(skeleton_branch_cfg.get("model", {}))
    skeleton_branch_graph_cfg = dict(skeleton_branch_cfg.get("graph", {}))
    regions_branch_model_cfg = dict(regions_branch_cfg.get("model", {}))
    logging_cfg = dict(config.get("logging", {}))

    subset = str(dataset_cfg.get("subset", "")).strip()
    num_classes = int(dataset_cfg.get("num_classes", -1))
    if subset != SUBSET:
        raise CheckError(f"dataset.subset must be {SUBSET}, got {subset or '<missing>'}.")
    if num_classes != NUM_CLASSES:
        raise CheckError(f"dataset.num_classes must be {NUM_CLASSES}, got {num_classes}.")
    if skeleton_cfg.get("keypoint_set") != "selected_31":
        raise CheckError("dataset.skeleton.keypoint_set must be selected_31.")
    if list(skeleton_cfg.get("expected_shape", [])) != SKELETON_EXPECTED_SHAPE:
        raise CheckError("dataset.skeleton.expected_shape is incorrect.")
    if list(regions_cfg.get("expected_shape", [])) != REGIONS_EXPECTED_SHAPE:
        raise CheckError("dataset.regions.expected_shape is incorrect.")
    region_order = list(regions_cfg.get("active_regions", regions_cfg.get("region_order", [])))
    normalized_regions = [str(item).strip() for item in region_order]
    if normalized_regions != ACTIVE_REGIONS:
        raise CheckError("dataset.regions.active_regions is incorrect.")
    if str(experiment_cfg.get("name", "")).strip() != "gated-fusion-nslt300-sel31-ce-regions":
        raise CheckError("experiment.name must be gated-fusion-nslt300-sel31-ce-regions.")
    if str(skeleton_branch_graph_cfg.get("layout", "")).strip() != "selected_31":
        raise CheckError("skeleton_branch.graph.layout must be selected_31.")
    if str(skeleton_branch_model_cfg.get("name", "")).strip().lower() != "stgcnpp":
        raise CheckError("skeleton_branch.model.name must be stgcnpp.")
    if int(skeleton_branch_model_cfg.get("num_nodes", -1)) != 31:
        raise CheckError("skeleton_branch.model.num_nodes must be 31.")
    if int(skeleton_branch_model_cfg.get("num_classes", -1)) != NUM_CLASSES:
        raise CheckError(
            f"skeleton_branch.model.num_classes must be {NUM_CLASSES}."
        )
    if str(regions_branch_model_cfg.get("name", "")).strip().lower() != "region_resnet18_gru":
        raise CheckError("regions_branch.model.name must be region_resnet18_gru.")
    if int(regions_branch_model_cfg.get("num_classes", -1)) != NUM_CLASSES:
        raise CheckError(
            f"regions_branch.model.num_classes must be {NUM_CLASSES}."
        )
    if int(regions_branch_model_cfg.get("num_regions", -1)) != len(ACTIVE_REGIONS):
        raise CheckError(
            f"regions_branch.model.num_regions must be {len(ACTIVE_REGIONS)}."
        )
    if str(logging_cfg.get("project", "")).strip() == "wlasl-gated-fusion-100":
        raise CheckError("logging.project still points to the NSLT100 template.")

    path_fields = {
        "skeleton_train_manifest": skeleton_cfg.get("manifests", {}).get("train"),
        "regions_train_manifest": regions_cfg.get("manifests", {}).get("train"),
        "skeleton_config_path": config.get("skeleton_branch", {}).get("config_path"),
        "regions_config_path": config.get("regions_branch", {}).get("config_path"),
        "skeleton_checkpoint_path": config.get("skeleton_branch", {}).get("checkpoint_path"),
        "regions_checkpoint_path": config.get("regions_branch", {}).get("checkpoint_path"),
    }
    for label, value in path_fields.items():
        text = str(value or "")
        if SUBSET not in text and "checkpoints/models" not in text:
            raise CheckError(f"{label} does not point to an NSLT300 path: {text or '<missing>'}.")

    return {
        "experiment_name": str(experiment_cfg.get("name", "")).strip() or None,
        "output_root": str(experiment_cfg.get("output_root", "")).strip() or None,
        "subset": subset,
        "num_classes": num_classes,
        "skeleton_branch_num_classes": int(skeleton_branch_model_cfg.get("num_classes", -1)),
        "regions_branch_num_classes": int(regions_branch_model_cfg.get("num_classes", -1)),
    }


def inspect_branch_split(
    *,
    rows: list[dict[str, str]],
    tensor_key: str,
    expected_shape: list[int],
    class_limit: int,
    required_keypoint_set: str | None = None,
) -> dict[str, Any]:
    if not rows:
        raise CheckError("Manifest has no data rows.")
    class_ids = [int(row["class_id"]) for row in rows]
    if min(class_ids) < 0 or max(class_ids) >= class_limit:
        raise CheckError(f"class_id range is invalid for {class_limit} classes: min={min(class_ids)} max={max(class_ids)}.")

    if required_keypoint_set is not None:
        keypoint_sets = sorted({str(row.get("keypoint_set", "")).strip() for row in rows})
        if keypoint_sets != [required_keypoint_set]:
            raise CheckError(f"keypoint_set mismatch: expected {required_keypoint_set}, got {keypoint_sets}.")
    else:
        keypoint_sets = []

    manifest_shape = parse_shape_literal(rows[0].get("tensor_shape", ""))
    if manifest_shape != expected_shape:
        raise CheckError(
            f"Manifest tensor_shape is not {expected_shape}: got {manifest_shape or '<missing>'}."
        )

    tensor_path_text = str(rows[0].get(tensor_key, "")).strip()
    if not tensor_path_text:
        raise CheckError(f"Manifest field {tensor_key} is empty for the first row.")
    tensor_path = Path(tensor_path_text)
    actual_shape = load_tensor_shape(tensor_path)
    if actual_shape != expected_shape:
        raise CheckError(
            f"Tensor shape mismatch for {tensor_path.as_posix()}: expected {expected_shape}, got {actual_shape}."
        )

    return {
        "count": len(rows),
        "class_min": min(class_ids),
        "class_max": max(class_ids),
        "distinct_classes": len(set(class_ids)),
        "manifest_tensor_shape": manifest_shape,
        "sample_tensor_path": tensor_path.as_posix(),
        "sample_tensor_shape": actual_shape,
        "keypoint_sets": keypoint_sets or None,
    }


def inspect_skeleton_branch_inputs(project_root: Path) -> dict[str, Any]:
    branch_root = (project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l").resolve()
    manifest_dir = branch_root / "manifests"
    manifest_paths = {
        split: manifest_dir / f"nslt300_selected_31_{split}.csv"
        for split in ("train", "val", "test")
    }
    canonical_tensor_root = branch_root / "tensors" / SUBSET
    fallback_tensor_root = branch_root / "graph_tensors" / "selected_31" / SUBSET

    warnings: list[str] = []
    errors: list[str] = []
    split_results: dict[str, Any] = {}
    all_ok = True

    if not branch_root.exists():
        errors.append(f"Missing skeleton branch root: {branch_root.as_posix()}")
        all_ok = False
    if not canonical_tensor_root.exists() and fallback_tensor_root.exists():
        warnings.append(
            "Canonical skeleton tensor root is missing; using graph_tensors/selected_31/nslt300 as the available source."
        )
    if not canonical_tensor_root.exists() and not fallback_tensor_root.exists():
        errors.append(
            "Missing skeleton tensor roots: "
            f"{canonical_tensor_root.as_posix()} and {fallback_tensor_root.as_posix()}"
        )
        all_ok = False

    for split, path in manifest_paths.items():
        if not path.exists():
            errors.append(f"Missing skeleton manifest: {path.as_posix()}")
            all_ok = False
            continue
        try:
            _fieldnames, rows = read_manifest(path)
            split_results[split] = inspect_branch_split(
                rows=rows,
                tensor_key="graph_tensor_path",
                expected_shape=SKELETON_EXPECTED_SHAPE,
                class_limit=NUM_CLASSES,
                required_keypoint_set="selected_31",
            )
        except Exception as exc:
            split_results[split] = {"error": str(exc)}
            errors.append(f"{split}: {exc}")
            all_ok = False

    return {
        "ok": all_ok,
        "branch_root": branch_root.as_posix(),
        "canonical_tensor_root": canonical_tensor_root.as_posix(),
        "fallback_tensor_root": fallback_tensor_root.as_posix(),
        "source_root": (
            canonical_tensor_root.as_posix()
            if canonical_tensor_root.exists()
            else fallback_tensor_root.as_posix()
            if fallback_tensor_root.exists()
            else None
        ),
        "manifest_paths": {split: path.as_posix() for split, path in manifest_paths.items()},
        "splits": split_results,
        "warnings": warnings,
        "errors": errors,
    }


def inspect_regions_branch_inputs(project_root: Path) -> dict[str, Any]:
    branch_root = (project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l").resolve()
    manifest_dir = branch_root / "manifests"
    manifest_paths = {split: manifest_dir / f"nslt300_{split}.csv" for split in ("train", "val", "test")}
    tensor_root = branch_root / "tensors" / SUBSET

    errors: list[str] = []
    split_results: dict[str, Any] = {}
    all_ok = True

    if not branch_root.exists():
        errors.append(f"Missing regions branch root: {branch_root.as_posix()}")
        all_ok = False
    if not tensor_root.exists():
        errors.append(f"Missing regions tensor root: {tensor_root.as_posix()}")
        all_ok = False

    for split, path in manifest_paths.items():
        if not path.exists():
            errors.append(f"Missing regions manifest: {path.as_posix()}")
            all_ok = False
            continue
        try:
            _fieldnames, rows = read_manifest(path)
            split_results[split] = inspect_branch_split(
                rows=rows,
                tensor_key="tensor_path",
                expected_shape=REGIONS_EXPECTED_SHAPE,
                class_limit=NUM_CLASSES,
            )
        except Exception as exc:
            split_results[split] = {"error": str(exc)}
            errors.append(f"{split}: {exc}")
            all_ok = False

    return {
        "ok": all_ok,
        "branch_root": branch_root.as_posix(),
        "tensor_root": tensor_root.as_posix(),
        "source_root": tensor_root.as_posix() if tensor_root.exists() else None,
        "manifest_paths": {split: path.as_posix() for split, path in manifest_paths.items()},
        "splits": split_results,
        "warnings": [],
        "errors": errors,
    }


def inspect_pairing(
    project_root: Path,
    *,
    pairing_min_coverage: float,
) -> dict[str, Any]:
    skeleton_manifest_dir = project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests"
    regions_manifest_dir = project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests"
    results: dict[str, Any] = {}
    errors: list[str] = []
    all_ok = True

    for split in ("train", "val", "test"):
        skeleton_path = skeleton_manifest_dir / f"nslt300_selected_31_{split}.csv"
        regions_path = regions_manifest_dir / f"nslt300_{split}.csv"
        if not skeleton_path.exists() or not regions_path.exists():
            results[split] = {
                "error": "Missing manifest(s) required for pairing.",
                "skeleton_manifest": skeleton_path.as_posix(),
                "regions_manifest": regions_path.as_posix(),
            }
            errors.append(f"{split}: missing manifests for pairing.")
            all_ok = False
            continue

        _s_fields, skeleton_rows = read_manifest(skeleton_path)
        _r_fields, regions_rows = read_manifest(regions_path)
        skeleton_map = {normalize_sample_id(row["sample_id"]): row for row in skeleton_rows}
        regions_map = {normalize_sample_id(row["sample_id"]): row for row in regions_rows}
        skeleton_ids = set(skeleton_map)
        regions_ids = set(regions_map)
        matched_ids = sorted(skeleton_ids & regions_ids)
        label_mismatch = sum(
            1
            for sample_id in matched_ids
            if str(skeleton_map[sample_id]["class_id"]).strip() != str(regions_map[sample_id]["class_id"]).strip()
        )
        gloss_mismatch = 0
        if matched_ids and all("gloss" in row for row in skeleton_rows) and all("gloss" in row for row in regions_rows):
            gloss_mismatch = sum(
                1
                for sample_id in matched_ids
                if str(skeleton_map[sample_id].get("gloss", "")).strip()
                != str(regions_map[sample_id].get("gloss", "")).strip()
            )
        denominator = min(len(skeleton_ids), len(regions_ids))
        coverage = (len(matched_ids) / denominator) if denominator else 0.0
        split_ok = label_mismatch == 0 and gloss_mismatch == 0 and coverage >= pairing_min_coverage
        if not split_ok:
            all_ok = False
            if label_mismatch != 0:
                errors.append(f"{split}: label_mismatch={label_mismatch}.")
            if gloss_mismatch != 0:
                errors.append(f"{split}: gloss_mismatch={gloss_mismatch}.")
            if coverage < pairing_min_coverage:
                errors.append(
                    f"{split}: matched coverage {coverage:.3f} is below the required {pairing_min_coverage:.3f}."
                )
        results[split] = {
            "skeleton_count": len(skeleton_ids),
            "regions_count": len(regions_ids),
            "matched_count": len(matched_ids),
            "missing_in_skeleton": len(regions_ids - skeleton_ids),
            "missing_in_regions": len(skeleton_ids - regions_ids),
            "label_mismatch": label_mismatch,
            "gloss_mismatch": gloss_mismatch,
            "coverage": round(coverage, 6),
            "ok": split_ok,
        }

    return {"ok": all_ok, "splits": results, "errors": errors}


def create_gated_fusion_config(
    *,
    project_root: Path,
    skeleton_checkpoint_path: Path,
    regions_checkpoint_path: Path,
    skeleton_config_path: Path,
    regions_config_path: Path,
) -> Path:
    base_path = (project_root / GATED_CONFIG_TEMPLATE).resolve()
    if not base_path.exists():
        raise CheckError(f"Missing base gated fusion config template: {base_path.as_posix()}")
    config = read_yaml(base_path)

    config.setdefault("experiment", {})
    config["experiment"]["name"] = "gated-fusion-nslt300-sel31-ce-regions"
    config["experiment"]["output_root"] = "outputs/fusion"

    config["dataset"] = {
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "skeleton": {
            "data_root": "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l",
            "keypoint_set": "selected_31",
            "expected_shape": list(SKELETON_EXPECTED_SHAPE),
            "manifests": {
                "train": "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_train.csv",
                "val": "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_val.csv",
                "test": "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_test.csv",
            },
            "return_metadata": True,
            "strict_shape_check": True,
        },
        "regions": {
            "data_root": "data/datasets/WLASL/branch_inputs/regions/rtmw_l",
            "expected_shape": list(REGIONS_EXPECTED_SHAPE),
            "region_order": list(ACTIVE_REGIONS),
            "active_regions": list(ACTIVE_REGIONS),
            "manifests": {
                "train": "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_train.csv",
                "val": "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_val.csv",
                "test": "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_test.csv",
            },
            "normalize": {"type": "imagenet"},
            "return_metadata": True,
            "strict_shape_check": True,
        },
    }

    config.setdefault("skeleton_branch", {})
    config["skeleton_branch"]["config_path"] = str(skeleton_config_path.relative_to(project_root).as_posix())
    config["skeleton_branch"]["checkpoint_path"] = str(skeleton_checkpoint_path.relative_to(project_root).as_posix())
    config["skeleton_branch"]["graph"] = {
        "layout": "selected_31",
        "strategy": "spatial",
        "add_self_links": True,
        "normalize_adjacency": True,
    }
    config["skeleton_branch"]["model"] = {
        "name": "stgcnpp",
        "in_channels": 3,
        "num_nodes": 31,
        "num_classes": NUM_CLASSES,
        "base_channels": 64,
        "dropout": 0.5,
    }
    config.setdefault("regions_branch", {})
    config["regions_branch"]["config_path"] = str(regions_config_path.relative_to(project_root).as_posix())
    config["regions_branch"]["checkpoint_path"] = str(regions_checkpoint_path.relative_to(project_root).as_posix())
    config["regions_branch"]["model"] = {
        "name": "region_resnet18_gru",
        "num_classes": NUM_CLASSES,
        "num_regions": len(ACTIVE_REGIONS),
        "in_channels": 3,
        "clip_len": 64,
        "crop_size": 112,
        "pretrained": False,
        "freeze_encoder": True,
        "encoder_name": "resnet18",
        "encoder_feature_dim": 512,
        "gru_hidden_size": 128,
        "gru_num_layers": 1,
        "bidirectional": True,
        "dropout": 0.5,
        "fusion": "concat",
        "use_valid_mask": True,
    }

    config.setdefault("fusion_model", {})
    config["fusion_model"]["name"] = "gated_feature_fusion"
    config["fusion_model"]["hidden_dim"] = 256
    config["fusion_model"]["proj_dropout"] = 0.2
    config["fusion_model"]["classifier_dropout"] = 0.5
    config["fusion_model"]["freeze_skeleton"] = True
    config["fusion_model"]["freeze_regions"] = True

    config.setdefault("train", {})
    config["train"]["epochs"] = 50
    config["train"]["device"] = "auto"
    config["train"]["optimizer"] = "adamw"
    config["train"]["learning_rate"] = 0.0005
    config["train"]["weight_decay"] = 0.001
    config["train"]["loss"] = "cross_entropy"
    config["train"]["grad_clip_norm"] = 1.0
    config["train"]["amp"] = False

    config.setdefault("early_stopping", {})
    config["early_stopping"]["enabled"] = True
    config["early_stopping"]["monitor_metric"] = "val/top5"
    config["early_stopping"]["monitor_mode"] = "max"
    config["early_stopping"]["patience"] = 8
    config["early_stopping"]["min_delta"] = 0.0

    config.setdefault("logging", {})
    config["logging"]["use_wandb"] = True
    config["logging"]["project"] = "wlasl-nslt300-gated-fusion"
    config["logging"]["entity_env"] = "WANDB_ENTITY"
    config["logging"]["run_name"] = "gated-fusion-nslt300-sel31-ce-regions"
    config["logging"]["tags"] = [
        "nslt300",
        "gated-fusion",
        "skeleton-selected-31",
        "regions-all",
        "left-hand",
        "right-hand",
        "face",
    ]
    config["logging"]["log_model"] = True

    output_path = (project_root / GATED_CONFIG_PATH).resolve()
    write_yaml(output_path, config)
    return output_path


def maybe_prepare_gated_fusion_config(
    *,
    project_root: Path,
    prerequisites_ready: bool,
    skeleton_checkpoint_path: Path | None,
    regions_checkpoint_path: Path | None,
    skeleton_config_path: Path | None,
    regions_config_path: Path | None,
) -> dict[str, Any]:
    output_path = (project_root / GATED_CONFIG_PATH).resolve()
    created = False
    if output_path.exists():
        try:
            details = verify_gated_fusion_config(output_path)
            return {"ok": True, "path": output_path.as_posix(), "created": False, "details": details}
        except Exception as exc:
            if not prerequisites_ready or None in (
                skeleton_checkpoint_path,
                regions_checkpoint_path,
                skeleton_config_path,
                regions_config_path,
            ):
                return {
                    "ok": False,
                    "path": output_path.as_posix(),
                    "created": False,
                    "error": str(exc),
                }

    if prerequisites_ready and None not in (
        skeleton_checkpoint_path,
        regions_checkpoint_path,
        skeleton_config_path,
        regions_config_path,
    ):
        created_path = create_gated_fusion_config(
            project_root=project_root,
            skeleton_checkpoint_path=skeleton_checkpoint_path,
            regions_checkpoint_path=regions_checkpoint_path,
            skeleton_config_path=skeleton_config_path,
            regions_config_path=regions_config_path,
        )
        created = True
        details = verify_gated_fusion_config(created_path)
        return {"ok": True, "path": created_path.as_posix(), "created": created, "details": details}

    return {
        "ok": False,
        "path": output_path.as_posix(),
        "created": created,
        "error": "Missing valid NSLT300 checkpoints/configs, so the gated fusion NSLT300 config was not created.",
    }


def build_summary(project_root: Path, *, pairing_min_coverage: float) -> dict[str, Any]:
    skeleton_inputs = inspect_skeleton_branch_inputs(project_root)
    regions_inputs = inspect_regions_branch_inputs(project_root)
    pairing = inspect_pairing(project_root, pairing_min_coverage=pairing_min_coverage)

    skeleton_checkpoint_candidates = list_existing_candidates(
        project_root,
        [
            "checkpoints/models/skeleton_nslt300/best.pt",
            "checkpoints/models/skeleton/best_nslt300.pt",
            "artifacts/fusion/nslt300/checkpoints/skeleton/best.pt",
            "outputs/skeleton/**/best.pt",
            "**/*skeleton*/best.pt",
            "**/*skeleton*/*.pt",
        ],
    )
    regions_checkpoint_candidates = list_existing_candidates(
        project_root,
        [
            "checkpoints/models/regions_nslt300/best.pt",
            "checkpoints/models/regions/best_nslt300.pt",
            "artifacts/fusion/nslt300/checkpoints/regions/best.pt",
            "outputs/regions/**/best.pt",
            "**/*region*/best.pt",
            "**/*region*/*.pt",
        ],
    )
    skeleton_config_candidates = list_existing_candidates(
        project_root,
        [
            "artifacts/fusion/nslt300/configs/skeleton/config_resolved.yaml",
            "artifacts/fusion/nslt300/configs/skeleton_config_resolved.yaml",
            "checkpoints/models/skeleton_nslt300/config_resolved.yaml",
            "outputs/skeleton/**/config_resolved.yaml",
        ],
    )
    regions_config_candidates = list_existing_candidates(
        project_root,
        [
            "artifacts/fusion/nslt300/configs/regions/config_resolved.yaml",
            "artifacts/fusion/nslt300/configs/regions_config_resolved.yaml",
            "checkpoints/models/regions_nslt300/config_resolved.yaml",
            "outputs/regions/**/config_resolved.yaml",
        ],
    )

    selected_skeleton_checkpoint, skeleton_checkpoint_results = evaluate_candidates(
        label="skeleton checkpoint",
        candidates=skeleton_checkpoint_candidates,
        verifier=verify_skeleton_checkpoint,
    )
    selected_regions_checkpoint, regions_checkpoint_results = evaluate_candidates(
        label="regions checkpoint",
        candidates=regions_checkpoint_candidates,
        verifier=verify_regions_checkpoint,
    )
    selected_skeleton_config, skeleton_config_results = evaluate_candidates(
        label="skeleton config",
        candidates=skeleton_config_candidates,
        verifier=verify_skeleton_config,
    )
    selected_regions_config, regions_config_results = evaluate_candidates(
        label="regions config",
        candidates=regions_config_candidates,
        verifier=verify_regions_config,
    )

    prerequisites_ready = all(
        [
            skeleton_inputs["ok"],
            regions_inputs["ok"],
            pairing["ok"],
            selected_skeleton_checkpoint is not None,
            selected_regions_checkpoint is not None,
            selected_skeleton_config is not None,
            selected_regions_config is not None,
        ]
    )
    gated_fusion_config = maybe_prepare_gated_fusion_config(
        project_root=project_root,
        prerequisites_ready=prerequisites_ready,
        skeleton_checkpoint_path=selected_skeleton_checkpoint,
        regions_checkpoint_path=selected_regions_checkpoint,
        skeleton_config_path=selected_skeleton_config,
        regions_config_path=selected_regions_config,
    )

    missing_items: list[str] = []
    if not skeleton_inputs["ok"]:
        if not (project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_train.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_train.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_val.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_val.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_test.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_test.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors/nslt300").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors/nslt300/")
    if not regions_inputs["ok"]:
        if not (project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_train.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_train.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_val.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_val.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_test.csv").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests/nslt300_test.csv")
        if not (project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/tensors/nslt300").exists():
            missing_items.append("data/datasets/WLASL/branch_inputs/regions/rtmw_l/tensors/nslt300/")
    if selected_skeleton_checkpoint is None:
        missing_items.append("skeleton checkpoint NSLT300 best.pt with selected_31 and 300-class output")
    if selected_regions_checkpoint is None:
        missing_items.append("regions checkpoint NSLT300 best.pt for RegionResNet18GRU with 300-class output")
    if selected_skeleton_config is None:
        missing_items.append("skeleton config_resolved.yaml NSLT300")
    if selected_regions_config is None:
        missing_items.append("regions config_resolved.yaml NSLT300 for RegionResNet18GRU")
    if not gated_fusion_config["ok"]:
        missing_items.append("configs/train/fusion/gated_feature/nslt300/gated_feature_fusion_ce.yaml")
    if not pairing["ok"]:
        missing_items.append("paired skeleton/regions manifests with sufficient sample_id coverage for train/val/test")

    ready = all(
        [
            skeleton_inputs["ok"],
            regions_inputs["ok"],
            selected_skeleton_checkpoint is not None,
            selected_regions_checkpoint is not None,
            selected_skeleton_config is not None,
            selected_regions_config is not None,
            gated_fusion_config["ok"],
            pairing["ok"],
        ]
    )

    return {
        "project_root": project_root.as_posix(),
        "ready": ready,
        "skeleton_inputs": skeleton_inputs,
        "regions_inputs": regions_inputs,
        "skeleton_checkpoint_candidates": [path.as_posix() for path in skeleton_checkpoint_candidates],
        "regions_checkpoint_candidates": [path.as_posix() for path in regions_checkpoint_candidates],
        "skeleton_checkpoint_results": skeleton_checkpoint_results,
        "regions_checkpoint_results": regions_checkpoint_results,
        "selected_skeleton_checkpoint": selected_skeleton_checkpoint.as_posix() if selected_skeleton_checkpoint else None,
        "selected_regions_checkpoint": selected_regions_checkpoint.as_posix() if selected_regions_checkpoint else None,
        "skeleton_config_candidates": [path.as_posix() for path in skeleton_config_candidates],
        "regions_config_candidates": [path.as_posix() for path in regions_config_candidates],
        "skeleton_config_results": skeleton_config_results,
        "regions_config_results": regions_config_results,
        "selected_skeleton_config": selected_skeleton_config.as_posix() if selected_skeleton_config else None,
        "selected_regions_config": selected_regions_config.as_posix() if selected_regions_config else None,
        "gated_fusion_config": gated_fusion_config,
        "pairing": pairing,
        "missing_items": missing_items,
    }


def format_candidate_section(title: str, candidates: list[str], results: list[dict[str, Any]]) -> list[str]:
    lines = [title]
    lines.append("Candidates:")
    if candidates:
        lines.extend(f"- {path}" for path in candidates)
    else:
        lines.append("- <none>")
    lines.append("Verification results:")
    for item in results:
        status = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"- [{status}] {item['path']}")
        if item.get("details"):
            lines.append(f"  details: {json.dumps(item['details'], ensure_ascii=False)}")
        if item.get("error"):
            lines.append(f"  error: {item['error']}")
    return lines


def create_report(summary: dict[str, Any], *, pairing_min_coverage: float) -> str:
    skeleton_inputs = summary["skeleton_inputs"]
    regions_inputs = summary["regions_inputs"]
    pairing = summary["pairing"]
    gated_fusion = summary["gated_fusion_config"]

    sections: list[str] = [
        "# Gated Fusion NSLT300 Requirement Check Report",
        "",
        "## 1. Muc tieu",
        "Check whether the repo is complete enough to package `wlasl-nslt300-gated-fusion-ready` for Kaggle.",
        "",
        "## 2. Ket luan READY hay NOT READY",
        f"Conclusion: {'READY' if summary['ready'] else 'NOT READY'}",
        "",
        "## 3. Skeleton branch inputs",
        f"Status: {'PASS' if skeleton_inputs['ok'] else 'FAIL'}",
        f"Source root: {skeleton_inputs['source_root'] or '<missing>'}",
        f"Canonical tensor root: {skeleton_inputs['canonical_tensor_root']}",
        f"Fallback tensor root: {skeleton_inputs['fallback_tensor_root']}",
        "Manifest files:",
        *[f"- {value}" for value in skeleton_inputs["manifest_paths"].values()],
        "Split verification:",
    ]
    for split, details in skeleton_inputs["splits"].items():
        sections.append(f"- {split}: {json.dumps(details, ensure_ascii=False)}")
    if skeleton_inputs["warnings"]:
        sections.append("Warnings:")
        sections.extend(f"- {item}" for item in skeleton_inputs["warnings"])
    if skeleton_inputs["errors"]:
        sections.append("Errors:")
        sections.extend(f"- {item}" for item in skeleton_inputs["errors"])

    sections.extend(
        [
            "",
            "## 4. Regions branch inputs",
            f"Status: {'PASS' if regions_inputs['ok'] else 'FAIL'}",
            f"Source root: {regions_inputs['source_root'] or '<missing>'}",
            f"Tensor root: {regions_inputs['tensor_root']}",
            "Manifest files:",
            *[f"- {value}" for value in regions_inputs["manifest_paths"].values()],
            "Split verification:",
        ]
    )
    for split, details in regions_inputs["splits"].items():
        sections.append(f"- {split}: {json.dumps(details, ensure_ascii=False)}")
    if regions_inputs["errors"]:
        sections.append("Errors:")
        sections.extend(f"- {item}" for item in regions_inputs["errors"])

    sections.extend(
        [
            "",
            "## 5. Skeleton checkpoint",
            f"Selected candidate: {summary['selected_skeleton_checkpoint'] or '<none>'}",
            *format_candidate_section(
                "Checkpoint audit:",
                summary["skeleton_checkpoint_candidates"],
                summary["skeleton_checkpoint_results"],
            ),
            "",
            "## 6. Regions checkpoint",
            f"Selected candidate: {summary['selected_regions_checkpoint'] or '<none>'}",
            *format_candidate_section(
                "Checkpoint audit:",
                summary["regions_checkpoint_candidates"],
                summary["regions_checkpoint_results"],
            ),
            "",
            "## 7. Skeleton config",
            f"Selected candidate: {summary['selected_skeleton_config'] or '<none>'}",
            *format_candidate_section(
                "Config audit:",
                summary["skeleton_config_candidates"],
                summary["skeleton_config_results"],
            ),
            "",
            "## 8. Regions config",
            f"Selected candidate: {summary['selected_regions_config'] or '<none>'}",
            *format_candidate_section(
                "Config audit:",
                summary["regions_config_candidates"],
                summary["regions_config_results"],
            ),
            "",
            "## 9. Gated Fusion config",
            f"Status: {'PASS' if gated_fusion['ok'] else 'FAIL'}",
            f"Path: {gated_fusion['path']}",
            f"Created by this check: {'yes' if gated_fusion.get('created') else 'no'}",
        ]
    )
    if gated_fusion.get("details"):
        sections.append(f"Details: {json.dumps(gated_fusion['details'], ensure_ascii=False)}")
    if gated_fusion.get("error"):
        sections.append(f"Error: {gated_fusion['error']}")

    sections.extend(
        [
            "",
            "## 10. Pairing check",
            f"Required minimum matched coverage per split: {pairing_min_coverage:.3f}",
            f"Status: {'PASS' if pairing['ok'] else 'FAIL'}",
        ]
    )
    for split, details in pairing["splits"].items():
        sections.append(f"- {split}: {json.dumps(details, ensure_ascii=False)}")
    if pairing["errors"]:
        sections.append("Errors:")
        sections.extend(f"- {item}" for item in pairing["errors"])

    sections.extend(
        [
            "",
            "## 11. File con thieu",
        ]
    )
    if summary["missing_items"]:
        sections.extend(f"- {item}" for item in summary["missing_items"])
    else:
        sections.append("- None")

    next_steps = [
        "Train or recover a valid NSLT300 skeleton checkpoint with selected_31 and 300 output classes.",
        "Train or recover a valid NSLT300 RegionResNet18GRU checkpoint with all regions and 300 output classes.",
        "Export resolved configs for both NSLT300 branch checkpoints.",
    ]
    if not pairing["ok"]:
        next_steps.append(
            "Fix skeleton/regions manifest pairing so sample_id coverage is high enough for all train/val/test splits."
        )
    if not (Path(summary["project_root"]) / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors/nslt300").exists():
        next_steps.append(
            "Optional but recommended: materialize the canonical skeleton tensor root at "
            "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors/nslt300/."
        )
    if summary["ready"]:
        next_steps = [
            "Repo is ready for packaging. The next step is running the dedicated NSLT300 packaging script.",
        ]

    sections.extend(
        [
            "",
            "## 12. Viec can lam tiep",
            *[f"- {item}" for item in next_steps],
            "",
        ]
    )
    return "\n".join(sections)


def print_summary(summary: dict[str, Any], *, report_path: Path) -> None:
    print(f"Skeleton branch inputs source: {summary['skeleton_inputs']['source_root'] or '<missing>'}")
    print(f"Regions branch inputs source: {summary['regions_inputs']['source_root'] or '<missing>'}")
    print("Skeleton checkpoint candidates:")
    for path in summary["skeleton_checkpoint_candidates"]:
        print(f"- {path}")
    print("Regions checkpoint candidates:")
    for path in summary["regions_checkpoint_candidates"]:
        print(f"- {path}")
    print("Skeleton config candidates:")
    for path in summary["skeleton_config_candidates"]:
        print(f"- {path}")
    print("Regions config candidates:")
    for path in summary["regions_config_candidates"]:
        print(f"- {path}")
    print(f"Gated Fusion config candidate: {summary['gated_fusion_config']['path']}")
    print(f"Conclusion: {'READY' if summary['ready'] else 'NOT READY'}")
    if summary["missing_items"]:
        print("Missing:")
        for item in summary["missing_items"]:
            print(f"- {item}")
    print(f"Report written to: {report_path.as_posix()}")


def main() -> int:
    args = build_parser().parse_args()
    project_root = resolve_path(args.project_root, root=Path.cwd())
    report_path = resolve_path(args.report_path, root=project_root)
    summary = build_summary(project_root, pairing_min_coverage=float(args.pairing_min_coverage))
    report_text = create_report(summary, pairing_min_coverage=float(args.pairing_min_coverage))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print_summary(summary, report_path=report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
