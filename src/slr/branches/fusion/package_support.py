"""Shared helpers for NSLT1000 gated-fusion packaging code."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slr.branches.regions.dataset import resolve_region_tensor_path
from slr.branches.skeleton.dataset import resolve_graph_tensor_path
from slr.utils.io import read_yaml


LOGGER = logging.getLogger(__name__)

SUBSET = "nslt1000"
NUM_CLASSES = 1000
PACKAGE_NAME_DEFAULT = "wlasl-nslt1000-gated-fusion-ready"
SKELETON_EXPECTED_SHAPE = (3, 150, 31, 1)
REGIONS_EXPECTED_SHAPE = (3, 3, 64, 112, 112)
ACTIVE_REGIONS = ("left_hand", "right_hand", "face")
SPLIT_COUNTS = {"train": 5001, "val": 1290, "test": 941}
TOTAL_SAMPLES = sum(SPLIT_COUNTS.values())
SAMPLE_ID_CANONICAL_WIDTH = 5
SAMPLE_SPLITS = ("train", "val", "test")
INTEGER_LIKE_FLOAT_RE = re.compile(r"^(?P<int>\d+)\.0+$")
READY_STATUS = "READY"
PARTIALLY_READY_STATUS = "PARTIALLY_READY"
NOT_READY_STATUS = "NOT_READY"


@dataclass(frozen=True)
class ManifestSet:
    """One split-indexed manifest bundle."""

    branch_name: str
    manifest_root: Path
    manifest_paths: dict[str, Path]
    fieldnames_by_split: dict[str, list[str]]
    rows_by_split: dict[str, list[dict[str, str]]]


@dataclass(frozen=True)
class AlignmentItem:
    """One paired skeleton/regions sample."""

    split: str
    logical_sample_id: str
    canonical_sample_id: str
    skeleton_row: dict[str, str]
    regions_row: dict[str, str]


def safe_str(value: Any, default: str = "") -> str:
    """Convert nullable values to stable stripped strings."""

    if value is None:
        return default
    text = str(value).strip()
    return text or default


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_bytes(size_bytes: int | float) -> str:
    """Render one byte count as human-readable text."""

    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def normalize_sample_id(value: object) -> str:
    """Normalize one sample ID for logical matching.

    Numeric aliases such as ``00593``, ``593`` and ``593.0`` collapse to the
    same logical identifier. Non-numeric IDs are preserved.
    """

    text = safe_str(value)
    if not text:
        return ""
    if text.isdigit():
        return str(int(text))
    match = INTEGER_LIKE_FLOAT_RE.fullmatch(text)
    if match is not None:
        return str(int(match.group("int")))
    return text


def canonical_sample_id(value: object, *, width: int = SAMPLE_ID_CANONICAL_WIDTH) -> str:
    """Return the canonical manifest/package sample ID."""

    logical = normalize_sample_id(value)
    if not logical:
        return ""
    if logical.isdigit():
        return logical.zfill(max(int(width), len(logical)))
    return logical


def ensure_exists(path: Path, label: str) -> None:
    """Raise a readable error when one required path is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path.as_posix()}")


def ensure_within_root(path: Path, *, root: Path) -> None:
    """Ensure one destination path stays inside the intended root."""

    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to operate outside root {resolved_root.as_posix()}: {resolved_path.as_posix()}"
        ) from exc


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one CSV manifest as fieldnames + rows."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path.as_posix()}")
        rows = [{key: safe_str(value) for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write one CSV manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist one JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_shape(value: Any) -> list[int] | None:
    """Parse one shape-like literal from config or manifest text."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
    elif isinstance(value, np.ndarray):
        payload = value.tolist()
    else:
        payload = value
    if isinstance(payload, (list, tuple)):
        try:
            return [int(item) for item in payload]
        except (TypeError, ValueError):
            return None
    return None


def load_tensor_shape(path: Path) -> list[int]:
    """Load the main array shape from one tensor file."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "data" in payload:
                array = payload["data"]
            elif "tensor" in payload:
                array = payload["tensor"]
            else:
                key = payload.files[0] if payload.files else None
                if key is None:
                    raise ValueError(f"No arrays found inside {path.as_posix()}")
                array = payload[key]
        return [int(item) for item in np.asarray(array).shape]
    if suffix == ".npy":
        return [int(item) for item in np.load(path, allow_pickle=False).shape]
    raise ValueError(f"Unsupported tensor file suffix: {path.as_posix()}")


def get_state_dict(payload: Any) -> dict[str, Any] | None:
    """Extract a state dict from one torch checkpoint payload."""

    if not isinstance(payload, dict):
        return None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    return None


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> tuple[int | None, list[str]]:
    """Infer classifier output width from common weight names."""

    preferred_keys = (
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "classifier.2.weight",
        "head.1.weight",
    )
    evidence: list[str] = []
    for key in preferred_keys:
        tensor = state_dict.get(key)
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        evidence.append(f"{key}={list(shape)}")
        return int(shape[0]), evidence
    return None, evidence


def inspect_skeleton_checkpoint(path: Path) -> dict[str, Any]:
    """Validate the expected NSLT1000 skeleton checkpoint properties."""

    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Skeleton checkpoint has no readable state dict: {path.as_posix()}")

    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")
    adjacency = state_dict.get("A")
    bn_weight = state_dict.get("data_bn.weight")
    num_classes, evidence = infer_num_classes_from_state_dict(state_dict)

    config_payload = payload.get("config", {}) if isinstance(payload, dict) else {}
    dataset_cfg = dict(config_payload.get("dataset", {})) if isinstance(config_payload, dict) else {}
    model_cfg = dict(config_payload.get("model", {})) if isinstance(config_payload, dict) else {}

    subset = safe_str(dataset_cfg.get("subset"))
    model_name = safe_str(model_cfg.get("name")).lower()
    classifier_shape = list(getattr(classifier_weight, "shape", []))
    bias_shape = list(getattr(classifier_bias, "shape", []))
    adjacency_shape = list(getattr(adjacency, "shape", []))
    bn_shape = list(getattr(bn_weight, "shape", []))
    feature_dim = int(classifier_shape[1]) if len(classifier_shape) == 2 else None

    if subset and subset != SUBSET:
        raise ValueError(f"Skeleton checkpoint subset must be {SUBSET}, got {subset!r}")
    if model_name and model_name != "stgcnpp":
        raise ValueError(f"Skeleton checkpoint model must be stgcnpp, got {model_name!r}")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Skeleton checkpoint num_classes must be {NUM_CLASSES}, got {num_classes!r}")
    if classifier_shape != [NUM_CLASSES, 256]:
        raise ValueError(
            "Skeleton checkpoint classifier.weight shape must be "
            f"[{NUM_CLASSES}, 256], got {classifier_shape or '<missing>'}"
        )
    if bias_shape != [NUM_CLASSES]:
        raise ValueError(
            f"Skeleton checkpoint classifier.bias shape must be [{NUM_CLASSES}], got {bias_shape or '<missing>'}"
        )
    if adjacency_shape != [3, 31, 31]:
        raise ValueError(
            f"Skeleton checkpoint A shape must be [3, 31, 31], got {adjacency_shape or '<missing>'}"
        )
    if bn_shape != [93]:
        raise ValueError(
            f"Skeleton checkpoint data_bn.weight shape must be [93], got {bn_shape or '<missing>'}"
        )

    return {
        "subset": subset or SUBSET,
        "model_name": model_name or "stgcnpp",
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "classifier_weight_shape": classifier_shape,
        "classifier_bias_shape": bias_shape,
        "adjacency_shape": adjacency_shape,
        "data_bn_weight_shape": bn_shape,
        "classifier_evidence": evidence,
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "best_metric": payload.get("best_metric") if isinstance(payload, dict) else None,
    }


def inspect_regions_checkpoint(path: Path) -> dict[str, Any]:
    """Validate the expected NSLT1000 regions checkpoint properties."""

    payload = torch.load(path, map_location="cpu")
    state_dict = get_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Regions checkpoint has no readable state dict: {path.as_posix()}")

    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")
    num_classes, evidence = infer_num_classes_from_state_dict(state_dict)
    config_payload = payload.get("config", {}) if isinstance(payload, dict) else {}
    dataset_cfg = dict(config_payload.get("dataset", {})) if isinstance(config_payload, dict) else {}
    model_cfg = dict(config_payload.get("model", {})) if isinstance(config_payload, dict) else {}

    subset = safe_str(dataset_cfg.get("subset"))
    model_name = safe_str(model_cfg.get("name")).lower()
    active_regions = [
        safe_str(item)
        for item in dataset_cfg.get("active_regions", dataset_cfg.get("region_order", []))
    ]
    classifier_shape = list(getattr(classifier_weight, "shape", []))
    bias_shape = list(getattr(classifier_bias, "shape", []))
    feature_dim = int(classifier_shape[1]) if len(classifier_shape) == 2 else None

    if subset and subset != SUBSET:
        raise ValueError(f"Regions checkpoint subset must be {SUBSET}, got {subset!r}")
    if model_name and model_name != "region_resnet18_gru":
        raise ValueError(f"Regions checkpoint model must be region_resnet18_gru, got {model_name!r}")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Regions checkpoint num_classes must be {NUM_CLASSES}, got {num_classes!r}")
    if classifier_shape != [NUM_CLASSES, 768]:
        raise ValueError(
            "Regions checkpoint classifier.weight shape must be "
            f"[{NUM_CLASSES}, 768], got {classifier_shape or '<missing>'}"
        )
    if bias_shape != [NUM_CLASSES]:
        raise ValueError(
            f"Regions checkpoint classifier.bias shape must be [{NUM_CLASSES}], got {bias_shape or '<missing>'}"
        )
    if active_regions and active_regions != list(ACTIVE_REGIONS):
        raise ValueError(
            f"Regions checkpoint active_regions must be {list(ACTIVE_REGIONS)}, got {active_regions}"
        )

    return {
        "subset": subset or SUBSET,
        "model_name": model_name or "region_resnet18_gru",
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "classifier_weight_shape": classifier_shape,
        "classifier_bias_shape": bias_shape,
        "active_regions": active_regions or list(ACTIVE_REGIONS),
        "classifier_evidence": evidence,
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "best_metric": payload.get("best_metric") if isinstance(payload, dict) else None,
    }


def inspect_skeleton_config(path: Path) -> dict[str, Any]:
    """Validate the expected NSLT1000 skeleton resolved config."""

    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    graph_cfg = dict(config.get("graph", {}))
    model_cfg = dict(config.get("model", {}))

    subset = safe_str(dataset_cfg.get("subset"))
    keypoint_set = safe_str(dataset_cfg.get("keypoint_set"))
    expected_shape = list(dataset_cfg.get("expected_shape", []))
    layout = safe_str(graph_cfg.get("layout"))
    model_name = safe_str(model_cfg.get("name")).lower()
    num_nodes = int(model_cfg.get("num_nodes", -1))
    num_classes = int(model_cfg.get("num_classes", -1))

    if subset != SUBSET:
        raise ValueError(f"Skeleton config dataset.subset must be {SUBSET}, got {subset!r}")
    if keypoint_set != "selected_31":
        raise ValueError(f"Skeleton config keypoint_set must be selected_31, got {keypoint_set!r}")
    if expected_shape != list(SKELETON_EXPECTED_SHAPE):
        raise ValueError(
            "Skeleton config expected_shape must be "
            f"{list(SKELETON_EXPECTED_SHAPE)}, got {expected_shape or '<missing>'}"
        )
    if layout != "selected_31":
        raise ValueError(f"Skeleton config graph.layout must be selected_31, got {layout!r}")
    if model_name != "stgcnpp":
        raise ValueError(f"Skeleton config model.name must be stgcnpp, got {model_name!r}")
    if num_nodes != 31:
        raise ValueError(f"Skeleton config model.num_nodes must be 31, got {num_nodes}")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Skeleton config model.num_classes must be {NUM_CLASSES}, got {num_classes}")

    return {
        "subset": subset,
        "keypoint_set": keypoint_set,
        "expected_shape": expected_shape,
        "graph_layout": layout,
        "model_name": model_name,
        "num_nodes": num_nodes,
        "num_classes": num_classes,
    }


def inspect_regions_config(path: Path) -> dict[str, Any]:
    """Validate the expected NSLT1000 regions resolved config."""

    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    model_cfg = dict(config.get("model", {}))

    subset = safe_str(dataset_cfg.get("subset"))
    expected_shape = list(dataset_cfg.get("expected_shape", []))
    active_regions = [
        safe_str(item)
        for item in dataset_cfg.get("active_regions", dataset_cfg.get("region_order", []))
    ]
    model_name = safe_str(model_cfg.get("name")).lower()
    num_classes = int(model_cfg.get("num_classes", -1))

    if subset != SUBSET:
        raise ValueError(f"Regions config dataset.subset must be {SUBSET}, got {subset!r}")
    if expected_shape != list(REGIONS_EXPECTED_SHAPE):
        raise ValueError(
            "Regions config expected_shape must be "
            f"{list(REGIONS_EXPECTED_SHAPE)}, got {expected_shape or '<missing>'}"
        )
    if model_name != "region_resnet18_gru":
        raise ValueError(f"Regions config model.name must be region_resnet18_gru, got {model_name!r}")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Regions config model.num_classes must be {NUM_CLASSES}, got {num_classes}")
    if active_regions != list(ACTIVE_REGIONS):
        raise ValueError(
            f"Regions config active_regions must be {list(ACTIVE_REGIONS)}, got {active_regions}"
        )

    return {
        "subset": subset,
        "expected_shape": expected_shape,
        "active_regions": active_regions,
        "model_name": model_name,
        "num_classes": num_classes,
    }


def skeleton_manifest_name(split: str) -> str:
    """Return the expected skeleton manifest filename for one split."""

    return f"{SUBSET}_selected_31_{split}.csv"


def regions_manifest_name(split: str) -> str:
    """Return the expected regions manifest filename for one split."""

    return f"{SUBSET}_{split}.csv"


def load_skeleton_manifest_set(manifest_root: Path) -> ManifestSet:
    """Load all expected skeleton manifests."""

    manifest_paths = {split: manifest_root / skeleton_manifest_name(split) for split in SAMPLE_SPLITS}
    fieldnames_by_split: dict[str, list[str]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split, path in manifest_paths.items():
        ensure_exists(path, f"skeleton manifest ({split})")
        fieldnames, rows = read_manifest(path)
        fieldnames_by_split[split] = fieldnames
        rows_by_split[split] = rows
    return ManifestSet(
        branch_name="skeleton",
        manifest_root=manifest_root,
        manifest_paths=manifest_paths,
        fieldnames_by_split=fieldnames_by_split,
        rows_by_split=rows_by_split,
    )


def load_regions_manifest_set(manifest_root: Path) -> ManifestSet:
    """Load all expected regions manifests."""

    manifest_paths = {split: manifest_root / regions_manifest_name(split) for split in SAMPLE_SPLITS}
    fieldnames_by_split: dict[str, list[str]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split, path in manifest_paths.items():
        ensure_exists(path, f"regions manifest ({split})")
        fieldnames, rows = read_manifest(path)
        fieldnames_by_split[split] = fieldnames
        rows_by_split[split] = rows
    return ManifestSet(
        branch_name="regions",
        manifest_root=manifest_root,
        manifest_paths=manifest_paths,
        fieldnames_by_split=fieldnames_by_split,
        rows_by_split=rows_by_split,
    )


def audit_manifest_rows(
    rows: list[dict[str, str]],
    *,
    branch_name: str,
    split: str,
) -> dict[str, Any]:
    """Inspect duplicate/collision issues for one manifest split."""

    logical_index: dict[str, dict[str, str]] = {}
    raw_aliases: dict[str, set[str]] = defaultdict(set)
    canonical_to_logical: dict[str, set[str]] = defaultdict(set)
    duplicate_logical_ids: list[str] = []
    empty_ids: list[int] = []

    for row_index, row in enumerate(rows, start=2):
        raw_sample_id = safe_str(row.get("sample_id"))
        logical_sample_id = normalize_sample_id(raw_sample_id)
        if not logical_sample_id:
            empty_ids.append(row_index)
            continue
        canonical_id = canonical_sample_id(logical_sample_id)
        raw_aliases[logical_sample_id].add(raw_sample_id)
        canonical_to_logical[canonical_id].add(logical_sample_id)
        if logical_sample_id in logical_index:
            duplicate_logical_ids.append(logical_sample_id)
            continue
        logical_index[logical_sample_id] = row

    collisions = {
        canonical_id: sorted(logical_ids)
        for canonical_id, logical_ids in canonical_to_logical.items()
        if len(logical_ids) > 1
    }
    alias_groups = {
        logical_id: sorted(raw_ids)
        for logical_id, raw_ids in raw_aliases.items()
        if len(raw_ids) > 1
    }
    return {
        "branch_name": branch_name,
        "split": split,
        "row_count": len(rows),
        "unique_logical_sample_ids": len(logical_index),
        "empty_sample_id_rows": empty_ids,
        "duplicate_logical_sample_ids": sorted(set(duplicate_logical_ids)),
        "duplicate_count": len(set(duplicate_logical_ids)),
        "canonical_collisions": collisions,
        "collision_count": len(collisions),
        "alias_groups": alias_groups,
        "logical_index": logical_index,
    }


def build_alignment(
    skeleton_manifests: ManifestSet,
    regions_manifests: ManifestSet,
) -> dict[str, Any]:
    """Build the normalized skeleton/regions alignment table."""

    all_items: list[AlignmentItem] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    global_errors: list[str] = []

    for split in SAMPLE_SPLITS:
        skeleton_rows = skeleton_manifests.rows_by_split[split]
        regions_rows = regions_manifests.rows_by_split[split]
        skeleton_audit = audit_manifest_rows(skeleton_rows, branch_name="skeleton", split=split)
        regions_audit = audit_manifest_rows(regions_rows, branch_name="regions", split=split)

        skeleton_index = skeleton_audit["logical_index"]
        regions_index = regions_audit["logical_index"]
        skeleton_ids = set(skeleton_index)
        regions_ids = set(regions_index)
        matched_ids = sorted(skeleton_ids & regions_ids)
        missing_in_skeleton = sorted(regions_ids - skeleton_ids)
        missing_in_regions = sorted(skeleton_ids - regions_ids)
        class_mismatch_ids: list[str] = []
        gloss_mismatch_ids: list[str] = []
        split_mismatch_ids: list[str] = []
        items: list[AlignmentItem] = []

        for logical_id in matched_ids:
            skeleton_row = skeleton_index[logical_id]
            regions_row = regions_index[logical_id]
            skeleton_class = safe_str(skeleton_row.get("class_id"))
            regions_class = safe_str(regions_row.get("class_id"))
            skeleton_gloss = safe_str(skeleton_row.get("gloss"))
            regions_gloss = safe_str(regions_row.get("gloss"))
            skeleton_split = safe_str(skeleton_row.get("split")).lower()
            regions_split = safe_str(regions_row.get("split")).lower()
            if skeleton_class != regions_class:
                class_mismatch_ids.append(logical_id)
            if skeleton_gloss != regions_gloss:
                gloss_mismatch_ids.append(logical_id)
            if skeleton_split != split or regions_split != split:
                split_mismatch_ids.append(logical_id)
            items.append(
                AlignmentItem(
                    split=split,
                    logical_sample_id=logical_id,
                    canonical_sample_id=canonical_sample_id(logical_id),
                    skeleton_row=skeleton_row,
                    regions_row=regions_row,
                )
            )

        split_errors: list[str] = []
        if skeleton_audit["duplicate_count"]:
            split_errors.append(
                f"skeleton duplicate logical sample IDs after normalization: {skeleton_audit['duplicate_logical_sample_ids'][:10]}"
            )
        if regions_audit["duplicate_count"]:
            split_errors.append(
                f"regions duplicate logical sample IDs after normalization: {regions_audit['duplicate_logical_sample_ids'][:10]}"
            )
        if skeleton_audit["collision_count"]:
            split_errors.append("skeleton canonical sample-id collisions detected")
        if regions_audit["collision_count"]:
            split_errors.append("regions canonical sample-id collisions detected")
        if skeleton_audit["empty_sample_id_rows"]:
            split_errors.append("skeleton manifest contains empty sample_id rows")
        if regions_audit["empty_sample_id_rows"]:
            split_errors.append("regions manifest contains empty sample_id rows")

        split_summary = {
            "split": split,
            "expected_count": SPLIT_COUNTS[split],
            "skeleton_count": len(skeleton_rows),
            "regions_count": len(regions_rows),
            "matched_count": len(matched_ids),
            "missing_in_skeleton_count": len(missing_in_skeleton),
            "missing_in_regions_count": len(missing_in_regions),
            "class_mismatch_count": len(class_mismatch_ids),
            "gloss_mismatch_count": len(gloss_mismatch_ids),
            "split_mismatch_count": len(split_mismatch_ids),
            "missing_in_skeleton_examples": missing_in_skeleton[:10],
            "missing_in_regions_examples": missing_in_regions[:10],
            "class_mismatch_examples": class_mismatch_ids[:10],
            "gloss_mismatch_examples": gloss_mismatch_ids[:10],
            "split_mismatch_examples": split_mismatch_ids[:10],
            "skeleton_audit": {key: value for key, value in skeleton_audit.items() if key != "logical_index"},
            "regions_audit": {key: value for key, value in regions_audit.items() if key != "logical_index"},
            "errors": split_errors,
            "items": items,
        }
        split_summaries[split] = split_summary
        all_items.extend(items)
        global_errors.extend(f"{split}: {message}" for message in split_errors)

    total_matched = sum(split_summaries[split]["matched_count"] for split in SAMPLE_SPLITS)
    summary = {
        "splits": split_summaries,
        "total_matched": total_matched,
        "expected_total": TOTAL_SAMPLES,
        "errors": global_errors,
    }
    return summary


def choose_skeleton_source_root(repo_root: Path) -> Path:
    """Select the preferred local skeleton tensor root."""

    candidates = (
        repo_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/tensors" / SUBSET,
        repo_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/graph_tensors/selected_31" / SUBSET,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find a local skeleton tensor source root. Checked: "
        + ", ".join(path.as_posix() for path in candidates)
    )


def choose_regions_branch_root(repo_root: Path) -> Path:
    """Return the logical regions branch root used for path remapping."""

    candidate = repo_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l"
    if candidate.exists():
        return candidate.resolve()
    union_candidate = repo_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l_union"
    if union_candidate.exists():
        return union_candidate.resolve()
    raise FileNotFoundError(
        "Could not find a regions branch root. Checked: "
        f"{candidate.as_posix()}, {union_candidate.as_posix()}"
    )


def _candidate_tensor_ids(raw_sample_id: str) -> list[str]:
    """Build ordered source tensor filename candidates."""

    logical = normalize_sample_id(raw_sample_id)
    canonical = canonical_sample_id(raw_sample_id)
    values: list[str] = []
    for candidate in (safe_str(raw_sample_id), logical, canonical):
        if candidate and candidate not in values:
            values.append(candidate)
    return values


def resolve_skeleton_source_tensor(
    row: dict[str, str],
    *,
    repo_root: Path,
    skeleton_source_root: Path | None = None,
) -> Path:
    """Resolve one source skeleton graph tensor path."""

    data_root = (repo_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l").resolve()
    path_text = safe_str(row.get("graph_tensor_path"))
    if path_text:
        try:
            return resolve_graph_tensor_path(
                path_text,
                project_root=repo_root,
                data_root=data_root,
            )
        except FileNotFoundError:
            pass

    source_root = skeleton_source_root or choose_skeleton_source_root(repo_root)
    split = safe_str(row.get("split")).lower()
    for sample_id in _candidate_tensor_ids(safe_str(row.get("sample_id"))):
        candidate = source_root / split / f"{sample_id}.npz"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve skeleton source tensor for sample_id="
        f"{safe_str(row.get('sample_id'))!r} split={split!r}"
    )


def resolve_regions_source_tensor(
    row: dict[str, str],
    *,
    repo_root: Path,
    regions_branch_root: Path | None = None,
) -> Path:
    """Resolve one source regions tensor path."""

    branch_root = regions_branch_root or choose_regions_branch_root(repo_root)
    path_text = safe_str(row.get("tensor_path"))
    if path_text:
        return resolve_region_tensor_path(
            path_text,
            project_root=repo_root,
            data_root=branch_root,
        )

    split = safe_str(row.get("split")).lower()
    source_roots = (
        branch_root / "tensors" / SUBSET,
        repo_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l_incremental/tensors" / SUBSET,
        repo_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/tensors" / "nslt300",
    )
    for sample_id in _candidate_tensor_ids(safe_str(row.get("sample_id"))):
        for source_root in source_roots:
            candidate = source_root / split / f"{sample_id}.npz"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve regions source tensor for sample_id="
        f"{safe_str(row.get('sample_id'))!r} split={split!r}"
    )


def summarize_tensor_resolution(
    alignment: dict[str, Any],
    *,
    repo_root: Path,
    check_all_tensor_paths: bool,
) -> dict[str, Any]:
    """Resolve source tensors and collect size/path summaries."""

    skeleton_source_root = choose_skeleton_source_root(repo_root)
    regions_branch_root = choose_regions_branch_root(repo_root)
    checked_items: list[AlignmentItem] = []
    errors: list[str] = []
    split_samples: dict[str, list[AlignmentItem]] = {
        split: list(alignment["splits"][split]["items"])
        for split in SAMPLE_SPLITS
    }
    selected_items: list[AlignmentItem] = []

    for split, items in split_samples.items():
        if check_all_tensor_paths:
            selected_items.extend(items)
        else:
            selected_items.extend(items[: min(3, len(items))])

    skeleton_bytes = 0
    regions_bytes = 0
    skeleton_count = 0
    regions_count = 0
    for item in selected_items:
        try:
            skeleton_source = resolve_skeleton_source_tensor(
                item.skeleton_row,
                repo_root=repo_root,
                skeleton_source_root=skeleton_source_root,
            )
            regions_source = resolve_regions_source_tensor(
                item.regions_row,
                repo_root=repo_root,
                regions_branch_root=regions_branch_root,
            )
            skeleton_bytes += int(skeleton_source.stat().st_size)
            regions_bytes += int(regions_source.stat().st_size)
            skeleton_count += 1
            regions_count += 1
            checked_items.append(item)
        except Exception as exc:
            errors.append(
                f"split={item.split} sample_id={item.logical_sample_id} tensor resolution failed: {exc}"
            )

    return {
        "checked_pair_count": len(checked_items),
        "check_all_tensor_paths": bool(check_all_tensor_paths),
        "skeleton_checked_count": skeleton_count,
        "regions_checked_count": regions_count,
        "skeleton_bytes_checked": skeleton_bytes,
        "regions_bytes_checked": regions_bytes,
        "errors": errors,
        "skeleton_source_root": skeleton_source_root.as_posix(),
        "regions_branch_root": regions_branch_root.as_posix(),
    }


def estimate_source_sizes(
    alignment: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Estimate full package tensor sizes using source-file metadata."""

    skeleton_source_root = choose_skeleton_source_root(repo_root)
    regions_branch_root = choose_regions_branch_root(repo_root)
    skeleton_bytes = 0
    regions_bytes = 0
    errors: list[str] = []
    sample_examples: list[dict[str, str]] = []

    for split in SAMPLE_SPLITS:
        for item in alignment["splits"][split]["items"]:
            try:
                skeleton_source = resolve_skeleton_source_tensor(
                    item.skeleton_row,
                    repo_root=repo_root,
                    skeleton_source_root=skeleton_source_root,
                )
                regions_source = resolve_regions_source_tensor(
                    item.regions_row,
                    repo_root=repo_root,
                    regions_branch_root=regions_branch_root,
                )
            except Exception as exc:
                errors.append(
                    f"split={split} sample_id={item.logical_sample_id} source resolution failed: {exc}"
                )
                continue
            skeleton_bytes += int(skeleton_source.stat().st_size)
            regions_bytes += int(regions_source.stat().st_size)
            if len(sample_examples) < 6:
                sample_examples.append(
                    {
                        "split": split,
                        "logical_sample_id": item.logical_sample_id,
                        "canonical_sample_id": item.canonical_sample_id,
                        "skeleton_source": skeleton_source.as_posix(),
                        "regions_source": regions_source.as_posix(),
                    }
                )

    return {
        "skeleton_bytes": skeleton_bytes,
        "regions_bytes": regions_bytes,
        "total_bytes": skeleton_bytes + regions_bytes,
        "sample_examples": sample_examples,
        "errors": errors,
        "skeleton_source_root": skeleton_source_root.as_posix(),
        "regions_branch_root": regions_branch_root.as_posix(),
    }


def get_free_disk_bytes(path: Path) -> int:
    """Return free disk bytes on the target filesystem."""

    usage = shutil.disk_usage(path.resolve())
    return int(usage.free)


def load_build_state(path: Path) -> dict[str, Any]:
    """Load build state when resuming one interrupted package build."""

    if not path.exists():
        return {"created_at": "", "updated_at": "", "samples": {}}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Build state must be a JSON object: {path.as_posix()}")
    payload.setdefault("samples", {})
    return payload


def save_build_state(path: Path, payload: dict[str, Any]) -> None:
    """Persist one build-state JSON file."""

    payload["updated_at"] = now_utc_iso()
    write_json(path, payload)


def materialize_file(
    src: Path,
    dst: Path,
    *,
    link_mode: str,
) -> str:
    """Materialize one file via hardlink/copy/auto fallback."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    selected_mode = str(link_mode).strip().lower()
    if selected_mode not in {"hardlink", "copy", "auto"}:
        raise ValueError(f"Unsupported link_mode: {link_mode!r}")

    if selected_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if selected_mode == "hardlink":
            raise
        shutil.copy2(src, dst)
        return "copy"


def verify_existing_file(path: Path) -> bool:
    """Return whether one existing destination file looks usable."""

    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def rewrite_skeleton_manifest_row(
    row: dict[str, str],
    *,
    canonical_sample_id_value: str,
    split: str,
) -> dict[str, str]:
    """Rewrite one skeleton row for the self-contained package."""

    updated = dict(row)
    updated["sample_id"] = canonical_sample_id_value
    updated["graph_tensor_path"] = f"tensors/{SUBSET}/{split}/{canonical_sample_id_value}.npz"
    for key in ("pose_path", "selected_path", "normalized_path", "error_message"):
        if key in updated:
            updated[key] = ""
    return updated


def rewrite_regions_manifest_row(
    row: dict[str, str],
    *,
    canonical_sample_id_value: str,
    split: str,
) -> dict[str, str]:
    """Rewrite one regions row for the self-contained package."""

    updated = dict(row)
    updated["sample_id"] = canonical_sample_id_value
    updated["tensor_path"] = f"tensors/{SUBSET}/{split}/{canonical_sample_id_value}.npz"
    updated["status"] = "ok"
    for key in ("crop_root", "preview_path", "error_message"):
        if key in updated:
            updated[key] = ""
    return updated


def build_kaggle_root(package_name: str) -> str:
    """Return the canonical Kaggle mount root for one package name."""

    safe_name = safe_str(package_name)
    if not safe_name:
        raise ValueError("package_name must not be empty")
    return f"/kaggle/input/{safe_name}/{safe_name}"


def build_sample_id_policy_description() -> dict[str, Any]:
    """Return the package sample-id policy block used in metadata."""

    return {
        "logical_normalization": "digits and integer-like float strings collapse to the same logical ID",
        "canonical_format": f"zero-pad numeric logical IDs to width {SAMPLE_ID_CANONICAL_WIDTH}",
        "examples": {
            "00593": normalize_sample_id("00593"),
            "593": normalize_sample_id("593"),
            "593.0": normalize_sample_id("593.0"),
            "canonical_593": canonical_sample_id("593"),
        },
    }


def build_package_metadata(
    *,
    package_name: str,
    created_at: str,
    package_version: str,
    link_mode: str,
) -> dict[str, Any]:
    """Create the portable package metadata payload."""

    return {
        "package_name": package_name,
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "created_at": created_at,
        "package_version": package_version,
        "skeleton_model": "stgcnpp",
        "regions_model": "region_resnet18_gru",
        "skeleton_feature_dim": 256,
        "regions_feature_dim": 768,
        "fusion_dim": 256,
        "active_regions": list(ACTIVE_REGIONS),
        "keypoint_set": "selected_31",
        "expected_shapes": {
            "skeleton": list(SKELETON_EXPECTED_SHAPE),
            "regions": list(REGIONS_EXPECTED_SHAPE),
        },
        "split_counts": dict(SPLIT_COUNTS),
        "total_samples": TOTAL_SAMPLES,
        "checkpoint_paths": {
            "skeleton": "checkpoints/skeleton/best.pt",
            "regions": "checkpoints/regions/best.pt",
        },
        "config_paths": {
            "fusion": "configs/gated_feature_fusion_nslt1000_kaggle.yaml",
            "skeleton": "configs/skeleton_config_resolved.yaml",
            "regions": "configs/regions_config_resolved.yaml",
        },
        "tensor_counts": {
            "skeleton": dict(SPLIT_COUNTS),
            "regions": dict(SPLIT_COUNTS),
        },
        "link_mode": link_mode,
        "sample_id_policy": build_sample_id_policy_description(),
        "source_description": (
            "Skeleton graph tensors are materialized from the local selected_31 NSLT1000 branch inputs. "
            "Regions tensors are materialized from the NSLT1000 union manifests, resolving each row to either "
            "the NSLT300 reusable tensor set or the NSLT1000 incremental tensor set before copying or linking "
            "into the canonical package layout."
        ),
    }


def create_package_readme(package_name: str) -> str:
    """Render the package README template."""

    kaggle_root = build_kaggle_root(package_name)
    return (
        f"# {package_name}\n\n"
        "## 1. Purpose\n"
        "This package is a fully self-contained Kaggle-ready bundle for Gated Feature Fusion on WLASL NSLT1000.\n\n"
        "## 2. Included contents\n"
        "- Skeleton checkpoint and resolved config\n"
        "- Regions checkpoint and resolved config\n"
        "- Gated Feature Fusion Kaggle config\n"
        "- Canonical skeleton manifests and tensors\n"
        "- Canonical regions manifests and tensors\n"
        "- Package verifier under `verify/`\n"
        "- `README.md` and `metadata.json`\n\n"
        "## 3. Directory structure\n"
        "```text\n"
        f"{package_name}/\n"
        "|-- README.md\n"
        "|-- metadata.json\n"
        "|-- checkpoints/\n"
        "|   |-- skeleton/best.pt\n"
        "|   `-- regions/best.pt\n"
        "|-- configs/\n"
        "|   |-- gated_feature_fusion_nslt1000_kaggle.yaml\n"
        "|   |-- skeleton_config_resolved.yaml\n"
        "|   `-- regions_config_resolved.yaml\n"
        "|-- branch_inputs/\n"
        "|   |-- skeleton/rtmw_l/...\n"
        "|   `-- regions/rtmw_l/...\n"
        "`-- verify/\n"
        "    `-- verify_package.py\n"
        "```\n\n"
        "## 4. Expected split counts\n"
        f"- train: {SPLIT_COUNTS['train']}\n"
        f"- val: {SPLIT_COUNTS['val']}\n"
        f"- test: {SPLIT_COUNTS['test']}\n"
        f"- total: {TOTAL_SAMPLES}\n\n"
        "## 5. Expected tensor shapes\n"
        f"- skeleton: {list(SKELETON_EXPECTED_SHAPE)}\n"
        f"- regions: {list(REGIONS_EXPECTED_SHAPE)}\n\n"
        "## 6. Regions materialization policy\n"
        "The package contains the full canonical 7,232 regions tensors inside `branch_inputs/regions/rtmw_l/tensors/nslt1000/`.\n"
        "The union manifests are resolved row by row so the final package no longer depends on mounting separate NSLT300 and NSLT1000 runtime roots.\n\n"
        "## 7. Sample-ID policy\n"
        "Numeric aliases such as `00593`, `593`, and `593.0` are treated as the same logical sample ID.\n"
        "Canonical package sample IDs are zero-padded to width 5 for numeric values.\n\n"
        "## 8. Hardlink note\n"
        "When packaging locally with `--link-mode hardlink` or `--link-mode auto`, hardlinks can reduce additional physical disk usage.\n"
        "After zipping or uploading to Kaggle, packaged files behave like normal files.\n"
        "Do not delete source tensors until package verification has passed.\n\n"
        "## 9. Verify the package\n"
        "```bash\n"
        f"python {kaggle_root}/verify/verify_package.py --package-root {kaggle_root}\n"
        "```\n\n"
        "## 10. Train on Kaggle\n"
        "```bash\n"
        "python scripts/train/train_gated_fusion.py \\\n"
        f"  --config {kaggle_root}/configs/gated_feature_fusion_nslt1000_kaggle.yaml\n"
        "```\n"
    )


def relative_package_path(path: Path, *, package_root: Path) -> str:
    """Return one package-internal relative path."""

    return path.resolve(strict=False).relative_to(package_root.resolve(strict=False)).as_posix()


def copy_report_files(source_root: Path, destination_root: Path) -> list[str]:
    """Copy available report files into the package tree."""

    copied: list[str] = []
    if not source_root.exists():
        return copied
    for item in sorted(source_root.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(source_root)
        target = destination_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied.append(target.as_posix())
    return copied


def inspect_fusion_config(path: Path) -> dict[str, Any]:
    """Validate the local NSLT1000 gated-fusion train config."""

    config = read_yaml(path)
    dataset_cfg = dict(config.get("dataset", {}))
    skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
    regions_cfg = dict(dataset_cfg.get("regions", {}))
    skeleton_branch_cfg = dict(config.get("skeleton_branch", {}))
    regions_branch_cfg = dict(config.get("regions_branch", {}))
    fusion_cfg = dict(config.get("fusion_model", {}))

    subset = safe_str(dataset_cfg.get("subset"))
    num_classes = int(dataset_cfg.get("num_classes", -1))
    skeleton_expected_shape = list(skeleton_cfg.get("expected_shape", []))
    regions_expected_shape = list(regions_cfg.get("expected_shape", []))
    active_regions = [
        safe_str(item)
        for item in regions_cfg.get("active_regions", regions_cfg.get("region_order", []))
    ]

    if subset != SUBSET:
        raise ValueError(f"Fusion config dataset.subset must be {SUBSET}, got {subset!r}")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Fusion config dataset.num_classes must be {NUM_CLASSES}, got {num_classes}")
    if skeleton_cfg.get("keypoint_set") != "selected_31":
        raise ValueError("Fusion config skeleton.keypoint_set must be selected_31")
    if skeleton_expected_shape != list(SKELETON_EXPECTED_SHAPE):
        raise ValueError("Fusion config skeleton.expected_shape is incorrect")
    if regions_expected_shape != list(REGIONS_EXPECTED_SHAPE):
        raise ValueError("Fusion config regions.expected_shape is incorrect")
    if active_regions != list(ACTIVE_REGIONS):
        raise ValueError("Fusion config regions.active_regions is incorrect")
    if int(skeleton_branch_cfg.get("model", {}).get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError("Fusion config skeleton_branch.model.num_classes is incorrect")
    if int(regions_branch_cfg.get("model", {}).get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError("Fusion config regions_branch.model.num_classes is incorrect")
    if int(fusion_cfg.get("hidden_dim", -1)) != 256:
        raise ValueError("Fusion config fusion_model.hidden_dim must be 256")

    return {
        "subset": subset,
        "num_classes": num_classes,
        "skeleton_keypoint_set": skeleton_cfg.get("keypoint_set"),
        "skeleton_expected_shape": skeleton_expected_shape,
        "regions_expected_shape": regions_expected_shape,
        "active_regions": active_regions,
        "fusion_dim": int(fusion_cfg.get("hidden_dim", 256)),
    }


def build_runtime_fusion_config(
    *,
    fusion_config_path: Path,
    skeleton_checkpoint: Path,
    regions_checkpoint: Path,
    skeleton_config: Path,
    regions_config: Path,
) -> dict[str, Any]:
    """Load one repo fusion config and replace branch artifact paths explicitly."""

    config = read_yaml(fusion_config_path)
    config["skeleton_branch"] = dict(config.get("skeleton_branch", {}))
    config["regions_branch"] = dict(config.get("regions_branch", {}))
    config["skeleton_branch"]["checkpoint_path"] = skeleton_checkpoint.as_posix()
    config["regions_branch"]["checkpoint_path"] = regions_checkpoint.as_posix()
    config["skeleton_branch"]["config_path"] = skeleton_config.as_posix()
    config["regions_branch"]["config_path"] = regions_config.as_posix()
    return config


def run_fusion_component_smoke(
    *,
    fusion_config_path: Path,
    skeleton_checkpoint: Path,
    regions_checkpoint: Path,
    skeleton_config: Path,
    regions_config: Path,
) -> dict[str, Any]:
    """Run a lightweight CPU forward/loss smoke test for the fused model."""

    from slr.branches.fusion.build import build_gated_feature_fusion_from_config

    runtime_config = build_runtime_fusion_config(
        fusion_config_path=fusion_config_path,
        skeleton_checkpoint=skeleton_checkpoint,
        regions_checkpoint=regions_checkpoint,
        skeleton_config=skeleton_config,
        regions_config=regions_config,
    )
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
    return {
        "ok": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu().item()),
        "skeleton_feature_shape": list(features["skeleton_feature"].shape),
        "region_feature_shape": list(features["region_feature"].shape),
        "skeleton_proj_shape": list(features["skeleton_proj"].shape),
        "region_proj_shape": list(features["region_proj"].shape),
        "gate_shape": list(features["gate"].shape),
        "fused_shape": list(features["fused"].shape),
        "model_info": info,
    }


def build_requirements_summary(
    *,
    repo_root: Path,
    skeleton_checkpoint: Path,
    regions_checkpoint: Path,
    skeleton_config: Path,
    regions_config: Path,
    skeleton_manifest_root: Path,
    regions_manifest_root: Path,
    fusion_config: Path,
    check_all_tensor_paths: bool,
) -> dict[str, Any]:
    """Build the end-to-end NSLT1000 packaging readiness summary."""

    errors: list[str] = []
    warnings: list[str] = []

    try:
        skeleton_checkpoint_info = inspect_skeleton_checkpoint(skeleton_checkpoint)
    except Exception as exc:
        skeleton_checkpoint_info = {"error": str(exc)}
        errors.append(f"skeleton checkpoint: {exc}")

    try:
        regions_checkpoint_info = inspect_regions_checkpoint(regions_checkpoint)
    except Exception as exc:
        regions_checkpoint_info = {"error": str(exc)}
        errors.append(f"regions checkpoint: {exc}")

    try:
        skeleton_config_info = inspect_skeleton_config(skeleton_config)
    except Exception as exc:
        skeleton_config_info = {"error": str(exc)}
        errors.append(f"skeleton config: {exc}")

    try:
        regions_config_info = inspect_regions_config(regions_config)
    except Exception as exc:
        regions_config_info = {"error": str(exc)}
        errors.append(f"regions config: {exc}")

    try:
        fusion_config_info = inspect_fusion_config(fusion_config)
    except Exception as exc:
        fusion_config_info = {"error": str(exc)}
        errors.append(f"fusion config: {exc}")

    alignment_summary: dict[str, Any] = {"errors": ["alignment not attempted"]}
    tensor_resolution_summary: dict[str, Any] = {"errors": ["tensor resolution not attempted"]}
    size_estimate_summary: dict[str, Any] = {"errors": ["size estimation not attempted"]}
    smoke_summary: dict[str, Any] = {"ok": False, "error": "smoke test not attempted"}
    skeleton_manifests: ManifestSet | None = None
    regions_manifests: ManifestSet | None = None

    if not errors:
        try:
            skeleton_manifests = load_skeleton_manifest_set(skeleton_manifest_root)
            regions_manifests = load_regions_manifest_set(regions_manifest_root)
            alignment_summary = build_alignment(skeleton_manifests, regions_manifests)
        except Exception as exc:
            errors.append(f"manifest/alignment: {exc}")
            alignment_summary = {"errors": [str(exc)]}

    if not errors and skeleton_manifests is not None and regions_manifests is not None:
        for split in SAMPLE_SPLITS:
            split_summary = alignment_summary["splits"][split]
            if split_summary["skeleton_count"] != SPLIT_COUNTS[split]:
                errors.append(
                    f"skeleton manifest count mismatch for {split}: expected {SPLIT_COUNTS[split]}, "
                    f"got {split_summary['skeleton_count']}"
                )
            if split_summary["regions_count"] != SPLIT_COUNTS[split]:
                errors.append(
                    f"regions manifest count mismatch for {split}: expected {SPLIT_COUNTS[split]}, "
                    f"got {split_summary['regions_count']}"
                )
            if split_summary["matched_count"] != SPLIT_COUNTS[split]:
                errors.append(
                    f"paired sample count mismatch for {split}: expected {SPLIT_COUNTS[split]}, "
                    f"got {split_summary['matched_count']}"
                )
            if split_summary["missing_in_skeleton_count"]:
                errors.append(f"missing regions-only samples detected for {split}")
            if split_summary["missing_in_regions_count"]:
                errors.append(f"missing skeleton-only samples detected for {split}")
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

        if alignment_summary.get("total_matched") != TOTAL_SAMPLES:
            errors.append(
                f"total matched sample count mismatch: expected {TOTAL_SAMPLES}, "
                f"got {alignment_summary.get('total_matched')}"
            )

    if not errors:
        tensor_resolution_summary = summarize_tensor_resolution(
            alignment_summary,
            repo_root=repo_root,
            check_all_tensor_paths=check_all_tensor_paths,
        )
        if tensor_resolution_summary["errors"]:
            errors.extend(tensor_resolution_summary["errors"])
        if not check_all_tensor_paths:
            warnings.append(
                "check-all-tensor-paths was disabled, so only a small tensor-path sample was resolved."
            )
        size_estimate_summary = estimate_source_sizes(alignment_summary, repo_root=repo_root)
        if size_estimate_summary["errors"]:
            errors.extend(size_estimate_summary["errors"])

    if not errors:
        try:
            smoke_summary = run_fusion_component_smoke(
                fusion_config_path=fusion_config,
                skeleton_checkpoint=skeleton_checkpoint,
                regions_checkpoint=regions_checkpoint,
                skeleton_config=skeleton_config,
                regions_config=regions_config,
            )
        except Exception as exc:
            smoke_summary = {"ok": False, "error": str(exc)}
            errors.append(f"fusion smoke test: {exc}")

    free_disk_bytes = get_free_disk_bytes(repo_root)
    total_source_bytes = int(size_estimate_summary.get("total_bytes", 0))
    estimated_additional_physical_bytes = (
        0 if total_source_bytes <= 0 else total_source_bytes
    )
    if free_disk_bytes < estimated_additional_physical_bytes:
        warnings.append(
            "available free disk appears smaller than the estimated additional physical size "
            f"required by a copy-based build ({format_bytes(estimated_additional_physical_bytes)})."
        )

    status = READY_STATUS if not errors else NOT_READY_STATUS
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "repo_root": repo_root.as_posix(),
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "package_name_default": PACKAGE_NAME_DEFAULT,
        "expected_shapes": {
            "skeleton": list(SKELETON_EXPECTED_SHAPE),
            "regions": list(REGIONS_EXPECTED_SHAPE),
        },
        "active_regions": list(ACTIVE_REGIONS),
        "split_counts": dict(SPLIT_COUNTS),
        "total_samples": TOTAL_SAMPLES,
        "check_all_tensor_paths": bool(check_all_tensor_paths),
        "paths": {
            "skeleton_checkpoint": skeleton_checkpoint.as_posix(),
            "regions_checkpoint": regions_checkpoint.as_posix(),
            "skeleton_config": skeleton_config.as_posix(),
            "regions_config": regions_config.as_posix(),
            "fusion_config": fusion_config.as_posix(),
            "skeleton_manifest_root": skeleton_manifest_root.as_posix(),
            "regions_manifest_root": regions_manifest_root.as_posix(),
        },
        "checkpoints": {
            "skeleton": skeleton_checkpoint_info,
            "regions": regions_checkpoint_info,
        },
        "configs": {
            "skeleton": skeleton_config_info,
            "regions": regions_config_info,
            "fusion": fusion_config_info,
        },
        "alignment": {
            "total_matched": alignment_summary.get("total_matched"),
            "expected_total": alignment_summary.get("expected_total"),
            "splits": {
                split: {
                    key: value
                    for key, value in alignment_summary.get("splits", {}).get(split, {}).items()
                    if key != "items"
                }
                for split in SAMPLE_SPLITS
            },
        },
        "tensor_resolution": tensor_resolution_summary,
        "size_estimate": {
            **size_estimate_summary,
            "skeleton_human": format_bytes(int(size_estimate_summary.get("skeleton_bytes", 0))),
            "regions_human": format_bytes(int(size_estimate_summary.get("regions_bytes", 0))),
            "total_human": format_bytes(int(size_estimate_summary.get("total_bytes", 0))),
            "estimated_additional_physical_bytes_copy_mode": estimated_additional_physical_bytes,
            "estimated_additional_physical_human_copy_mode": format_bytes(estimated_additional_physical_bytes),
        },
        "disk": {
            "free_bytes": free_disk_bytes,
            "free_human": format_bytes(free_disk_bytes),
        },
        "fusion_smoke": smoke_summary,
        "ready_to_execute": status == READY_STATUS,
    }
