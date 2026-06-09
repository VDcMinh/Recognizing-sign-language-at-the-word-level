"""Check whether NSLT300 region tensors can be reused incrementally for NSLT1000."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn, resolve_region_tensor_path
from slr.branches.regions.region_schema import REGION_NAMES
from slr.utils.io import ensure_dir, stringify_path, write_dataframe_csv, write_json


ALLOWED_SPLITS = ("train", "val", "test")
DEFAULT_REPORT_ROOT = Path("reports/regions/nslt1000_incremental_feasibility")
DEFAULT_PREVIEW_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l_incremental_check")
DEFAULT_INCREMENTAL_FUTURE_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l_incremental")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check incremental reuse feasibility from NSLT300 region tensors to NSLT1000."
    )
    parser.add_argument(
        "--regions-root",
        type=Path,
        required=True,
        help="Existing regions branch root, for example data/datasets/WLASL/branch_inputs/regions/rtmw_l.",
    )
    parser.add_argument(
        "--nslt1000-standardized-root",
        type=Path,
        required=True,
        help="Standardized dataset root containing manifests/, for example data/datasets/WLASL/standardized.",
    )
    parser.add_argument(
        "--subset-source",
        type=str,
        default="nslt300",
        help="Subset providing reusable tensors.",
    )
    parser.add_argument(
        "--subset-target",
        type=str,
        default="nslt1000",
        help="Target subset that may reuse tensors.",
    )
    parser.add_argument(
        "--active-regions",
        type=str,
        default="left_hand,right_hand,face",
        help="Comma-separated region order.",
    )
    parser.add_argument(
        "--expected-shape",
        type=str,
        default="3,3,64,112,112",
        help="Comma-separated expected tensor shape.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Expected number of target classes.",
    )
    parser.add_argument(
        "--create-preview",
        action="store_true",
        help="Create small preview manifests and run loader checks against them.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=12,
        help="Maximum total preview rows to create across all splits.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="Output folder for summary JSON and CSV artifacts.",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=DEFAULT_PREVIEW_ROOT,
        help="Output folder for preview manifests used in loader compatibility checks.",
    )
    return parser


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(shape) != 5:
        raise ValueError(f"expected-shape must contain 5 integers, got {shape}.")
    return shape


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in ("sample_id", "video_id", "gloss", "split", "status", "tensor_path", "preview_path", "crop_root"):
        if column in working.columns:
            working[column] = working[column].fillna("").astype(str).str.strip()
    if "split" in working.columns:
        working["split"] = working["split"].str.lower()
    if "status" in working.columns:
        working["status"] = working["status"].str.lower()
    if "class_id" in working.columns:
        working["class_id"] = pd.to_numeric(working["class_id"], errors="coerce").astype("Int64")
    return working


def _load_manifest_set(root: Path, subset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ALLOWED_SPLITS:
        path = root / "manifests" / f"{subset}_{split}.csv"
        frame = pd.read_csv(path)
        frame["manifest_path"] = stringify_path(path)
        frame["split_file"] = split
        frames.append(_normalize_manifest(frame))
    output = pd.concat(frames, ignore_index=True)
    output["sample_id"] = output["sample_id"].astype(str).str.strip()
    return output


def _load_regions_manifest_set(root: Path, subset: str) -> pd.DataFrame:
    manifests_root = root / "manifests"
    frames: list[pd.DataFrame] = []
    for split in ALLOWED_SPLITS:
        path = manifests_root / f"{subset}_{split}.csv"
        frame = pd.read_csv(path)
        frame["manifest_path"] = stringify_path(path)
        frame["split_file"] = split
        frames.append(_normalize_manifest(frame))
    output = pd.concat(frames, ignore_index=True)
    output["sample_id"] = output["sample_id"].astype(str).str.strip()
    return output


def _repo_relative_path(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    try:
        return stringify_path(candidate.resolve().relative_to(project_root.resolve()))
    except Exception:
        return stringify_path(candidate)


def _split_summary(source: pd.DataFrame, target: pd.DataFrame, split: str) -> dict[str, Any]:
    source_ids = set(source.loc[source["split"] == split, "sample_id"])
    target_ids = set(target.loc[target["split"] == split, "sample_id"])
    return {
        "split": split,
        "source_rows": int((source["split"] == split).sum()),
        "target_rows": int((target["split"] == split).sum()),
        "reusable_from_source": int(len(source_ids & target_ids)),
        "missing_for_target": int(len(target_ids - source_ids)),
        "source_not_found_in_target": int(len(source_ids - target_ids)),
    }


def _pick_preview_rows(
    union_frame: pd.DataFrame,
    preview_limit: int,
) -> dict[str, pd.DataFrame]:
    per_split_frames: dict[str, pd.DataFrame] = {}
    if preview_limit <= 0:
        return {split: union_frame.iloc[0:0].copy() for split in ALLOWED_SPLITS}

    base = preview_limit // len(ALLOWED_SPLITS)
    remainder = preview_limit % len(ALLOWED_SPLITS)
    split_limits = {
        split: base + (1 if index < remainder else 0)
        for index, split in enumerate(ALLOWED_SPLITS)
    }

    for split in ALLOWED_SPLITS:
        split_frame = union_frame[union_frame["split"] == split].copy()
        if split_frame.empty or split_limits[split] <= 0:
            per_split_frames[split] = split_frame.iloc[0:0].copy()
            continue

        reusable = split_frame[split_frame["reuse_source"] == "nslt300"].sort_values("sample_id")
        missing = split_frame[split_frame["reuse_source"] == "missing"].sort_values("sample_id")
        reusable_quota = min(len(reusable), max(1, split_limits[split] // 2)) if not reusable.empty else 0
        missing_quota = min(len(missing), split_limits[split] - reusable_quota)
        rows = [reusable.head(reusable_quota), missing.head(missing_quota)]
        combined = pd.concat(rows, ignore_index=True) if rows else split_frame.iloc[0:0].copy()
        if len(combined) < split_limits[split]:
            already = set(combined["sample_id"].tolist())
            remainder_rows = split_frame[~split_frame["sample_id"].isin(already)].sort_values("sample_id")
            combined = pd.concat(
                [combined, remainder_rows.head(split_limits[split] - len(combined))],
                ignore_index=True,
            )
        per_split_frames[split] = combined.head(split_limits[split]).reset_index(drop=True)

    return per_split_frames


def _preview_manifest_columns() -> list[str]:
    return [
        "sample_id",
        "video_id",
        "class_id",
        "gloss",
        "split",
        "tensor_path",
        "status",
        "reuse_source",
        "needs_extraction",
        "source_subset",
        "original_nslt1000_class_id",
        "original_nslt1000_gloss",
        "tensor_shape",
        "preview_path",
        "crop_root",
        "notes",
    ]


def _verify_tensor_shapes(
    reusable_frame: pd.DataFrame,
    *,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    limit: int,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checked_rows: list[dict[str, Any]] = []
    sample_rows = reusable_frame.sort_values(["split", "sample_id"]).head(limit)
    for _, row in sample_rows.iterrows():
        resolved = resolve_region_tensor_path(row["tensor_path"], project_root=project_root)
        with np.load(resolved, allow_pickle=False) as payload:
            data = np.asarray(payload["data"])
            region_names = payload["region_names"].tolist() if "region_names" in payload else []
        checked_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "tensor_path": row["tensor_path"],
                "resolved_tensor_path": stringify_path(resolved),
                "file_size_bytes": int(resolved.stat().st_size),
                "shape": json.dumps(list(data.shape)),
                "shape_ok": bool(tuple(data.shape) == expected_shape),
                "region_names": json.dumps(region_names),
                "region_names_ok": bool(region_names == active_regions),
            }
        )

    checked_frame = pd.DataFrame(checked_rows)
    summary = {
        "checked_reusable_tensor_count": int(len(checked_frame)),
        "shape_pass_count": int(checked_frame["shape_ok"].sum()) if not checked_frame.empty else 0,
        "shape_fail_count": int((~checked_frame["shape_ok"]).sum()) if not checked_frame.empty else 0,
        "shape_fail_examples": checked_frame.loc[~checked_frame["shape_ok"], "sample_id"].head(10).tolist()
        if not checked_frame.empty
        else [],
    }
    return checked_frame, summary


def _write_preview_manifests(
    preview_by_split: dict[str, pd.DataFrame],
    preview_root: Path,
) -> dict[str, str]:
    manifests_root = ensure_dir(preview_root / "manifests")
    output_paths: dict[str, str] = {}
    columns = _preview_manifest_columns()
    for split, frame in preview_by_split.items():
        output_path = manifests_root / f"nslt1000_{split}_preview.csv"
        write_dataframe_csv(frame.reindex(columns=columns), output_path)
        output_paths[split] = stringify_path(output_path)
    return output_paths


def _run_loader_preview_check(
    preview_paths: dict[str, str],
    *,
    preview_root: Path,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    num_classes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, manifest_path in preview_paths.items():
        preview_frame = pd.read_csv(manifest_path)
        ok_preview_frame = preview_frame.loc[preview_frame["status"].astype(str).str.lower() == "ok"].copy()
        if ok_preview_frame.empty:
            continue
        dataset = RegionClipDataset(
            manifest_path=manifest_path,
            project_root=Path.cwd(),
            data_root=preview_root,
            split=split,
            expected_shape=expected_shape,
            num_classes=num_classes,
            region_order=REGION_NAMES,
            active_regions=active_regions,
            return_metadata=True,
            strict_shape_check=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(4, len(dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=region_collate_fn,
        )
        batch = next(iter(loader))
        batch_shape = tuple(batch["data"].shape)
        for index in range(len(dataset)):
            sample = dataset[index]
            rows.append(
                {
                    "split": split,
                    "sample_id": sample["sample_id"],
                    "class_id": int(sample["class_id"]),
                    "resolved_path": sample["path"],
                    "sample_shape": json.dumps(list(sample["data"].shape)),
                    "region_names": json.dumps(sample.get("region_names", [])),
                    "batch_shape": json.dumps(list(batch_shape)),
                    "loaded_ok": True,
                }
            )

    frame = pd.DataFrame(rows)
    summary = {
        "loader_preview_pass": bool(not frame.empty),
        "loaded_preview_rows": int(len(frame)),
        "preview_splits_checked": sorted(frame["split"].unique().tolist()) if not frame.empty else [],
    }
    return frame, summary


def _format_size(num_bytes: float) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes:.2f} B"


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    active_regions = _parse_csv_list(args.active_regions)
    expected_shape = _parse_shape(args.expected_shape)
    if active_regions != list(REGION_NAMES):
        raise ValueError(f"Expected active-regions {list(REGION_NAMES)}, got {active_regions}.")

    report_root = ensure_dir(args.report_root)
    preview_root = ensure_dir(args.preview_root)
    source_frame = _load_regions_manifest_set(args.regions_root, args.subset_source)
    target_frame = _load_manifest_set(args.nslt1000_standardized_root, args.subset_target)

    source_frame = source_frame[(source_frame["status"] == "ok") & (source_frame["tensor_path"] != "")].copy()
    if "status" in target_frame.columns:
        target_frame = target_frame[target_frame["status"].isin(["", "ok"])].copy()

    source_ids = set(source_frame["sample_id"])
    target_ids = set(target_frame["sample_id"])
    intersection_ids = source_ids & target_ids
    target_missing_ids = target_ids - source_ids
    source_only_ids = source_ids - target_ids

    source_lookup = source_frame.set_index("sample_id", drop=False)
    target_lookup = target_frame.set_index("sample_id", drop=False)
    compare = source_frame[
        ["sample_id", "class_id", "gloss", "split", "tensor_path", "status", "preview_path", "crop_root", "tensor_shape"]
    ].merge(
        target_frame[["sample_id", "class_id", "gloss", "split", "video_id", "status"]],
        on="sample_id",
        how="inner",
        suffixes=("_source", "_target"),
    )
    compare["same_split"] = compare["split_source"] == compare["split_target"]
    compare["same_class_id"] = compare["class_id_source"] == compare["class_id_target"]
    compare["same_gloss"] = compare["gloss_source"] == compare["gloss_target"]

    sample_overlap = target_frame[["sample_id", "video_id", "class_id", "gloss", "split", "status"]].copy()
    sample_overlap["reusable_from_source"] = sample_overlap["sample_id"].isin(source_ids)
    sample_overlap["needs_extraction"] = ~sample_overlap["reusable_from_source"]
    sample_overlap["source_split"] = sample_overlap["sample_id"].map(source_lookup["split"]) if not source_lookup.empty else ""
    sample_overlap["source_tensor_path"] = sample_overlap["sample_id"].map(source_lookup["tensor_path"]) if not source_lookup.empty else ""
    sample_overlap["source_tensor_path"] = sample_overlap["source_tensor_path"].fillna("").astype(str)

    split_summaries = [_split_summary(source_frame, target_frame, split) for split in ALLOWED_SPLITS]

    union_rows: list[dict[str, Any]] = []
    for _, row in target_frame.sort_values(["split", "sample_id"]).iterrows():
        sample_id = row["sample_id"]
        reusable = sample_id in source_ids
        source_row = source_lookup.loc[sample_id] if reusable else None
        if reusable:
            tensor_path = _repo_relative_path(source_row["tensor_path"], project_root)
            preview_path = _repo_relative_path(source_row.get("preview_path", ""), project_root) if _safe_str(source_row.get("preview_path")) else ""
            crop_root = _repo_relative_path(source_row.get("crop_root", ""), project_root) if _safe_str(source_row.get("crop_root")) else ""
            tensor_shape = _safe_str(source_row.get("tensor_shape")) or json.dumps(list(expected_shape))
            status = "ok"
            reuse_source = args.subset_source
            notes = "reuse_existing_tensor"
        else:
            tensor_path = stringify_path(
                DEFAULT_INCREMENTAL_FUTURE_ROOT / "tensors" / args.subset_target / row["split"] / f"{sample_id}.npz"
            )
            preview_path = ""
            crop_root = ""
            tensor_shape = json.dumps(list(expected_shape))
            status = "pending_extraction"
            reuse_source = "missing"
            notes = "future_incremental_extraction_target"

        union_rows.append(
            {
                "sample_id": sample_id,
                "video_id": _safe_str(row.get("video_id")),
                "class_id": int(row["class_id"]),
                "gloss": _safe_str(row["gloss"]),
                "split": _safe_str(row["split"]),
                "tensor_path": tensor_path,
                "status": status,
                "reuse_source": reuse_source,
                "needs_extraction": bool(not reusable),
                "source_subset": args.subset_target,
                "original_nslt1000_class_id": int(row["class_id"]),
                "original_nslt1000_gloss": _safe_str(row["gloss"]),
                "tensor_shape": tensor_shape,
                "preview_path": preview_path,
                "crop_root": crop_root,
                "notes": notes,
            }
        )
    union_frame = pd.DataFrame(union_rows)

    absolute_tensor_path_count = int(source_frame["tensor_path"].map(lambda value: Path(str(value)).is_absolute()).sum())
    relative_tensor_path_count = int(len(source_frame) - absolute_tensor_path_count)

    total_existing_tensor_bytes = 0
    for tensor_path in source_frame["tensor_path"].tolist():
        resolved = resolve_region_tensor_path(tensor_path, project_root=project_root)
        total_existing_tensor_bytes += int(resolved.stat().st_size)
    avg_tensor_size_bytes = total_existing_tensor_bytes / max(len(source_frame), 1)

    reusable_tensor_check, shape_summary = _verify_tensor_shapes(
        source_frame,
        expected_shape=expected_shape,
        active_regions=active_regions,
        limit=min(max(args.preview_limit, 1), len(source_frame)),
        project_root=project_root,
    )

    preview_manifest_paths: dict[str, str] = {}
    loader_preview_check = pd.DataFrame()
    loader_summary = {"loader_preview_pass": False, "loaded_preview_rows": 0, "preview_splits_checked": []}
    if args.create_preview:
        preview_by_split = _pick_preview_rows(union_frame, args.preview_limit)
        preview_manifest_paths = _write_preview_manifests(preview_by_split, preview_root)
        loader_preview_check, loader_summary = _run_loader_preview_check(
            preview_manifest_paths,
            preview_root=preview_root,
            expected_shape=expected_shape,
            active_regions=active_regions,
            num_classes=args.num_classes,
        )

    sample_overlap_path = report_root / "sample_overlap.csv"
    class_mapping_path = report_root / "class_id_mapping_check.csv"
    split_mapping_path = report_root / "split_mapping_check.csv"
    reusable_tensor_check_path = report_root / "reusable_tensor_check.csv"
    loader_preview_check_path = report_root / "loader_preview_check.csv"

    write_dataframe_csv(sample_overlap, sample_overlap_path)
    write_dataframe_csv(
        compare[
            [
                "sample_id",
                "class_id_source",
                "gloss_source",
                "split_source",
                "class_id_target",
                "gloss_target",
                "split_target",
                "same_class_id",
                "same_gloss",
            ]
        ],
        class_mapping_path,
    )
    write_dataframe_csv(
        compare[
            [
                "sample_id",
                "split_source",
                "split_target",
                "same_split",
                "class_id_source",
                "class_id_target",
                "gloss_source",
                "gloss_target",
            ]
        ],
        split_mapping_path,
    )
    write_dataframe_csv(reusable_tensor_check, reusable_tensor_check_path)
    if args.create_preview:
        write_dataframe_csv(loader_preview_check, loader_preview_check_path)

    feasible_status = "FEASIBLE"
    if int((~compare["same_gloss"]).sum()) > 0:
        feasible_status = "NOT FEASIBLE"
    elif args.create_preview and not loader_summary["loader_preview_pass"]:
        feasible_status = "FEASIBLE WITH LOADER/PATH PATCH"
    elif int((~compare["same_split"]).sum()) > 0:
        feasible_status = "NEEDS MANUAL REVIEW"

    summary = {
        "status": feasible_status,
        "source_subset": args.subset_source,
        "target_subset": args.subset_target,
        "active_regions": active_regions,
        "expected_shape": list(expected_shape),
        "num_classes": int(args.num_classes),
        "sample_id_containment": {
            "source_total_rows": int(len(source_frame)),
            "target_total_rows": int(len(target_frame)),
            "source_unique_sample_ids": int(len(source_ids)),
            "target_unique_sample_ids": int(len(target_ids)),
            "intersection_count": int(len(intersection_ids)),
            "source_samples_found_in_target": int(len(intersection_ids)),
            "source_samples_missing_in_target": int(len(source_only_ids)),
            "target_samples_reusable_from_source": int(len(intersection_ids)),
            "target_missing_samples_to_extract": int(len(target_missing_ids)),
            "source_missing_examples": sorted(list(source_only_ids))[:10],
            "target_missing_examples": sorted(list(target_missing_ids))[:10],
            "per_split": split_summaries,
        },
        "split_consistency": {
            "split_match_count": int(compare["same_split"].sum()),
            "split_mismatch_count": int((~compare["same_split"]).sum()),
            "split_mismatch_examples": compare.loc[
                ~compare["same_split"],
                ["sample_id", "split_source", "split_target"],
            ].head(10).to_dict(orient="records"),
        },
        "class_id_mapping": {
            "same_class_id_count": int(compare["same_class_id"].sum()),
            "different_class_id_count": int((~compare["same_class_id"]).sum()),
            "class_id_mismatch_examples": compare.loc[
                ~compare["same_class_id"],
                ["sample_id", "class_id_source", "gloss_source", "class_id_target", "gloss_target"],
            ].head(10).to_dict(orient="records"),
        },
        "gloss_mapping": {
            "same_gloss_count": int(compare["same_gloss"].sum()),
            "different_gloss_count": int((~compare["same_gloss"]).sum()),
            "gloss_mismatch_examples": compare.loc[
                ~compare["same_gloss"],
                ["sample_id", "gloss_source", "gloss_target"],
            ].head(10).to_dict(orient="records"),
        },
        "tensor_path_reuse": {
            "manifest_has_tensor_path": bool("tensor_path" in source_frame.columns),
            "absolute_tensor_path_count": absolute_tensor_path_count,
            "relative_tensor_path_count": relative_tensor_path_count,
            "loader_resolution_behavior": {
                "uses_tensor_path_directly": True,
                "rebuilds_path_from_subset_split_sample_id": False,
                "supports_absolute_tensor_path": True,
                "supports_relative_tensor_path": True,
                "relative_path_resolution": ["project_root", "data_root"],
                "manifest_relative_resolution": False,
                "filters_status_ok": True,
                "shape_check_independent_of_subset_name": True,
            },
        },
        "reusable_tensor_shape_check": {
            **shape_summary,
        },
        "loader_preview": {
            **loader_summary,
            "preview_manifest_paths": preview_manifest_paths,
        },
        "disk_estimate": {
            "target_total_samples": int(len(target_frame)),
            "reusable_samples_from_source": int(len(intersection_ids)),
            "missing_samples_to_extract": int(len(target_missing_ids)),
            "reuse_percentage": float(len(intersection_ids) / max(len(target_frame), 1)),
            "missing_percentage": float(len(target_missing_ids) / max(len(target_frame), 1)),
            "estimated_tensor_size_per_sample_bytes": float(avg_tensor_size_bytes),
            "estimated_tensor_size_per_sample_human": _format_size(avg_tensor_size_bytes),
            "estimated_full_extraction_bytes": float(avg_tensor_size_bytes * len(target_frame)),
            "estimated_full_extraction_human": _format_size(avg_tensor_size_bytes * len(target_frame)),
            "estimated_missing_only_bytes": float(avg_tensor_size_bytes * len(target_missing_ids)),
            "estimated_missing_only_human": _format_size(avg_tensor_size_bytes * len(target_missing_ids)),
            "estimated_saved_bytes": float(avg_tensor_size_bytes * len(intersection_ids)),
            "estimated_saved_human": _format_size(avg_tensor_size_bytes * len(intersection_ids)),
            "existing_source_tensor_bytes": int(total_existing_tensor_bytes),
            "existing_source_tensor_human": _format_size(total_existing_tensor_bytes),
        },
        "union_manifest_preview": {
            "preview_root": stringify_path(preview_root),
            "preview_limit": int(args.preview_limit),
            "preview_row_count": int(sum(len(frame) for frame in _pick_preview_rows(union_frame, args.preview_limit).values()))
            if args.create_preview
            else 0,
        },
        "outputs": {
            "summary_json": stringify_path(report_root / "summary.json"),
            "sample_overlap_csv": stringify_path(sample_overlap_path),
            "class_id_mapping_csv": stringify_path(class_mapping_path),
            "split_mapping_csv": stringify_path(split_mapping_path),
            "reusable_tensor_check_csv": stringify_path(reusable_tensor_check_path),
            "loader_preview_check_csv": stringify_path(loader_preview_check_path) if args.create_preview else "",
        },
    }

    write_json(summary, report_root / "summary.json", indent=2)

    print("== Incremental Feasibility Summary ==")
    print(f"status: {summary['status']}")
    print(f"source_subset: {args.subset_source}")
    print(f"target_subset: {args.subset_target}")
    print(f"source_rows: {summary['sample_id_containment']['source_total_rows']}")
    print(f"target_rows: {summary['sample_id_containment']['target_total_rows']}")
    print(f"reusable_samples: {summary['sample_id_containment']['target_samples_reusable_from_source']}")
    print(f"missing_samples_to_extract: {summary['sample_id_containment']['target_missing_samples_to_extract']}")
    print(f"split_mismatch_count: {summary['split_consistency']['split_mismatch_count']}")
    print(f"class_id_mismatch_count: {summary['class_id_mapping']['different_class_id_count']}")
    print(f"gloss_mismatch_count: {summary['gloss_mapping']['different_gloss_count']}")
    print(f"loader_preview_pass: {summary['loader_preview']['loader_preview_pass']}")
    print(f"estimated_full_extraction: {summary['disk_estimate']['estimated_full_extraction_human']}")
    print(f"estimated_missing_only: {summary['disk_estimate']['estimated_missing_only_human']}")
    print(f"estimated_saved: {summary['disk_estimate']['estimated_saved_human']}")
    print(f"summary_json: {report_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
