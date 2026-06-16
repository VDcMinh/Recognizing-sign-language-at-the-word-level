"""Preflight checker for the NSLT1000 incremental regions pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from scripts.common.regions_nslt1000_incremental_common import (
    ALLOWED_SPLITS,
    DEFAULT_ACTIVE_REGIONS,
    DEFAULT_BASE_ROOT,
    DEFAULT_BASE_SUBSET,
    DEFAULT_EXPECTED_SHAPE,
    DEFAULT_INCREMENTAL_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_TARGET_SOURCE_ROOT,
    DEFAULT_TARGET_SUBSET,
    FEASIBILITY_REPORT_PATH,
    FEASIBILITY_SCRIPT_PATH,
    build_overlap_frames,
    estimate_size_from_base,
    format_size,
    get_free_disk_bytes,
    build_lookup_by_sample,
    load_manifest_set,
    load_pose_manifest,
    load_standardized_manifest,
    loader_preview_pass,
    parse_csv_list,
    parse_shape,
    render_markdown_table,
    repo_relative,
    lookup_row,
    resolve_pose_tensor_path,
    resolve_standardized_frames_path,
    safe_str,
    save_report_pair,
    summarize_counts,
    tensor_check,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check whether the NSLT1000 incremental regions pipeline is ready to run."
    )
    parser.add_argument("--regions-base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--target-source-root", type=Path, default=DEFAULT_TARGET_SOURCE_ROOT)
    parser.add_argument("--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--base-subset", type=str, default=DEFAULT_BASE_SUBSET)
    parser.add_argument("--target-subset", type=str, default=DEFAULT_TARGET_SUBSET)
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--active-regions", type=str, default=",".join(DEFAULT_ACTIVE_REGIONS))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--loader-preview-limit", type=int, default=2)
    parser.add_argument("--tensor-sample-limit", type=int, default=2)
    return parser


def _required_paths(
    *,
    base_root: Path,
    target_source_root: Path,
    target_subset: str,
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [
        ("feasibility_report", FEASIBILITY_REPORT_PATH),
        ("feasibility_script", FEASIBILITY_SCRIPT_PATH),
    ]
    for split in ALLOWED_SPLITS:
        paths.extend(
            [
                (f"base_manifest_{split}", base_root / "manifests" / f"{DEFAULT_BASE_SUBSET}_{split}.csv"),
                (f"base_tensor_dir_{split}", base_root / "tensors" / DEFAULT_BASE_SUBSET / split),
                (f"target_manifest_{split}", target_source_root / "manifests" / f"{target_subset}_{split}.csv"),
                (f"target_frames_dir_{split}", target_source_root / "frames" / target_subset / split),
                (f"target_pose_manifest_{split}", target_source_root.parent / "pose" / "rtmw_l" / "manifests" / f"{target_subset}_{split}.csv"),
                (f"target_pose_dir_{split}", target_source_root.parent / "pose" / "rtmw_l" / "wholebody_133" / target_subset / split),
            ]
        )
    paths.extend(
        [
            ("prepare_regions_script", Path("scripts/preprocess/prepare_regions_branch_inputs.py")),
            ("dataset_loader", Path("src/slr/branches/regions/dataset.py")),
            ("dataset_checker", Path("scripts/verify/check_region_dataset.py")),
            ("crop_config", Path("configs/preprocessing/regions/region_crops_nslt1000.yaml")),
            ("train_config", Path("configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml")),
        ]
    )
    return paths


def _sample_tensor_checks(
    compare: pd.DataFrame,
    *,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    sample_limit: int,
) -> list[dict[str, Any]]:
    reusable = compare[compare["reusable_from_base"]].copy()
    if reusable.empty:
        return []
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        split_frame = reusable[reusable["split_target"] == split].sort_values("sample_id").head(sample_limit)
        for _, row in split_frame.iterrows():
            check = tensor_check(
                row["tensor_path"],
                expected_shape=expected_shape,
                active_regions=active_regions,
                project_root=Path.cwd(),
                data_root=DEFAULT_BASE_ROOT,
            )
            rows.append(
                {
                    "split": split,
                    "sample_id": safe_str(row.get("sample_id")),
                    "tensor_path": safe_str(row.get("tensor_path")),
                    "resolved_tensor_path": check.resolved_path,
                    "exists": check.exists,
                    "valid": check.valid,
                    "shape": check.shape,
                    "region_order": check.region_order,
                    "size_bytes": check.size_bytes,
                    "error": check.error,
                }
            )
    return rows


def _source_input_checks(
    target_manifests: dict[str, pd.DataFrame],
    pose_manifests: dict[str, pd.DataFrame],
    *,
    target_source_root: Path,
    target_subset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}

    for split in ALLOWED_SPLITS:
        target_frame = target_manifests[split]
        pose_lookup = build_lookup_by_sample(pose_manifests[split])
        checked = 0
        missing_frames = 0
        missing_pose = 0
        for _, row in target_frame.iterrows():
            checked += 1
            frames_path = resolve_standardized_frames_path(row, target_source_root=target_source_root, subset=target_subset)
            pose_row = lookup_row(row, pose_lookup)
            pose_path = resolve_pose_tensor_path(
                pose_row if pose_row is not None else row,
                target_source_root=target_source_root,
                subset=target_subset,
            )
            if not frames_path.exists():
                missing_frames += 1
                missing_rows.append(
                    {
                        "split": split,
                        "sample_id": safe_str(row.get("sample_id")),
                        "missing_type": "frames_dir",
                        "path": repo_relative(frames_path),
                    }
                )
            if pose_row is None or not pose_path.exists():
                missing_pose += 1
                missing_rows.append(
                    {
                        "split": split,
                        "sample_id": safe_str(row.get("sample_id")),
                        "missing_type": "pose_path",
                        "path": repo_relative(pose_path),
                    }
                )
        split_summary[split] = {
            "checked_rows": int(checked),
            "missing_frames": int(missing_frames),
            "missing_pose": int(missing_pose),
        }
    summary = {
        "checked_rows_total": int(sum(item["checked_rows"] for item in split_summary.values())),
        "missing_frames_total": int(sum(item["missing_frames"] for item in split_summary.values())),
        "missing_pose_total": int(sum(item["missing_pose"] for item in split_summary.values())),
        "per_split": split_summary,
    }
    return summary, missing_rows


def _build_loader_preview_manifest(compare: pd.DataFrame, manifest_path: Path, *, expected_shape: tuple[int, ...]) -> pd.DataFrame:
    reusable = compare[compare["reusable_from_base"]].sort_values(["split_target", "sample_id"]).copy()
    preview = reusable.head(6).copy()
    preview["video_id"] = preview["video_id_target"]
    preview["class_id"] = preview["class_id_target"].astype(int)
    preview["gloss"] = preview["gloss_target"]
    preview["split"] = preview["split_target"]
    preview["status"] = "ok"
    preview["tensor_shape"] = json.dumps(list(expected_shape))
    preview["preview_path"] = ""
    preview["crop_root"] = ""
    preview["notes"] = "preflight_loader_preview"
    columns = [
        "instance_uid",
        "sample_id",
        "video_id",
        "gloss",
        "class_id",
        "split",
        "tensor_path",
        "crop_root",
        "preview_path",
        "tensor_shape",
        "status",
        "notes",
    ]
    preview = preview.rename(columns={"instance_uid_target": "instance_uid"})
    for column in columns:
        if column not in preview.columns:
            preview[column] = ""
    preview = preview.loc[:, columns]
    preview.to_csv(manifest_path, index=False, encoding="utf-8")
    return preview


def _render_report(summary: dict[str, Any]) -> str:
    status = summary["status"]
    required_rows = [
        [name, "OK" if info["exists"] else "MISSING", info["path"]]
        for name, info in summary["required_paths"].items()
    ]
    overlap = summary["overlap"]
    source_inputs = summary["source_inputs"]
    disk = summary["disk"]
    warnings = summary["warnings"]
    tensor_rows = summary["tensor_samples"]

    lines = [
        "# NSLT1000 Incremental Pipeline Preflight Report",
        "",
        f"Status: `{status}`",
        "",
        "## 1. Required Paths",
        "",
        render_markdown_table(["Item", "Status", "Path"], required_rows),
        "",
        "## 2. Overlap Summary",
        "",
        render_markdown_table(
            ["Split", "Target Rows", "Reusable", "Missing"],
            [
                [split, overlap["per_split"][split]["target_rows"], overlap["per_split"][split]["reused_base_rows"], overlap["per_split"][split]["missing_rows"]]
                for split in ALLOWED_SPLITS
            ],
        ),
        "",
        f"Reusable rows: `{overlap['reused_base_rows']}`",
        f"Missing rows: `{overlap['missing_rows']}`",
        f"Split mismatches: `{overlap['split_mismatch_count']}`",
        f"Class ID mismatches: `{overlap['class_id_mismatch_count']}`",
        f"Gloss mismatches: `{overlap['gloss_mismatch_count']}`",
        "",
        "## 3. Source Inputs",
        "",
        render_markdown_table(
            ["Split", "Checked", "Missing Frames", "Missing Pose"],
            [
                [
                    split,
                    source_inputs["per_split"][split]["checked_rows"],
                    source_inputs["per_split"][split]["missing_frames"],
                    source_inputs["per_split"][split]["missing_pose"],
                ]
                for split in ALLOWED_SPLITS
            ],
        ),
        "",
        f"Missing frames total: `{source_inputs['missing_frames_total']}`",
        f"Missing pose total: `{source_inputs['missing_pose_total']}`",
        "",
        "## 4. Sample Tensor Checks",
        "",
    ]
    if tensor_rows:
        lines.extend(
            [
                render_markdown_table(
                    ["Split", "Sample ID", "Valid", "Shape", "Region Order", "Error"],
                    [
                        [
                            row["split"],
                            row["sample_id"],
                            row["valid"],
                            row["shape"],
                            row["region_order"],
                            row["error"] or "",
                        ]
                        for row in tensor_rows
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["No reusable base tensors were available to sample.", ""])

    lines.extend(
        [
            "## 5. Loader Compatibility",
            "",
            f"Loader preview pass: `{summary['loader_preview']['ok']}`",
            f"Rows loaded: `{summary['loader_preview'].get('rows_loaded', 0)}`",
            f"Batch shape: `{summary['loader_preview'].get('batch_shape', [])}`",
            "",
            "## 6. Disk Estimate",
            "",
            f"Average reusable tensor size: `{disk['mean_tensor_size_human']}`",
            f"Estimated missing-only output size: `{disk['estimated_missing_only_human']}`",
            f"Current free disk: `{disk['free_disk_human']}`",
            "",
            "## 7. Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 8. Conclusion",
            "",
            f"`{status}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    active_regions = parse_csv_list(args.active_regions)
    expected_shape = parse_shape(args.expected_shape)
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    required_paths = {
        name: {"path": repo_relative(path), "exists": bool(path.exists())}
        for name, path in _required_paths(
            base_root=args.regions_base_root,
            target_source_root=args.target_source_root,
            target_subset=args.target_subset,
        )
    }

    target_frame, base_frame, compare = build_overlap_frames(
        base_root=args.regions_base_root,
        target_source_root=args.target_source_root,
        base_subset=args.base_subset,
        target_subset=args.target_subset,
        expected_shape=expected_shape,
        active_regions=active_regions,
        verify_base_payload=False,
    )
    overlap_summary = summarize_counts(compare)
    size_summary = estimate_size_from_base(base_frame, overlap_summary["missing_rows"])
    free_disk_bytes = get_free_disk_bytes(args.incremental_root.parent if args.incremental_root.parent.exists() else Path.cwd())
    size_summary["free_disk_bytes"] = free_disk_bytes
    size_summary["free_disk_human"] = format_size(free_disk_bytes)

    target_manifests = load_manifest_set(load_standardized_manifest, args.target_source_root, args.target_subset)
    pose_manifests = load_manifest_set(load_pose_manifest, args.target_source_root, args.target_subset)
    source_input_summary, missing_input_rows = _source_input_checks(
        target_manifests,
        pose_manifests,
        target_source_root=args.target_source_root,
        target_subset=args.target_subset,
    )

    tensor_rows = _sample_tensor_checks(
        compare,
        expected_shape=expected_shape,
        active_regions=active_regions,
        sample_limit=args.tensor_sample_limit,
    )

    loader_summary: dict[str, Any]
    with TemporaryDirectory(prefix="nslt1000_incremental_preflight_") as temp_dir:
        preview_manifest = Path(temp_dir) / "preview.csv"
        preview_frame = _build_loader_preview_manifest(compare, preview_manifest, expected_shape=expected_shape)
        if preview_frame.empty:
            loader_summary = {"ok": False, "rows_loaded": 0, "batch_shape": [], "error": "no_reusable_rows_for_preview"}
        else:
            try:
                loader_summary = loader_preview_pass(
                    preview_manifest,
                    expected_shape=expected_shape,
                    active_regions=active_regions,
                    num_classes=args.num_classes,
                    limit=args.loader_preview_limit,
                )
            except Exception as exc:
                loader_summary = {"ok": False, "rows_loaded": 0, "batch_shape": [], "error": str(exc)}

    warnings: list[str] = []
    if size_summary["estimated_missing_only_bytes"] > free_disk_bytes:
        warnings.append(
            "Estimated missing-only output size is larger than current free disk. "
            "Implementation can still proceed, but full extraction should not be started yet."
        )
    if source_input_summary["missing_frames_total"] > 0 or source_input_summary["missing_pose_total"] > 0:
        warnings.append("Some source frames or pose files are missing.")
    if overlap_summary["split_mismatch_count"] > 0:
        warnings.append("Overlap split mismatches were found.")
    if overlap_summary["class_id_mismatch_count"] > 0:
        warnings.append("Overlap class_id mismatches were found.")
    if overlap_summary["gloss_mismatch_count"] > 0:
        warnings.append("Overlap gloss mismatches were found.")
    if any(not info["exists"] for info in required_paths.values()):
        warnings.append("One or more required files or directories are missing.")
    if any(not row["valid"] for row in tensor_rows):
        warnings.append("Sampled reusable tensors failed shape or region-order validation.")
    if not loader_summary.get("ok", False):
        warnings.append("Loader preview check failed.")

    ready = (
        all(info["exists"] for info in required_paths.values())
        and overlap_summary["split_mismatch_count"] == 0
        and overlap_summary["class_id_mismatch_count"] == 0
        and overlap_summary["gloss_mismatch_count"] == 0
        and source_input_summary["missing_frames_total"] == 0
        and source_input_summary["missing_pose_total"] == 0
        and all(row["valid"] for row in tensor_rows)
        and bool(loader_summary.get("ok", False))
    )

    summary = {
        "status": "READY" if ready else "NOT READY",
        "required_paths": required_paths,
        "overlap": overlap_summary,
        "source_inputs": source_input_summary,
        "missing_input_examples": missing_input_rows[:25],
        "tensor_samples": tensor_rows,
        "loader_preview": loader_summary,
        "disk": size_summary,
        "warnings": warnings,
        "output_paths": {
            "summary_json": repo_relative(report_root / "preflight_summary.json"),
            "report_md": repo_relative(report_root / "preflight_report.md"),
        },
    }
    report_text = _render_report(summary)
    save_report_pair(
        summary,
        report_text,
        summary_path=report_root / "preflight_summary.json",
        report_path=report_root / "preflight_report.md",
    )
    print(summary["status"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
