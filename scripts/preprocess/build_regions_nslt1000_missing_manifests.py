"""Build NSLT1000 missing-only manifests for incremental regions extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    build_overlap_frames,
    build_lookup_by_sample,
    ensure_incremental_layout,
    estimate_size_from_base,
    format_size,
    load_manifest_set,
    load_pose_manifest,
    load_standardized_manifest,
    parse_shape,
    repo_relative,
    lookup_row,
    resolve_pose_tensor_path,
    resolve_standardized_frames_path,
    safe_str,
    save_manifest,
    save_report_pair,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build missing-only NSLT1000 manifests for incremental regions extraction."
    )
    parser.add_argument("--regions-base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--target-source-root", type=Path, default=DEFAULT_TARGET_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--base-subset", type=str, default=DEFAULT_BASE_SUBSET)
    parser.add_argument("--target-subset", type=str, default=DEFAULT_TARGET_SUBSET)
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def _missing_rows_for_split(
    compare: pd.DataFrame,
    *,
    split: str,
    target_manifest: pd.DataFrame,
    pose_manifest: pd.DataFrame,
    target_source_root: Path,
    output_root: Path,
    target_subset: str,
) -> pd.DataFrame:
    pose_lookup = build_lookup_by_sample(pose_manifest)
    missing_ids = set(compare.loc[(compare["split_target"] == split) & (~compare["reusable_from_base"]), "sample_id"].astype(str).tolist())
    rows: list[dict[str, Any]] = []
    for _, row in target_manifest.iterrows():
        sample_id = safe_str(row.get("sample_id"))
        if sample_id not in missing_ids:
            continue
        pose_row = lookup_row(row, pose_lookup)
        frames_path = resolve_standardized_frames_path(row, target_source_root=target_source_root, subset=target_subset)
        pose_path = resolve_pose_tensor_path(
            pose_row if pose_row is not None else row,
            target_source_root=target_source_root,
            subset=target_subset,
        )
        expected_tensor_path = output_root / "tensors" / target_subset / split / f"{sample_id}.npz"
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": safe_str(row.get("video_id")),
                "class_id": int(row.get("class_id")),
                "gloss": safe_str(row.get("gloss")),
                "split": split,
                "source_frames_path": repo_relative(frames_path) if frames_path.exists() else repo_relative(frames_path),
                "pose_path": repo_relative(pose_path) if pose_path.exists() else repo_relative(pose_path),
                "expected_tensor_path": repo_relative(expected_tensor_path),
                "status": "pending_extraction",
                "needs_extraction": True,
                "tensor_path": "",
                "tensor_shape": "",
                "error_message": "",
                "processed_at": "",
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "video_id",
                "class_id",
                "gloss",
                "split",
                "source_frames_path",
                "pose_path",
                "expected_tensor_path",
                "status",
                "needs_extraction",
                "tensor_path",
                "tensor_shape",
                "error_message",
                "processed_at",
            ]
        )
    return output.sort_values(["sample_id", "video_id"]).reset_index(drop=True)


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# NSLT1000 Missing Manifest Summary",
        "",
        f"Total target rows: `{summary['counts']['total_target_rows']}`",
        f"Reusable NSLT300 rows: `{summary['counts']['reused_base_rows']}`",
        f"Missing rows to extract: `{summary['counts']['missing_rows']}`",
        "",
        "## Per Split",
        "",
        "| Split | Missing Rows | Output Manifest |",
        "| --- | ---: | --- |",
    ]
    for split in ALLOWED_SPLITS:
        lines.append(
            f"| {split} | {summary['counts']['per_split'][split]['missing_rows']} | "
            f"{summary['outputs']['manifests'][split]} |"
        )
    lines.extend(
        [
            "",
            "## Estimated Disk",
            "",
            f"Average reusable tensor size: `{summary['disk']['mean_tensor_size_human']}`",
            f"Estimated incremental output size: `{summary['disk']['estimated_missing_only_human']}`",
            "",
            "## Status",
            "",
            "Manifests were built from current source manifests and current reusable NSLT300 tensors.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    expected_shape = parse_shape(args.expected_shape)
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    output_paths = ensure_incremental_layout(args.output_root)

    manifest_paths = {
        split: output_paths["manifests"] / f"{args.target_subset}_missing_{split}.csv"
        for split in ALLOWED_SPLITS
    }
    if not args.overwrite:
        for path in manifest_paths.values():
            if path.exists():
                raise FileExistsError(f"Output manifest already exists: {path}. Re-run with --overwrite.")

    target_frame, base_frame, compare = build_overlap_frames(
        base_root=args.regions_base_root,
        target_source_root=args.target_source_root,
        base_subset=args.base_subset,
        target_subset=args.target_subset,
        expected_shape=expected_shape,
        active_regions=list(DEFAULT_ACTIVE_REGIONS),
        verify_base_payload=False,
    )
    target_manifests = load_manifest_set(load_standardized_manifest, args.target_source_root, args.target_subset)
    pose_manifests = load_manifest_set(load_pose_manifest, args.target_source_root, args.target_subset)

    all_missing_rows: list[pd.DataFrame] = []
    for split in ALLOWED_SPLITS:
        split_frame = _missing_rows_for_split(
            compare,
            split=split,
            target_manifest=target_manifests[split],
            pose_manifest=pose_manifests[split],
            target_source_root=args.target_source_root,
            output_root=args.output_root,
            target_subset=args.target_subset,
        )
        save_manifest(split_frame, manifest_paths[split])
        all_missing_rows.append(split_frame)

    missing_all = pd.concat(all_missing_rows, ignore_index=True) if all_missing_rows else pd.DataFrame()
    missing_all_path = report_root / "missing_samples_all.csv"
    missing_all.to_csv(missing_all_path, index=False, encoding="utf-8")

    counts = {
        "total_target_rows": int(len(target_frame)),
        "reused_base_rows": int(compare["reusable_from_base"].sum()),
        "missing_rows": int((~compare["reusable_from_base"]).sum()),
        "per_split": {
            split: {
                "missing_rows": int(len(frame)),
            }
            for split, frame in zip(ALLOWED_SPLITS, all_missing_rows)
        },
    }
    disk = estimate_size_from_base(base_frame, counts["missing_rows"])
    summary = {
        "counts": counts,
        "disk": disk,
        "outputs": {
            "manifests": {split: repo_relative(path) for split, path in manifest_paths.items()},
            "missing_samples_all_csv": repo_relative(missing_all_path),
            "summary_json": repo_relative(report_root / "missing_summary.json"),
        },
    }
    report_text = _render_report(summary)
    save_report_pair(
        summary,
        report_text,
        summary_path=report_root / "missing_summary.json",
        report_path=report_root / "missing_report.md",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
