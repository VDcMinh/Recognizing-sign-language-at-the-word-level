"""Verify the full NSLT1000 union dataset manifests and tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from scripts.common.regions_nslt1000_incremental_common import (
    ALLOWED_SPLITS,
    DEFAULT_ACTIVE_REGIONS,
    DEFAULT_EXPECTED_SHAPE,
    DEFAULT_REPORT_ROOT,
    DEFAULT_TARGET_SUBSET,
    format_size,
    parse_csv_list,
    parse_shape,
    repo_relative,
    save_report_pair,
    tensor_check,
)
from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify NSLT1000 union manifests and dataset loader compatibility."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset", type=str, default=DEFAULT_TARGET_SUBSET)
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--active-regions", type=str, default=",".join(DEFAULT_ACTIVE_REGIONS))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--check-all-paths", action="store_true")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def _manifest_path(root: Path, subset: str, split: str) -> Path:
    return root / "manifests" / f"{subset}_{split}.csv"


def _verify_loader(manifest_path: Path, split: str, expected_shape: tuple[int, ...], active_regions: list[str], num_classes: int) -> dict[str, Any]:
    dataset = RegionClipDataset(
        manifest_path=manifest_path,
        project_root=Path.cwd(),
        data_root=None,
        split=split,
        expected_shape=expected_shape,
        num_classes=num_classes,
        region_order=DEFAULT_ACTIVE_REGIONS,
        active_regions=active_regions,
        return_metadata=True,
        strict_shape_check=True,
        limit=2,
    )
    loader = DataLoader(dataset, batch_size=min(2, len(dataset)), num_workers=0, shuffle=False, collate_fn=region_collate_fn)
    batch = next(iter(loader))
    return {
        "ok": True,
        "rows_loaded": int(len(dataset)),
        "batch_shape": [int(value) for value in batch["data"].shape],
    }


def _verify_split(
    *,
    root: Path,
    subset: str,
    split: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    num_classes: int,
    check_all_paths: bool,
) -> dict[str, Any]:
    manifest_path = _manifest_path(root, subset, split)
    frame = pd.read_csv(manifest_path)
    frame["sample_id"] = frame["sample_id"].fillna("").astype(str).str.strip()
    frame["status"] = frame["status"].fillna("").astype(str).str.lower()
    frame["tensor_path"] = frame["tensor_path"].fillna("").astype(str)
    duplicate_sample_ids = int(frame["sample_id"].duplicated().sum())
    status_not_ok = int((frame["status"] != "ok").sum())
    class_id_series = pd.to_numeric(frame["class_id"], errors="coerce")
    class_id_out_of_range = int((~class_id_series.between(0, num_classes - 1)).sum())
    invalid_tensor = 0
    missing_tensor = 0
    checked_bytes = 0
    if check_all_paths:
        for _, row in frame.iterrows():
            check = tensor_check(
                row["tensor_path"],
                expected_shape=expected_shape,
                active_regions=active_regions,
                project_root=Path.cwd(),
            )
            checked_bytes += check.size_bytes
            if not check.exists:
                missing_tensor += 1
            elif not check.valid:
                invalid_tensor += 1

    loader_result = _verify_loader(manifest_path, split, expected_shape, active_regions, num_classes)
    return {
        "manifest_path": repo_relative(manifest_path),
        "row_count": int(len(frame)),
        "duplicate_sample_id": duplicate_sample_ids,
        "missing_tensor": int(missing_tensor),
        "invalid_tensor": int(invalid_tensor),
        "status_not_ok": int(status_not_ok),
        "class_id_out_of_range": int(class_id_out_of_range),
        "loader": loader_result,
        "checked_tensor_bytes": int(checked_bytes),
        "checked_tensor_human": format_size(checked_bytes),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# NSLT1000 Union Dataset Verification",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Split | Rows | Duplicate sample_id | Missing Tensor | Invalid Tensor | status != ok | Loader |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in ALLOWED_SPLITS:
        row = summary["splits"][split]
        lines.append(
            f"| {split} | {row['row_count']} | {row['duplicate_sample_id']} | {row['missing_tensor']} | "
            f"{row['invalid_tensor']} | {row['status_not_ok']} | {row['loader']['ok']} |"
        )
    lines.append("")
    lines.append("`VERIFY PASS`" if summary["status"] == "VERIFY PASS" else "`VERIFY FAIL`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    expected_shape = parse_shape(args.expected_shape)
    active_regions = parse_csv_list(args.active_regions)
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    split_results = {
        split: _verify_split(
            root=args.data_root,
            subset=args.subset,
            split=split,
            expected_shape=expected_shape,
            active_regions=active_regions,
            num_classes=args.num_classes,
            check_all_paths=args.check_all_paths,
        )
        for split in ALLOWED_SPLITS
    }
    gloss_map_ok = True
    combined = pd.concat(
        [pd.read_csv(_manifest_path(args.data_root, args.subset, split)) for split in ALLOWED_SPLITS],
        ignore_index=True,
    )
    if not combined.empty:
        mapping_counts = combined.groupby("class_id")["gloss"].nunique(dropna=True)
        gloss_map_ok = bool((mapping_counts <= 1).all())

    verify_pass = (
        all(result["duplicate_sample_id"] == 0 for result in split_results.values())
        and all(result["missing_tensor"] == 0 for result in split_results.values())
        and all(result["invalid_tensor"] == 0 for result in split_results.values())
        and all(result["status_not_ok"] == 0 for result in split_results.values())
        and all(result["class_id_out_of_range"] == 0 for result in split_results.values())
        and all(result["loader"]["ok"] for result in split_results.values())
        and gloss_map_ok
    )
    summary = {
        "status": "VERIFY PASS" if verify_pass else "VERIFY FAIL",
        "splits": split_results,
        "gloss_mapping_ok": gloss_map_ok,
        "output_paths": {
            "summary_json": repo_relative(report_root / "union_verify_summary.json"),
            "report_md": repo_relative(report_root / "union_verify_report.md"),
        },
    }
    report_text = _render_report(summary)
    save_report_pair(
        summary,
        report_text,
        summary_path=report_root / "union_verify_summary.json",
        report_path=report_root / "union_verify_report.md",
    )
    print(summary["status"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
