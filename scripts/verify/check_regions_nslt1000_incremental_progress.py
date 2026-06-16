"""Check progress for incremental NSLT1000 region tensor extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.regions_nslt1000_incremental_common import (
    ALLOWED_SPLITS,
    DEFAULT_ACTIVE_REGIONS,
    DEFAULT_EXPECTED_SHAPE,
    DEFAULT_INCREMENTAL_ROOT,
    DEFAULT_REPORT_ROOT,
    format_size,
    parse_csv_list,
    parse_shape,
    repo_relative,
    save_report_pair,
    tensor_check,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check extraction progress for incremental NSLT1000 region tensors."
    )
    parser.add_argument("--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--active-regions", type=str, default=",".join(DEFAULT_ACTIVE_REGIONS))
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def _manifest_path(root: Path, split: str) -> Path:
    return root / "manifests" / f"nslt1000_missing_{split}.csv"


def _progress_for_split(
    *,
    root: Path,
    split: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
) -> dict[str, Any]:
    manifest_path = _manifest_path(root, split)
    if not manifest_path.exists():
        return {
            "split": split,
            "manifest_exists": False,
            "expected_missing": 0,
            "completed": 0,
            "valid_tensors": 0,
            "invalid_tensors": 0,
            "failed": 0,
            "remaining": 0,
            "completion_percentage": 0.0,
            "disk_used_bytes": 0,
            "estimated_remaining_disk_bytes": 0,
        }

    frame = pd.read_csv(manifest_path)
    frame["status"] = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    frame["tensor_path"] = frame.get("tensor_path", frame.get("expected_tensor_path", pd.Series("", index=frame.index))).fillna("").astype(str)
    valid_tensors = 0
    invalid_tensors = 0
    valid_sizes: list[int] = []
    for _, row in frame.iterrows():
        tensor_path = row.get("tensor_path") or row.get("expected_tensor_path")
        if not str(tensor_path).strip():
            continue
        check = tensor_check(
            tensor_path,
            expected_shape=expected_shape,
            active_regions=active_regions,
            project_root=Path.cwd(),
            data_root=root,
        )
        if check.valid:
            valid_tensors += 1
            valid_sizes.append(check.size_bytes)
        elif check.exists:
            invalid_tensors += 1

    expected_missing = int(len(frame))
    completed = int((frame["status"] == "ok").sum())
    failed = int(frame["status"].isin(["error", "failed", "invalid_existing_tensor"]).sum())
    remaining = int(expected_missing - completed)
    mean_size = (sum(valid_sizes) / len(valid_sizes)) if valid_sizes else 0.0
    estimated_remaining_disk = int(mean_size * max(remaining, 0))
    disk_used = int(sum(valid_sizes))
    return {
        "split": split,
        "manifest_exists": True,
        "expected_missing": expected_missing,
        "completed": completed,
        "valid_tensors": int(valid_tensors),
        "invalid_tensors": int(invalid_tensors),
        "failed": failed,
        "remaining": remaining,
        "completion_percentage": round((completed / expected_missing) * 100.0, 4) if expected_missing else 0.0,
        "disk_used_bytes": disk_used,
        "disk_used_human": format_size(disk_used),
        "estimated_remaining_disk_bytes": estimated_remaining_disk,
        "estimated_remaining_disk_human": format_size(estimated_remaining_disk),
        "manifest_path": repo_relative(manifest_path),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# NSLT1000 Incremental Extraction Progress",
        "",
        "| Split | Expected Missing | Completed | Valid Tensors | Invalid Tensors | Failed | Remaining | Completion % | Disk Used | Est. Remaining Disk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for split in ALLOWED_SPLITS:
        row = summary["splits"][split]
        lines.append(
            f"| {split} | {row['expected_missing']} | {row['completed']} | {row['valid_tensors']} | "
            f"{row['invalid_tensors']} | {row['failed']} | {row['remaining']} | "
            f"{row['completion_percentage']:.2f} | {row['disk_used_human']} | {row['estimated_remaining_disk_human']} |"
        )
    lines.extend(
        [
            "",
            f"Overall completion: `{summary['overall_completion_percentage']:.2f}%`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    expected_shape = parse_shape(args.expected_shape)
    active_regions = parse_csv_list(args.active_regions)
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    split_summary = {
        split: _progress_for_split(
            root=args.incremental_root,
            split=split,
            expected_shape=expected_shape,
            active_regions=active_regions,
        )
        for split in ALLOWED_SPLITS
    }
    expected_total = sum(item["expected_missing"] for item in split_summary.values())
    completed_total = sum(item["completed"] for item in split_summary.values())
    summary = {
        "splits": split_summary,
        "expected_missing_total": int(expected_total),
        "completed_total": int(completed_total),
        "overall_completion_percentage": round((completed_total / expected_total) * 100.0, 4) if expected_total else 0.0,
        "output_paths": {
            "summary_json": repo_relative(report_root / "progress_summary.json"),
            "report_md": repo_relative(report_root / "progress_report.md"),
        },
    }
    report_text = _render_report(summary)
    save_report_pair(
        summary,
        report_text,
        summary_path=report_root / "progress_summary.json",
        report_path=report_root / "progress_report.md",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
