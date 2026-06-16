"""Build full NSLT1000 union manifests from base and incremental region tensors."""

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
    DEFAULT_TARGET_SOURCE_ROOT,
    DEFAULT_TARGET_SUBSET,
    DEFAULT_UNION_ROOT,
    UNION_MANIFEST_COLUMNS,
    build_overlap_frames,
    load_manifest_set,
    load_standardized_manifest,
    normalize_sample_id,
    normalize_sample_id_column,
    parse_shape,
    repo_relative,
    safe_str,
    tensor_check,
    write_dataframe_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build union manifests for NSLT1000 from NSLT300 reuse plus incremental tensors."
    )
    parser.add_argument("--regions-base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--target-source-root", type=Path, default=DEFAULT_TARGET_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_UNION_ROOT)
    parser.add_argument("--base-subset", type=str, default=DEFAULT_BASE_SUBSET)
    parser.add_argument("--target-subset", type=str, default=DEFAULT_TARGET_SUBSET)
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _missing_manifest(root: Path, split: str) -> Path:
    return root / "manifests" / f"nslt1000_missing_{split}.csv"


def _load_incremental_rows(root: Path, split: str, expected_shape: tuple[int, ...]) -> pd.DataFrame:
    path = _missing_manifest(root, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing incremental manifest: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "sample_id": "string",
            "video_id": "string",
            "gloss": "string",
            "split": "string",
            "status": "string",
            "tensor_path": "string",
            "expected_tensor_path": "string",
            "tensor_shape": "string",
        },
    )
    frame = normalize_sample_id_column(frame, frame_name=f"incremental_manifest:{path.name}")
    frame["status"] = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    frame["tensor_path"] = frame.get("tensor_path", frame.get("expected_tensor_path", pd.Series("", index=frame.index))).fillna("").astype(str)
    if "tensor_shape" not in frame.columns:
        frame["tensor_shape"] = json.dumps(list(expected_shape))
    return frame


def _empty_lookup() -> pd.DataFrame:
    return pd.DataFrame().set_index(pd.Index([], name="sample_id"))


def _build_union_frame_for_split(
    *,
    split: str,
    target_manifest: pd.DataFrame,
    base_lookup: pd.DataFrame,
    incremental_lookup: pd.DataFrame,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    base_subset: str,
    target_subset: str,
    incremental_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    union_rows: list[dict[str, Any]] = []
    incomplete_rows: list[str] = []

    for _, row in target_manifest.sort_values(["sample_id", "video_id"]).iterrows():
        sample_id = normalize_sample_id(row["sample_id"])
        common_values = {
            "sample_id": sample_id,
            "video_id": safe_str(row.get("video_id")),
            "class_id": int(row["class_id"]),
            "gloss": safe_str(row.get("gloss")),
            "split": split,
            "region_order": json.dumps(list(active_regions)),
            "source_subset": target_subset,
        }

        if sample_id in base_lookup.index and bool(base_lookup.loc[sample_id, "base_row_valid"]):
            base_row = base_lookup.loc[sample_id]
            tensor_path = repo_relative(base_row["resolved_tensor_path"] or base_row["tensor_path"])
            union_rows.append(
                {
                    **common_values,
                    "tensor_path": tensor_path,
                    "tensor_shape": base_row["tensor_shape"] or json.dumps(list(expected_shape)),
                    "status": "ok",
                    "reuse_source": base_subset,
                    "needs_extraction": False,
                }
            )
            continue

        status = "pending_extraction"
        tensor_path = repo_relative(incremental_root / "tensors" / target_subset / split / f"{sample_id}.npz")
        tensor_shape = json.dumps(list(expected_shape))

        if sample_id in incremental_lookup.index:
            incremental_row = incremental_lookup.loc[sample_id]
            tensor_path = safe_str(
                incremental_row.get("tensor_path") or incremental_row.get("expected_tensor_path"),
                default=tensor_path,
            )
            tensor_shape = safe_str(incremental_row.get("tensor_shape"), default=tensor_shape) or tensor_shape
            status = safe_str(incremental_row.get("status")).lower()
            if status == "ok":
                check = tensor_check(
                    tensor_path,
                    expected_shape=expected_shape,
                    active_regions=active_regions,
                    project_root=Path.cwd(),
                    data_root=incremental_root,
                )
                if not check.valid:
                    status = "invalid_tensor"

        needs_extraction = status != "ok"
        if needs_extraction:
            incomplete_rows.append(sample_id)

        union_rows.append(
            {
                **common_values,
                "tensor_path": repo_relative(tensor_path),
                "tensor_shape": tensor_shape,
                "status": status,
                "reuse_source": "nslt1000_incremental",
                "needs_extraction": needs_extraction,
            }
        )

    union_frame = pd.DataFrame(union_rows)
    if union_frame.empty:
        return pd.DataFrame(columns=UNION_MANIFEST_COLUMNS), incomplete_rows
    return union_frame.loc[:, list(UNION_MANIFEST_COLUMNS)], incomplete_rows


def main() -> int:
    args = build_parser().parse_args()
    expected_shape = parse_shape(args.expected_shape)
    if args.strict and args.allow_incomplete:
        raise ValueError("Choose either --strict or --allow-incomplete, not both.")

    output_manifest_root = args.output_root / "manifests"
    output_manifest_root.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        for split in ALLOWED_SPLITS:
            target = output_manifest_root / f"{args.target_subset}_{split}.csv"
            if target.exists():
                raise FileExistsError(f"Union manifest already exists: {target}. Re-run with --overwrite.")

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
    base_lookup = base_frame.set_index("sample_id", drop=False)
    incomplete_rows: list[str] = []
    union_frames: dict[str, pd.DataFrame] = {}

    for split in ALLOWED_SPLITS:
        incremental_frame = _load_incremental_rows(args.incremental_root, split, expected_shape)
        incremental_lookup = (
            incremental_frame.set_index("sample_id", drop=False)
            if not incremental_frame.empty
            else _empty_lookup()
        )
        union_frame, split_incomplete = _build_union_frame_for_split(
            split=split,
            target_manifest=target_manifests[split],
            base_lookup=base_lookup,
            incremental_lookup=incremental_lookup,
            expected_shape=expected_shape,
            active_regions=list(DEFAULT_ACTIVE_REGIONS),
            base_subset=args.base_subset,
            target_subset=args.target_subset,
            incremental_root=args.incremental_root,
        )
        union_frames[split] = union_frame
        incomplete_rows.extend(split_incomplete)

    if incomplete_rows and not args.allow_incomplete:
        raise RuntimeError(
            "Union manifests were not built in strict-ready form because some missing samples still "
            f"lack valid tensors. Example sample_ids: {sorted(set(incomplete_rows))[:20]}"
        )

    for split, union_frame in union_frames.items():
        write_dataframe_csv(
            union_frame,
            output_manifest_root / f"{args.target_subset}_{split}.csv",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "status": "ok" if not incomplete_rows else "incomplete",
                "output_root": repo_relative(args.output_root),
                "incomplete_count": len(incomplete_rows),
                "incomplete_examples": sorted(set(incomplete_rows))[:20],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
