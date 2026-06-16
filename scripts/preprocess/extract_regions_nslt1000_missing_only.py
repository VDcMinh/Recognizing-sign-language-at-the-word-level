"""Extract only the missing NSLT1000 region tensors incrementally."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common.regions_nslt1000_incremental_common import (
    ALLOWED_SPLITS,
    DEFAULT_ACTIVE_REGIONS,
    DEFAULT_EXPECTED_SHAPE,
    DEFAULT_INCREMENTAL_ROOT,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_TARGET_SOURCE_ROOT,
    DEFAULT_TARGET_SUBSET,
    determine_chunk,
    ensure_incremental_layout,
    format_size,
    get_free_disk_bytes,
    build_lookup_by_sample,
    load_manifest_set,
    load_pose_manifest,
    load_standardized_manifest,
    load_state,
    lookup_row,
    normalize_sample_id,
    normalize_sample_id_column,
    parse_csv_list,
    parse_shape,
    repo_relative,
    save_manifest,
    save_state,
    state_file_path,
    tensor_check,
)
from slr.branches.regions import build_crops


TEXT_COLUMNS = [
    "status",
    "tensor_path",
    "expected_tensor_path",
    "tensor_shape",
    "error_message",
    "processed_at",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Extract only the missing NSLT1000 region tensors from one incremental manifest."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_PREPROCESS_CONFIG)
    parser.add_argument("--target-source-root", type=Path, default=DEFAULT_TARGET_SOURCE_ROOT)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite-invalid", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--next-chunk", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-state-every", type=int, default=1)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--expected-shape", type=str, default=",".join(str(value) for value in DEFAULT_EXPECTED_SHAPE))
    parser.add_argument("--active-regions", type=str, default=",".join(DEFAULT_ACTIVE_REGIONS))
    return parser


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _selected_splits(args: argparse.Namespace) -> list[str]:
    if args.split:
        return [str(args.split).strip().lower()]
    if args.splits:
        return [split for split in parse_csv_list(args.splits) if split in ALLOWED_SPLITS]
    if args.manifest is not None:
        if "train" in args.manifest.name.lower():
            return ["train"]
        if "val" in args.manifest.name.lower():
            return ["val"]
        if "test" in args.manifest.name.lower():
            return ["test"]
    raise ValueError("Provide --split, --splits, or a split-specific --manifest path.")


def _manifest_path_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.manifest is not None:
        return args.manifest
    return args.output_root / "manifests" / f"{DEFAULT_TARGET_SUBSET}_missing_{split}.csv"


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": "string", "video_id": "string", "gloss": "string", "split": "string"})
    if "sample_id" not in frame.columns or "expected_tensor_path" not in frame.columns:
        raise ValueError(f"Missing required columns in incremental manifest: {path}")
    frame = normalize_sample_id_column(frame, frame_name=f"incremental_manifest:{path.name}")
    frame["split"] = frame["split"].fillna("").astype(str).str.strip().str.lower()
    for column in TEXT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].astype("object")
    frame["status"] = frame["status"].fillna("").astype(str).str.strip().str.lower()
    return frame


def _sample_filter(frame: pd.DataFrame, sample_ids_file: Path | None) -> pd.DataFrame:
    if sample_ids_file is None:
        return frame
    sample_ids = {
        normalize_sample_id(line.strip())
        for line in sample_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return frame[frame["sample_id"].isin(sample_ids)].copy().reset_index(drop=True)


def _configure_build_crops(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path]]:
    config = build_crops.load_config(args.config, subset_override=DEFAULT_TARGET_SUBSET)
    config["dataset"]["subset"] = DEFAULT_TARGET_SUBSET
    config["output"]["root"] = args.output_root
    config["output"]["crops_root"] = args.output_root / "crops"
    config["output"]["tensors_root"] = args.output_root / "tensors"
    config["output"]["manifests_root"] = args.output_root / "manifests"
    config["output"]["previews_root"] = args.output_root / "previews"
    config["output"]["reports_root"] = args.output_root / "reports"
    config["output"]["logs_root"] = args.output_root / "logs"
    config["output"]["metadata_path"] = args.output_root / "metadata.json"
    config["crop"]["clip_length"] = int(DEFAULT_EXPECTED_SHAPE[2])
    config["crop"]["crop_size"] = int(DEFAULT_EXPECTED_SHAPE[3])
    config["options"]["overwrite"] = bool(args.overwrite or args.overwrite_invalid)
    config["options"]["save_crops"] = False
    config["options"]["save_tensors"] = not args.dry_run
    config["options"]["save_previews"] = bool(args.save_preview)
    paths = build_crops.resolve_paths(config)
    if not args.dry_run:
        ensure_incremental_layout(args.output_root)
        for key in ("logs_root", "previews_subset_root", "tensors_subset_root", "reports_root"):
            paths[key].mkdir(parents=True, exist_ok=True)
    return config, paths


def _summary_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False)


def _check_free_disk(output_root: Path, min_free_gb: float) -> tuple[int, bool]:
    free_bytes = get_free_disk_bytes(output_root.parent if output_root.parent.exists() else Path.cwd())
    min_free_bytes = int(float(min_free_gb) * (1024**3))
    return free_bytes, free_bytes >= min_free_bytes


def _update_row(frame: pd.DataFrame, sample_id: str, payload: dict[str, Any]) -> None:
    mask = frame["sample_id"] == sample_id
    for key, value in payload.items():
        frame.loc[mask, key] = value


def _process_split(
    *,
    args: argparse.Namespace,
    split: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
) -> dict[str, Any]:
    manifest_path = _manifest_path_for_split(args, split)
    frame = _load_manifest(manifest_path)
    frame = frame[frame["split"] == split].copy().reset_index(drop=True)
    frame = _sample_filter(frame, args.sample_ids_file)
    if frame.empty:
        return {
            "split": split,
            "total_rows": 0,
            "message": "No rows selected.",
        }

    selected_frame, chunk_meta = determine_chunk(
        frame,
        chunk_size=args.chunk_size,
        chunk_index=args.chunk_index,
        start_index=args.start_index,
        max_samples=args.max_samples,
        next_chunk=args.next_chunk,
    )
    standardized_map = load_manifest_set(load_standardized_manifest, args.target_source_root, DEFAULT_TARGET_SUBSET)
    pose_map = load_manifest_set(load_pose_manifest, args.target_source_root, DEFAULT_TARGET_SUBSET)
    standardized_lookup = build_lookup_by_sample(standardized_map[split])
    pose_lookup = build_lookup_by_sample(pose_map[split])
    state_path = state_file_path(args.output_root, split)
    state = load_state(state_path)
    state["split"] = split
    state.setdefault("samples", {})
    config, paths = _configure_build_crops(args)

    free_disk_bytes, disk_ok = _check_free_disk(args.output_root, args.min_free_gb)
    if not disk_ok:
        raise RuntimeError(
            f"Current free disk {format_size(free_disk_bytes)} is below --min-free-gb {args.min_free_gb:.2f}."
        )

    already_completed = int((frame["status"] == "ok").sum())
    newly_processed = 0
    failed = 0
    skipped = 0
    processed_since_save = 0

    for _, row in selected_frame.iterrows():
        sample_id = row["sample_id"]
        state_entry = state["samples"].get(sample_id, {})
        expected_tensor_path = Path(str(row["expected_tensor_path"]))
        existing_check = tensor_check(
            expected_tensor_path,
            expected_shape=expected_shape,
            active_regions=active_regions,
            project_root=Path.cwd(),
            data_root=args.output_root,
        )

        if args.resume and str(state_entry.get("status", "")).lower() == "ok":
            skipped += 1
            continue
        if existing_check.valid and (args.skip_existing or args.resume):
            if not args.dry_run:
                payload = {
                    "status": "ok",
                    "tensor_path": repo_relative(existing_check.resolved_path),
                    "tensor_shape": json.dumps(existing_check.shape),
                    "error_message": "",
                    "processed_at": _now_iso(),
                }
                _update_row(frame, sample_id, payload)
                state["samples"][sample_id] = dict(payload)
            skipped += 1
            continue
        if existing_check.exists and not existing_check.valid and not args.overwrite_invalid:
            payload = {
                "status": "invalid_existing_tensor",
                "tensor_path": repo_relative(existing_check.resolved_path) if existing_check.resolved_path else repo_relative(expected_tensor_path),
                "tensor_shape": json.dumps(existing_check.shape),
                "error_message": existing_check.error or "invalid_existing_tensor",
                "processed_at": _now_iso(),
            }
            if not args.dry_run:
                _update_row(frame, sample_id, payload)
                state["samples"][sample_id] = dict(payload)
            failed += 1
            if args.fail_fast and not args.continue_on_error:
                break
            continue

        standardized_row = lookup_row(row, standardized_lookup)
        if standardized_row is None:
            message = f"Could not find standardized row for sample_id={sample_id}"
            payload = {
                "status": "error",
                "tensor_path": repo_relative(expected_tensor_path),
                "tensor_shape": "",
                "error_message": message,
                "processed_at": _now_iso(),
            }
            if not args.dry_run:
                _update_row(frame, sample_id, payload)
                state["samples"][sample_id] = dict(payload)
            failed += 1
            if args.fail_fast and not args.continue_on_error:
                break
            continue

        pose_row = lookup_row(standardized_row, pose_lookup)
        try:
            if args.dry_run:
                result = {
                    "status": "dry_run",
                    "tensor_path": repo_relative(expected_tensor_path),
                    "tensor_shape": json.dumps(list(expected_shape)),
                    "error_message": "",
                }
            else:
                result, _stats = build_crops.process_sample(
                    standardized_row,
                    pose_row,
                    config,
                    paths,
                    dry_run=False,
                )
        except Exception as exc:
            result = {
                "status": "error",
                "tensor_path": repo_relative(expected_tensor_path),
                "tensor_shape": "",
                "error_message": str(exc),
            }

        if result["status"] == "ok":
            newly_processed += 1
        elif result["status"] == "dry_run":
            skipped += 1
        else:
            failed += 1

        if not args.dry_run:
            payload = {
                "status": result["status"],
                "tensor_path": repo_relative(result.get("tensor_path") or expected_tensor_path),
                "tensor_shape": result.get("tensor_shape", ""),
                "error_message": result.get("error_message", ""),
                "processed_at": _now_iso(),
            }
            _update_row(frame, sample_id, payload)
            state["samples"][sample_id] = dict(payload)
            processed_since_save += 1
            if processed_since_save >= max(1, args.save_state_every):
                save_manifest(frame, manifest_path)
                state["updated_at"] = _now_iso()
                save_state(state_path, state)
                processed_since_save = 0

        if result["status"] != "ok" and args.fail_fast and not args.continue_on_error:
            break

    if not args.dry_run:
        save_manifest(frame, manifest_path)
        state["updated_at"] = _now_iso()
        save_state(state_path, state)

    remaining = int((frame["status"].fillna("").astype(str).str.lower() != "ok").sum())
    estimated_remaining_bytes = 0
    if newly_processed > 0:
        produced_sizes = []
        for sample_id, entry in state["samples"].items():
            if str(entry.get("status", "")).lower() != "ok":
                continue
            tensor_path = entry.get("tensor_path")
            check = tensor_check(
                tensor_path,
                expected_shape=expected_shape,
                active_regions=active_regions,
                project_root=Path.cwd(),
                data_root=args.output_root,
            )
            if check.valid:
                produced_sizes.append(check.size_bytes)
        if produced_sizes:
            estimated_remaining_bytes = int((sum(produced_sizes) / len(produced_sizes)) * remaining)

    summary = {
        "split": split,
        "manifest": repo_relative(manifest_path),
        "total_rows": int(chunk_meta["total_rows"]),
        "current_chunk_range": [int(chunk_meta["chunk_start"]), int(chunk_meta["chunk_end"])],
        "chunk_index": int(chunk_meta["chunk_index"]),
        "selected_rows": int(len(selected_frame)),
        "already_completed": int(already_completed),
        "newly_processed": int(newly_processed),
        "failed": int(failed),
        "skipped": int(skipped),
        "remaining": int(remaining),
        "estimated_disk_required_bytes": int(estimated_remaining_bytes),
        "estimated_disk_required_human": format_size(estimated_remaining_bytes),
        "current_free_disk_bytes": int(free_disk_bytes),
        "current_free_disk_human": format_size(free_disk_bytes),
        "dry_run": bool(args.dry_run),
    }
    return summary


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    expected_shape = parse_shape(args.expected_shape)
    active_regions = parse_csv_list(args.active_regions)
    if active_regions != list(DEFAULT_ACTIVE_REGIONS):
        raise ValueError(f"Expected active_regions={list(DEFAULT_ACTIVE_REGIONS)}, got {active_regions}.")

    summaries = [
        _process_split(
            args=args,
            split=split,
            expected_shape=expected_shape,
            active_regions=active_regions,
        )
        for split in _selected_splits(args)
    ]
    for summary in summaries:
        print(_summary_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
