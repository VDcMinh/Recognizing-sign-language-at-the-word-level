"""Build stable WLASL index manifests from raw metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from slr.data.manifests import (
    CLASS_MAP_COLUMNS,
    MASTER_INSTANCE_COLUMNS,
    SUBSET_MANIFEST_COLUMNS,
)
from slr.data.validation import (
    validate_manifest_schema,
    validate_no_nulls_for_keys,
    validate_split_values,
)
from slr.utils.io import (
    ensure_dir,
    read_json,
    read_text_lines,
    read_yaml,
    write_dataframe_csv,
    write_json,
    write_text,
)
from slr.utils.logging import setup_logger


ALLOWED_SPLITS = ("train", "val", "test")
DEFAULT_CONFIG_PATH = Path("configs/preprocessing/index/index.yaml")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the index stage."""

    parser = argparse.ArgumentParser(
        description="Build stable WLASL index manifests from raw metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the index-layer configuration file.",
    )
    return parser.parse_args()


def path_to_str(path: Path) -> str:
    """Convert a path to a stable POSIX-like string."""

    return path.as_posix()


def normalize_split(value: Any) -> str | None:
    """Normalize split strings to train/val/test when possible."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text == "validation":
        return "val"
    return text


def add_note(notes: list[str], note: str | None) -> None:
    """Append a note once while preserving insertion order."""

    if note and note not in notes:
        notes.append(note)


def split_notes(value: str | None) -> list[str]:
    """Split a note string back into a list."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not value:
        return []
    return [part for part in str(value).split(";") if part]


def join_notes(*note_groups: list[str] | tuple[str, ...] | str | None) -> str:
    """Join multiple note sources into a stable semicolon-separated string."""

    merged: list[str] = []
    for group in note_groups:
        if group is None:
            continue
        if isinstance(group, str):
            items = split_notes(group)
        else:
            items = [item for item in group if item]
        for item in items:
            add_note(merged, item)
    return ";".join(merged)


def normalize_video_id(value: Any, width: int = 5) -> str | None:
    """Normalize a video identifier to a zero-padded string."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        token = str(int(value))
    elif isinstance(value, int):
        token = str(value)
    elif isinstance(value, float):
        if value.is_integer():
            token = str(int(value))
        else:
            token = str(value).strip()
    else:
        token = str(value).strip()

    if not token:
        return None

    token = Path(token).name
    token = Path(token).stem
    return token.zfill(width) if token.isdigit() else token


def resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    """Resolve a possibly-relative path under a base directory."""

    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def infer_nslt_mapping(metadata_dir: Path) -> dict[str, str]:
    """Infer NSLT metadata files from the metadata directory when not configured."""

    mapping: dict[str, str] = {}
    for path in sorted(metadata_dir.glob("nslt_*.json")):
        suffix = path.stem.replace("nslt_", "")
        mapping[f"nslt{suffix}"] = path.name
    return mapping


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and normalize the index-layer configuration."""

    config = read_yaml(config_path)

    if "index" in config and "dataset" not in config:
        index_cfg = config.get("index", {})
        config = {
            "dataset": {
                "name": "WLASL",
                "root": "data/datasets/WLASL",
                "raw_root": "data/datasets/WLASL/raw",
                "metadata_dir": "data/datasets/WLASL/raw/metadata",
                "videos_dir": "data/datasets/WLASL/raw/videos",
            },
            "metadata": {
                "master_file": "WLASL_v0.3.json",
                "class_list_file": Path(index_cfg.get("class_list_file", "wlasl_class_list.txt")).name,
                "missing_file": "missing.txt",
                "nslt_files": index_cfg.get("metadata_filename_map", {}),
            },
            "output": {
                "index_root": index_cfg.get("output_dir", "data/datasets/WLASL/index"),
            },
            "options": {
                "video_id_width": 5,
                "video_extension": ".mp4",
                "keep_nslt_only": True,
                "write_available_manifests": True,
                "fail_on_missing_raw_metadata": True,
            },
        }

    dataset_cfg = config.get("dataset", {})
    metadata_cfg = config.get("metadata", {})
    output_cfg = config.get("output", {})
    options_cfg = config.get("options", {})

    dataset_root = Path(dataset_cfg.get("root", "data/datasets/WLASL"))
    raw_root = Path(dataset_cfg.get("raw_root", dataset_root / "raw"))
    metadata_dir = Path(dataset_cfg.get("metadata_dir", raw_root / "metadata"))
    videos_dir = Path(dataset_cfg.get("videos_dir", raw_root / "videos"))

    nslt_files = metadata_cfg.get("nslt_files") or infer_nslt_mapping(metadata_dir)
    if not isinstance(nslt_files, dict) or not nslt_files:
        raise ValueError("No NSLT metadata files were configured or discovered.")

    master_path = resolve_path(metadata_dir, metadata_cfg.get("master_file"))
    class_list_path = resolve_path(metadata_dir, metadata_cfg.get("class_list_file"))
    missing_path = resolve_path(metadata_dir, metadata_cfg.get("missing_file"))
    nslt_paths = {
        subset: resolve_path(metadata_dir, filename)
        for subset, filename in nslt_files.items()
    }

    resolved = {
        "dataset": {
            "name": dataset_cfg.get("name", "WLASL"),
            "root": dataset_root,
            "raw_root": raw_root,
            "metadata_dir": metadata_dir,
            "videos_dir": videos_dir,
        },
        "metadata": {
            "master_file": master_path,
            "class_list_file": class_list_path,
            "missing_file": missing_path,
            "nslt_files": nslt_paths,
        },
        "output": {
            "index_root": Path(output_cfg.get("index_root", dataset_root / "index")),
        },
        "options": {
            "video_id_width": int(options_cfg.get("video_id_width", 5)),
            "video_extension": str(options_cfg.get("video_extension", ".mp4")),
            "keep_nslt_only": bool(options_cfg.get("keep_nslt_only", True)),
            "write_available_manifests": bool(
                options_cfg.get("write_available_manifests", True)
            ),
            "fail_on_missing_raw_metadata": bool(
                options_cfg.get("fail_on_missing_raw_metadata", True)
            ),
        },
        "config_path": config_path,
    }

    fail_on_missing = resolved["options"]["fail_on_missing_raw_metadata"]
    required_paths = [
        resolved["metadata"]["master_file"],
        resolved["metadata"]["class_list_file"],
        *resolved["metadata"]["nslt_files"].values(),
    ]
    if fail_on_missing:
        missing_required = [path for path in required_paths if path is None or not path.exists()]
        if missing_required:
            raise FileNotFoundError(
                "Required raw metadata files are missing: "
                + ", ".join(path_to_str(path) for path in missing_required if path is not None)
            )

    return resolved


def load_master_metadata(master_path: Path, video_id_width: int) -> tuple[pd.DataFrame, dict[int, str]]:
    """Load and flatten `WLASL_v0.3.json` into a dataframe."""

    payload = read_json(master_path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected master metadata list at {master_path}.")

    rows: list[dict[str, Any]] = []
    master_gloss_by_id: dict[int, str] = {}

    for gloss_id, gloss_entry in enumerate(payload):
        if not isinstance(gloss_entry, dict):
            continue

        gloss = gloss_entry.get("gloss")
        master_gloss_by_id[gloss_id] = gloss
        instances = gloss_entry.get("instances") or []

        if not isinstance(instances, list):
            instances = []

        for instance in instances:
            instance = instance or {}
            notes: list[str] = []
            raw_video_id = instance.get("video_id")
            video_id = normalize_video_id(raw_video_id, width=video_id_width)
            if video_id is None:
                video_id = ""
                add_note(notes, "missing_video_id")
            if video_id and not video_id.isdigit():
                add_note(notes, "invalid_video_id_format")

            split_source = normalize_split(instance.get("split"))
            if split_source not in ALLOWED_SPLITS:
                add_note(notes, "invalid_split")

            for field_name in (
                "bbox",
                "fps",
                "source",
                "url",
                "signer_id",
                "variation_id",
                "split",
                "frame_start",
                "frame_end",
            ):
                if field_name not in instance:
                    add_note(notes, f"missing_{field_name}")

            bbox = instance.get("bbox")

            rows.append(
                {
                    "instance_uid": f"wlasl:{video_id}",
                    "sample_id": video_id,
                    "instance_id": instance.get("instance_id"),
                    "video_id": video_id,
                    "gloss": gloss,
                    "gloss_id": gloss_id,
                    "class_id": gloss_id,
                    "metadata_source": "WLASL_v0.3",
                    "source": instance.get("source"),
                    "url": instance.get("url"),
                    "variation_id": instance.get("variation_id"),
                    "split_source": split_source,
                    "subset_membership": "",
                    "raw_video_filename": None,
                    "raw_video_path": None,
                    "is_present_locally": False,
                    "local_file_size_bytes": None,
                    "start_frame": instance.get("frame_start"),
                    "end_frame": instance.get("frame_end"),
                    "bbox": None if bbox is None else json.dumps(bbox, ensure_ascii=False),
                    "fps": instance.get("fps"),
                    "signer_id": instance.get("signer_id"),
                    "notes": join_notes(notes),
                }
            )

    frame = pd.DataFrame(rows, columns=MASTER_INSTANCE_COLUMNS)
    return frame, master_gloss_by_id


def load_class_list(class_list_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the class-id to gloss mapping file."""

    rows: list[dict[str, Any]] = []
    invalid_lines: list[str] = []

    for line in read_text_lines(class_list_path, drop_empty=True):
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            invalid_lines.append(line)
            continue
        class_id_raw, gloss = parts
        try:
            class_id = int(class_id_raw)
        except ValueError:
            invalid_lines.append(line)
            continue
        rows.append({"class_id": class_id, "class_list_gloss": gloss.strip()})

    frame = pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)
    return frame, invalid_lines


def load_missing_ids(missing_path: Path | None, video_id_width: int) -> set[str]:
    """Load the optional `missing.txt` video-id list."""

    if missing_path is None or not missing_path.exists():
        return set()
    return {
        normalized
        for line in read_text_lines(missing_path, drop_empty=True)
        if (normalized := normalize_video_id(line, width=video_id_width)) is not None
    }


def load_nslt_metadata(
    nslt_paths: dict[str, Path], video_id_width: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all NSLT subset manifests into a long dataframe."""

    rows: list[dict[str, Any]] = []
    source_file_map: dict[str, str] = {}

    for subset_name, path in nslt_paths.items():
        source_file_map[subset_name] = path.name
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected NSLT metadata dictionary at {path}.")

        for raw_video_id, entry in payload.items():
            entry = entry or {}
            notes: list[str] = []
            video_id = normalize_video_id(raw_video_id, width=video_id_width)
            if video_id is None:
                video_id = ""
                add_note(notes, "missing_video_id")
            if video_id and not video_id.isdigit():
                add_note(notes, "invalid_video_id_format")

            split = normalize_split(entry.get("subset"))
            if split not in ALLOWED_SPLITS:
                add_note(notes, "invalid_split")

            action = entry.get("action")
            class_id = None
            start_frame = None
            end_frame = None
            if isinstance(action, list):
                if len(action) >= 1:
                    try:
                        class_id = int(action[0])
                    except (TypeError, ValueError):
                        add_note(notes, "invalid_action_class_id")
                else:
                    add_note(notes, "missing_action_class_id")
                if len(action) >= 2:
                    start_frame = action[1]
                else:
                    add_note(notes, "missing_action_start_frame")
                if len(action) >= 3:
                    end_frame = action[2]
                else:
                    add_note(notes, "missing_action_end_frame")
            else:
                add_note(notes, "invalid_action")

            rows.append(
                {
                    "subset_name": subset_name,
                    "video_id": video_id,
                    "split": split,
                    "class_id": class_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source_file": path.name,
                    "notes": join_notes(notes),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["subset_name", "split", "class_id", "video_id"],
            na_position="last",
        ).reset_index(drop=True)
    return frame, source_file_map


def scan_local_videos(
    videos_dir: Path, video_extension: str, video_id_width: int
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    """Scan local raw videos and build a normalized lookup."""

    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    duplicate_lines: list[str] = []
    suffix = video_extension.lower()

    for path in sorted(videos_dir.iterdir()) if videos_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() != suffix:
            continue

        video_id = normalize_video_id(path.name, width=video_id_width)
        if video_id is None:
            continue

        row = {
            "video_id": video_id,
            "path": path_to_str(path),
            "size_bytes": path.stat().st_size,
            "original_name": path.name,
        }
        rows.append(row)

        if video_id in lookup:
            duplicate_lines.append(
                "LOCAL_VIDEO_ID_COLLISION|"
                f"{video_id}|existing={lookup[video_id]['path']}|duplicate={path_to_str(path)}"
            )
            continue
        lookup[video_id] = row

    frame = pd.DataFrame(rows)
    return frame, lookup, duplicate_lines


def build_master_instances(
    master_frame: pd.DataFrame,
    local_video_lookup: dict[str, dict[str, Any]],
    subset_membership: dict[str, list[str]],
    videos_dir: Path,
    video_extension: str,
) -> pd.DataFrame:
    """Attach local-video and subset-membership information to master rows."""

    rows: list[dict[str, Any]] = []

    for record in master_frame.to_dict(orient="records"):
        video_id = record["video_id"]
        local_info = local_video_lookup.get(video_id)
        raw_filename = f"{video_id}{video_extension}" if video_id else None
        expected_path = videos_dir / raw_filename if raw_filename else videos_dir
        notes = split_notes(record.get("notes"))

        if local_info is None:
            add_note(notes, "video_missing_locally")

        rows.append(
            {
                **record,
                "subset_membership": ";".join(subset_membership.get(video_id, [])),
                "raw_video_filename": raw_filename,
                "raw_video_path": path_to_str(
                    Path(local_info["path"]) if local_info is not None else expected_path
                ),
                "is_present_locally": local_info is not None,
                "local_file_size_bytes": (
                    int(local_info["size_bytes"]) if local_info is not None else None
                ),
                "notes": join_notes(notes),
            }
        )

    frame = pd.DataFrame(rows, columns=MASTER_INSTANCE_COLUMNS)
    frame = frame.sort_values(
        by=["video_id", "class_id", "instance_id"], na_position="last"
    ).reset_index(drop=True)
    return frame


def build_available_and_missing(
    master_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split master instances into available and missing local-video partitions."""

    available = master_frame[master_frame["is_present_locally"]].copy()
    missing = master_frame[~master_frame["is_present_locally"]].copy()
    return available.reset_index(drop=True), missing.reset_index(drop=True)


def build_nslt_only_instances(
    nslt_frame: pd.DataFrame,
    master_frame: pd.DataFrame,
    class_list_map: dict[int, str],
    local_video_lookup: dict[str, dict[str, Any]],
    videos_dir: Path,
    video_extension: str,
) -> pd.DataFrame:
    """Build rows for video IDs that exist only in NSLT manifests."""

    master_ids = set(master_frame["video_id"])
    rows: list[dict[str, Any]] = []

    nslt_only_ids = sorted(set(nslt_frame["video_id"]) - master_ids)

    for video_id in nslt_only_ids:
        subset_rows = nslt_frame[nslt_frame["video_id"] == video_id].copy()
        subsets = sorted(set(subset_rows["subset_name"]))
        splits = sorted({value for value in subset_rows["split"] if value in ALLOWED_SPLITS})
        class_ids = sorted(
            {int(value) for value in subset_rows["class_id"].dropna().tolist()}
        )

        notes: list[str] = ["nslt_id_not_in_WLASL_v0.3"]
        if len(class_ids) > 1:
            add_note(notes, "multiple_nslt_class_ids")
        if len(splits) > 1:
            add_note(notes, "multiple_nslt_splits")

        local_info = local_video_lookup.get(video_id)
        if local_info is None:
            add_note(notes, "video_missing_locally")

        class_id = class_ids[0] if class_ids else None
        gloss = class_list_map.get(class_id) if class_id is not None else None
        raw_filename = f"{video_id}{video_extension}"
        expected_path = videos_dir / raw_filename

        rows.append(
            {
                "instance_uid": f"nslt_only:{video_id}",
                "sample_id": video_id,
                "instance_id": None,
                "video_id": video_id,
                "gloss": gloss,
                "gloss_id": class_id,
                "class_id": class_id,
                "metadata_source": ";".join(subsets),
                "source": None,
                "url": None,
                "variation_id": None,
                "split_source": ";".join(splits),
                "subset_membership": ";".join(subsets),
                "raw_video_filename": raw_filename,
                "raw_video_path": path_to_str(
                    Path(local_info["path"]) if local_info is not None else expected_path
                ),
                "is_present_locally": local_info is not None,
                "local_file_size_bytes": (
                    int(local_info["size_bytes"]) if local_info is not None else None
                ),
                "start_frame": subset_rows["start_frame"].iloc[0] if not subset_rows.empty else None,
                "end_frame": subset_rows["end_frame"].iloc[0] if not subset_rows.empty else None,
                "bbox": None,
                "fps": None,
                "signer_id": None,
                "notes": join_notes(notes, subset_rows["notes"].tolist()),
            }
        )

    frame = pd.DataFrame(rows, columns=MASTER_INSTANCE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            by=["video_id", "class_id"], na_position="last"
        ).reset_index(drop=True)
    return frame


def build_class_map(
    class_list_frame: pd.DataFrame,
    master_frame: pd.DataFrame,
    nslt_frame: pd.DataFrame,
    master_gloss_by_id: dict[int, str],
) -> pd.DataFrame:
    """Build the class-id to gloss table plus per-class coverage stats."""

    class_list_map = dict(
        zip(class_list_frame["class_id"], class_list_frame["class_list_gloss"], strict=False)
    )
    class_ids = sorted(
        set(class_list_map)
        | set(master_gloss_by_id)
        | set(int(value) for value in nslt_frame["class_id"].dropna().tolist())
    )

    master_total = master_frame.groupby("class_id").size().to_dict()
    master_available = (
        master_frame[master_frame["is_present_locally"]].groupby("class_id").size().to_dict()
    )
    subset_counts = {
        subset: nslt_frame[nslt_frame["subset_name"] == subset].groupby("class_id").size().to_dict()
        for subset in sorted(set(nslt_frame["subset_name"]))
    }

    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        class_list_gloss = class_list_map.get(class_id)
        master_gloss = master_gloss_by_id.get(class_id)
        gloss = class_list_gloss or master_gloss

        notes: list[str] = []
        if class_list_gloss is None:
            add_note(notes, "missing_from_class_list")
        if master_gloss is None:
            add_note(notes, "missing_from_master")
        if class_list_gloss and master_gloss and class_list_gloss != master_gloss:
            add_note(notes, "gloss_mismatch")

        total = int(master_total.get(class_id, 0))
        available = int(master_available.get(class_id, 0))

        rows.append(
            {
                "class_id": class_id,
                "gloss_id": class_id,
                "gloss": gloss,
                "class_list_gloss": class_list_gloss,
                "master_gloss": master_gloss,
                "gloss_match": (
                    bool(class_list_gloss == master_gloss)
                    if class_list_gloss is not None and master_gloss is not None
                    else False
                ),
                "master_total": total,
                "master_available": available,
                "master_missing": total - available,
                "nslt100_total": int(subset_counts.get("nslt100", {}).get(class_id, 0)),
                "nslt300_total": int(subset_counts.get("nslt300", {}).get(class_id, 0)),
                "nslt1000_total": int(subset_counts.get("nslt1000", {}).get(class_id, 0)),
                "nslt2000_total": int(subset_counts.get("nslt2000", {}).get(class_id, 0)),
                "notes": join_notes(notes),
            }
        )

    frame = pd.DataFrame(rows, columns=CLASS_MAP_COLUMNS)
    return frame.sort_values("class_id").reset_index(drop=True)


def resolve_unique(values: pd.Series) -> tuple[str | None, list[str]]:
    """Resolve a set of split values into one string plus conflict notes."""

    unique_values = sorted(
        {
            str(value)
            for value in values
            if value is not None and not pd.isna(value) and str(value) != ""
        }
    )
    if not unique_values:
        return None, []
    if len(unique_values) == 1:
        return unique_values[0], []
    return ";".join(unique_values), ["split_conflict"]


def build_video_to_split(
    master_frame: pd.DataFrame,
    nslt_frame: pd.DataFrame,
    local_video_lookup: dict[str, dict[str, Any]],
    subset_names: list[str],
    include_nslt_only: bool = False,
) -> pd.DataFrame:
    """Build master-only or master-plus-NSLT split lookup tables."""

    master_ids = set(master_frame["video_id"])
    nslt_ids = set(nslt_frame["video_id"])
    video_ids = sorted(master_ids | nslt_ids) if include_nslt_only else sorted(master_ids)

    rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        master_rows = master_frame[master_frame["video_id"] == video_id]
        nslt_rows = nslt_frame[nslt_frame["video_id"] == video_id]
        notes: list[str] = []

        master_split, master_split_notes = resolve_unique(master_rows["split_source"])
        for note in master_split_notes:
            add_note(notes, f"master_{note}")

        subset_split_values: dict[str, str | None] = {}
        for subset in subset_names:
            subset_rows = nslt_rows[nslt_rows["subset_name"] == subset]
            subset_split, subset_split_notes = resolve_unique(subset_rows["split"])
            subset_split_values[subset] = subset_split
            for note in subset_split_notes:
                add_note(notes, f"{subset}_{note}")
            if master_split and subset_split and master_split != subset_split:
                add_note(notes, f"{subset}_split_mismatch")

        is_in_master = video_id in master_ids
        is_in_any_nslt = video_id in nslt_ids
        is_present_locally = video_id in local_video_lookup

        if not is_in_master:
            add_note(notes, "nslt_id_not_in_WLASL_v0.3")
        if not is_present_locally:
            add_note(notes, "video_missing_locally")

        row = {
            "video_id": video_id,
            "master_split": master_split,
            "is_in_master": is_in_master,
            "is_in_any_nslt": is_in_any_nslt,
            "is_present_locally": is_present_locally,
            "notes": join_notes(notes),
        }
        for subset in subset_names:
            row[f"{subset}_split"] = subset_split_values.get(subset)
        rows.append(row)

    ordered_columns = [
        "video_id",
        "master_split",
        *[f"{subset}_split" for subset in subset_names],
        "is_in_master",
        "is_in_any_nslt",
        "is_present_locally",
        "notes",
    ]
    frame = pd.DataFrame(rows, columns=ordered_columns)
    frame = frame.rename(
        columns={f"{subset}_split": f"{subset}_split" for subset in subset_names}
    )
    return frame.sort_values("video_id").reset_index(drop=True)


def build_subset_manifests(
    master_frame: pd.DataFrame,
    nslt_frame: pd.DataFrame,
    class_list_map: dict[int, str],
    nslt_source_files: dict[str, str],
    local_video_lookup: dict[str, dict[str, Any]],
    videos_dir: Path,
    video_extension: str,
    subset_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Build full and available manifests for each NSLT subset."""

    master_lookup = (
        master_frame.drop_duplicates("video_id").set_index("video_id").to_dict(orient="index")
    )
    subset_outputs: dict[str, dict[str, Any]] = {}

    for subset_name in subset_names:
        subset_rows = nslt_frame[nslt_frame["subset_name"] == subset_name].copy()
        rows: list[dict[str, Any]] = []

        for record in subset_rows.to_dict(orient="records"):
            video_id = record["video_id"]
            master_row = master_lookup.get(video_id)
            notes = split_notes(record.get("notes"))
            local_info = local_video_lookup.get(video_id)

            if master_row is None:
                add_note(notes, "nslt_id_not_in_WLASL_v0.3")
            if local_info is None:
                add_note(notes, "video_missing_locally")

            if master_row is not None:
                if (
                    master_row.get("split_source")
                    and record.get("split")
                    and master_row["split_source"] != record["split"]
                ):
                    add_note(notes, "master_nslt_split_mismatch")
                if (
                    master_row.get("class_id") is not None
                    and record.get("class_id") is not None
                    and int(master_row["class_id"]) != int(record["class_id"])
                ):
                    add_note(notes, "master_nslt_class_id_mismatch")

            raw_filename = f"{video_id}{video_extension}"
            expected_path = videos_dir / raw_filename
            gloss = (
                master_row.get("gloss")
                if master_row is not None and master_row.get("gloss") is not None
                else class_list_map.get(record.get("class_id"))
            )

            rows.append(
                {
                    "instance_uid": (
                        master_row["instance_uid"]
                        if master_row is not None
                        else f"nslt_only:{video_id}"
                    ),
                    "sample_id": video_id,
                    "instance_id": (
                        master_row.get("instance_id") if master_row is not None else None
                    ),
                    "video_id": video_id,
                    "gloss": gloss,
                    "class_id": record.get("class_id"),
                    "split": record.get("split"),
                    "video_path": path_to_str(
                        Path(local_info["path"]) if local_info is not None else expected_path
                    ),
                    "is_present_locally": local_info is not None,
                    "start_frame": record.get("start_frame"),
                    "end_frame": record.get("end_frame"),
                    "master_start_frame": (
                        master_row.get("start_frame") if master_row is not None else None
                    ),
                    "master_end_frame": (
                        master_row.get("end_frame") if master_row is not None else None
                    ),
                    "bbox": master_row.get("bbox") if master_row is not None else None,
                    "fps": master_row.get("fps") if master_row is not None else None,
                    "signer_id": master_row.get("signer_id") if master_row is not None else None,
                    "source": master_row.get("source") if master_row is not None else None,
                    "url": master_row.get("url") if master_row is not None else None,
                    "notes": join_notes(notes),
                }
            )

        manifest = pd.DataFrame(rows, columns=SUBSET_MANIFEST_COLUMNS)
        if not manifest.empty:
            manifest = manifest.sort_values(
                by=["split", "class_id", "video_id"], na_position="last"
            ).reset_index(drop=True)

        available = manifest[manifest["is_present_locally"]].copy().reset_index(drop=True)

        class_ids = sorted(
            int(value) for value in subset_rows["class_id"].dropna().astype(int).unique().tolist()
        )
        id_to_gloss = {
            str(class_id): class_list_map.get(class_id, "")
            for class_id in class_ids
        }
        label_map = {
            "subset": subset_name,
            "source_file": nslt_source_files[subset_name],
            "num_classes": len(class_ids),
            "class_ids": class_ids,
            "id_to_gloss": id_to_gloss,
            "gloss_to_id": {
                gloss: class_id
                for class_id, gloss in ((cid, id_to_gloss[str(cid)]) for cid in class_ids)
                if gloss
            },
        }

        subset_outputs[subset_name] = {
            "all": manifest,
            "available": available,
            "label_map": label_map,
        }

    return subset_outputs


def build_reports(
    class_map_frame: pd.DataFrame,
    master_frame: pd.DataFrame,
    available_frame: pd.DataFrame,
    missing_frame: pd.DataFrame,
    nslt_only_frame: pd.DataFrame,
    subset_manifests: dict[str, dict[str, Any]],
    local_videos_frame: pd.DataFrame,
    missing_txt_ids: set[str],
) -> dict[str, Any]:
    """Build report payloads for markdown, JSON, and invalid-id text output."""

    referenced_ids = set(master_frame["video_id"]) | {
        video_id
        for payload in subset_manifests.values()
        for video_id in payload["all"]["video_id"].tolist()
    }
    local_unreferenced = (
        local_videos_frame[~local_videos_frame["video_id"].isin(referenced_ids)]
        .sort_values("video_id")
        .reset_index(drop=True)
        if not local_videos_frame.empty
        else pd.DataFrame(columns=["video_id", "path", "size_bytes", "original_name"])
    )

    coverage_by_split: dict[str, Any] = {"master": {}, "subsets": {}}
    for split in ALLOWED_SPLITS:
        total = int((master_frame["split_source"] == split).sum())
        available = int(
            ((master_frame["split_source"] == split) & master_frame["is_present_locally"]).sum()
        )
        coverage_by_split["master"][split] = {
            "total": total,
            "available": available,
            "missing": total - available,
        }
    coverage_by_split["master"]["overall"] = {
        "total": int(len(master_frame)),
        "available": int(len(available_frame)),
        "missing": int(len(missing_frame)),
    }

    for subset_name, payload in subset_manifests.items():
        manifest = payload["all"]
        available_manifest = payload["available"]
        coverage_by_split["subsets"][subset_name] = {}
        for split in ALLOWED_SPLITS:
            total = int((manifest["split"] == split).sum())
            available = int((available_manifest["split"] == split).sum())
            coverage_by_split["subsets"][subset_name][split] = {
                "total": total,
                "available": available,
                "missing": total - available,
            }
        coverage_by_split["subsets"][subset_name]["overall"] = {
            "total": int(len(manifest)),
            "available": int(len(available_manifest)),
            "missing": int(len(manifest) - len(available_manifest)),
        }

    coverage_by_class = class_map_frame.loc[
        :,
        [
            "class_id",
            "class_list_gloss",
            "master_gloss",
            "gloss_match",
            "master_total",
            "master_available",
            "master_missing",
            "nslt100_total",
            "nslt300_total",
            "nslt1000_total",
            "nslt2000_total",
        ],
    ].to_dict(orient="records")

    duplicated_instance_ids = Counter(
        int(value)
        for value in master_frame["instance_id"].dropna().tolist()
    )
    duplicated_instance_ids = {
        instance_id: count
        for instance_id, count in duplicated_instance_ids.items()
        if count > 1
    }

    actual_missing_master_ids = set(missing_frame["video_id"])
    missing_txt_lists_present = sorted(missing_txt_ids & set(available_frame["video_id"]))
    missing_txt_omits_missing = sorted(actual_missing_master_ids - missing_txt_ids)

    invalid_lines: list[str] = []
    for record in missing_frame.to_dict(orient="records"):
        invalid_lines.append(
            "MASTER_VIDEO_MISSING_LOCALLY|"
            f"{record['video_id']}|{record['instance_uid']}|gloss={record['gloss']}"
        )
    for record in nslt_only_frame.to_dict(orient="records"):
        invalid_lines.append(
            "NSLT_ID_NOT_IN_MASTER|"
            f"{record['video_id']}|{record['instance_uid']}|subsets={record['subset_membership']}"
        )
        if not bool(record["is_present_locally"]):
            invalid_lines.append(
                "NSLT_ONLY_VIDEO_MISSING_LOCALLY|"
                f"{record['video_id']}|{record['instance_uid']}|subsets={record['subset_membership']}"
            )
    for record in local_unreferenced.to_dict(orient="records"):
        invalid_lines.append(
            "LOCAL_VIDEO_NOT_REFERENCED_BY_METADATA|"
            f"{record['video_id']}|path={record['path']}"
        )
    for record in class_map_frame[class_map_frame["notes"].str.contains("gloss_mismatch", na=False)].to_dict(
        orient="records"
    ):
        invalid_lines.append(
            "GLOSS_MISMATCH|"
            f"class_id={record['class_id']}|class_list={record['class_list_gloss']}|master={record['master_gloss']}"
        )
    for instance_id, count in sorted(duplicated_instance_ids.items()):
        invalid_lines.append(
            f"DUPLICATED_INSTANCE_ID|instance_id={instance_id}|count={count}"
        )
    for video_id in missing_txt_lists_present:
        invalid_lines.append(
            f"MISSING_TXT_LISTS_PRESENT_VIDEO|{video_id}|path=present_locally"
        )
    for video_id in missing_txt_omits_missing:
        invalid_lines.append(
            f"MISSING_TXT_OMITS_MASTER_MISSING|{video_id}|status=missing_locally"
        )

    summary_lines = [
        "# WLASL Index Summary",
        "",
        "## Overview",
        "",
        f"- Master instances: {len(master_frame)}",
        f"- Available local videos (master rows): {len(available_frame)}",
        f"- Missing local videos (master rows): {len(missing_frame)}",
        f"- NSLT-only rows: {len(nslt_only_frame)}",
        f"- Local videos not referenced by metadata: {len(local_unreferenced)}",
        f"- Duplicate `instance_id` values: {len(duplicated_instance_ids)}",
        f"- Gloss mismatches between class list and master: {int(class_map_frame['notes'].str.contains('gloss_mismatch', na=False).sum())}",
        "- Video ID normalization: strip path/suffix, keep stem, zero-pad to 5 digits when numeric.",
        "",
        "## Subset Coverage",
        "",
    ]

    for subset_name, payload in subset_manifests.items():
        manifest = payload["all"]
        available_manifest = payload["available"]
        summary_lines.append(f"### {subset_name}")
        summary_lines.append("")
        for split in ALLOWED_SPLITS:
            total = int((manifest["split"] == split).sum())
            available = int((available_manifest["split"] == split).sum())
            summary_lines.append(
                f"- {split}: total={total}, available={available}, missing={total - available}"
            )
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Warnings",
            "",
            f"- Missing master-local videos: {len(missing_frame)}",
            f"- NSLT-only IDs: {len(nslt_only_frame)}",
            f"- Duplicate `instance_id` groups: {len(duplicated_instance_ids)}",
            f"- Gloss mismatches: {int(class_map_frame['notes'].str.contains('gloss_mismatch', na=False).sum())}",
        ]
    )

    if missing_txt_lists_present or missing_txt_omits_missing:
        summary_lines.append(
            f"- `missing.txt` discrepancies: listed-present={len(missing_txt_lists_present)}, omitted-missing={len(missing_txt_omits_missing)}"
        )

    return {
        "dataset_summary_md": "\n".join(summary_lines) + "\n",
        "coverage_by_split": coverage_by_split,
        "coverage_by_class": coverage_by_class,
        "invalid_ids_text": "\n".join(sorted(invalid_lines)) + ("\n" if invalid_lines else ""),
        "local_unreferenced": local_unreferenced,
    }


def write_outputs(
    index_root: Path,
    master_frame: pd.DataFrame,
    available_frame: pd.DataFrame,
    missing_frame: pd.DataFrame,
    nslt_only_frame: pd.DataFrame,
    class_map_frame: pd.DataFrame,
    video_to_split_frame: pd.DataFrame,
    video_to_split_all_frame: pd.DataFrame,
    subset_manifests: dict[str, dict[str, Any]],
    reports: dict[str, Any],
    write_available_manifests: bool,
) -> int:
    """Write all index-layer outputs and return the number of files created."""

    reports_dir = ensure_dir(index_root / "reports")
    subsets_dir = ensure_dir(index_root / "subsets")
    subsets_available_dir = ensure_dir(index_root / "subsets_available")

    written_files = 0

    write_dataframe_csv(master_frame, index_root / "master_instances.csv")
    write_dataframe_csv(available_frame, index_root / "available_instances.csv")
    write_dataframe_csv(missing_frame, index_root / "missing_instances.csv")
    write_dataframe_csv(nslt_only_frame, index_root / "nslt_only_instances.csv")
    write_dataframe_csv(class_map_frame, index_root / "class_id_to_gloss.csv")
    write_dataframe_csv(video_to_split_frame, index_root / "video_to_split.csv")
    write_dataframe_csv(video_to_split_all_frame, index_root / "video_to_split_all.csv")
    written_files += 7

    for subset_name, payload in subset_manifests.items():
        subset_root = ensure_dir(subsets_dir / subset_name)
        manifest = payload["all"]
        for split in ALLOWED_SPLITS:
            write_dataframe_csv(
                manifest[manifest["split"] == split].reset_index(drop=True),
                subset_root / f"{split}.csv",
            )
            written_files += 1
        write_json(payload["label_map"], subset_root / "label_map.json")
        written_files += 1

        if write_available_manifests:
            subset_available_root = ensure_dir(subsets_available_dir / subset_name)
            available_manifest = payload["available"]
            for split in ALLOWED_SPLITS:
                write_dataframe_csv(
                    available_manifest[available_manifest["split"] == split].reset_index(drop=True),
                    subset_available_root / f"{split}.csv",
                )
                written_files += 1
            write_json(payload["label_map"], subset_available_root / "label_map.json")
            written_files += 1

    write_text(reports["dataset_summary_md"], reports_dir / "dataset_summary.md")
    write_json(reports["coverage_by_split"], reports_dir / "coverage_by_split.json")
    write_json(reports["coverage_by_class"], reports_dir / "coverage_by_class.json")
    write_text(reports["invalid_ids_text"], reports_dir / "invalid_ids.txt")
    written_files += 4

    return written_files


def run(config_path: Path) -> int:
    """Run the end-to-end WLASL index build."""

    config = load_config(config_path)
    index_root = ensure_dir(config["output"]["index_root"])
    logs_dir = ensure_dir(index_root / "logs")
    logger = setup_logger(__name__, logs_dir / "build_index.log")

    logger.info("Starting WLASL index build.")
    logger.info("Config path: %s", path_to_str(config["config_path"]))

    metadata_cfg = config["metadata"]
    options_cfg = config["options"]
    dataset_cfg = config["dataset"]
    subset_names = list(metadata_cfg["nslt_files"].keys())

    logger.info("Master metadata: %s", path_to_str(metadata_cfg["master_file"]))
    logger.info("Class list: %s", path_to_str(metadata_cfg["class_list_file"]))
    for subset_name, path in metadata_cfg["nslt_files"].items():
        logger.info("NSLT metadata [%s]: %s", subset_name, path_to_str(path))

    class_list_frame, invalid_class_lines = load_class_list(metadata_cfg["class_list_file"])
    master_base_frame, master_gloss_by_id = load_master_metadata(
        metadata_cfg["master_file"],
        video_id_width=options_cfg["video_id_width"],
    )
    nslt_frame, nslt_source_files = load_nslt_metadata(
        metadata_cfg["nslt_files"],
        video_id_width=options_cfg["video_id_width"],
    )
    local_videos_frame, local_video_lookup, local_duplicate_lines = scan_local_videos(
        dataset_cfg["videos_dir"],
        video_extension=options_cfg["video_extension"],
        video_id_width=options_cfg["video_id_width"],
    )
    missing_txt_ids = load_missing_ids(
        metadata_cfg["missing_file"],
        video_id_width=options_cfg["video_id_width"],
    )

    subset_membership = (
        nslt_frame.groupby("video_id")["subset_name"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
        if not nslt_frame.empty
        else {}
    )
    class_list_map = dict(
        zip(class_list_frame["class_id"], class_list_frame["class_list_gloss"], strict=False)
    )

    master_frame = build_master_instances(
        master_base_frame,
        local_video_lookup=local_video_lookup,
        subset_membership=subset_membership,
        videos_dir=dataset_cfg["videos_dir"],
        video_extension=options_cfg["video_extension"],
    )
    available_frame, missing_frame = build_available_and_missing(master_frame)
    nslt_only_frame = build_nslt_only_instances(
        nslt_frame,
        master_frame=master_frame,
        class_list_map=class_list_map,
        local_video_lookup=local_video_lookup,
        videos_dir=dataset_cfg["videos_dir"],
        video_extension=options_cfg["video_extension"],
    )
    class_map_frame = build_class_map(
        class_list_frame,
        master_frame=master_frame,
        nslt_frame=nslt_frame,
        master_gloss_by_id=master_gloss_by_id,
    )
    video_to_split_frame = build_video_to_split(
        master_frame,
        nslt_frame=nslt_frame,
        local_video_lookup=local_video_lookup,
        subset_names=subset_names,
        include_nslt_only=False,
    )
    video_to_split_all_frame = build_video_to_split(
        master_frame,
        nslt_frame=nslt_frame,
        local_video_lookup=local_video_lookup,
        subset_names=subset_names,
        include_nslt_only=True,
    )
    subset_manifests = build_subset_manifests(
        master_frame,
        nslt_frame=nslt_frame,
        class_list_map=class_list_map,
        nslt_source_files=nslt_source_files,
        local_video_lookup=local_video_lookup,
        videos_dir=dataset_cfg["videos_dir"],
        video_extension=options_cfg["video_extension"],
        subset_names=subset_names,
    )

    master_frame = validate_manifest_schema(master_frame, MASTER_INSTANCE_COLUMNS, name="master_instances")
    available_frame = validate_manifest_schema(
        available_frame, MASTER_INSTANCE_COLUMNS, name="available_instances"
    )
    missing_frame = validate_manifest_schema(
        missing_frame, MASTER_INSTANCE_COLUMNS, name="missing_instances"
    )
    nslt_only_frame = validate_manifest_schema(
        nslt_only_frame, MASTER_INSTANCE_COLUMNS, name="nslt_only_instances"
    )
    class_map_frame = validate_manifest_schema(
        class_map_frame, CLASS_MAP_COLUMNS, name="class_id_to_gloss"
    )

    expected_video_to_split_columns = [
        "video_id",
        "master_split",
        *[f"{subset}_split" for subset in subset_names],
        "is_in_master",
        "is_in_any_nslt",
        "is_present_locally",
        "notes",
    ]
    video_to_split_frame = validate_manifest_schema(
        video_to_split_frame, expected_video_to_split_columns, name="video_to_split"
    )
    video_to_split_all_frame = validate_manifest_schema(
        video_to_split_all_frame,
        expected_video_to_split_columns,
        name="video_to_split_all",
    )
    validate_no_nulls_for_keys(master_frame, ("instance_uid", "sample_id", "video_id"), name="master_instances")
    validate_split_values(master_frame["split_source"], context="master split values")
    validate_split_values(nslt_frame["split"], context="NSLT split values")

    for subset_name, payload in subset_manifests.items():
        payload["all"] = validate_manifest_schema(
            payload["all"],
            SUBSET_MANIFEST_COLUMNS,
            name=f"subsets/{subset_name}",
        )
        payload["available"] = validate_manifest_schema(
            payload["available"],
            SUBSET_MANIFEST_COLUMNS,
            name=f"subsets_available/{subset_name}",
        )
        validate_split_values(
            payload["all"]["split"], context=f"{subset_name} split values"
        )

    reports = build_reports(
        class_map_frame=class_map_frame,
        master_frame=master_frame,
        available_frame=available_frame,
        missing_frame=missing_frame,
        nslt_only_frame=nslt_only_frame,
        subset_manifests=subset_manifests,
        local_videos_frame=local_videos_frame,
        missing_txt_ids=missing_txt_ids,
    )

    if invalid_class_lines:
        reports["invalid_ids_text"] += "".join(
            f"INVALID_CLASS_LIST_LINE|{line}\n" for line in invalid_class_lines
        )
    if local_duplicate_lines:
        reports["invalid_ids_text"] += "".join(f"{line}\n" for line in local_duplicate_lines)

    written_files = write_outputs(
        index_root=index_root,
        master_frame=master_frame,
        available_frame=available_frame,
        missing_frame=missing_frame,
        nslt_only_frame=nslt_only_frame,
        class_map_frame=class_map_frame,
        video_to_split_frame=video_to_split_frame,
        video_to_split_all_frame=video_to_split_all_frame,
        subset_manifests=subset_manifests,
        reports=reports,
        write_available_manifests=options_cfg["write_available_manifests"],
    )

    logger.info("Master rows: %s", len(master_frame))
    logger.info("Available rows: %s", len(available_frame))
    logger.info("Missing rows: %s", len(missing_frame))
    logger.info("NSLT-only rows: %s", len(nslt_only_frame))
    for subset_name, payload in subset_manifests.items():
        manifest = payload["all"]
        available_manifest = payload["available"]
        for split in ALLOWED_SPLITS:
            logger.info(
                "Subset %s split %s: total=%s available=%s",
                subset_name,
                split,
                int((manifest["split"] == split).sum()),
                int((available_manifest["split"] == split).sum()),
            )
    logger.info(
        "Warnings: missing_master=%s nslt_only=%s local_unref=%s gloss_mismatch=%s",
        len(missing_frame),
        len(nslt_only_frame),
        len(reports["local_unreferenced"]),
        int(class_map_frame["notes"].str.contains("gloss_mismatch", na=False).sum()),
    )
    logger.info("Wrote %s output files under %s", written_files, path_to_str(index_root))
    logger.info("Finished WLASL index build.")
    return 0


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    run(args.config)
