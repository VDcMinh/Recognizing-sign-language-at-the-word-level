"""Dataset loader for packaged FullBBox-I3D RGB clips."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from slr.branches.appearance.transforms import build_appearance_transform
from slr.data.validation import require_columns, validate_split_values
from slr.utils.io import read_csv, read_yaml
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)
ALLOWED_SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = (
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "frames_dir",
    "num_frames",
    "status",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class AppearanceSampleRecord:
    """Resolved manifest row for one RGB clip."""

    sample_id: str
    video_id: str
    gloss: str
    class_id: int
    split: str
    frames_dir: Path
    num_frames: int
    manifest_index: int


def _normalize_split(value: Any) -> str:
    """Normalize split values to lower-case train/val/test tokens."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_str(value: Any, default: str = "") -> str:
    """Convert nullable values to stable strings."""

    if value is None:
        return default
    try:
        is_na = pd.isna(value)
    except TypeError:
        is_na = False
    if isinstance(is_na, (bool,)) and is_na:
        return default
    return str(value)


def _parse_int(value: Any, field_name: str) -> int:
    """Convert one manifest value to int with a readable error."""

    if value is None or pd.isna(value):
        raise ValueError(f"{field_name} is missing.")
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be convertible to int, got {value!r}.") from exc


def _resolve_manifest_path(package_root: Path, manifest_path: str | Path) -> Path:
    """Resolve one manifest path relative to the package root when needed."""

    raw = Path(manifest_path)
    return raw.resolve() if raw.is_absolute() else (package_root / raw).resolve()


def _resolve_package_root(package_root: str | Path) -> Path:
    """Resolve package_root to an absolute path."""

    return Path(package_root).resolve()


def _sort_frame_paths(frame_dir: Path) -> list[Path]:
    """Sort frame names numerically when possible, otherwise lexically."""

    frame_paths = [
        path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not frame_paths:
        return []
    if all(path.stem.isdigit() for path in frame_paths):
        return sorted(frame_paths, key=lambda path: int(path.stem))
    return sorted(frame_paths, key=lambda path: path.name)


def _uniform_sample_indices(num_frames: int, clip_len: int) -> list[int]:
    """Deterministically sample `clip_len` positions from a longer clip."""

    if num_frames < clip_len:
        raise ValueError("uniform sampling requires num_frames >= clip_len.")
    if num_frames == clip_len:
        return list(range(num_frames))
    step = float(num_frames - 1) / float(clip_len - 1)
    return [int(round(index * step)) for index in range(clip_len)]


def _random_jittered_indices(num_frames: int, clip_len: int) -> list[int]:
    """Random temporal jitter sampling for training clips."""

    if num_frames < clip_len:
        raise ValueError("random sampling requires num_frames >= clip_len.")
    if num_frames == clip_len:
        return list(range(num_frames))

    boundaries = [int(math.floor(value)) for value in torch.linspace(0, num_frames, steps=clip_len + 1).tolist()]
    indices: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        end = max(start + 1, end)
        if end - start <= 1:
            indices.append(min(start, num_frames - 1))
        else:
            indices.append(random.randint(start, min(end - 1, num_frames - 1)))
    return indices


def _pad_indices(num_frames: int, clip_len: int) -> list[int]:
    """Repeat the last frame until the target clip length is reached."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive to pad clip indices.")
    indices = list(range(num_frames))
    indices.extend([num_frames - 1] * max(0, clip_len - num_frames))
    return indices[:clip_len]


def build_label_maps_from_manifest(manifest: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    """Build `class_id -> gloss` and `gloss -> class_id` mappings from the manifest."""

    if manifest.empty:
        return {}, {}

    require_columns(manifest, ("class_id", "gloss"), name="appearance_manifest")
    working = manifest.loc[:, ["class_id", "gloss"]].copy()
    working["class_id"] = working["class_id"].apply(lambda value: _parse_int(value, "class_id"))
    working["gloss"] = working["gloss"].fillna("").astype(str)

    id_to_gloss: dict[int, str] = {}
    gloss_to_id: dict[str, int] = {}
    for class_id, group in working.groupby("class_id", sort=True):
        glosses = sorted({value.strip() for value in group["gloss"].tolist() if value.strip()})
        gloss = glosses[0] if glosses else ""
        id_to_gloss[int(class_id)] = gloss
        if gloss and gloss not in gloss_to_id:
            gloss_to_id[gloss] = int(class_id)
    return id_to_gloss, gloss_to_id


class AppearanceClipDataset(Dataset):
    """Load standardized full-bbox RGB clips from a Kaggle-ready package."""

    def __init__(
        self,
        *,
        package_root: str | Path,
        manifest_path: str | Path,
        split: str,
        clip_len: int,
        input_size: int,
        sampling_strategy: str = "auto",
        transform_mode: str = "train",
        preprocessing_config: dict[str, Any] | None = None,
        num_classes: int | None = None,
        return_metadata: bool = True,
        limit: int | None = None,
        logger=LOGGER,
    ) -> None:
        self.package_root = _resolve_package_root(package_root)
        self.manifest_path = _resolve_manifest_path(self.package_root, manifest_path)
        self.split = _normalize_split(split)
        self.clip_len = int(clip_len)
        self.input_size = int(input_size)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.transform_mode = str(transform_mode).strip().lower()
        self.preprocessing_config = dict(preprocessing_config or {})
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.return_metadata = bool(return_metadata)
        self.limit = limit
        self.logger = logger

        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r}. Expected one of {ALLOWED_SPLITS}.")
        if self.clip_len <= 0:
            raise ValueError("clip_len must be positive.")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")

        self.transform = build_appearance_transform(
            input_size=self.input_size,
            transform_mode=self.transform_mode,
            config=self.preprocessing_config,
        )
        self.manifest = self._load_manifest()
        self.id_to_gloss, self.gloss_to_id = build_label_maps_from_manifest(self.manifest)
        self.records = self._build_records()
        if not self.records:
            raise ValueError(
                f"No usable samples were found in manifest {self.manifest_path} after filtering."
            )

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict[str, Any],
        *,
        split: str,
        package_root: str | Path | None = None,
        limit: int | None = None,
        return_metadata: bool | None = None,
        logger=LOGGER,
    ) -> "AppearanceClipDataset":
        """Instantiate a dataset from one training config dict or YAML file."""

        if isinstance(config, (str, Path)):
            resolved = read_yaml(config)
        elif isinstance(config, dict):
            resolved = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}.")

        data_cfg = resolved.get("data", {})
        preprocessing_cfg = resolved.get("preprocessing", {})
        manifest_map = {
            "train": data_cfg.get("train_manifest"),
            "val": data_cfg.get("val_manifest"),
            "test": data_cfg.get("test_manifest"),
        }
        manifest_path = manifest_map[_normalize_split(split)]
        effective_package_root = package_root or data_cfg.get("package_root")
        if effective_package_root is None:
            raise ValueError("package_root must be provided either in config.data.package_root or via CLI override.")
        sampling_strategy = str(data_cfg.get("sampling_strategy", "auto"))
        transform_mode = "train" if _normalize_split(split) == "train" else "eval"

        return cls(
            package_root=effective_package_root,
            manifest_path=manifest_path,
            split=split,
            clip_len=int(data_cfg.get("clip_len", 32)),
            input_size=int(data_cfg.get("input_size", 224)),
            sampling_strategy=sampling_strategy,
            transform_mode=transform_mode,
            preprocessing_config=preprocessing_cfg,
            num_classes=resolved.get("model", {}).get("num_classes"),
            return_metadata=True if return_metadata is None else return_metadata,
            limit=limit,
            logger=logger,
        )

    def _load_manifest(self) -> pd.DataFrame:
        """Read, validate, and normalize one packaged manifest CSV."""

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {self.manifest_path}")

        manifest = read_csv(self.manifest_path)
        require_columns(manifest, REQUIRED_COLUMNS, name="appearance_manifest")
        validate_split_values(manifest["split"].fillna("").astype(str).str.strip().str.lower(), context="appearance split values")

        working = manifest.copy()
        working["status"] = working["status"].fillna("").astype(str).str.strip().str.lower()
        working["split"] = working["split"].fillna("").astype(str).str.strip().str.lower()
        working["frames_dir"] = working["frames_dir"].fillna("").astype(str).str.strip()
        working["sample_id"] = working["sample_id"].fillna("").astype(str).str.strip()
        working["video_id"] = working["video_id"].fillna("").astype(str).str.strip()
        working["gloss"] = working["gloss"].fillna("").astype(str).str.strip()

        working = working[working["status"] == "ok"].copy()
        working = working[working["split"] == self.split].copy()
        working = working[working["frames_dir"] != ""].copy()
        working = working.reset_index(drop=True)

        if self.limit is not None:
            working = working.head(int(self.limit)).reset_index(drop=True)
        return working

    def _build_records(self) -> list[AppearanceSampleRecord]:
        """Resolve usable sample records from the filtered manifest."""

        records: list[AppearanceSampleRecord] = []
        for manifest_index, row in self.manifest.iterrows():
            sample_id = _safe_str(row.get("sample_id")).strip()
            video_id = _safe_str(row.get("video_id")).strip()
            gloss = _safe_str(row.get("gloss")).strip()
            class_id = _parse_int(row.get("class_id"), "class_id")
            num_frames = _parse_int(row.get("num_frames"), "num_frames")
            if self.num_classes is not None and not 0 <= class_id < self.num_classes:
                raise ValueError(
                    f"class_id {class_id} is outside the expected range [0, {self.num_classes - 1}]"
                )

            frames_dir = (self.package_root / Path(_safe_str(row.get("frames_dir")).replace("\\", "/"))).resolve()
            if not frames_dir.exists():
                raise FileNotFoundError(
                    f"Resolved frame directory does not exist for sample_id={sample_id}: {frames_dir.as_posix()}"
                )
            frame_paths = _sort_frame_paths(frames_dir)
            if not frame_paths:
                raise FileNotFoundError(
                    f"Frame directory does not contain readable image files for sample_id={sample_id}: {frames_dir.as_posix()}"
                )
            if len(frame_paths) != num_frames:
                raise ValueError(
                    f"Frame count mismatch for sample_id={sample_id}: manifest={num_frames} actual={len(frame_paths)}"
                )

            records.append(
                AppearanceSampleRecord(
                    sample_id=sample_id,
                    video_id=video_id,
                    gloss=gloss,
                    class_id=class_id,
                    split=self.split,
                    frames_dir=frames_dir,
                    num_frames=num_frames,
                    manifest_index=int(manifest_index),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _sample_indices(self, num_frames: int) -> list[int]:
        """Choose temporal indices according to split and sampling strategy."""

        if num_frames >= self.clip_len:
            strategy = self.sampling_strategy
            if strategy == "auto":
                strategy = "random" if self.split == "train" else "uniform"
            if strategy == "random":
                return _random_jittered_indices(num_frames, self.clip_len)
            if strategy in {"uniform", "deterministic"}:
                return _uniform_sample_indices(num_frames, self.clip_len)
            raise ValueError(
                f"Unsupported sampling_strategy {self.sampling_strategy!r}. Expected auto, random, uniform, deterministic."
            )
        return _pad_indices(num_frames, self.clip_len)

    def _load_clip_frames(self, frame_dir: Path, indices: list[int]) -> list[Image.Image]:
        """Load one selected clip from disk as RGB PIL frames."""

        frame_paths = _sort_frame_paths(frame_dir)
        if not frame_paths:
            raise FileNotFoundError(f"No frame files found under {frame_dir.as_posix()}")

        frames: list[Image.Image] = []
        for index in indices:
            frame_path = frame_paths[index]
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
        return frames

    def __getitem__(self, index: int) -> dict[str, Any] | tuple[torch.Tensor, int]:
        record = self.records[index]
        indices = self._sample_indices(record.num_frames)
        frames = self._load_clip_frames(record.frames_dir, indices)
        video = self.transform(frames)
        if tuple(video.shape[:2]) != (3, self.clip_len):
            raise ValueError(
                f"Transformed clip has invalid shape for sample_id={record.sample_id}: {tuple(video.shape)}"
            )

        label = int(record.class_id)
        if not self.return_metadata:
            return video, label

        return {
            "video": video,
            "label": label,
            "sample_id": record.sample_id,
            "gloss": record.gloss,
            "num_frames": int(record.num_frames),
            "video_id": record.video_id,
            "split": record.split,
            "frame_dir": str(record.frames_dir),
            "selected_indices": list(indices),
        }


def appearance_collate_fn(batch: list[dict[str, Any] | tuple[torch.Tensor, int]]) -> dict[str, Any]:
    """Collate appearance samples into `B x C x T x H x W` tensors."""

    if not batch:
        raise ValueError("Batch is empty.")

    first = batch[0]
    if isinstance(first, tuple):
        videos = [item[0] for item in batch]
        labels = [int(item[1]) for item in batch]
        return {
            "video": torch.stack(videos, dim=0),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }

    videos = [item["video"] for item in batch]
    labels = [int(item["label"]) for item in batch]
    return {
        "video": torch.stack(videos, dim=0),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "sample_id": [str(item["sample_id"]) for item in batch],
        "gloss": [str(item["gloss"]) for item in batch],
        "num_frames": [int(item["num_frames"]) for item in batch],
        "video_id": [str(item["video_id"]) for item in batch],
        "split": [str(item["split"]) for item in batch],
        "frame_dir": [str(item["frame_dir"]) for item in batch],
        "selected_indices": [list(item["selected_indices"]) for item in batch],
    }


__all__ = ["AppearanceClipDataset", "AppearanceSampleRecord", "appearance_collate_fn"]
