"""Dataset interfaces for region branch tensors."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from slr.branches.regions.region_schema import DEFAULT_TENSOR_SHAPE, REGION_NAMES
from slr.branches.regions.transforms import (
    apply_region_dataset_normalization,
    apply_region_clip_augmentation,
    normalize_region_normalization_config,
    normalize_region_augmentation_config,
)
from slr.data.manifests import REGION_INPUT_MANIFEST_COLUMNS
from slr.data.validation import require_columns
from slr.utils.io import read_csv, read_yaml, remap_wlasl_path
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)
ALLOWED_SPLITS = ("train", "val", "test")
REQUIRED_DATASET_COLUMNS = (
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "tensor_path",
    "status",
)


@dataclass(frozen=True)
class RegionSampleRecord:
    """Resolved manifest row used by :class:`RegionClipDataset`."""

    sample_id: str
    video_id: str
    gloss: str
    class_id: int
    split: str
    tensor_path: Path
    preview_path: str
    crop_root: str
    manifest_index: int
    notes: str = ""


def build_label_maps_from_manifest(
    manifest: pd.DataFrame,
    *,
    logger=LOGGER,
) -> tuple[dict[int, str], dict[str, int]]:
    """Build ``class_id -> gloss`` and ``gloss -> class_id`` mappings from one manifest."""

    if manifest.empty:
        return {}, {}

    require_columns(manifest, ("class_id", "gloss"), name="region_manifest")
    working = manifest.loc[:, ["class_id", "gloss"]].copy()
    working["class_id"] = working["class_id"].apply(lambda value: _parse_int(value, "class_id"))
    working["gloss"] = working["gloss"].fillna("").astype(str)

    id_to_gloss: dict[int, str] = {}
    gloss_to_id: dict[str, int] = {}
    duplicate_glosses: list[str] = []

    for class_id, group in working.groupby("class_id", sort=True):
        glosses = sorted({value.strip() for value in group["gloss"].tolist() if value.strip()})
        gloss = glosses[0] if glosses else ""
        id_to_gloss[int(class_id)] = gloss
        if gloss:
            existing = gloss_to_id.get(gloss)
            if existing is not None and existing != int(class_id):
                duplicate_glosses.append(gloss)
            else:
                gloss_to_id[gloss] = int(class_id)

    class_ids = sorted(id_to_gloss)
    expected_contiguous = list(range(min(class_ids), max(class_ids) + 1)) if class_ids else []
    if class_ids and class_ids != expected_contiguous:
        logger.warning(
            "class_id values are not contiguous. min=%s max=%s unique=%s",
            min(class_ids),
            max(class_ids),
            len(class_ids),
        )
    if duplicate_glosses:
        logger.warning(
            "Found gloss values mapped to multiple class IDs: %s",
            ", ".join(sorted(set(duplicate_glosses))),
        )
    return id_to_gloss, gloss_to_id


def _normalize_split(value: Any) -> str:
    """Normalize split values to lowercase train/val/test tokens."""

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
    if isinstance(is_na, (bool, np.bool_)) and is_na:
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


def _coerce_torch_dtype(value: Any) -> torch.dtype:
    """Resolve user-configured torch dtypes."""

    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    text = str(value).strip().lower()
    mapping = {
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "torch.float64": torch.float64,
        "double": torch.float64,
    }
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype value: {value!r}.") from exc


def _parse_shape_value(value: Any) -> tuple[int, ...] | None:
    """Parse shape-like manifest/config values into integer tuples."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse shape string: {value!r}.") from exc
        value = parsed
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid shape value: {value!r}.") from exc
    raise ValueError(f"Unsupported shape value: {value!r}.")


def resolve_region_tensor_path(
    path_text: str | Path,
    *,
    project_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve one region tensor path across local and remapped workspaces."""

    text = _safe_str(path_text).strip()
    if not text:
        raise ValueError("tensor_path is empty.")

    project_path = Path(project_root or Path.cwd()).resolve()
    remapped = remap_wlasl_path(text, project_root=project_path, dataset_root=data_root)
    if remapped.exists():
        return remapped.resolve()

    raw_path = Path(text)
    candidates = [raw_path] if raw_path.is_absolute() else [project_path / raw_path]
    if data_root is not None:
        data_path = Path(data_root).resolve()
        candidates.append(data_path / raw_path)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve region tensor path {text!r}. Tried: {candidate_text or '<none>'}."
    )


def load_region_train_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and normalize one training-oriented regions config."""

    config = read_yaml(config_path)
    dataset_cfg = config.get("dataset", {})
    dataloader_cfg = config.get("dataloader", {})

    project_path = Path(project_root or Path.cwd()).resolve()
    manifests_cfg = dataset_cfg.get("manifests", {})
    resolved_manifests = {
        split: (Path(value) if Path(value).is_absolute() else (project_path / Path(value)))
        for split, value in manifests_cfg.items()
    }

    data_root_value = dataset_cfg.get("data_root")
    data_root = None
    if data_root_value:
        raw_root = Path(data_root_value)
        data_root = raw_root if raw_root.is_absolute() else (project_path / raw_root)

    return {
        "config_path": Path(config_path),
        "project_root": project_path,
        "dataset": {
            "name": str(dataset_cfg.get("name", "WLASL")),
            "subset": str(dataset_cfg.get("subset", "nslt100")),
            "data_root": data_root,
            "regions": tuple(dataset_cfg.get("regions", REGION_NAMES)),
            "num_classes": int(dataset_cfg.get("num_classes", 100)),
            "expected_shape": _parse_shape_value(dataset_cfg.get("expected_shape")) or DEFAULT_TENSOR_SHAPE,
            "manifests": resolved_manifests,
            "normalize": normalize_region_normalization_config(dataset_cfg.get("normalize")),
            "return_metadata": bool(dataset_cfg.get("return_metadata", True)),
            "strict_shape_check": bool(dataset_cfg.get("strict_shape_check", True)),
        },
        "dataloader": {
            "batch_size": int(dataloader_cfg.get("batch_size", 8)),
            "num_workers": int(dataloader_cfg.get("num_workers", 0)),
            "pin_memory": bool(dataloader_cfg.get("pin_memory", False)),
        },
        "augmentation": config.get("augmentation", {}),
    }


class RegionClipDataset(Dataset):
    """Load precomputed region clip tensors and integer class labels."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        project_root: str | Path | None = None,
        data_root: str | Path | None = None,
        split: str | None = None,
        expected_shape: Sequence[int] | str | None = None,
        num_classes: int | None = None,
        return_metadata: bool = True,
        dtype: Any = torch.float32,
        strict_shape_check: bool = True,
        strict_path_check: bool | None = None,
        augmentation_config: dict[str, Any] | None = None,
        normalization_config: dict[str, Any] | None = None,
        limit: int | None = None,
        logger=LOGGER,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.split = _normalize_split(split) if split is not None else None
        parsed_expected_shape = _parse_shape_value(expected_shape) if expected_shape is not None else None
        self.expected_shape = parsed_expected_shape or DEFAULT_TENSOR_SHAPE
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.return_metadata = bool(return_metadata)
        self.dtype = _coerce_torch_dtype(dtype)
        self.strict_shape_check = bool(strict_shape_check)
        self.strict_path_check = self.strict_shape_check if strict_path_check is None else bool(strict_path_check)
        self.augmentation_config = normalize_region_augmentation_config(augmentation_config)
        self.normalization_config = normalize_region_normalization_config(normalization_config)
        self.apply_augmentation = bool(self.augmentation_config["enabled"]) and self.split == "train"
        self.limit = limit
        self.logger = logger

        if self.split is not None and self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {split!r}. Expected one of {ALLOWED_SPLITS}.")
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("num_classes must be positive when provided.")

        self.manifest = self._load_manifest()
        self.id_to_gloss, self.gloss_to_id = build_label_maps_from_manifest(self.manifest, logger=self.logger)
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
        split: str | None = None,
        return_metadata: bool | None = None,
        limit: int | None = None,
        dtype: Any = torch.float32,
        logger=LOGGER,
    ) -> "RegionClipDataset":
        """Instantiate a dataset from one train config dict or YAML file."""

        if isinstance(config, (str, Path)):
            resolved = load_region_train_config(config)
        elif isinstance(config, dict):
            resolved = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}.")

        dataset_cfg = resolved.get("dataset", {})
        manifest_map = dataset_cfg.get("manifests", {})
        dataset_split = _normalize_split(split) if split is not None else None
        selected_split = dataset_split or "train"
        try:
            manifest_path = manifest_map[selected_split]
        except KeyError as exc:
            raise KeyError(f"Manifest for split {selected_split!r} is missing from config.") from exc

        return cls(
            manifest_path=manifest_path,
            project_root=resolved.get("project_root"),
            data_root=dataset_cfg.get("data_root"),
            split=dataset_split,
            expected_shape=dataset_cfg.get("expected_shape"),
            num_classes=dataset_cfg.get("num_classes"),
            return_metadata=dataset_cfg.get("return_metadata", True)
            if return_metadata is None
            else return_metadata,
            dtype=dtype,
            strict_shape_check=bool(dataset_cfg.get("strict_shape_check", True)),
            augmentation_config=resolved.get("augmentation"),
            normalization_config=dataset_cfg.get("normalize"),
            limit=limit,
            logger=logger,
        )

    def _load_manifest(self) -> pd.DataFrame:
        """Read and normalize the region manifest CSV."""

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {self.manifest_path}")

        dtype_map = {
            column: "string"
            for column in REGION_INPUT_MANIFEST_COLUMNS
            if column not in {"class_id", "num_frames_original", "num_frames_used"}
        }
        manifest = read_csv(self.manifest_path, dtype=dtype_map)
        require_columns(manifest, REQUIRED_DATASET_COLUMNS, name="region_manifest")

        working = manifest.copy()
        working["status"] = working["status"].fillna("").astype(str).str.strip().str.lower()
        working["split"] = working["split"].fillna("").astype(str).str.strip().str.lower()
        working["tensor_path"] = working["tensor_path"].fillna("").astype(str).str.strip()
        working["sample_id"] = working["sample_id"].fillna("").astype(str).str.strip()
        working["video_id"] = working["video_id"].fillna("").astype(str).str.strip()
        working["gloss"] = working["gloss"].fillna("").astype(str).str.strip()
        working["preview_path"] = working.get("preview_path", pd.Series("", index=working.index)).fillna("").astype(str)
        working["crop_root"] = working.get("crop_root", pd.Series("", index=working.index)).fillna("").astype(str)

        working = working[working["status"] == "ok"].copy()
        if self.split is not None:
            working = working[working["split"] == self.split].copy()
        working = working[working["tensor_path"] != ""].copy()
        working = working.reset_index(drop=True)

        if self.limit is not None:
            working = working.head(int(self.limit)).reset_index(drop=True)
        return working

    def _validate_manifest_row_shape(self, row: pd.Series) -> None:
        """Validate one manifest row's declared tensor shape."""

        tensor_shape_text = _safe_str(row.get("tensor_shape")).strip()
        if not tensor_shape_text:
            return
        manifest_shape = _parse_shape_value(tensor_shape_text)
        if manifest_shape is None:
            return
        if tuple(manifest_shape) != tuple(self.expected_shape):
            message = (
                f"Manifest tensor_shape {manifest_shape} does not match expected_shape "
                f"{self.expected_shape} for sample_id={row.get('sample_id')}."
            )
            if self.strict_shape_check:
                raise ValueError(message)
            self.logger.warning(message)

    def _build_records(self) -> list[RegionSampleRecord]:
        """Resolve usable records from the filtered manifest."""

        records: list[RegionSampleRecord] = []
        skipped = 0

        for manifest_index, row in self.manifest.iterrows():
            try:
                self._validate_manifest_row_shape(row)
                class_id = _parse_int(row.get("class_id"), "class_id")
                if self.num_classes is not None and not 0 <= class_id < int(self.num_classes):
                    raise ValueError(
                        f"class_id {class_id} is outside the expected range [0, {int(self.num_classes) - 1}]."
                    )
                tensor_path = resolve_region_tensor_path(
                    row.get("tensor_path", ""),
                    project_root=self.project_root,
                    data_root=self.data_root,
                )
            except Exception as exc:
                if self.strict_path_check:
                    raise
                skipped += 1
                self.logger.warning(
                    "Skipping manifest row index=%s sample_id=%s: %s",
                    manifest_index,
                    row.get("sample_id"),
                    exc,
                )
                continue

            records.append(
                RegionSampleRecord(
                    sample_id=_safe_str(row.get("sample_id")).strip(),
                    video_id=_safe_str(row.get("video_id")).strip(),
                    gloss=_safe_str(row.get("gloss")).strip(),
                    class_id=class_id,
                    split=_normalize_split(row.get("split")),
                    tensor_path=tensor_path,
                    preview_path=_safe_str(row.get("preview_path")).strip(),
                    crop_root=_safe_str(row.get("crop_root")).strip(),
                    manifest_index=int(manifest_index),
                    notes=_safe_str(row.get("notes")).strip(),
                )
            )

        if skipped:
            self.logger.warning("Skipped %s manifest rows while building region dataset.", skipped)
        return records

    def __len__(self) -> int:
        """Return the number of resolved region samples."""

        return len(self.records)

    def _load_region_tensor(self, path: Path) -> dict[str, np.ndarray]:
        """Load one region tensor npz file."""

        with np.load(path, allow_pickle=False) as payload:
            if "data" not in payload:
                raise KeyError(f"Region tensor file {path} does not contain the required 'data' key.")
            data = np.asarray(payload["data"], dtype=np.uint8)
            valid_mask = np.asarray(payload["valid_mask"], dtype=np.uint8) if "valid_mask" in payload else None
            bbox_source = np.asarray(payload["bbox_source"], dtype=np.uint8) if "bbox_source" in payload else None
            bboxes = np.asarray(payload["bboxes"], dtype=np.float32) if "bboxes" in payload else None
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32) if "frame_indices" in payload else None

        if self.strict_shape_check and tuple(data.shape) != tuple(self.expected_shape):
            raise ValueError(
                f"Region tensor shape mismatch for {path}: expected {self.expected_shape}, got {tuple(data.shape)}."
            )
        return {
            "data": data,
            "valid_mask": valid_mask,
            "bbox_source": bbox_source,
            "bboxes": bboxes,
            "frame_indices": frame_indices,
        }

    def __getitem__(self, index: int) -> dict[str, Any] | tuple[torch.Tensor, int]:
        """Return one region tensor sample and its integer class label."""

        record = self.records[index]
        payload = self._load_region_tensor(record.tensor_path)
        data = torch.as_tensor(payload["data"].astype(np.float32) / 255.0, dtype=self.dtype)
        label = int(record.class_id)
        valid_mask_tensor = (
            torch.as_tensor(payload["valid_mask"], dtype=torch.float32)
            if payload["valid_mask"] is not None
            else None
        )

        if self.apply_augmentation:
            data, valid_mask_tensor = apply_region_clip_augmentation(
                data,
                valid_mask=valid_mask_tensor,
                config=self.augmentation_config,
            )
        data = apply_region_dataset_normalization(
            data,
            config=self.normalization_config,
        )

        if not self.return_metadata:
            return data, label

        bbox_source = payload["bbox_source"]
        bboxes = payload["bboxes"]
        frame_indices = payload["frame_indices"]
        metadata = {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "gloss": record.gloss,
            "class_id": label,
            "split": record.split,
            "path": str(record.tensor_path),
            "preview_path": record.preview_path,
            "crop_root": record.crop_root,
            "notes": record.notes,
        }
        return {
            "data": data,
            "valid_mask": valid_mask_tensor,
            "bbox_source": torch.as_tensor(bbox_source, dtype=torch.long) if bbox_source is not None else None,
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32) if bboxes is not None else None,
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.long) if frame_indices is not None else None,
            "label": label,
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "gloss": record.gloss,
            "class_id": label,
            "split": record.split,
            "path": str(record.tensor_path),
            "preview_path": record.preview_path,
            "crop_root": record.crop_root,
            "notes": record.notes,
            "metadata": metadata,
        }


def region_collate_fn(batch: Sequence[dict[str, Any] | tuple[torch.Tensor, int]]) -> dict[str, Any]:
    """Collate region samples into batched tensors plus optional metadata."""

    if not batch:
        raise ValueError("Batch is empty.")

    first = batch[0]
    if isinstance(first, tuple):
        data_tensors = [item[0] for item in batch]
        labels = [int(item[1]) for item in batch]
        return {
            "data": torch.stack(data_tensors, dim=0),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }

    data_tensors = [item["data"] for item in batch]
    labels = [int(item["label"]) for item in batch]
    output = {
        "data": torch.stack(data_tensors, dim=0),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "metadata": [dict(item.get("metadata", {})) for item in batch],
    }

    if all(item.get("valid_mask") is not None for item in batch):
        output["valid_mask"] = torch.stack([item["valid_mask"] for item in batch], dim=0)
    if all(item.get("bbox_source") is not None for item in batch):
        output["bbox_source"] = torch.stack([item["bbox_source"] for item in batch], dim=0)
    if all(item.get("bboxes") is not None for item in batch):
        output["bboxes"] = torch.stack([item["bboxes"] for item in batch], dim=0)
    if all(item.get("frame_indices") is not None for item in batch):
        output["frame_indices"] = torch.stack([item["frame_indices"] for item in batch], dim=0)
    return output


__all__ = [
    "RegionClipDataset",
    "RegionSampleRecord",
    "build_label_maps_from_manifest",
    "load_region_train_config",
    "region_collate_fn",
    "resolve_region_tensor_path",
]
