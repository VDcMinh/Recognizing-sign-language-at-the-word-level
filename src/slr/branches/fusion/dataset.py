"""Paired dataset utilities for skeleton-region fusion experiments."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from slr.branches.fusion.package_support import normalize_sample_id
from slr.branches.regions.dataset import RegionClipDataset
from slr.branches.skeleton.dataset import SkeletonGraphDataset
from slr.utils.io import read_yaml
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)
ALLOWED_SPLITS = ("train", "val", "test")


def _resolve_path(
    value: str | Path | None,
    *,
    project_root: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not str(path):
        return None
    return path if path.is_absolute() else (project_root / path)


def _normalize_sample_id_key(value: Any) -> str:
    """Normalize sample IDs so zero-padded numeric IDs align across branches."""

    return normalize_sample_id(value)


def load_paired_skeleton_regions_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and normalize one paired fusion config."""

    raw_config = read_yaml(config_path)
    dataset_cfg = raw_config.get("dataset", {})
    dataloader_cfg = raw_config.get("dataloader", {})
    project_path = Path(project_root or Path.cwd()).resolve()

    skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
    skeleton_manifests = {
        split: _resolve_path(path_value, project_root=project_path)
        for split, path_value in dict(skeleton_cfg.get("manifests", {})).items()
    }
    regions_cfg = dict(dataset_cfg.get("regions", {}))
    regions_manifests = {
        split: _resolve_path(path_value, project_root=project_path)
        for split, path_value in dict(regions_cfg.get("manifests", {})).items()
    }

    return {
        "config_path": Path(config_path),
        "project_root": project_path,
        "dataset": {
            "subset": str(dataset_cfg.get("subset", "nslt100")),
            "num_classes": int(dataset_cfg.get("num_classes", 100)),
            "skeleton": {
                "data_root": _resolve_path(skeleton_cfg.get("data_root"), project_root=project_path),
                "keypoint_set": str(skeleton_cfg.get("keypoint_set", "selected_31")),
                "expected_shape": list(skeleton_cfg.get("expected_shape", [3, 150, 31, 1])),
                "manifests": skeleton_manifests,
                "return_metadata": bool(skeleton_cfg.get("return_metadata", True)),
                "strict_shape_check": bool(skeleton_cfg.get("strict_shape_check", True)),
            },
            "regions": {
                "data_root": _resolve_path(regions_cfg.get("data_root"), project_root=project_path),
                "expected_shape": list(regions_cfg.get("expected_shape", [3, 3, 64, 112, 112])),
                "region_order": list(regions_cfg.get("region_order", ["left_hand", "right_hand", "face"])),
                "active_regions": list(regions_cfg.get("active_regions", ["left_hand", "right_hand", "face"])),
                "manifests": regions_manifests,
                "normalize": copy.deepcopy(regions_cfg.get("normalize", {"type": "imagenet"})),
                "return_metadata": bool(regions_cfg.get("return_metadata", True)),
                "strict_shape_check": bool(regions_cfg.get("strict_shape_check", True)),
            },
        },
        "dataloader": {
            "batch_size": int(dataloader_cfg.get("batch_size", 4)),
            "num_workers": int(dataloader_cfg.get("num_workers", 0)),
            "pin_memory": bool(dataloader_cfg.get("pin_memory", False)),
            "shuffle": bool(dataloader_cfg.get("shuffle", False)),
        },
    }


class PairedSkeletonRegionsDataset(Dataset):
    """Load skeleton and region tensors aligned by ``sample_id``."""

    def __init__(
        self,
        *,
        skeleton_manifest_path: str | Path,
        regions_manifest_path: str | Path,
        project_root: str | Path | None = None,
        split: str | None = None,
        num_classes: int | None = None,
        skeleton_data_root: str | Path | None = None,
        skeleton_keypoint_set: str = "selected_31",
        skeleton_expected_shape: Sequence[int] | None = None,
        skeleton_strict_shape_check: bool = True,
        regions_data_root: str | Path | None = None,
        regions_expected_shape: Sequence[int] | None = None,
        region_order: Sequence[str] | None = None,
        active_regions: Sequence[str] | None = None,
        regions_normalization_config: dict[str, Any] | None = None,
        regions_strict_shape_check: bool = True,
        limit: int | None = None,
        dtype: Any = torch.float32,
        logger=LOGGER,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.split = str(split).strip().lower() if split is not None else None
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.limit = int(limit) if limit is not None else None
        self.logger = logger

        if self.split is not None and self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r}. Expected one of {ALLOWED_SPLITS}.")

        self.skeleton_dataset = SkeletonGraphDataset(
            manifest_path=skeleton_manifest_path,
            project_root=self.project_root,
            data_root=skeleton_data_root,
            keypoint_set=str(skeleton_keypoint_set),
            split=self.split,
            expected_shape=skeleton_expected_shape,
            num_classes=self.num_classes,
            return_metadata=True,
            dtype=dtype,
            strict_shape_check=bool(skeleton_strict_shape_check),
            limit=None,
            logger=self.logger,
        )
        self.regions_dataset = RegionClipDataset(
            manifest_path=regions_manifest_path,
            project_root=self.project_root,
            data_root=regions_data_root,
            split=self.split,
            expected_shape=regions_expected_shape,
            num_classes=self.num_classes,
            region_order=region_order,
            active_regions=active_regions,
            return_metadata=True,
            dtype=dtype,
            strict_shape_check=bool(regions_strict_shape_check),
            normalization_config=regions_normalization_config,
            limit=None,
            logger=self.logger,
        )

        self.alignment_report: dict[str, Any] = {}
        self.pairs = self._build_pairs()
        if not self.pairs:
            raise ValueError("No matched skeleton/regions samples remained after alignment.")

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict[str, Any],
        *,
        split: str | None = None,
        limit: int | None = None,
        dtype: Any = torch.float32,
        logger=LOGGER,
    ) -> "PairedSkeletonRegionsDataset":
        """Instantiate from one fusion config path or normalized config dict."""

        if isinstance(config, (str, Path)):
            resolved = load_paired_skeleton_regions_config(config)
        elif isinstance(config, dict):
            resolved = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}.")

        dataset_cfg = resolved.get("dataset", {})
        skeleton_cfg = dataset_cfg.get("skeleton", {})
        regions_cfg = dataset_cfg.get("regions", {})
        selected_split = str(split).strip().lower() if split is not None else "train"

        skeleton_manifest_map = dict(skeleton_cfg.get("manifests", {}))
        regions_manifest_map = dict(regions_cfg.get("manifests", {}))
        if selected_split not in skeleton_manifest_map:
            raise KeyError(f"Skeleton manifest for split {selected_split!r} is missing.")
        if selected_split not in regions_manifest_map:
            raise KeyError(f"Regions manifest for split {selected_split!r} is missing.")

        return cls(
            skeleton_manifest_path=skeleton_manifest_map[selected_split],
            regions_manifest_path=regions_manifest_map[selected_split],
            project_root=resolved.get("project_root"),
            split=selected_split,
            num_classes=dataset_cfg.get("num_classes"),
            skeleton_data_root=skeleton_cfg.get("data_root"),
            skeleton_keypoint_set=str(skeleton_cfg.get("keypoint_set", "selected_31")),
            skeleton_expected_shape=skeleton_cfg.get("expected_shape"),
            skeleton_strict_shape_check=bool(skeleton_cfg.get("strict_shape_check", True)),
            regions_data_root=regions_cfg.get("data_root"),
            regions_expected_shape=regions_cfg.get("expected_shape"),
            region_order=regions_cfg.get("region_order"),
            active_regions=regions_cfg.get("active_regions"),
            regions_normalization_config=regions_cfg.get("normalize"),
            regions_strict_shape_check=bool(regions_cfg.get("strict_shape_check", True)),
            limit=limit,
            dtype=dtype,
            logger=logger,
        )

    def _build_index_by_sample_id(
        self,
        records: Sequence[Any],
        *,
        branch_name: str,
    ) -> dict[str, int]:
        index_by_sample_id: dict[str, int] = {}
        duplicate_sample_ids: list[str] = []

        for index, record in enumerate(records):
            sample_id = _normalize_sample_id_key(record.sample_id)
            if not sample_id:
                raise ValueError(f"Encountered empty sample_id in {branch_name} dataset.")
            if sample_id in index_by_sample_id:
                duplicate_sample_ids.append(sample_id)
                continue
            index_by_sample_id[sample_id] = index

        if duplicate_sample_ids:
            preview = ", ".join(sorted(set(duplicate_sample_ids))[:5])
            raise ValueError(
                f"Duplicate sample_id values detected in {branch_name} dataset. "
                f"Examples: {preview}"
            )
        return index_by_sample_id

    def _build_pairs(self) -> list[tuple[str, int, int]]:
        skeleton_index = self._build_index_by_sample_id(
            self.skeleton_dataset.records,
            branch_name="skeleton",
        )
        regions_index = self._build_index_by_sample_id(
            self.regions_dataset.records,
            branch_name="regions",
        )

        skeleton_ids = set(skeleton_index)
        regions_ids = set(regions_index)
        missing_in_skeleton = sorted(regions_ids - skeleton_ids)
        missing_in_regions = sorted(skeleton_ids - regions_ids)

        matched_pairs: list[tuple[str, int, int]] = []
        label_mismatches: list[str] = []
        for skeleton_record in self.skeleton_dataset.records:
            sample_id = _normalize_sample_id_key(skeleton_record.sample_id)
            if sample_id not in regions_index:
                continue
            skeleton_idx = skeleton_index[sample_id]
            regions_idx = regions_index[sample_id]
            regions_record = self.regions_dataset.records[regions_idx]
            if int(skeleton_record.class_id) != int(regions_record.class_id):
                label_mismatches.append(
                    f"sample_id={sample_id} skeleton_label={int(skeleton_record.class_id)} "
                    f"regions_label={int(regions_record.class_id)}"
                )
                continue
            matched_pairs.append((sample_id, skeleton_idx, regions_idx))

        if label_mismatches:
            preview = "; ".join(label_mismatches[:5])
            raise ValueError(
                "Label mismatches detected between skeleton and regions manifests. "
                f"count={len(label_mismatches)} examples={preview}"
            )

        if self.limit is not None:
            matched_pairs = matched_pairs[: self.limit]

        self.alignment_report = {
            "split": self.split or "all",
            "skeleton_count": len(self.skeleton_dataset),
            "regions_count": len(self.regions_dataset),
            "matched_count": len(matched_pairs),
            "missing_in_skeleton": len(missing_in_skeleton),
            "missing_in_regions": len(missing_in_regions),
            "label_mismatch": len(label_mismatches),
            "missing_in_skeleton_examples": missing_in_skeleton[:5],
            "missing_in_regions_examples": missing_in_regions[:5],
            "skeleton_keypoint_set": str(self.skeleton_dataset.keypoint_set),
            "active_regions": list(self.regions_dataset.active_regions),
        }
        return matched_pairs

    def get_alignment_report(self) -> dict[str, Any]:
        """Return a copy of the latest alignment summary."""

        return copy.deepcopy(self.alignment_report)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_key, skeleton_index, regions_index = self.pairs[index]
        skeleton_item = self.skeleton_dataset[skeleton_index]
        regions_item = self.regions_dataset[regions_index]

        skeleton_sample_id = str(skeleton_item["sample_id"])
        regions_sample_id = str(regions_item["sample_id"])
        if _normalize_sample_id_key(skeleton_sample_id) != _normalize_sample_id_key(regions_sample_id):
            raise RuntimeError(
                "Aligned pair sample_id mismatch. "
                f"skeleton={skeleton_sample_id} regions={regions_sample_id}"
            )

        gloss = str(skeleton_item["gloss"] or regions_item["gloss"])
        label = int(skeleton_item["label"])
        item = {
            "sample_id": sample_key,
            "label": label,
            "gloss": gloss,
            "skeleton": skeleton_item["data"],
            "regions": regions_item["data"],
            "regions_valid_mask": regions_item.get("valid_mask"),
            "metadata": {
                "skeleton": {
                    "sample_id": skeleton_sample_id,
                    "path": skeleton_item["path"],
                    "video_id": skeleton_item["video_id"],
                    "split": skeleton_item["split"],
                },
                "regions": {
                    **dict(regions_item.get("metadata", {})),
                    "sample_id": regions_sample_id,
                },
            },
        }
        return item


def paired_skeleton_regions_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate aligned skeleton-region samples into batched tensors."""

    if not batch:
        raise ValueError("Batch is empty.")

    output = {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "glosses": [str(item["gloss"]) for item in batch],
        "skeleton": torch.stack([item["skeleton"] for item in batch], dim=0),
        "regions": torch.stack([item["regions"] for item in batch], dim=0),
        "labels": torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "metadata": [copy.deepcopy(item.get("metadata", {})) for item in batch],
    }
    if all(item.get("regions_valid_mask") is not None for item in batch):
        output["regions_valid_mask"] = torch.stack(
            [item["regions_valid_mask"] for item in batch],
            dim=0,
        )
    return output


__all__ = [
    "PairedSkeletonRegionsDataset",
    "load_paired_skeleton_regions_config",
    "paired_skeleton_regions_collate_fn",
]
