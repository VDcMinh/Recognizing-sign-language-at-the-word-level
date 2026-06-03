"""Evaluate late fusion between skeleton and regions checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn
from slr.branches.regions.models import build_region_model
from slr.branches.regions.train import (
    resolve_training_config as resolve_regions_training_config,
)
from slr.branches.regions.train import select_device as select_regions_device
from slr.branches.skeleton.dataset import SkeletonGraphDataset, skeleton_collate_fn
from slr.branches.skeleton.graph import SkeletonGraph
from slr.branches.skeleton.models import build_skeleton_model
from slr.branches.skeleton.train import (
    resolve_training_config as resolve_skeleton_training_config,
)
from slr.branches.skeleton.train import select_device as select_skeleton_device
from slr.training.checkpointing import load_checkpoint
from slr.training.losses import build_loss_from_config
from slr.training.metrics import accuracy_topk
from slr.utils.io import ensure_dir, read_yaml, write_dataframe_csv, write_json, write_text
from slr.utils.logging import setup_logger


DEFAULT_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TOPK = (1, 5, 10)
SELECTABLE_METRICS = {"top1", "top5", "top10"}


@dataclass(frozen=True)
class BranchSpec:
    config_path: Path
    checkpoint_path: Path
    data_root: Path | None
    val_manifest: Path | None
    test_manifest: Path | None


@dataclass(frozen=True)
class FusionSpec:
    subset: str
    output_dir: Path
    skeleton: BranchSpec
    regions: BranchSpec
    splits: tuple[str, ...]
    alphas: tuple[float, ...]
    select_metric: str
    fusion_space: str
    reuse_logits: bool
    dry_run: bool
    fusion_config_path: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export branch logits, check alignment, and evaluate late fusion."
    )
    parser.add_argument("--fusion-config", type=Path, default=None, help="Optional fusion config YAML.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset name, for example nslt100.")
    parser.add_argument("--skeleton-config", type=Path, default=None, help="Skeleton resolved config YAML.")
    parser.add_argument("--skeleton-checkpoint", type=Path, default=None, help="Skeleton checkpoint.")
    parser.add_argument("--regions-config", type=Path, default=None, help="Regions resolved config YAML.")
    parser.add_argument("--regions-checkpoint", type=Path, default=None, help="Regions checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Fusion output workspace.")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits to export, for example val,test.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated fusion alphas, for example 0.0,0.5,1.0.",
    )
    parser.add_argument(
        "--select-metric",
        type=str,
        default=None,
        choices=sorted(SELECTABLE_METRICS),
        help="Validation metric used to pick the best alpha.",
    )
    parser.add_argument(
        "--fusion-space",
        type=str,
        default=None,
        choices=["logits", "probs"],
        help="Fuse either raw logits or per-branch probabilities.",
    )
    parser.add_argument("--reuse-logits", action="store_true", help="Reuse cached .npz logits when available.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/configs without running models.")
    parser.add_argument("--skeleton-data-root", type=Path, default=None, help="Optional skeleton data_root override.")
    parser.add_argument("--regions-data-root", type=Path, default=None, help="Optional regions data_root override.")
    parser.add_argument("--skeleton-val-manifest", type=Path, default=None, help="Optional skeleton val manifest override.")
    parser.add_argument("--skeleton-test-manifest", type=Path, default=None, help="Optional skeleton test manifest override.")
    parser.add_argument("--regions-val-manifest", type=Path, default=None, help="Optional regions val manifest override.")
    parser.add_argument("--regions-test-manifest", type=Path, default=None, help="Optional regions test manifest override.")
    return parser


def _default_skeleton_args() -> argparse.Namespace:
    return SimpleNamespace(
        run_name=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        dropout=None,
        device=None,
        seed=None,
        no_wandb=True,
        wandb_project=None,
        wandb_entity=None,
        output_root=None,
        num_workers=None,
        limit_train=None,
        limit_val=None,
        limit_test=None,
        dry_run=False,
    )


def _default_regions_args() -> argparse.Namespace:
    return SimpleNamespace(
        config=None,
        output_dir=None,
        run_name=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        device=None,
        seed=None,
        data_root=None,
        train_manifest=None,
        val_manifest=None,
        test_manifest=None,
        limit_train=None,
        limit_val=None,
        limit_test=None,
        no_wandb=True,
        wandb_project=None,
        wandb_entity=None,
        dry_run=False,
    )


def _parse_comma_list(value: str | None, *, lower: bool = False) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",")]
    cleaned = [item.lower() if lower else item for item in items if item]
    return cleaned


def _parse_alphas(value: str | None, fallback: list[float]) -> tuple[float, ...]:
    if value is None:
        candidates = fallback
    else:
        candidates = [float(item) for item in _parse_comma_list(value)]
    if not candidates:
        raise ValueError("alphas must not be empty.")
    normalized: list[float] = []
    for alpha in candidates:
        value_f = float(alpha)
        if not 0.0 <= value_f <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {value_f}.")
        normalized.append(value_f)
    return tuple(normalized)


def _resolve_fusion_type(fusion_cfg: dict[str, Any]) -> str:
    fusion_type = str(fusion_cfg.get("type", "late_logits")).strip().lower()
    if "prob" in fusion_type:
        return "probs"
    return "logits"


def _required_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required argument: {label}.")
    return Path(value)


def resolve_spec(args: argparse.Namespace) -> FusionSpec:
    fusion_config = read_yaml(args.fusion_config) if args.fusion_config else {}
    fusion_cfg = fusion_config.get("fusion", {})
    skeleton_cfg = fusion_config.get("skeleton", {})
    regions_cfg = fusion_config.get("regions", {})

    subset = str(args.subset or fusion_config.get("subset") or "nslt100")
    output_dir = Path(args.output_dir or fusion_config.get("workspace") or f"artifacts/fusion/{subset}")
    splits = tuple(_parse_comma_list(args.splits, lower=True) or list(DEFAULT_SPLITS))
    if "val" not in splits:
        raise ValueError("splits must include 'val' so the best alpha can be selected on validation.")
    unknown_splits = [split for split in splits if split not in {"train", "val", "test"}]
    if unknown_splits:
        raise ValueError(f"Unsupported split values: {unknown_splits}.")

    select_metric = str(args.select_metric or fusion_cfg.get("select_metric") or "top5").strip().lower()
    if select_metric not in SELECTABLE_METRICS:
        raise ValueError(
            f"select_metric must be one of {sorted(SELECTABLE_METRICS)}, got {select_metric!r}."
        )

    alphas = _parse_alphas(args.alphas, fallback=list(fusion_cfg.get("alphas", DEFAULT_ALPHAS)))
    fusion_space = str(args.fusion_space or _resolve_fusion_type(fusion_cfg)).strip().lower()

    skeleton = BranchSpec(
        config_path=_required_path(args.skeleton_config or skeleton_cfg.get("config"), "--skeleton-config"),
        checkpoint_path=_required_path(
            args.skeleton_checkpoint or skeleton_cfg.get("checkpoint"),
            "--skeleton-checkpoint",
        ),
        data_root=args.skeleton_data_root,
        val_manifest=args.skeleton_val_manifest,
        test_manifest=args.skeleton_test_manifest,
    )
    regions = BranchSpec(
        config_path=_required_path(args.regions_config or regions_cfg.get("config"), "--regions-config"),
        checkpoint_path=_required_path(
            args.regions_checkpoint or regions_cfg.get("checkpoint"),
            "--regions-checkpoint",
        ),
        data_root=args.regions_data_root,
        val_manifest=args.regions_val_manifest,
        test_manifest=args.regions_test_manifest,
    )
    return FusionSpec(
        subset=subset,
        output_dir=output_dir,
        skeleton=skeleton,
        regions=regions,
        splits=splits,
        alphas=alphas,
        select_metric=select_metric,
        fusion_space=fusion_space,
        reuse_logits=bool(args.reuse_logits),
        dry_run=bool(args.dry_run),
        fusion_config_path=args.fusion_config,
    )


def _branch_missing_paths(branch_name: str, spec: BranchSpec) -> list[str]:
    missing: list[str] = []
    if not spec.config_path.exists():
        missing.append(f"{branch_name} config: {spec.config_path.as_posix()}")
    if not spec.checkpoint_path.exists():
        missing.append(f"{branch_name} checkpoint: {spec.checkpoint_path.as_posix()}")
    return missing


def _override_split_paths(config: dict[str, Any], spec: BranchSpec) -> dict[str, Any]:
    resolved = dict(config)
    resolved["dataset"] = dict(config.get("dataset", {}))
    resolved["dataset"]["manifests"] = dict(resolved["dataset"].get("manifests", {}))
    if spec.data_root is not None:
        resolved["dataset"]["data_root"] = str(spec.data_root.as_posix())
    if spec.val_manifest is not None:
        resolved["dataset"]["manifests"]["val"] = str(spec.val_manifest.as_posix())
    if spec.test_manifest is not None:
        resolved["dataset"]["manifests"]["test"] = str(spec.test_manifest.as_posix())
    resolved["dataset"]["return_metadata"] = True
    return resolved


def load_skeleton_runtime_config(spec: BranchSpec) -> dict[str, Any]:
    config = resolve_skeleton_training_config(spec.config_path, _default_skeleton_args())
    return _override_split_paths(config, spec)


def load_regions_runtime_config(spec: BranchSpec) -> dict[str, Any]:
    config = resolve_regions_training_config(spec.config_path, _default_regions_args())
    config = _override_split_paths(config, spec)
    config["model"] = dict(config.get("model", {}))
    config["model"]["pretrained"] = False
    return config


def build_skeleton_loader(config: dict[str, Any], *, split: str, device: torch.device) -> tuple[Any, DataLoader]:
    dataset = SkeletonGraphDataset.from_config(config, split=split)
    dataloader_cfg = config["dataloader"]
    loader = DataLoader(
        dataset,
        batch_size=int(dataloader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        pin_memory=bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda",
        collate_fn=skeleton_collate_fn,
    )
    return dataset, loader


def build_regions_loader(config: dict[str, Any], *, split: str, device: torch.device) -> tuple[Any, DataLoader]:
    dataset = RegionClipDataset.from_config(config, split=split)
    train_cfg = config["train"]
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)) and device.type == "cuda",
        collate_fn=region_collate_fn,
    )
    return dataset, loader


def logits_cache_path(output_dir: Path, *, split: str, branch_name: str) -> Path:
    return output_dir / "logits" / f"{split}_{branch_name}_logits.npz"


def can_skip_branch_export(spec: FusionSpec, branch_name: str) -> bool:
    if not spec.reuse_logits:
        return False
    return all(
        logits_cache_path(spec.output_dir, split=split, branch_name=branch_name).exists()
        for split in spec.splits
    )


def _to_numpy_strings(values: list[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=str)


def export_skeleton_logits(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_path: Path,
    reuse_logits: bool,
    logger,
) -> dict[str, Any]:
    if reuse_logits and output_path.exists():
        logger.info("Reusing cached skeleton logits: %s", output_path)
        return load_logits_payload(output_path)

    device = select_skeleton_device(str(config["train"]["device"]))
    dataset, loader = build_skeleton_loader(config, split=split, device=device)
    graph_cfg = config["graph"]
    graph = SkeletonGraph(
        layout=str(graph_cfg["layout"]),
        strategy=str(graph_cfg["strategy"]),
        normalize=bool(graph_cfg["normalize_adjacency"]),
        add_self_links=bool(graph_cfg["add_self_links"]),
    )
    model = build_skeleton_model(config["model"], graph).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    criterion = build_loss_from_config(config)

    sample_ids: list[str] = []
    glosses: list[str] = []
    labels: list[int] = []
    pred_top1: list[int] = []
    logits_batches: list[np.ndarray] = []
    loss_sum = 0.0
    num_samples = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            data = batch["data"].to(device, non_blocking=device.type == "cuda")
            batch_labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
            logits = model(data)
            batch_loss = criterion(logits, batch_labels)
            batch_size = int(batch_labels.shape[0])
            loss_sum += float(batch_loss.item()) * batch_size
            num_samples += batch_size

            metadata = batch.get("metadata", [])
            sample_ids.extend(str(item.get("sample_id", "")) for item in metadata)
            glosses.extend(str(item.get("gloss", "")) for item in metadata)
            labels.extend(int(value) for value in batch_labels.detach().cpu().numpy().tolist())
            pred_top1.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().numpy().tolist())
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))

    payload = {
        "sample_ids": _to_numpy_strings(sample_ids),
        "labels": np.asarray(labels, dtype=np.int64),
        "glosses": _to_numpy_strings(glosses),
        "logits": np.concatenate(logits_batches, axis=0) if logits_batches else np.empty((0, 0), dtype=np.float32),
        "pred_top1": np.asarray(pred_top1, dtype=np.int64),
        "loss_mean": np.asarray(loss_sum / max(num_samples, 1), dtype=np.float32),
    }
    save_logits_payload(output_path, payload)
    logger.info("Saved skeleton logits for split=%s to %s", split, output_path)
    return load_logits_payload(output_path)


def export_regions_logits(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_path: Path,
    reuse_logits: bool,
    logger,
) -> dict[str, Any]:
    if reuse_logits and output_path.exists():
        logger.info("Reusing cached regions logits: %s", output_path)
        return load_logits_payload(output_path)

    device = select_regions_device(str(config["train"]["device"]), logger=logger)
    _, loader = build_regions_loader(config, split=split, device=device)
    model = build_region_model(config["model"]).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    criterion = build_loss_from_config(config)

    sample_ids: list[str] = []
    glosses: list[str] = []
    labels: list[int] = []
    pred_top1: list[int] = []
    logits_batches: list[np.ndarray] = []
    loss_sum = 0.0
    num_samples = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            data = batch["data"].to(device, non_blocking=device.type == "cuda")
            batch_labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
            valid_mask = batch.get("valid_mask")
            logits = model(
                data,
                valid_mask=(
                    valid_mask.to(device, non_blocking=device.type == "cuda")
                    if valid_mask is not None
                    else None
                ),
            )
            batch_loss = criterion(logits, batch_labels)
            batch_size = int(batch_labels.shape[0])
            loss_sum += float(batch_loss.item()) * batch_size
            num_samples += batch_size

            metadata = batch.get("metadata", [])
            sample_ids.extend(str(item.get("sample_id", "")) for item in metadata)
            glosses.extend(str(item.get("gloss", "")) for item in metadata)
            labels.extend(int(value) for value in batch_labels.detach().cpu().numpy().tolist())
            pred_top1.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().numpy().tolist())
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))

    payload = {
        "sample_ids": _to_numpy_strings(sample_ids),
        "labels": np.asarray(labels, dtype=np.int64),
        "glosses": _to_numpy_strings(glosses),
        "logits": np.concatenate(logits_batches, axis=0) if logits_batches else np.empty((0, 0), dtype=np.float32),
        "pred_top1": np.asarray(pred_top1, dtype=np.int64),
        "loss_mean": np.asarray(loss_sum / max(num_samples, 1), dtype=np.float32),
    }
    save_logits_payload(output_path, payload)
    logger.info("Saved regions logits for split=%s to %s", split, output_path)
    return load_logits_payload(output_path)


def save_logits_payload(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **payload)


def load_logits_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def align_logits_by_sample_id(
    *,
    split: str,
    skeleton_payload: dict[str, Any],
    regions_payload: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    skeleton_ids = [str(value) for value in skeleton_payload["sample_ids"].tolist()]
    regions_ids = [str(value) for value in regions_payload["sample_ids"].tolist()]
    skeleton_index = {sample_id: idx for idx, sample_id in enumerate(skeleton_ids)}
    regions_index = {sample_id: idx for idx, sample_id in enumerate(regions_ids)}

    matched_ids = sorted(set(skeleton_index).intersection(regions_index))
    missing_in_skeleton = sorted(set(regions_index).difference(skeleton_index))
    missing_in_regions = sorted(set(skeleton_index).difference(regions_index))

    skeleton_match_indices = np.asarray([skeleton_index[sample_id] for sample_id in matched_ids], dtype=np.int64)
    regions_match_indices = np.asarray([regions_index[sample_id] for sample_id in matched_ids], dtype=np.int64)

    skeleton_labels = np.asarray(skeleton_payload["labels"], dtype=np.int64)[skeleton_match_indices]
    regions_labels = np.asarray(regions_payload["labels"], dtype=np.int64)[regions_match_indices]
    label_mismatch_mask = skeleton_labels != regions_labels
    label_mismatch = int(label_mismatch_mask.sum())

    report = {
        "split": split,
        "skeleton_count": int(len(skeleton_ids)),
        "regions_count": int(len(regions_ids)),
        "matched_count": int(len(matched_ids)),
        "missing_in_skeleton": int(len(missing_in_skeleton)),
        "missing_in_regions": int(len(missing_in_regions)),
        "label_mismatch": label_mismatch,
        "missing_in_skeleton_examples": missing_in_skeleton[:10],
        "missing_in_regions_examples": missing_in_regions[:10],
    }
    if not matched_ids:
        write_json(report, report_path)
        raise ValueError(
            f"No matched sample_id values were found for split={split}. "
            f"See {report_path.as_posix()} for details."
        )
    if label_mismatch:
        mismatch_ids = [
            matched_ids[index]
            for index, is_mismatch in enumerate(label_mismatch_mask.tolist())
            if bool(is_mismatch)
        ]
        report["label_mismatch_examples"] = mismatch_ids[:10]
    write_json(report, report_path)

    if label_mismatch:
        raise ValueError(
            f"Label mismatch detected for split={split}. See {report_path.as_posix()} for details."
        )

    skeleton_glosses = _slice_string_array(skeleton_payload.get("glosses"), skeleton_match_indices)
    regions_glosses = _slice_string_array(regions_payload.get("glosses"), regions_match_indices)
    merged_glosses = np.asarray(
        [
            skeleton_gloss if skeleton_gloss else regions_gloss
            for skeleton_gloss, regions_gloss in zip(skeleton_glosses.tolist(), regions_glosses.tolist())
        ],
        dtype=str,
    )

    return {
        "sample_ids": np.asarray(matched_ids, dtype=str),
        "labels": skeleton_labels,
        "glosses": merged_glosses,
        "skeleton_logits": np.asarray(skeleton_payload["logits"], dtype=np.float32)[skeleton_match_indices],
        "regions_logits": np.asarray(regions_payload["logits"], dtype=np.float32)[regions_match_indices],
        "alignment_report": report,
    }


def _slice_string_array(values: Any, indices: np.ndarray) -> np.ndarray:
    if values is None:
        return np.asarray([""] * int(indices.shape[0]), dtype=str)
    array = np.asarray(values, dtype=str)
    return array[indices]


def fuse_outputs(
    *,
    skeleton_logits: np.ndarray,
    regions_logits: np.ndarray,
    alpha: float,
    fusion_space: str,
) -> np.ndarray:
    if fusion_space == "logits":
        return alpha * skeleton_logits + (1.0 - alpha) * regions_logits
    skeleton_probs = softmax_numpy(skeleton_logits)
    regions_probs = softmax_numpy(regions_logits)
    return alpha * skeleton_probs + (1.0 - alpha) * regions_probs


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(logits, dtype=torch.float32)
    probs = F.softmax(tensor, dim=1)
    return probs.detach().cpu().numpy()


def compute_metrics(outputs: np.ndarray, labels: np.ndarray, *, fusion_space: str) -> dict[str, float]:
    metrics = accuracy_topk(outputs, labels, topk=DEFAULT_TOPK)
    tensor_outputs = torch.as_tensor(outputs, dtype=torch.float32)
    tensor_labels = torch.as_tensor(labels, dtype=torch.long)
    if fusion_space == "logits":
        loss = F.cross_entropy(tensor_outputs, tensor_labels)
    else:
        loss = F.nll_loss(torch.log(tensor_outputs.clamp_min(1e-12)), tensor_labels)
    return {
        "loss": float(loss.item()),
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "top10": float(metrics["top10"]),
        "num_samples": int(labels.shape[0]),
    }


def sweep_alphas(
    *,
    aligned_payload: dict[str, Any],
    alphas: tuple[float, ...],
    fusion_space: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    labels = np.asarray(aligned_payload["labels"], dtype=np.int64)
    skeleton_logits = np.asarray(aligned_payload["skeleton_logits"], dtype=np.float32)
    regions_logits = np.asarray(aligned_payload["regions_logits"], dtype=np.float32)

    for alpha in alphas:
        fused = fuse_outputs(
            skeleton_logits=skeleton_logits,
            regions_logits=regions_logits,
            alpha=float(alpha),
            fusion_space=fusion_space,
        )
        metrics = compute_metrics(fused, labels, fusion_space=fusion_space)
        row = {
            "alpha": float(alpha),
            "top1": float(metrics["top1"]),
            "top5": float(metrics["top5"]),
            "top10": float(metrics["top10"]),
            "num_samples": int(metrics["num_samples"]),
            "loss": float(metrics["loss"]),
        }
        rows.append(row)
        if best_row is None:
            best_row = row
            continue
        current_metric = float(row["top5"])
        best_metric = float(best_row["top5"])
        if current_metric > best_metric or (
            np.isclose(current_metric, best_metric) and float(row["alpha"]) > float(best_row["alpha"])
        ):
            best_row = row

    if best_row is None:
        raise RuntimeError("No alpha rows were produced during sweep.")

    frame = pd.DataFrame(rows)
    return frame, best_row


def select_best_alpha(sweep_frame: pd.DataFrame, *, select_metric: str) -> dict[str, Any]:
    best_row: dict[str, Any] | None = None
    for row in sweep_frame.to_dict(orient="records"):
        if best_row is None:
            best_row = row
            continue
        current_metric = float(row[select_metric])
        best_metric = float(best_row[select_metric])
        if current_metric > best_metric or (
            np.isclose(current_metric, best_metric) and float(row["alpha"]) > float(best_row["alpha"])
        ):
            best_row = row
    if best_row is None:
        raise RuntimeError("Validation alpha sweep was empty.")
    return best_row


def evaluate_alpha(
    *,
    aligned_payload: dict[str, Any],
    alpha: float,
    fusion_space: str,
) -> dict[str, float]:
    fused = fuse_outputs(
        skeleton_logits=np.asarray(aligned_payload["skeleton_logits"], dtype=np.float32),
        regions_logits=np.asarray(aligned_payload["regions_logits"], dtype=np.float32),
        alpha=float(alpha),
        fusion_space=fusion_space,
    )
    return compute_metrics(
        fused,
        np.asarray(aligned_payload["labels"], dtype=np.int64),
        fusion_space=fusion_space,
    )


def write_runtime_summary(
    *,
    spec: FusionSpec,
    val_alignment: dict[str, Any],
    test_alignment: dict[str, Any] | None,
    selected_alpha: float,
    select_metric: str,
    val_best: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    baselines: dict[str, Any],
) -> None:
    report_path = spec.output_dir / "reports" / "late_fusion_summary.md"
    lines = [
        "# Skeleton + Regions Late Fusion Summary",
        "",
        f"- subset: `{spec.subset}`",
        f"- fusion_config: `{spec.fusion_config_path.as_posix()}`" if spec.fusion_config_path else "- fusion_config: not used",
        f"- fusion_space: `{spec.fusion_space}`",
        f"- selected_by: `val_{select_metric}`",
        f"- selected_alpha: `{selected_alpha:.2f}`",
        "- tie_break: prefer larger alpha when validation metric ties",
        "",
        "## Validation alignment",
        "",
        f"- skeleton_count: {val_alignment['skeleton_count']}",
        f"- regions_count: {val_alignment['regions_count']}",
        f"- matched_count: {val_alignment['matched_count']}",
        f"- missing_in_skeleton: {val_alignment['missing_in_skeleton']}",
        f"- missing_in_regions: {val_alignment['missing_in_regions']}",
        f"- label_mismatch: {val_alignment['label_mismatch']}",
        "",
        "## Validation best",
        "",
        f"- top1: {float(val_best['top1']):.4f}",
        f"- top5: {float(val_best['top5']):.4f}",
        f"- top10: {float(val_best['top10']):.4f}",
        f"- loss: {float(val_best['loss']):.4f}",
        "",
        "## Baselines",
        "",
        f"- skeleton_only val_top5: {float(baselines['skeleton_only']['val']['top5']):.4f}",
        f"- regions_only val_top5: {float(baselines['regions_only']['val']['top5']):.4f}",
    ]
    if test_alignment is not None:
        lines.extend(
            [
                "",
                "## Test alignment",
                "",
                f"- skeleton_count: {test_alignment['skeleton_count']}",
                f"- regions_count: {test_alignment['regions_count']}",
                f"- matched_count: {test_alignment['matched_count']}",
                f"- missing_in_skeleton: {test_alignment['missing_in_skeleton']}",
                f"- missing_in_regions: {test_alignment['missing_in_regions']}",
                f"- label_mismatch: {test_alignment['label_mismatch']}",
            ]
        )
    if test_metrics is not None:
        lines.extend(
            [
                "",
                "## Test metrics",
                "",
                f"- top1: {float(test_metrics['top1']):.4f}",
                f"- top5: {float(test_metrics['top5']):.4f}",
                f"- top10: {float(test_metrics['top10']):.4f}",
                f"- loss: {float(test_metrics['loss']):.4f}",
                "",
                f"- skeleton_only test_top5: {float(baselines['skeleton_only']['test']['top5']):.4f}",
                f"- regions_only test_top5: {float(baselines['regions_only']['test']['top5']):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- `{(spec.output_dir / 'logits' / 'val_skeleton_logits.npz').as_posix()}`",
            f"- `{(spec.output_dir / 'logits' / 'val_regions_logits.npz').as_posix()}`",
            f"- `{(spec.output_dir / 'reports' / 'val_alignment_report.json').as_posix()}`",
            f"- `{(spec.output_dir / 'reports' / 'val_alpha_sweep.csv').as_posix()}`",
            f"- `{(spec.output_dir / 'reports' / 'test_fusion_metrics.json').as_posix()}`",
        ]
    )
    if "test" in spec.splits:
        lines.extend(
            [
                f"- `{(spec.output_dir / 'logits' / 'test_skeleton_logits.npz').as_posix()}`",
                f"- `{(spec.output_dir / 'logits' / 'test_regions_logits.npz').as_posix()}`",
                f"- `{(spec.output_dir / 'reports' / 'test_alignment_report.json').as_posix()}`",
            ]
        )
    write_text("\n".join(lines) + "\n", report_path)


def write_dry_run_summary(spec: FusionSpec, missing: list[str]) -> None:
    report_path = spec.output_dir / "reports" / "dry_run_summary.md"
    lines = [
        "# Late Fusion Dry Run",
        "",
        f"- subset: `{spec.subset}`",
        f"- output_dir: `{spec.output_dir.as_posix()}`",
        f"- fusion_space: `{spec.fusion_space}`",
        f"- splits: `{','.join(spec.splits)}`",
        f"- select_metric: `{spec.select_metric}`",
        "",
    ]
    if missing:
        lines.append("## Missing files")
        lines.append("")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Full fusion evaluation requires the files above to be copied into the workspace.")
    else:
        lines.append("All required config/checkpoint files are present.")
    write_text("\n".join(lines) + "\n", report_path)


def build_test_metrics_payload(
    *,
    spec: FusionSpec,
    selected_alpha: float,
    val_best: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    baselines: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "subset": spec.subset,
        "fusion_space": spec.fusion_space,
        "selected_alpha": float(selected_alpha),
        "selected_by": f"val_{spec.select_metric}",
        "tie_break": "prefer larger alpha when validation metrics tie",
        "val_best": {
            "loss": float(val_best["loss"]),
            "top1": float(val_best["top1"]),
            "top5": float(val_best["top5"]),
            "top10": float(val_best["top10"]),
            "num_samples": int(val_best["num_samples"]),
        },
        "test": None
        if test_metrics is None
        else {
            "loss": float(test_metrics["loss"]),
            "top1": float(test_metrics["top1"]),
            "top5": float(test_metrics["top5"]),
            "top10": float(test_metrics["top10"]),
            "num_samples": int(test_metrics["num_samples"]),
        },
        "baselines": baselines,
    }
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    spec = resolve_spec(args)

    ensure_dir(spec.output_dir)
    ensure_dir(spec.output_dir / "logits")
    ensure_dir(spec.output_dir / "reports")

    logger = setup_logger("slr.fusion.skeleton_regions")
    skip_skeleton_export = can_skip_branch_export(spec, "skeleton")
    skip_regions_export = can_skip_branch_export(spec, "regions")

    missing: list[str] = []
    if not skip_skeleton_export:
        missing.extend(_branch_missing_paths("skeleton", spec.skeleton))
    if not skip_regions_export:
        missing.extend(_branch_missing_paths("regions", spec.regions))

    if spec.dry_run:
        if spec.skeleton.config_path.exists() and not skip_skeleton_export:
            load_skeleton_runtime_config(spec.skeleton)
        if spec.regions.config_path.exists() and not skip_regions_export:
            load_regions_runtime_config(spec.regions)
        write_dry_run_summary(spec, missing)
        if missing:
            print("Dry run completed. Missing files:")
            for item in missing:
                print(f"- {item}")
            print()
            print("Copy the missing files, then rerun without --dry-run.")
        else:
            print("Dry run completed. Required config and checkpoint paths exist.")
        return 0
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    skeleton_config = None if skip_skeleton_export else load_skeleton_runtime_config(spec.skeleton)
    regions_config = None if skip_regions_export else load_regions_runtime_config(spec.regions)

    logits_by_split: dict[str, dict[str, Any]] = {}
    for split in spec.splits:
        logits_by_split[split] = {
            "skeleton": (
                load_logits_payload(logits_cache_path(spec.output_dir, split=split, branch_name="skeleton"))
                if skip_skeleton_export
                else export_skeleton_logits(
                    config=skeleton_config,
                    checkpoint_path=spec.skeleton.checkpoint_path,
                    split=split,
                    output_path=logits_cache_path(spec.output_dir, split=split, branch_name="skeleton"),
                    reuse_logits=spec.reuse_logits,
                    logger=logger,
                )
            ),
            "regions": (
                load_logits_payload(logits_cache_path(spec.output_dir, split=split, branch_name="regions"))
                if skip_regions_export
                else export_regions_logits(
                    config=regions_config,
                    checkpoint_path=spec.regions.checkpoint_path,
                    split=split,
                    output_path=logits_cache_path(spec.output_dir, split=split, branch_name="regions"),
                    reuse_logits=spec.reuse_logits,
                    logger=logger,
                )
            ),
        }

    aligned_by_split: dict[str, dict[str, Any]] = {}
    for split in spec.splits:
        aligned_by_split[split] = align_logits_by_sample_id(
            split=split,
            skeleton_payload=logits_by_split[split]["skeleton"],
            regions_payload=logits_by_split[split]["regions"],
            report_path=spec.output_dir / "reports" / f"{split}_alignment_report.json",
        )

    val_sweep_frame, _ = sweep_alphas(
        aligned_payload=aligned_by_split["val"],
        alphas=spec.alphas,
        fusion_space=spec.fusion_space,
    )
    best_val_row = select_best_alpha(val_sweep_frame, select_metric=spec.select_metric)
    write_dataframe_csv(
        val_sweep_frame.loc[:, ["alpha", "top1", "top5", "top10", "num_samples"]],
        spec.output_dir / "reports" / "val_alpha_sweep.csv",
    )

    selected_alpha = float(best_val_row["alpha"])
    test_metrics = (
        evaluate_alpha(
            aligned_payload=aligned_by_split["test"],
            alpha=selected_alpha,
            fusion_space=spec.fusion_space,
        )
        if "test" in aligned_by_split
        else None
    )
    baselines = {
        "skeleton_only": {
            "alpha": 1.0,
            "val": evaluate_alpha(
                aligned_payload=aligned_by_split["val"],
                alpha=1.0,
                fusion_space=spec.fusion_space,
            ),
            "test": None
            if "test" not in aligned_by_split
            else evaluate_alpha(
                aligned_payload=aligned_by_split["test"],
                alpha=1.0,
                fusion_space=spec.fusion_space,
            ),
        },
        "regions_only": {
            "alpha": 0.0,
            "val": evaluate_alpha(
                aligned_payload=aligned_by_split["val"],
                alpha=0.0,
                fusion_space=spec.fusion_space,
            ),
            "test": None
            if "test" not in aligned_by_split
            else evaluate_alpha(
                aligned_payload=aligned_by_split["test"],
                alpha=0.0,
                fusion_space=spec.fusion_space,
            ),
        },
    }
    test_metrics_payload = build_test_metrics_payload(
        spec=spec,
        selected_alpha=selected_alpha,
        val_best=best_val_row,
        test_metrics=test_metrics,
        baselines=baselines,
    )
    write_json(test_metrics_payload, spec.output_dir / "reports" / "test_fusion_metrics.json")
    write_runtime_summary(
        spec=spec,
        val_alignment=aligned_by_split["val"]["alignment_report"],
        test_alignment=(
            aligned_by_split["test"]["alignment_report"] if "test" in aligned_by_split else None
        ),
        selected_alpha=selected_alpha,
        select_metric=spec.select_metric,
        val_best=best_val_row,
        test_metrics=test_metrics,
        baselines=baselines,
    )
    logger.info(
        "Late fusion complete | selected_alpha=%.2f val_%s=%.4f",
        selected_alpha,
        spec.select_metric,
        float(best_val_row[spec.select_metric]),
    )
    print(json.dumps(test_metrics_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
