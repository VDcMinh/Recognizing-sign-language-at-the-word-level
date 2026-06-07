"""Training entrypoint for gated skeleton-regions feature fusion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from slr.branches.fusion import (
    PairedSkeletonRegionsDataset,
    build_gated_feature_fusion_from_config,
    load_gated_feature_fusion_config,
    paired_skeleton_regions_collate_fn,
)
from slr.training.losses import build_loss_from_config
from slr.training.metrics import AverageMeter, accuracy_topk
from slr.training.optim import build_optimizer, build_scheduler
from slr.training.seed import set_seed
from slr.training.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    log_wandb_model_artifact,
)
from slr.utils.io import ensure_dir, write_dataframe_csv, write_json, write_yaml
from slr.utils.logging import setup_logger


DEFAULT_TOPK = (1, 5, 10)
FUSION_HEAD_MODULES = (
    "skeleton_proj",
    "region_proj",
    "gate_network",
    "classifier",
)
DEFAULT_REGION_NAMES = ["left_hand", "right_hand", "face"]


class GateStatsTracker:
    """Aggregate gate statistics across validation or evaluation batches."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def update(self, gate: torch.Tensor) -> None:
        values = gate.detach().float().reshape(-1)
        if values.numel() <= 0:
            return
        self.count += int(values.numel())
        self.sum += float(values.sum().item())
        self.sum_sq += float(values.square().sum().item())
        batch_min = float(values.min().item())
        batch_max = float(values.max().item())
        self.min_value = batch_min if self.min_value is None else min(self.min_value, batch_min)
        self.max_value = batch_max if self.max_value is None else max(self.max_value, batch_max)

    def as_dict(self, *, prefix: str) -> dict[str, float]:
        if self.count <= 0:
            return {
                f"{prefix}/gate_mean": 0.0,
                f"{prefix}/gate_std": 0.0,
                f"{prefix}/gate_min": 0.0,
                f"{prefix}/gate_max": 0.0,
            }

        mean = self.sum / self.count
        variance = max((self.sum_sq / self.count) - (mean * mean), 0.0)
        return {
            f"{prefix}/gate_mean": float(mean),
            f"{prefix}/gate_std": float(variance**0.5),
            f"{prefix}/gate_min": float(self.min_value if self.min_value is not None else 0.0),
            f"{prefix}/gate_max": float(self.max_value if self.max_value is not None else 0.0),
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for gated fusion training."""

    parser = argparse.ArgumentParser(
        description="Train the gated feature fusion head over skeleton and regions backbones."
    )
    parser.add_argument("--config", type=Path, required=True, help="Fusion training config YAML.")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment.name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override train.learning_rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override train.weight_decay.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override experiment.output_root.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit the number of train batches per epoch for quick debugging.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit the number of validation batches per epoch for quick debugging.",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val subset size.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override logging.project.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Override logging.entity.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset alignment, model wiring, and one forward/loss pass only.",
    )
    return parser


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    return value


def _normalize_training_config(config_path: Path) -> dict[str, Any]:
    resolved = _stringify_paths(load_gated_feature_fusion_config(config_path))
    resolved["config_path"] = str(config_path.as_posix())
    resolved["project_root"] = str(Path(resolved.get("project_root", Path.cwd())).as_posix())

    experiment = resolved.setdefault("experiment", {})
    dataset = resolved.setdefault("dataset", {})
    dataloader = resolved.setdefault("dataloader", {})
    train_cfg = resolved.setdefault("train", {})
    scheduler = resolved.setdefault("scheduler", {})
    early_stopping = resolved.setdefault("early_stopping", {})
    logging_cfg = resolved.setdefault("logging", {})
    runtime = resolved.setdefault("runtime", {})
    fusion_model = resolved.setdefault("fusion_model", {})
    skeleton_branch = resolved.setdefault("skeleton_branch", {})
    regions_branch = resolved.setdefault("regions_branch", {})

    experiment.setdefault("name", "gated-fusion-nslt100-sel31-ce-regions")
    experiment.setdefault("seed", 42)
    experiment.setdefault("output_root", "outputs/fusion")
    experiment.setdefault("monitor_metric", "val/top5")
    experiment.setdefault("monitor_mode", "max")
    experiment.setdefault("save_every_epoch", False)

    dataloader.setdefault("batch_size", 8)
    dataloader.setdefault("num_workers", 2)
    dataloader.setdefault("pin_memory", True)
    dataloader.setdefault("shuffle_train", bool(dataloader.get("shuffle", True)))

    train_cfg.setdefault("epochs", 50)
    train_cfg.setdefault("device", str(runtime.get("device", "auto")))
    train_cfg.setdefault("optimizer", "adamw")
    train_cfg.setdefault("learning_rate", 1e-3)
    train_cfg.setdefault("weight_decay", 1e-4)
    train_cfg.setdefault("loss", "cross_entropy")
    train_cfg.setdefault("grad_clip_norm", 1.0)
    train_cfg.setdefault("amp", False)

    scheduler.setdefault("enabled", True)
    scheduler.setdefault("name", "cosine")
    scheduler.setdefault("min_lr", 1e-6)
    scheduler.setdefault("step_size", 10)
    scheduler.setdefault("gamma", 0.1)

    early_stopping.setdefault("enabled", True)
    early_stopping.setdefault("monitor_metric", experiment["monitor_metric"])
    early_stopping.setdefault("monitor_mode", experiment["monitor_mode"])
    early_stopping.setdefault("patience", 8)
    early_stopping.setdefault("min_delta", 0.0)

    logging_cfg.setdefault("use_wandb", False)
    logging_cfg.setdefault("entity_env", "WANDB_ENTITY")
    logging_cfg.setdefault("project", "wlasl-gated-fusion-100")
    logging_cfg.setdefault("run_name", experiment["name"])
    logging_cfg.setdefault("tags", [])
    logging_cfg.setdefault("log_model", True)

    runtime.setdefault("device", str(train_cfg["device"]))
    runtime.setdefault("limit_train", None)
    runtime.setdefault("limit_val", None)
    runtime.setdefault("max_train_batches", None)
    runtime.setdefault("max_val_batches", None)

    fusion_model.setdefault("name", "gated_feature_fusion")
    fusion_model.setdefault("hidden_dim", 256)
    fusion_model.setdefault("proj_dropout", 0.2)
    fusion_model.setdefault("classifier_dropout", 0.5)
    fusion_model.setdefault("freeze_skeleton", True)
    fusion_model.setdefault("freeze_regions", True)

    skeleton_branch.setdefault(
        "config_path",
        "checkpoints/models/skeleton/config_resolved.yaml",
    )
    skeleton_branch.setdefault(
        "checkpoint_path",
        skeleton_branch.get("checkpoint", "checkpoints/models/skeleton/best.pt"),
    )
    regions_branch.setdefault(
        "config_path",
        "checkpoints/models/regions/config_resolved.yaml",
    )
    regions_branch.setdefault(
        "checkpoint_path",
        regions_branch.get("checkpoint", "checkpoints/models/regions/best.pt"),
    )

    run_name = str(experiment["name"])
    output_root = Path(experiment["output_root"])
    experiment["output_dir"] = str((output_root / run_name).as_posix())
    logging_cfg["run_name"] = str(logging_cfg.get("run_name") or run_name)
    return resolved


def resolve_training_config(config_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, normalize, and validate one gated fusion training config."""

    config = _normalize_training_config(config_path)
    resolved = copy.deepcopy(config)
    overrides = dict(overrides or {})

    if overrides.get("run_name"):
        resolved["experiment"]["name"] = str(overrides["run_name"])
        resolved["logging"]["run_name"] = str(overrides["run_name"])
    if overrides.get("epochs") is not None:
        resolved["train"]["epochs"] = int(overrides["epochs"])
    if overrides.get("batch_size") is not None:
        resolved["dataloader"]["batch_size"] = int(overrides["batch_size"])
    if overrides.get("learning_rate") is not None:
        resolved["train"]["learning_rate"] = float(overrides["learning_rate"])
    if overrides.get("weight_decay") is not None:
        resolved["train"]["weight_decay"] = float(overrides["weight_decay"])
    if overrides.get("device") is not None:
        resolved["train"]["device"] = str(overrides["device"])
        resolved["runtime"]["device"] = str(overrides["device"])
    if overrides.get("seed") is not None:
        resolved["experiment"]["seed"] = int(overrides["seed"])
    if overrides.get("output_root") is not None:
        resolved["experiment"]["output_root"] = str(Path(overrides["output_root"]).as_posix())
    if overrides.get("num_workers") is not None:
        resolved["dataloader"]["num_workers"] = int(overrides["num_workers"])
    if overrides.get("limit_train") is not None:
        resolved["runtime"]["limit_train"] = int(overrides["limit_train"])
    if overrides.get("limit_val") is not None:
        resolved["runtime"]["limit_val"] = int(overrides["limit_val"])
    if overrides.get("max_train_batches") is not None:
        resolved["runtime"]["max_train_batches"] = int(overrides["max_train_batches"])
    if overrides.get("max_val_batches") is not None:
        resolved["runtime"]["max_val_batches"] = int(overrides["max_val_batches"])
    if overrides.get("no_wandb"):
        resolved["logging"]["use_wandb"] = False
    if overrides.get("wandb_project") is not None:
        resolved["logging"]["project"] = str(overrides["wandb_project"])
    if overrides.get("wandb_entity") is not None:
        resolved["logging"]["entity"] = str(overrides["wandb_entity"])

    run_name = str(resolved["experiment"]["name"])
    resolved["logging"]["run_name"] = str(resolved["logging"].get("run_name") or run_name)
    resolved["experiment"]["output_dir"] = str(
        (Path(resolved["experiment"]["output_root"]) / run_name).as_posix()
    )
    _validate_training_config(resolved)
    return resolved


def _validate_training_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    skeleton_cfg = dataset["skeleton"]
    regions_cfg = dataset["regions"]
    train_cfg = config["train"]
    experiment = config["experiment"]
    early_stopping = config["early_stopping"]

    skeleton_expected_shape = tuple(int(value) for value in skeleton_cfg["expected_shape"])
    regions_expected_shape = tuple(int(value) for value in regions_cfg["expected_shape"])
    active_regions = [str(name) for name in regions_cfg.get("active_regions", [])]

    if str(skeleton_cfg.get("keypoint_set")) != "selected_31":
        raise ValueError("fusion dataset.skeleton.keypoint_set must stay 'selected_31'.")
    if skeleton_expected_shape != (3, 150, 31, 1):
        raise ValueError(
            f"fusion dataset.skeleton.expected_shape must be (3, 150, 31, 1), got {skeleton_expected_shape}."
        )
    if regions_expected_shape != (3, 3, 64, 112, 112):
        raise ValueError(
            "fusion dataset.regions.expected_shape must be "
            f"(3, 3, 64, 112, 112), got {regions_expected_shape}."
        )
    if active_regions != DEFAULT_REGION_NAMES:
        raise ValueError(
            "fusion dataset.regions.active_regions must remain "
            f"{DEFAULT_REGION_NAMES}, got {active_regions}."
        )
    if int(dataset["num_classes"]) <= 0:
        raise ValueError("dataset.num_classes must be positive.")
    if int(train_cfg["epochs"]) <= 0:
        raise ValueError("train.epochs must be positive.")
    if int(config["dataloader"]["batch_size"]) <= 0:
        raise ValueError("dataloader.batch_size must be positive.")
    if float(train_cfg["learning_rate"]) <= 0:
        raise ValueError("train.learning_rate must be positive.")
    if str(train_cfg.get("loss", "cross_entropy")).strip().lower() != "cross_entropy":
        raise ValueError("fusion training currently supports train.loss='cross_entropy' only.")
    if str(experiment["monitor_mode"]).strip().lower() not in {"min", "max"}:
        raise ValueError("experiment.monitor_mode must be 'min' or 'max'.")
    if str(early_stopping["monitor_mode"]).strip().lower() not in {"min", "max"}:
        raise ValueError("early_stopping.monitor_mode must be 'min' or 'max'.")
    if int(early_stopping["patience"]) < 0:
        raise ValueError("early_stopping.patience must be >= 0.")
    for key in ("max_train_batches", "max_val_batches"):
        value = config["runtime"].get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"runtime.{key} must be positive when provided.")


def select_device(device_name: str) -> torch.device:
    """Resolve auto to CUDA when available, otherwise CPU."""

    requested = str(device_name).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    return torch.device(device_name)


def build_paired_datasets(
    config: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, PairedSkeletonRegionsDataset]:
    """Instantiate paired datasets for the requested splits."""

    runtime_cfg = config.get("runtime", {})
    limit_by_split = {
        "train": runtime_cfg.get("limit_train"),
        "val": runtime_cfg.get("limit_val"),
    }
    datasets: dict[str, PairedSkeletonRegionsDataset] = {}
    for split in splits:
        datasets[split] = PairedSkeletonRegionsDataset.from_config(
            config,
            split=split,
            limit=limit_by_split.get(split),
        )
    return datasets


def build_paired_dataloaders(
    config: dict[str, Any],
    datasets: dict[str, PairedSkeletonRegionsDataset],
    *,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Build DataLoaders for paired skeleton-regions datasets."""

    dataloader_cfg = config["dataloader"]
    batch_size = int(dataloader_cfg["batch_size"])
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda"
    shuffle_train = bool(dataloader_cfg.get("shuffle_train", True))

    dataloaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train if split == "train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=paired_skeleton_regions_collate_fn,
        )
    return dataloaders


def _validate_batch_shapes(batch: dict[str, Any], config: dict[str, Any]) -> None:
    skeleton_expected = tuple(int(value) for value in config["dataset"]["skeleton"]["expected_shape"])
    regions_expected = tuple(int(value) for value in config["dataset"]["regions"]["expected_shape"])
    skeleton_shape = tuple(int(value) for value in batch["skeleton"].shape[1:])
    regions_shape = tuple(int(value) for value in batch["regions"].shape[1:])
    if skeleton_shape != skeleton_expected:
        raise ValueError(
            f"Expected skeleton sample shape {skeleton_expected}, got {skeleton_shape}."
        )
    if regions_shape != regions_expected:
        raise ValueError(
            f"Expected regions sample shape {regions_expected}, got {regions_shape}."
        )
    valid_mask = batch.get("regions_valid_mask")
    if valid_mask is not None:
        expected_mask_shape = (
            int(batch["regions"].shape[0]),
            int(regions_expected[0]),
            int(regions_expected[2]),
        )
        if tuple(int(value) for value in valid_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"Expected regions_valid_mask shape {expected_mask_shape}, got {tuple(valid_mask.shape)}."
            )


def _head_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    for module_name in FUSION_HEAD_MODULES:
        module = getattr(model, module_name)
        parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("No trainable fusion-head parameters were found.")
    return parameters


def _count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def _is_improved(
    current: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported mode {mode!r}. Expected 'min' or 'max'.")


def _build_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(config["experiment"]["output_dir"])
    return {
        "output_dir": output_dir,
        "best_checkpoint": output_dir / "best.pt",
        "last_checkpoint": output_dir / "last.pt",
        "config_resolved": output_dir / "config_resolved.yaml",
        "history_csv": output_dir / "training_history.csv",
        "summary_json": output_dir / "training_summary.json",
        "train_log": output_dir / "train.log",
    }


def save_fusion_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    best_metric: float | None,
    best_metric_name: str,
    config: dict[str, Any],
    last_metrics: dict[str, Any],
) -> Path:
    """Persist one fusion checkpoint payload."""

    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": None if best_metric is None else float(best_metric),
        "best_metric_name": str(best_metric_name),
        "last_metrics": dict(last_metrics),
        "config": _stringify_paths(copy.deepcopy(config)),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_fusion_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load one fusion checkpoint and restore state dicts."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'model_state_dict'.")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def run_one_epoch(
    *,
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
    config: dict[str, Any],
    optimizer=None,
    scaler: GradScaler | None = None,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    max_batches: int | None = None,
    collect_gate_stats: bool = False,
    trainable_parameters: list[torch.nn.Parameter] | None = None,
) -> dict[str, float]:
    """Run one train or validation epoch over the paired fusion loader."""

    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    topk_meters = {f"top{k}": AverageMeter(f"top{k}") for k in DEFAULT_TOPK}
    gate_tracker = GateStatsTracker() if collect_gate_stats else None
    batches_ran = 0
    samples_ran = 0

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        _validate_batch_shapes(batch, config)
        skeleton = batch["skeleton"].to(device, non_blocking=device.type == "cuda")
        regions = batch["regions"].to(device, non_blocking=device.type == "cuda")
        labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
        valid_mask = batch.get("regions_valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=device.type == "cuda")
        batch_size = int(labels.shape[0])

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                outputs = model(
                    skeleton,
                    regions,
                    return_features=collect_gate_stats,
                    regions_valid_mask=valid_mask,
                )
                if collect_gate_stats:
                    logits, features = outputs
                else:
                    logits = outputs
                    features = None
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and trainable_parameters is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and trainable_parameters is not None:
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, float(grad_clip_norm))
                    optimizer.step()

        metrics = accuracy_topk(logits.detach(), labels.detach(), topk=DEFAULT_TOPK)
        loss_meter.update(float(loss.item()), n=batch_size)
        for key, meter in topk_meters.items():
            meter.update(float(metrics[key]), n=batch_size)
        if gate_tracker is not None and features is not None:
            gate_tracker.update(features["gate"])
        batches_ran += 1
        samples_ran += batch_size

    if batches_ran <= 0:
        raise RuntimeError("No batches were processed in the current epoch.")

    epoch_metrics = {
        "loss": float(loss_meter.avg),
        "top1": float(topk_meters["top1"].avg),
        "top5": float(topk_meters["top5"].avg),
        "top10": float(topk_meters["top10"].avg),
        "num_batches": int(batches_ran),
        "num_samples": int(samples_ran),
    }
    if gate_tracker is not None:
        epoch_metrics.update(gate_tracker.as_dict(prefix="val"))
    return epoch_metrics


def _log_epoch_summary(logger, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
    logger.info(
        "Epoch %s/%s | train_loss=%.4f train_top1=%.4f train_top5=%.4f train_top10=%.4f "
        "val_loss=%.4f val_top1=%.4f val_top5=%.4f val_top10=%.4f "
        "val_gate_mean=%.4f val_gate_std=%.4f",
        epoch,
        total_epochs,
        metrics["train/loss"],
        metrics["train/top1"],
        metrics["train/top5"],
        metrics["train/top10"],
        metrics["val/loss"],
        metrics["val/top1"],
        metrics["val/top5"],
        metrics["val/top10"],
        metrics["val/gate_mean"],
        metrics["val/gate_std"],
    )


def _write_training_outputs(
    *,
    output_paths: dict[str, Path],
    config: dict[str, Any],
    history_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    ensure_dir(output_paths["output_dir"])
    write_yaml(_stringify_paths(copy.deepcopy(config)), output_paths["config_resolved"])
    write_dataframe_csv(pd.DataFrame(history_rows), output_paths["history_csv"])
    write_json(summary, output_paths["summary_json"])


def _build_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "seed": args.seed,
        "output_root": args.output_root,
        "num_workers": args.num_workers,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "no_wandb": args.no_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }


def run_training(config_path: Path, args: argparse.Namespace) -> int:
    """Train the gated fusion head and persist the requested artifacts."""

    resolved_config = resolve_training_config(config_path, _build_overrides_from_args(args))
    output_paths = _build_output_paths(resolved_config)
    ensure_dir(output_paths["output_dir"])
    logger = setup_logger(
        "slr.branches.fusion.train",
        log_file=None if args.dry_run else output_paths["train_log"],
    )
    device = select_device(str(resolved_config["train"]["device"]))

    logger.info("Resolved run_name=%s device=%s", resolved_config["experiment"]["name"], device)
    logger.info(
        "Fusion head only training | freeze_skeleton=%s freeze_regions=%s",
        resolved_config["fusion_model"]["freeze_skeleton"],
        resolved_config["fusion_model"]["freeze_regions"],
    )

    skeleton_checkpoint = Path(resolved_config["skeleton_branch"]["checkpoint_path"])
    regions_checkpoint = Path(resolved_config["regions_branch"]["checkpoint_path"])
    if not skeleton_checkpoint.exists():
        raise FileNotFoundError(f"Skeleton checkpoint does not exist: {skeleton_checkpoint}")
    if not regions_checkpoint.exists():
        raise FileNotFoundError(f"Regions checkpoint does not exist: {regions_checkpoint}")

    set_seed(int(resolved_config["experiment"]["seed"]))
    datasets = build_paired_datasets(resolved_config, splits=("train", "val"))
    dataloaders = build_paired_dataloaders(resolved_config, datasets, device=device)
    model, build_info = build_gated_feature_fusion_from_config(resolved_config, device=device)
    if not bool(build_info["skeleton"]["checkpoint_loaded"]):
        raise RuntimeError("Skeleton checkpoint was not loaded into the fusion backbone.")
    if not bool(build_info["regions"]["checkpoint_loaded"]):
        raise RuntimeError("Regions checkpoint was not loaded into the fusion backbone.")

    trainable_parameters = _head_parameters(model)
    trainable_parameter_count = _count_parameters(trainable_parameters)
    logger.info(
        "Datasets | train=%s val=%s | trainable_head_params=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        trainable_parameter_count,
    )

    criterion = build_loss_from_config(resolved_config)
    sample_batch = next(iter(dataloaders["train"]))
    _validate_batch_shapes(sample_batch, resolved_config)
    with torch.no_grad():
        logits = model(
            sample_batch["skeleton"].to(device),
            sample_batch["regions"].to(device),
            regions_valid_mask=(
                sample_batch["regions_valid_mask"].to(device)
                if sample_batch.get("regions_valid_mask") is not None
                else None
            ),
        )
        dry_loss = criterion(logits, sample_batch["labels"].to(device))

    if args.dry_run:
        logger.info(
            "Dry run successful | skeleton_batch=%s regions_batch=%s logits=%s loss=%.4f",
            tuple(sample_batch["skeleton"].shape),
            tuple(sample_batch["regions"].shape),
            tuple(logits.shape),
            float(dry_loss.item()),
        )
        return 0

    optimizer = build_optimizer(trainable_parameters, resolved_config["train"])
    scheduler = build_scheduler(
        optimizer,
        resolved_config["scheduler"],
        epochs=int(resolved_config["train"]["epochs"]),
    )

    amp_requested = bool(resolved_config["train"].get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and device.type != "cuda":
        logger.warning("AMP was requested but CUDA is unavailable; AMP has been disabled.")
    scaler = GradScaler(enabled=amp_enabled)

    write_yaml(_stringify_paths(copy.deepcopy(resolved_config)), output_paths["config_resolved"])
    wandb_run = init_wandb_run(
        resolved_config=resolved_config,
        logging_cfg=resolved_config["logging"],
        run_name=str(resolved_config["logging"]["run_name"]),
        logger=logger,
        cli_entity=args.wandb_entity,
    )

    history_rows: list[dict[str, Any]] = []
    monitor_metric = str(resolved_config["experiment"]["monitor_metric"])
    monitor_mode = str(resolved_config["experiment"]["monitor_mode"]).strip().lower()
    early_cfg = resolved_config["early_stopping"]
    early_metric_name = str(early_cfg["monitor_metric"])
    early_mode = str(early_cfg["monitor_mode"]).strip().lower()
    early_patience = int(early_cfg["patience"])
    early_min_delta = float(early_cfg["min_delta"])
    grad_clip_norm_raw = resolved_config["train"].get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm_raw is None else float(grad_clip_norm_raw)

    best_metric: float | None = None
    best_epoch = 0
    best_row: dict[str, Any] | None = None
    early_best_metric: float | None = None
    early_best_epoch = 0
    early_wait = 0
    stopped_epoch: int | None = None

    if bool(early_cfg.get("enabled", False)):
        logger.info(
            "Early stopping enabled | monitor=%s mode=%s patience=%s min_delta=%s",
            early_metric_name,
            early_mode,
            early_patience,
            early_min_delta,
        )

    try:
        total_epochs = int(resolved_config["train"]["epochs"])
        for epoch in range(1, total_epochs + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=dataloaders["train"],
                criterion=criterion,
                device=device,
                config=resolved_config,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
                max_batches=resolved_config["runtime"].get("max_train_batches"),
                collect_gate_stats=False,
                trainable_parameters=trainable_parameters,
            )
            val_metrics = run_one_epoch(
                model=model,
                loader=dataloaders["val"],
                criterion=criterion,
                device=device,
                config=resolved_config,
                amp_enabled=amp_enabled,
                max_batches=resolved_config["runtime"].get("max_val_batches"),
                collect_gate_stats=True,
            )

            if scheduler is not None:
                scheduler.step()

            lr = float(optimizer.param_groups[0]["lr"])
            flat_metrics = {
                "epoch": epoch,
                "lr": lr,
                "train/loss": float(train_metrics["loss"]),
                "train/top1": float(train_metrics["top1"]),
                "train/top5": float(train_metrics["top5"]),
                "train/top10": float(train_metrics["top10"]),
                "val/loss": float(val_metrics["loss"]),
                "val/top1": float(val_metrics["top1"]),
                "val/top5": float(val_metrics["top5"]),
                "val/top10": float(val_metrics["top10"]),
                "val/gate_mean": float(val_metrics["val/gate_mean"]),
                "val/gate_std": float(val_metrics["val/gate_std"]),
                "val/gate_min": float(val_metrics["val/gate_min"]),
                "val/gate_max": float(val_metrics["val/gate_max"]),
            }
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": flat_metrics["train/loss"],
                "train_top1": flat_metrics["train/top1"],
                "train_top5": flat_metrics["train/top5"],
                "train_top10": flat_metrics["train/top10"],
                "val_loss": flat_metrics["val/loss"],
                "val_top1": flat_metrics["val/top1"],
                "val_top5": flat_metrics["val/top5"],
                "val_top10": flat_metrics["val/top10"],
                "val_gate_mean": flat_metrics["val/gate_mean"],
                "val_gate_std": flat_metrics["val/gate_std"],
                "val_gate_min": flat_metrics["val/gate_min"],
                "val_gate_max": flat_metrics["val/gate_max"],
            }
            history_rows.append(row)
            _log_epoch_summary(logger, epoch, total_epochs, flat_metrics)
            log_wandb_metrics(wandb_run, flat_metrics, step=epoch)

            if monitor_metric not in flat_metrics:
                raise KeyError(f"Monitor metric {monitor_metric!r} is missing from epoch metrics.")
            current_metric = float(flat_metrics[monitor_metric])
            if _is_improved(current_metric, best_metric, mode=monitor_mode):
                best_metric = current_metric
                best_epoch = epoch
                best_row = dict(row)
                save_fusion_checkpoint(
                    output_paths["best_checkpoint"],
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    best_metric_name=monitor_metric,
                    config=resolved_config,
                    last_metrics=flat_metrics,
                )

            save_fusion_checkpoint(
                output_paths["last_checkpoint"],
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                best_metric_name=monitor_metric,
                config=resolved_config,
                last_metrics=flat_metrics,
            )

            if bool(resolved_config["experiment"].get("save_every_epoch", False)):
                save_fusion_checkpoint(
                    output_paths["output_dir"] / f"epoch_{epoch:03d}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    best_metric_name=monitor_metric,
                    config=resolved_config,
                    last_metrics=flat_metrics,
                )

            if bool(early_cfg.get("enabled", False)):
                if early_metric_name not in flat_metrics:
                    raise KeyError(
                        f"Early stopping metric {early_metric_name!r} is missing from epoch metrics."
                    )
                current_early_metric = float(flat_metrics[early_metric_name])
                if _is_improved(
                    current_early_metric,
                    early_best_metric,
                    mode=early_mode,
                    min_delta=early_min_delta,
                ):
                    early_best_metric = current_early_metric
                    early_best_epoch = epoch
                    early_wait = 0
                else:
                    early_wait += 1
                    logger.info(
                        "Early stopping wait %s/%s | monitor=%s current=%.4f best=%.4f",
                        early_wait,
                        early_patience,
                        early_metric_name,
                        current_early_metric,
                        0.0 if early_best_metric is None else early_best_metric,
                    )
                    if early_wait >= early_patience:
                        stopped_epoch = epoch
                        logger.info("Early stopping triggered at epoch %s", stopped_epoch)
                        logger.info("Best epoch: %s", early_best_epoch)
                        logger.info(
                            "Best %s: %.4f",
                            early_metric_name,
                            0.0 if early_best_metric is None else early_best_metric,
                        )
                        break

        if best_epoch <= 0 or best_row is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        if wandb_run is not None and bool(resolved_config["logging"].get("log_model", True)):
            artifact_name = f"{resolved_config['experiment']['name']}-{getattr(wandb_run, 'id', 'best')}"
            log_wandb_model_artifact(
                wandb_run,
                output_paths["best_checkpoint"],
                artifact_name=artifact_name,
                aliases=["best"],
            )

        summary = {
            "run_name": str(resolved_config["experiment"]["name"]),
            "output_dir": str(output_paths["output_dir"].as_posix()),
            "config_path": str(config_path.as_posix()),
            "config_resolved_path": str(output_paths["config_resolved"].as_posix()),
            "best_checkpoint": str(output_paths["best_checkpoint"].as_posix()),
            "last_checkpoint": str(output_paths["last_checkpoint"].as_posix()),
            "training_history_csv": str(output_paths["history_csv"].as_posix()),
            "monitor_metric": str(monitor_metric),
            "monitor_mode": str(monitor_mode),
            "best_epoch": int(best_epoch),
            "best_metric": float(best_metric if best_metric is not None else 0.0),
            "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
            "epochs_completed": int(len(history_rows)),
            "metrics": {
                "best_val_loss": float(best_row["val_loss"]),
                "best_val_top1": float(best_row["val_top1"]),
                "best_val_top5": float(best_row["val_top5"]),
                "best_val_top10": float(best_row["val_top10"]),
                "best_val_gate_mean": float(best_row["val_gate_mean"]),
                "best_val_gate_std": float(best_row["val_gate_std"]),
                "best_val_gate_min": float(best_row["val_gate_min"]),
                "best_val_gate_max": float(best_row["val_gate_max"]),
                "final_train_loss": float(history_rows[-1]["train_loss"]),
                "final_val_loss": float(history_rows[-1]["val_loss"]),
            },
            "dataset": {
                "subset": str(resolved_config["dataset"]["subset"]),
                "num_classes": int(resolved_config["dataset"]["num_classes"]),
                "skeleton_keypoint_set": str(resolved_config["dataset"]["skeleton"]["keypoint_set"]),
                "regions_active": list(resolved_config["dataset"]["regions"]["active_regions"]),
                "num_samples": {
                    "train": int(len(datasets["train"])),
                    "val": int(len(datasets["val"])),
                },
                "alignment": {
                    "train": datasets["train"].get_alignment_report(),
                    "val": datasets["val"].get_alignment_report(),
                },
            },
            "fusion_model": {
                "name": str(resolved_config["fusion_model"]["name"]),
                "hidden_dim": int(resolved_config["fusion_model"]["hidden_dim"]),
                "freeze_skeleton": bool(resolved_config["fusion_model"]["freeze_skeleton"]),
                "freeze_regions": bool(resolved_config["fusion_model"]["freeze_regions"]),
                "trainable_modules": list(FUSION_HEAD_MODULES),
                "trainable_parameter_count": int(trainable_parameter_count),
            },
            "branch_info": build_info,
            "early_stopping": {
                "enabled": bool(early_cfg.get("enabled", False)),
                "monitor_metric": str(early_metric_name),
                "monitor_mode": str(early_mode),
                "patience": int(early_patience),
                "min_delta": float(early_min_delta),
                "best_epoch": int(early_best_epoch) if early_best_epoch > 0 else None,
                "best_metric": (
                    float(early_best_metric) if early_best_metric is not None else None
                ),
                "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
            },
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = summary["best_epoch"]
            wandb_run.summary["best_metric"] = summary["best_metric"]
            wandb_run.summary["best_val_top1"] = summary["metrics"]["best_val_top1"]
            wandb_run.summary["best_val_top5"] = summary["metrics"]["best_val_top5"]
            wandb_run.summary["best_val_top10"] = summary["metrics"]["best_val_top10"]
            wandb_run.summary["best_val_gate_mean"] = summary["metrics"]["best_val_gate_mean"]
            wandb_run.summary["best_val_gate_std"] = summary["metrics"]["best_val_gate_std"]

        _write_training_outputs(
            output_paths=output_paths,
            config=resolved_config,
            history_rows=history_rows,
            summary=summary,
        )
        logger.info(
            "Training finished | best_epoch=%s %s=%.4f best_checkpoint=%s",
            best_epoch,
            monitor_metric,
            float(best_metric if best_metric is not None else 0.0),
            output_paths["best_checkpoint"],
        )
        return 0
    finally:
        finish_wandb_run(wandb_run)


def main() -> int:
    """CLI entrypoint for gated fusion training."""

    parser = build_parser()
    args = parser.parse_args()
    return run_training(args.config, args)


__all__ = [
    "FUSION_HEAD_MODULES",
    "GateStatsTracker",
    "build_paired_dataloaders",
    "build_paired_datasets",
    "build_parser",
    "load_fusion_checkpoint",
    "main",
    "resolve_training_config",
    "run_one_epoch",
    "run_training",
    "save_fusion_checkpoint",
    "select_device",
]
