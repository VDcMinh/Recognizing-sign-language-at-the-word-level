"""Training and evaluation entrypoints for region tensor baselines."""

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

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn
from slr.branches.regions.models import build_region_model
from slr.training.checkpointing import load_checkpoint, save_checkpoint
from slr.training.losses import (
    build_loss_from_config,
    get_label_smoothing_epsilon,
    get_loss_name,
)
from slr.training.metrics import AverageMeter, accuracy_topk
from slr.training.optim import build_optimizer, build_scheduler
from slr.training.seed import set_seed
from slr.training.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    log_wandb_model_artifact,
)
from slr.utils.io import ensure_dir, read_yaml, write_dataframe_csv, write_json, write_yaml
from slr.utils.logging import setup_logger


DEFAULT_TOPK = (1, 5, 10)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for regions baseline training."""

    parser = argparse.ArgumentParser(
        description="Train a regions baseline on local region clip tensors."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment.output_dir.")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override train.optimizer.lr.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override train.optimizer.weight_decay.")
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset.data_root.")
    parser.add_argument("--train-manifest", type=Path, default=None, help="Override dataset.manifests.train.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Override dataset.manifests.val.")
    parser.add_argument("--test-manifest", type=Path, default=None, help="Override dataset.manifests.test.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val subset size.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test subset size.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override logging.project.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Override the W&B entity.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, datasets, model, and one forward/loss pass without training.",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for local checkpoint evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a regions checkpoint against train/val/test tensor manifests."
    )
    parser.add_argument("--config", type=Path, required=True, help="Resolved config YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file to load.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train.batch_size override.")
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    return parser


def _attach_loss_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Attach resolved loss metadata to the runtime section."""

    runtime_cfg = config.setdefault("runtime", {})
    runtime_cfg["loss_type"] = get_loss_name(config)
    runtime_cfg["label_smoothing_epsilon"] = float(get_label_smoothing_epsilon(config))
    return config


def _format_loss_log(config: dict[str, Any]) -> str:
    """Render one short startup log line for the configured loss."""

    loss_type = str(config["runtime"]["loss_type"])
    epsilon = float(config["runtime"]["label_smoothing_epsilon"])
    if loss_type == "standard_label_smoothing":
        return f"Loss: {loss_type} epsilon={epsilon:g}"
    return f"Loss: {loss_type}"


def _normalize_training_config(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Fill missing sections and defaults for regions baseline training."""

    resolved = copy.deepcopy(config)
    resolved["config_path"] = str(config_path.as_posix())
    resolved["project_root"] = str(Path.cwd().as_posix())

    experiment = resolved.setdefault("experiment", {})
    dataset = resolved.setdefault("dataset", {})
    model = resolved.setdefault("model", {})
    train_cfg = resolved.setdefault("train", {})
    logging_cfg = resolved.setdefault("logging", {})
    runtime = resolved.setdefault("runtime", {})
    early_stopping_cfg = resolved.setdefault("early_stopping", {})
    augmentation_cfg = resolved.setdefault("augmentation", {})

    experiment.setdefault("name", "regions-resnet18-gru")
    experiment.setdefault("output_dir", f"outputs/regions/{experiment['name']}")
    experiment.setdefault("monitor_metric", train_cfg.get("save_best_metric", "val_top1"))
    experiment.setdefault("monitor_mode", "max")
    experiment.setdefault("save_every_epoch", False)
    experiment.setdefault("seed", 42)

    dataset.setdefault("name", "WLASL")
    dataset.setdefault("subset", "nslt100")
    dataset.setdefault("data_root", "data/datasets/WLASL/branch_inputs/regions/rtmw_l")
    dataset.setdefault("num_classes", 100)
    dataset.setdefault("expected_shape", [3, 3, 64, 112, 112])
    dataset.setdefault("region_order", ["left_hand", "right_hand", "face"])
    normalize_cfg = dataset.setdefault("normalize", {})
    dataset.setdefault("return_metadata", True)
    dataset.setdefault("strict_shape_check", True)
    manifests = dataset.setdefault("manifests", {})
    for split in ("train", "val", "test"):
        manifests.setdefault(split, "")

    model.setdefault("name", "region_resnet18_gru")
    model.setdefault("num_classes", int(dataset.get("num_classes", 100)))
    model.setdefault("num_regions", int(dataset.get("expected_shape", [3])[0]))
    model.setdefault("in_channels", int(dataset.get("expected_shape", [3, 3])[1]))
    model.setdefault("clip_len", int(dataset.get("expected_shape", [3, 3, 64])[2]))
    model.setdefault("crop_size", int(dataset.get("expected_shape", [3, 3, 64, 112, 112])[3]))
    model.setdefault("cnn_feature_dim", 256)
    model.setdefault("pretrained", True)
    model.setdefault("freeze_encoder", True)
    model.setdefault("encoder_name", "resnet18")
    model.setdefault("encoder_feature_dim", 512)
    model.setdefault("gru_hidden_size", 128)
    model.setdefault("gru_num_layers", 1)
    model.setdefault("bidirectional", True)
    model.setdefault("dropout", 0.5)
    model.setdefault("fusion", "concat")
    model.setdefault("use_valid_mask", True)
    normalize_cfg.setdefault(
        "type",
        "imagenet" if str(model.get("name", "")).strip().lower() == "region_resnet18_gru" else "none",
    )

    train_cfg.setdefault("epochs", 30)
    train_cfg.setdefault("batch_size", 8)
    train_cfg.setdefault("num_workers", 2)
    train_cfg.setdefault("pin_memory", True)
    train_cfg.setdefault("shuffle_train", True)
    train_cfg.setdefault("device", "auto")
    train_cfg.setdefault("loss", "cross_entropy")
    train_cfg.setdefault("grad_clip_norm", None)
    train_cfg.setdefault("amp", False)
    train_cfg.setdefault("save_best_metric", "val_top1")
    optimizer_cfg = train_cfg.setdefault("optimizer", {})
    optimizer_cfg.setdefault("name", "adamw")
    optimizer_cfg.setdefault("lr", 3e-4)
    optimizer_cfg.setdefault("weight_decay", 1e-4)
    scheduler_cfg = train_cfg.setdefault("scheduler", {})
    scheduler_cfg.setdefault("enabled", True)
    scheduler_cfg.setdefault("name", "cosine")
    scheduler_cfg.setdefault("min_lr", 1e-6)

    logging_cfg.setdefault("use_wandb", False)
    logging_cfg.setdefault("entity_env", "WANDB_ENTITY")
    logging_cfg.setdefault("project", "wlasl-regions")
    logging_cfg.setdefault("run_name", experiment["name"])
    logging_cfg.setdefault("tags", [])
    logging_cfg.setdefault("log_model", True)

    runtime.setdefault("limit_train", None)
    runtime.setdefault("limit_val", None)
    runtime.setdefault("limit_test", None)

    early_stopping_cfg.setdefault("enabled", False)
    early_stopping_cfg.setdefault("monitor", "val_top5")
    early_stopping_cfg.setdefault(
        "mode",
        "min" if str(early_stopping_cfg.get("monitor", "")).strip().lower() == "val_loss" else "max",
    )
    early_stopping_cfg.setdefault("patience", 4)
    early_stopping_cfg.setdefault("min_delta", 0.0)

    augmentation_cfg.setdefault("enabled", False)
    color_jitter_cfg = augmentation_cfg.setdefault("color_jitter", {})
    color_jitter_cfg.setdefault("enabled", False)
    color_jitter_cfg.setdefault("brightness", 0.2)
    color_jitter_cfg.setdefault("contrast", 0.2)
    color_jitter_cfg.setdefault("saturation", 0.15)
    color_jitter_cfg.setdefault("hue", 0.05)
    random_resized_crop_cfg = augmentation_cfg.setdefault("random_resized_crop", {})
    random_resized_crop_cfg.setdefault("enabled", False)
    random_resized_crop_cfg.setdefault("scale", [0.85, 1.0])
    random_erasing_cfg = augmentation_cfg.setdefault("random_erasing", {})
    random_erasing_cfg.setdefault("enabled", False)
    random_erasing_cfg.setdefault("p", 0.15)
    temporal_dropout_cfg = augmentation_cfg.setdefault("temporal_dropout", {})
    temporal_dropout_cfg.setdefault("enabled", False)
    temporal_dropout_cfg.setdefault("p", 0.10)
    region_dropout_cfg = augmentation_cfg.setdefault("region_dropout", {})
    region_dropout_cfg.setdefault("enabled", False)
    region_dropout_cfg.setdefault("p", 0.10)
    return resolved


def _resolve_output_dir(existing_output_dir: str, run_name: str, override_output_dir: Path | None) -> str:
    """Resolve the final output directory from config defaults and CLI overrides."""

    if override_output_dir is not None:
        return str(Path(override_output_dir).as_posix())

    configured = Path(existing_output_dir)
    if configured.name:
        return str((configured.parent / run_name).as_posix())
    return str((configured / run_name).as_posix())


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply user CLI overrides to one normalized config."""

    resolved = copy.deepcopy(config)

    if args.run_name:
        resolved["experiment"]["name"] = str(args.run_name)
        resolved["logging"]["run_name"] = str(args.run_name)
    else:
        resolved["logging"]["run_name"] = str(
            resolved["logging"].get("run_name") or resolved["experiment"]["name"]
        )

    if args.epochs is not None:
        resolved["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        resolved["train"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        resolved["train"]["optimizer"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        resolved["train"]["optimizer"]["weight_decay"] = float(args.weight_decay)
    if args.device is not None:
        resolved["train"]["device"] = str(args.device)
    if args.seed is not None:
        resolved["experiment"]["seed"] = int(args.seed)
    if args.data_root is not None:
        resolved["dataset"]["data_root"] = str(Path(args.data_root).as_posix())
    if args.train_manifest is not None:
        resolved["dataset"]["manifests"]["train"] = str(Path(args.train_manifest).as_posix())
    if args.val_manifest is not None:
        resolved["dataset"]["manifests"]["val"] = str(Path(args.val_manifest).as_posix())
    if args.test_manifest is not None:
        resolved["dataset"]["manifests"]["test"] = str(Path(args.test_manifest).as_posix())
    if args.no_wandb:
        resolved["logging"]["use_wandb"] = False
    if args.wandb_project is not None:
        resolved["logging"]["project"] = str(args.wandb_project)
    if args.wandb_entity is not None:
        resolved["logging"]["entity"] = str(args.wandb_entity)

    resolved["runtime"]["limit_train"] = args.limit_train
    resolved["runtime"]["limit_val"] = args.limit_val
    resolved["runtime"]["limit_test"] = args.limit_test

    run_name = str(resolved["experiment"]["name"])
    resolved["logging"]["run_name"] = str(resolved["logging"].get("run_name") or run_name)
    resolved["experiment"]["output_dir"] = _resolve_output_dir(
        str(resolved["experiment"]["output_dir"]),
        run_name,
        args.output_dir,
    )
    return resolved


def resolve_training_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load, normalize, and override one regions training config."""

    config = read_yaml(config_path)
    normalized = _normalize_training_config(config, config_path=config_path)
    resolved = apply_cli_overrides(normalized, args)
    resolved = _attach_loss_metadata(resolved)

    expected_shape = tuple(int(value) for value in resolved["dataset"]["expected_shape"])
    if len(expected_shape) != 5:
        raise ValueError("dataset.expected_shape must contain [R, C, T, H, W].")
    if expected_shape[0] != int(resolved["model"]["num_regions"]):
        raise ValueError(
            f"dataset.expected_shape[0]={expected_shape[0]} does not match model.num_regions="
            f"{resolved['model']['num_regions']}."
        )
    if expected_shape[1] != int(resolved["model"]["in_channels"]):
        raise ValueError(
            f"dataset.expected_shape[1]={expected_shape[1]} does not match model.in_channels="
            f"{resolved['model']['in_channels']}."
        )
    if expected_shape[2] != int(resolved["model"]["clip_len"]):
        raise ValueError(
            f"dataset.expected_shape[2]={expected_shape[2]} does not match model.clip_len="
            f"{resolved['model']['clip_len']}."
        )
    if expected_shape[3] != int(resolved["model"]["crop_size"]) or expected_shape[4] != int(
        resolved["model"]["crop_size"]
    ):
        raise ValueError("dataset crop size must match model.crop_size.")
    if int(resolved["model"]["num_classes"]) != int(resolved["dataset"]["num_classes"]):
        raise ValueError("model.num_classes must match dataset.num_classes.")
    if len(resolved["dataset"]["region_order"]) != int(resolved["model"]["num_regions"]):
        raise ValueError("dataset.region_order length must match model.num_regions.")
    _resolve_early_stopping_config(resolved)
    return resolved


def select_device(device_name: str, *, logger) -> torch.device:
    """Resolve ``auto`` and gracefully fall back to CPU when CUDA is unavailable."""

    requested = str(device_name).strip().lower()
    if requested == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device requested=auto resolved=%s", resolved)
        return resolved
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_region_datasets(config: dict[str, Any]) -> dict[str, RegionClipDataset]:
    """Instantiate train/val/test datasets from one resolved config."""

    runtime_cfg = config.get("runtime", {})
    limits = {
        "train": runtime_cfg.get("limit_train"),
        "val": runtime_cfg.get("limit_val"),
        "test": runtime_cfg.get("limit_test"),
    }

    datasets: dict[str, RegionClipDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = RegionClipDataset.from_config(
            config,
            split=split,
            limit=limits.get(split),
        )
    return datasets


def build_region_dataloaders(
    config: dict[str, Any],
    datasets: dict[str, RegionClipDataset],
    *,
    device: torch.device,
    batch_size_override: int | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for each requested split."""

    train_cfg = config["train"]
    batch_size = int(batch_size_override or train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False)) and device.type == "cuda"
    shuffle_train = bool(train_cfg.get("shuffle_train", True))

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=region_collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=region_collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=region_collate_fn,
        ),
    }


def _normalize_optimizer_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map nested optimizer config to the shared helper format."""

    optimizer_cfg = dict(train_cfg.get("optimizer", {}))
    return {
        "optimizer": str(optimizer_cfg.get("name", "adamw")),
        "learning_rate": float(optimizer_cfg.get("lr", 3e-4)),
        "weight_decay": float(optimizer_cfg.get("weight_decay", 1e-4)),
        "momentum": float(optimizer_cfg.get("momentum", 0.9)),
        "nesterov": bool(optimizer_cfg.get("nesterov", True)),
    }


def _normalize_scheduler_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map nested scheduler config to the shared helper format."""

    scheduler_cfg = dict(train_cfg.get("scheduler", {}))
    return {
        "enabled": bool(scheduler_cfg.get("enabled", True)),
        "name": str(scheduler_cfg.get("name", "cosine")),
        "min_lr": float(scheduler_cfg.get("min_lr", 1e-6)),
        "step_size": int(scheduler_cfg.get("step_size", 10)),
        "gamma": float(scheduler_cfg.get("gamma", 0.1)),
    }


def _log_optimizer_config(logger, train_cfg: dict[str, Any]) -> None:
    """Log the resolved optimizer configuration used for training."""

    optimizer_cfg = dict(train_cfg.get("optimizer", {}))
    logger.info("Optimizer: %s", str(optimizer_cfg.get("name", "adamw")).strip().lower())
    logger.info("Learning rate: %s", float(optimizer_cfg.get("lr", 3e-4)))
    logger.info("Weight decay: %s", float(optimizer_cfg.get("weight_decay", 1e-4)))


def _validate_batch_shape(batch_data: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    if batch_data.ndim != 6:
        raise ValueError(
            "Expected batched region tensors with shape (N, R, C, T, H, W), "
            f"got {tuple(batch_data.shape)}."
        )
    actual = tuple(int(value) for value in batch_data.shape[1:])
    if actual != expected_shape:
        raise ValueError(f"Expected batch sample shape {expected_shape}, got {actual}.")


def run_one_epoch_with_shape(
    *,
    expected_shape: tuple[int, ...],
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
    scaler: GradScaler | None = None,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Run one epoch with explicit batch shape validation."""

    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    topk_meters = {f"top{k}": AverageMeter(f"top{k}") for k in DEFAULT_TOPK}

    for batch in loader:
        data = batch["data"]
        labels = batch["labels"]
        valid_mask = batch.get("valid_mask")

        _validate_batch_shape(data, expected_shape)
        data = data.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        valid_mask = (
            valid_mask.to(device=device, non_blocking=device.type == "cuda")
            if valid_mask is not None
            else None
        )
        batch_size = int(labels.shape[0])

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                logits = model(data, valid_mask=valid_mask)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()

        metrics = accuracy_topk(logits.detach(), labels.detach(), topk=DEFAULT_TOPK)
        loss_meter.update(float(loss.item()), n=batch_size)
        for key, meter in topk_meters.items():
            meter.update(float(metrics[key]), n=batch_size)

    epoch_metrics = {"loss": loss_meter.avg}
    for key, meter in topk_meters.items():
        epoch_metrics[key] = meter.avg
    return epoch_metrics


def _build_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    return {
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "best_checkpoint": checkpoints_dir / "best.pt",
        "last_checkpoint": checkpoints_dir / "last.pt",
        "config_resolved": output_dir / "config_resolved.yaml",
        "metrics_json": output_dir / "metrics.json",
        "train_log_csv": output_dir / "train_log.csv",
        "summary_json": output_dir / "summary.json",
        "train_log_txt": output_dir / "train.log",
        "eval_best_json": output_dir / "eval_test_best.json",
    }


def _write_training_outputs(
    *,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    epoch_records: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
    summary: dict[str, Any],
    eval_result: dict[str, Any],
) -> None:
    ensure_dir(output_paths["output_dir"])
    ensure_dir(output_paths["checkpoints_dir"])
    write_yaml(config, output_paths["config_resolved"])
    write_json(metrics_summary, output_paths["metrics_json"])
    write_json(summary, output_paths["summary_json"])
    write_json(eval_result, output_paths["eval_best_json"])
    write_dataframe_csv(pd.DataFrame(epoch_records), output_paths["train_log_csv"])


def _resolve_metric_spec(metric_name: str) -> tuple[str, str]:
    """Resolve metric names like ``val_top1`` to runtime keys plus modes."""

    metric_mapping = {
        "val_top1": ("val/top1", "max"),
        "val_top5": ("val/top5", "max"),
        "val_top10": ("val/top10", "max"),
        "val_loss": ("val/loss", "min"),
    }
    return metric_mapping.get(str(metric_name).strip().lower(), ("val/top1", "max"))


def _resolve_monitor_metric(config: dict[str, Any]) -> tuple[str, str]:
    """Resolve the checkpoint monitor metric from train config."""

    monitor_metric = str(config["train"].get("save_best_metric", "val_top1")).strip().lower()
    return _resolve_metric_spec(monitor_metric)


def _resolve_early_stopping_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve early stopping settings while preserving older configs."""

    early_cfg = dict(config.get("early_stopping", {}))
    enabled = bool(early_cfg.get("enabled", False))
    monitor = str(early_cfg.get("monitor", "val_top5")).strip().lower()
    metric_key, inferred_mode = _resolve_metric_spec(monitor)
    mode = str(early_cfg.get("mode", inferred_mode)).strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be either 'min' or 'max'.")
    return {
        "enabled": enabled,
        "monitor": monitor,
        "metric_key": metric_key,
        "mode": mode,
        "patience": int(early_cfg.get("patience", 4)),
        "min_delta": float(early_cfg.get("min_delta", 0.0)),
    }


def _is_improved(current: float, best: float | None, *, mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported monitor mode {mode!r}. Expected 'max' or 'min'.")


def _log_epoch_summary(logger, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
    logger.info(
        "Epoch %s/%s | train_loss=%.4f train_top1=%.4f train_top5=%.4f train_top10=%.4f "
        "val_loss=%.4f val_top1=%.4f val_top5=%.4f val_top10=%.4f",
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
    )


def run_training(config_path: Path, args: argparse.Namespace) -> int:
    """Train a RegionCNNGRU baseline and persist all required outputs."""

    resolved_config = resolve_training_config(config_path, args)
    output_paths = _build_output_paths(resolved_config)
    ensure_dir(output_paths["output_dir"])
    ensure_dir(output_paths["checkpoints_dir"])

    logger = setup_logger(
        "slr.branches.regions.train",
        log_file=None if args.dry_run else output_paths["train_log_txt"],
    )
    device = select_device(str(resolved_config["train"]["device"]), logger=logger)
    expected_shape = tuple(int(value) for value in resolved_config["dataset"]["expected_shape"])
    run_name = str(resolved_config["experiment"]["name"])

    logger.info("Resolved run_name=%s device=%s", run_name, device)
    logger.info(_format_loss_log(resolved_config))
    _log_optimizer_config(logger, resolved_config["train"])
    set_seed(int(resolved_config["experiment"]["seed"]))

    datasets = build_region_datasets(resolved_config)
    dataloaders = build_region_dataloaders(resolved_config, datasets, device=device)
    model = build_region_model(resolved_config["model"]).to(device)

    logger.info(
        "Datasets | train=%s val=%s test=%s | model=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        model.__class__.__name__,
    )

    criterion = build_loss_from_config(resolved_config)
    sample_batch = next(iter(dataloaders["train"]))
    _validate_batch_shape(sample_batch["data"], expected_shape)

    with torch.no_grad():
        logits = model(
            sample_batch["data"].to(device),
            valid_mask=(
                sample_batch["valid_mask"].to(device)
                if sample_batch.get("valid_mask") is not None
                else None
            ),
        )
        dry_loss = criterion(logits, sample_batch["labels"].to(device))

    if args.dry_run:
        logger.info(
            "Dry run successful | batch_shape=%s logits_shape=%s loss=%.4f output_dir=%s",
            tuple(sample_batch["data"].shape),
            tuple(logits.shape),
            float(dry_loss.item()),
            output_paths["output_dir"],
        )
        return 0

    optimizer = build_optimizer(model.parameters(), _normalize_optimizer_config(resolved_config["train"]))
    scheduler = build_scheduler(
        optimizer,
        _normalize_scheduler_config(resolved_config["train"]),
        epochs=int(resolved_config["train"]["epochs"]),
    )

    amp_requested = bool(resolved_config["train"].get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and device.type != "cuda":
        logger.warning("AMP was requested but CUDA is unavailable; AMP has been disabled.")
    scaler = GradScaler(enabled=amp_enabled)

    wandb_run = init_wandb_run(
        resolved_config=resolved_config,
        logging_cfg=resolved_config["logging"],
        run_name=str(resolved_config["logging"]["run_name"]),
        logger=logger,
        cli_entity=args.wandb_entity,
    )

    best_metric: float | None = None
    best_epoch = 0
    best_row: dict[str, Any] | None = None
    epoch_records: list[dict[str, Any]] = []
    monitor_metric, monitor_mode = _resolve_monitor_metric(resolved_config)
    early_stopping_cfg = _resolve_early_stopping_config(resolved_config)
    early_stopping_best_metric: float | None = None
    early_stopping_best_epoch = 0
    early_stopping_wait = 0
    stopped_epoch: int | None = None

    if early_stopping_cfg["enabled"]:
        logger.info(
            "Early stopping enabled | monitor=%s mode=%s patience=%s min_delta=%s",
            early_stopping_cfg["monitor"],
            early_stopping_cfg["mode"],
            early_stopping_cfg["patience"],
            early_stopping_cfg["min_delta"],
        )

    try:
        total_epochs = int(resolved_config["train"]["epochs"])
        grad_clip_norm_raw = resolved_config["train"].get("grad_clip_norm")
        grad_clip_norm = None if grad_clip_norm_raw is None else float(grad_clip_norm_raw)

        for epoch in range(1, total_epochs + 1):
            train_metrics = run_one_epoch_with_shape(
                expected_shape=expected_shape,
                model=model,
                loader=dataloaders["train"],
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
            )
            val_metrics = run_one_epoch_with_shape(
                expected_shape=expected_shape,
                model=model,
                loader=dataloaders["val"],
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
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
            }
            epoch_records.append(row)
            _log_epoch_summary(logger, epoch, total_epochs, flat_metrics)
            log_wandb_metrics(wandb_run, flat_metrics, step=epoch)

            current_metric = float(flat_metrics[monitor_metric])
            is_best = _is_improved(current_metric, best_metric, mode=monitor_mode)
            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                best_row = dict(row)
                save_checkpoint(
                    output_paths["best_checkpoint"],
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    last_metrics=flat_metrics,
                    config=resolved_config,
                    keypoint_set="regions",
                    num_classes=int(resolved_config["dataset"]["num_classes"]),
                    num_nodes=int(resolved_config["model"]["num_regions"]),
                    model_name=str(resolved_config["model"]["name"]),
                    class_id_to_gloss=datasets["train"].id_to_gloss,
                )

            save_checkpoint(
                output_paths["last_checkpoint"],
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                last_metrics=flat_metrics,
                config=resolved_config,
                keypoint_set="regions",
                num_classes=int(resolved_config["dataset"]["num_classes"]),
                num_nodes=int(resolved_config["model"]["num_regions"]),
                model_name=str(resolved_config["model"]["name"]),
                class_id_to_gloss=datasets["train"].id_to_gloss,
            )

            if bool(resolved_config["experiment"].get("save_every_epoch", False)):
                save_checkpoint(
                    output_paths["checkpoints_dir"] / f"epoch_{epoch:03d}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    last_metrics=flat_metrics,
                    config=resolved_config,
                    keypoint_set="regions",
                    num_classes=int(resolved_config["dataset"]["num_classes"]),
                    num_nodes=int(resolved_config["model"]["num_regions"]),
                    model_name=str(resolved_config["model"]["name"]),
                    class_id_to_gloss=datasets["train"].id_to_gloss,
                )

            if early_stopping_cfg["enabled"]:
                current_early_metric = float(flat_metrics[early_stopping_cfg["metric_key"]])
                early_metric_improved = _is_improved(
                    current_early_metric,
                    early_stopping_best_metric,
                    mode=early_stopping_cfg["mode"],
                    min_delta=float(early_stopping_cfg["min_delta"]),
                )
                if early_metric_improved:
                    early_stopping_best_metric = current_early_metric
                    early_stopping_best_epoch = epoch
                    early_stopping_wait = 0
                else:
                    early_stopping_wait += 1
                    logger.info(
                        "Early stopping wait %s/%s | monitor=%s current=%.4f best=%.4f",
                        early_stopping_wait,
                        early_stopping_cfg["patience"],
                        early_stopping_cfg["monitor"],
                        current_early_metric,
                        0.0 if early_stopping_best_metric is None else early_stopping_best_metric,
                    )
                    if early_stopping_wait >= int(early_stopping_cfg["patience"]):
                        stopped_epoch = epoch
                        logger.info(
                            "Early stopping triggered at epoch=%s | monitor=%s best_epoch=%s best_metric=%.4f patience=%s",
                            stopped_epoch,
                            early_stopping_cfg["monitor"],
                            early_stopping_best_epoch,
                            0.0 if early_stopping_best_metric is None else early_stopping_best_metric,
                            early_stopping_cfg["patience"],
                        )
                        break

        if best_epoch <= 0:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        load_checkpoint(output_paths["best_checkpoint"], model, map_location=device)
        test_metrics = run_one_epoch_with_shape(
            expected_shape=expected_shape,
            model=model,
            loader=dataloaders["test"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )
        eval_result = {
            "split": "test",
            "checkpoint": str(output_paths["best_checkpoint"].as_posix()),
            "epoch": int(best_epoch),
            "loss": float(test_metrics["loss"]),
            "top1": float(test_metrics["top1"]),
            "top5": float(test_metrics["top5"]),
            "top10": float(test_metrics["top10"]),
        }
        log_wandb_metrics(
            wandb_run,
            {
                "test/loss": eval_result["loss"],
                "test/top1": eval_result["top1"],
                "test/top5": eval_result["top5"],
                "test/top10": eval_result["top10"],
            },
            step=best_epoch,
        )

        if wandb_run is not None and bool(resolved_config["logging"].get("log_model", True)):
            artifact_name = f"{run_name}-{getattr(wandb_run, 'id', 'best')}"
            log_wandb_model_artifact(
                wandb_run,
                output_paths["best_checkpoint"],
                artifact_name=artifact_name,
                aliases=["best"],
            )

        metrics_summary = {
            "loss_type": str(resolved_config["runtime"]["loss_type"]),
            "label_smoothing_epsilon": float(resolved_config["runtime"]["label_smoothing_epsilon"]),
            "best_epoch": int(best_epoch),
            "best_val_top1": float(best_row["val_top1"]) if best_row is not None else 0.0,
            "best_val_top5": float(best_row["val_top5"]) if best_row is not None else 0.0,
            "best_val_top10": float(best_row["val_top10"]) if best_row is not None else 0.0,
            "best_val_loss": float(best_row["val_loss"]) if best_row is not None else 0.0,
            "test_loss": float(test_metrics["loss"]),
            "test_top1": float(test_metrics["top1"]),
            "test_top5": float(test_metrics["top5"]),
            "test_top10": float(test_metrics["top10"]),
            "final_train_loss": float(epoch_records[-1]["train_loss"]),
            "final_val_loss": float(epoch_records[-1]["val_loss"]),
        }

        summary = {
            "run_name": run_name,
            "loss_type": str(resolved_config["runtime"]["loss_type"]),
            "label_smoothing_epsilon": float(resolved_config["runtime"]["label_smoothing_epsilon"]),
            "branch": "regions",
            "model_name": str(resolved_config["model"]["name"]),
            "output_dir": str(output_paths["output_dir"].as_posix()),
            "best_checkpoint": str(output_paths["best_checkpoint"].as_posix()),
            "last_checkpoint": str(output_paths["last_checkpoint"].as_posix()),
            "config_path": str(config_path.as_posix()),
            "dataset": {
                "subset": str(resolved_config["dataset"]["subset"]),
                "num_classes": int(resolved_config["dataset"]["num_classes"]),
                "expected_shape": list(expected_shape),
                "region_order": list(resolved_config["dataset"]["region_order"]),
                "normalize": dict(resolved_config["dataset"].get("normalize", {})),
                "num_samples": {
                    "train": len(datasets["train"]),
                    "val": len(datasets["val"]),
                    "test": len(datasets["test"]),
                },
            },
            "model": {
                "name": str(resolved_config["model"]["name"]),
                "pretrained": bool(resolved_config["model"].get("pretrained", False)),
                "freeze_encoder": bool(resolved_config["model"].get("freeze_encoder", False)),
                "encoder_name": str(resolved_config["model"].get("encoder_name", "")),
                "encoder_feature_dim": int(resolved_config["model"].get("encoder_feature_dim", 0)),
                "gru_hidden_size": int(resolved_config["model"].get("gru_hidden_size", 0)),
                "gru_num_layers": int(resolved_config["model"].get("gru_num_layers", 0)),
                "bidirectional": bool(resolved_config["model"].get("bidirectional", False)),
                "fusion": str(resolved_config["model"].get("fusion", "")),
                "use_valid_mask": bool(resolved_config["model"].get("use_valid_mask", False)),
            },
            "early_stopping": {
                "enabled": bool(early_stopping_cfg["enabled"]),
                "monitor": str(early_stopping_cfg["monitor"]),
                "mode": str(early_stopping_cfg["mode"]),
                "patience": int(early_stopping_cfg["patience"]),
                "min_delta": float(early_stopping_cfg["min_delta"]),
                "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
                "best_epoch": int(early_stopping_best_epoch) if early_stopping_cfg["enabled"] else None,
                "best_metric": (
                    float(early_stopping_best_metric)
                    if early_stopping_cfg["enabled"] and early_stopping_best_metric is not None
                    else None
                ),
            },
            "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
            "best_epoch": int(early_stopping_best_epoch) if early_stopping_cfg["enabled"] else None,
            "best_metric": (
                float(early_stopping_best_metric)
                if early_stopping_cfg["enabled"] and early_stopping_best_metric is not None
                else None
            ),
            "monitor": str(early_stopping_cfg["monitor"]) if early_stopping_cfg["enabled"] else None,
            "patience": int(early_stopping_cfg["patience"]) if early_stopping_cfg["enabled"] else None,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        _write_training_outputs(
            config=resolved_config,
            output_paths=output_paths,
            epoch_records=epoch_records,
            metrics_summary=metrics_summary,
            summary=summary,
            eval_result=eval_result,
        )
        logger.info("Training finished. Best epoch=%s test_top1=%.4f", best_epoch, test_metrics["top1"])
        return 0
    finally:
        finish_wandb_run(wandb_run)


def run_evaluation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint against a selected split."""

    raw_config = read_yaml(config_path)
    config = _normalize_training_config(raw_config, config_path=config_path)
    if batch_size is not None:
        config["train"]["batch_size"] = int(batch_size)
    if device_name is not None:
        config["train"]["device"] = str(device_name)
    config = _attach_loss_metadata(config)

    logger = setup_logger("slr.branches.regions.evaluate")
    device = select_device(str(config["train"]["device"]), logger=logger)
    expected_shape = tuple(int(value) for value in config["dataset"]["expected_shape"])
    datasets = build_region_datasets(config)
    dataloaders = build_region_dataloaders(
        config,
        datasets,
        device=device,
        batch_size_override=batch_size,
    )
    model = build_region_model(config["model"]).to(device)
    criterion = build_loss_from_config(config)
    payload = load_checkpoint(checkpoint_path, model, map_location=device)
    metrics = run_one_epoch_with_shape(
        expected_shape=expected_shape,
        model=model,
        loader=dataloaders[split],
        criterion=criterion,
        device=device,
        amp_enabled=bool(config["train"].get("amp", False)) and device.type == "cuda",
    )

    result = {
        "split": split,
        "checkpoint": str(checkpoint_path.as_posix()),
        "epoch": int(payload.get("epoch", 0)),
        "loss": float(metrics["loss"]),
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "top10": float(metrics["top10"]),
    }
    logger.info(
        "Evaluation split=%s loss=%.4f top1=%.4f top5=%.4f top10=%.4f",
        split,
        result["loss"],
        result["top1"],
        result["top5"],
        result["top10"],
    )

    output_dir_text = config.get("experiment", {}).get("output_dir")
    if output_dir_text:
        output_dir = Path(output_dir_text)
        ensure_dir(output_dir)
        output_path = output_dir / f"eval_{split}_{checkpoint_path.stem}.json"
        write_json(result, output_path)
    return result


def main() -> int:
    """CLI entrypoint for regions baseline training."""

    parser = build_parser()
    args = parser.parse_args()
    return run_training(args.config, args)


def evaluate_main() -> int:
    """CLI entrypoint for regions checkpoint evaluation."""

    parser = build_evaluate_parser()
    args = parser.parse_args()
    result = run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


__all__ = [
    "build_evaluate_parser",
    "build_parser",
    "build_region_dataloaders",
    "build_region_datasets",
    "evaluate_main",
    "main",
    "resolve_training_config",
    "run_evaluation",
    "run_one_epoch_with_shape",
    "run_training",
]
