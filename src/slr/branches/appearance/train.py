"""Training and evaluation entrypoints for FullBBox-I3D appearance baselines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from slr.branches.appearance.dataset import AppearanceClipDataset, appearance_collate_fn
from slr.branches.appearance.models import build_appearance_model
from slr.training.checkpointing import load_checkpoint, save_checkpoint
from slr.training.losses import build_loss_from_config, get_label_smoothing_epsilon, get_loss_name
from slr.training.metrics import AverageMeter, accuracy_topk
from slr.training.optim import build_optimizer, build_scheduler
from slr.training.seed import set_seed
from slr.training.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    log_wandb_model_artifact,
)
from slr.utils.io import ensure_dir, read_yaml, write_json, write_yaml
from slr.utils.logging import setup_logger


DEFAULT_TOPK = (1, 5, 10)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for appearance training."""

    parser = argparse.ArgumentParser(
        description="Train a FullBBox-I3D appearance baseline on packaged RGB clips."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    parser.add_argument("--package-root", type=Path, default=None, help="Override data.package_root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override run.output_dir.")
    parser.add_argument("--device", type=str, default=None, help="Override training.device.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume or evaluate from this checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate one checkpoint only.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run.name.")
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val subset size.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test subset size.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, datasets, model, and one forward/loss pass without training.",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for appearance checkpoint evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate an appearance checkpoint against one packaged split."
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
    parser.add_argument("--package-root", type=Path, default=None, help="Override data.package_root.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training.batch_size override.")
    parser.add_argument("--device", type=str, default=None, help="Override training.device.")
    return parser


def _attach_loss_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Attach resolved loss metadata to runtime config."""

    runtime_cfg = config.setdefault("runtime", {})
    loss_config = {
        "train": {"loss": str(config["training"]["loss"]["name"]).strip().lower()},
        "label_smoothing": dict(config["training"].get("label_smoothing", {})),
    }
    runtime_cfg["loss_type"] = get_loss_name(loss_config)
    runtime_cfg["label_smoothing_epsilon"] = float(get_label_smoothing_epsilon(loss_config))
    return config


def _normalize_training_config(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Fill missing sections and defaults for appearance training."""

    resolved = copy.deepcopy(config)
    resolved["config_path"] = str(config_path.as_posix())
    resolved["project_root"] = str(Path.cwd().as_posix())

    run_cfg = resolved.setdefault("run", {})
    data_cfg = resolved.setdefault("data", {})
    model_cfg = resolved.setdefault("model", {})
    preprocessing_cfg = resolved.setdefault("preprocessing", {})
    training_cfg = resolved.setdefault("training", {})
    eval_cfg = resolved.setdefault("eval", {})
    wandb_cfg = resolved.setdefault("wandb", {})
    runtime_cfg = resolved.setdefault("runtime", {})

    run_cfg.setdefault("name", "fullbbox_i3d_nslt100_ce")
    run_cfg.setdefault("output_dir", f"outputs/appearance/{run_cfg['name']}")
    run_cfg.setdefault("seed", 42)
    run_cfg.setdefault("monitor_metric", training_cfg.get("monitor_metric", "val_top5"))
    run_cfg.setdefault("monitor_mode", "max")

    data_cfg.setdefault("package_root", "")
    data_cfg.setdefault("train_manifest", "manifests/nslt100_train.csv")
    data_cfg.setdefault("val_manifest", "manifests/nslt100_val.csv")
    data_cfg.setdefault("test_manifest", "manifests/nslt100_test.csv")
    data_cfg.setdefault("clip_len", 32)
    data_cfg.setdefault("input_size", 224)
    data_cfg.setdefault("num_workers", 2)
    data_cfg.setdefault("pin_memory", True)
    data_cfg.setdefault("sampling_strategy", "auto")

    model_cfg.setdefault("name", "fullbbox_i3d")
    model_cfg.setdefault("in_channels", 3)
    model_cfg.setdefault("num_classes", 100)
    model_cfg.setdefault("dropout", 0.5)
    model_cfg.setdefault("feature_dim", 1024)
    model_cfg.setdefault("pretrained_path", None)
    model_cfg.setdefault("freeze_backbone", False)
    model_cfg.setdefault("return_features", False)

    preprocessing_cfg.setdefault("resize_mode", "letterbox")
    preprocessing_cfg.setdefault("mean", [0.45, 0.45, 0.45])
    preprocessing_cfg.setdefault("std", [0.225, 0.225, 0.225])
    train_aug = preprocessing_cfg.setdefault("train_augmentation", {})
    train_aug.setdefault("color_jitter", False)
    train_aug.setdefault("horizontal_flip", False)
    train_aug.setdefault("horizontal_flip_prob", 0.5)

    training_cfg.setdefault("epochs", 50)
    training_cfg.setdefault("batch_size", 2)
    training_cfg.setdefault("gradient_accumulation_steps", 1)
    training_cfg.setdefault("amp", True)
    training_cfg.setdefault("device", "auto")
    training_cfg.setdefault("resume_checkpoint", None)
    training_cfg.setdefault("monitor_metric", "val_top5")
    training_cfg.setdefault("save_best", True)
    training_cfg.setdefault("grad_clip_norm", None)
    training_cfg.setdefault("eval_split", "test")
    loss_cfg = training_cfg.setdefault("loss", {})
    loss_cfg.setdefault("name", "cross_entropy")
    training_cfg.setdefault("label_smoothing", {})
    optimizer_cfg = training_cfg.setdefault("optimizer", {})
    optimizer_cfg.setdefault("name", "adamw")
    optimizer_cfg.setdefault("lr", 1e-4)
    optimizer_cfg.setdefault("weight_decay", 1e-2)
    optimizer_cfg.setdefault("momentum", 0.9)
    optimizer_cfg.setdefault("nesterov", True)
    scheduler_cfg = training_cfg.setdefault("scheduler", {})
    scheduler_cfg.setdefault("enabled", True)
    scheduler_cfg.setdefault("name", "cosine")
    scheduler_cfg.setdefault("min_lr", 1e-6)
    scheduler_cfg.setdefault("step_size", 10)
    scheduler_cfg.setdefault("gamma", 0.1)
    early_cfg = training_cfg.setdefault("early_stopping", {})
    early_cfg.setdefault("enabled", True)
    early_cfg.setdefault("monitor", "val_top5")
    early_cfg.setdefault("mode", "max")
    early_cfg.setdefault("patience", 10)
    early_cfg.setdefault("min_delta", 0.0)

    eval_cfg.setdefault("topk", [1, 5, 10])

    wandb_cfg.setdefault("enabled", False)
    wandb_cfg.setdefault("project", "wlasl-appearance-i3d")
    wandb_cfg.setdefault("entity_env", "WANDB_ENTITY")
    wandb_cfg.setdefault("run_name", run_cfg["name"])
    wandb_cfg.setdefault("tags", ["appearance", "fullbbox", "i3d-style", "nslt100"])
    wandb_cfg.setdefault("log_model", True)

    runtime_cfg.setdefault("limit_train", None)
    runtime_cfg.setdefault("limit_val", None)
    runtime_cfg.setdefault("limit_test", None)
    return resolved


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to a normalized config."""

    resolved = copy.deepcopy(config)
    if args.package_root is not None:
        resolved["data"]["package_root"] = str(Path(args.package_root).as_posix())
    if args.output_dir is not None:
        resolved["run"]["output_dir"] = str(Path(args.output_dir).as_posix())
    if args.device is not None:
        resolved["training"]["device"] = str(args.device)
    if args.epochs is not None:
        resolved["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        resolved["training"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        resolved["data"]["num_workers"] = int(args.num_workers)
    if args.resume is not None:
        resolved["training"]["resume_checkpoint"] = str(Path(args.resume).as_posix())
    if args.run_name is not None:
        resolved["run"]["name"] = str(args.run_name)
        resolved["wandb"]["run_name"] = str(args.run_name)
    else:
        resolved["wandb"]["run_name"] = str(resolved["wandb"].get("run_name") or resolved["run"]["name"])
    if args.seed is not None:
        resolved["run"]["seed"] = int(args.seed)
    if args.no_wandb:
        resolved["wandb"]["enabled"] = False

    resolved["runtime"]["limit_train"] = args.limit_train
    resolved["runtime"]["limit_val"] = args.limit_val
    resolved["runtime"]["limit_test"] = args.limit_test
    return resolved


def resolve_training_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load, normalize, validate, and override one appearance config."""

    config = read_yaml(config_path)
    normalized = _normalize_training_config(config, config_path=config_path)
    resolved = apply_cli_overrides(normalized, args)
    resolved = _attach_loss_metadata(resolved)
    resolved["eval"]["topk"] = list(_resolve_topk_from_config(resolved))

    if not str(resolved["data"]["package_root"]).strip():
        raise ValueError("data.package_root must be provided in config or via --package-root.")
    if int(resolved["model"]["in_channels"]) != 3:
        raise ValueError("Appearance RGB stream expects model.in_channels=3.")
    if int(resolved["model"]["num_classes"]) <= 0:
        raise ValueError("model.num_classes must be positive.")
    if int(resolved["data"]["clip_len"]) <= 0:
        raise ValueError("data.clip_len must be positive.")
    if int(resolved["data"]["input_size"]) <= 0:
        raise ValueError("data.input_size must be positive.")
    if int(resolved["training"]["gradient_accumulation_steps"]) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")
    return resolved


def select_device(device_name: str, *, logger) -> torch.device:
    """Resolve auto and gracefully fall back when CUDA is unavailable."""

    requested = str(device_name).strip().lower()
    if requested == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device requested=auto resolved=%s", resolved)
        return resolved
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_appearance_datasets(config: dict[str, Any]) -> dict[str, AppearanceClipDataset]:
    """Instantiate train/val/test datasets from one resolved config."""

    runtime_cfg = config.get("runtime", {})
    return {
        "train": AppearanceClipDataset.from_config(
            config,
            split="train",
            limit=runtime_cfg.get("limit_train"),
        ),
        "val": AppearanceClipDataset.from_config(
            config,
            split="val",
            limit=runtime_cfg.get("limit_val"),
        ),
        "test": AppearanceClipDataset.from_config(
            config,
            split="test",
            limit=runtime_cfg.get("limit_test"),
        ),
    }


def build_appearance_dataloaders(
    config: dict[str, Any],
    datasets: dict[str, AppearanceClipDataset],
    *,
    device: torch.device,
    batch_size_override: int | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for each requested split."""

    data_cfg = config["data"]
    training_cfg = config["training"]
    batch_size = int(batch_size_override or training_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=appearance_collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=appearance_collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=appearance_collate_fn,
        ),
    }


def _normalize_optimizer_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map nested optimizer config to the shared helper format."""

    optimizer_cfg = dict(training_cfg.get("optimizer", {}))
    return {
        "optimizer": str(optimizer_cfg.get("name", "adamw")),
        "learning_rate": float(optimizer_cfg.get("lr", 1e-4)),
        "weight_decay": float(optimizer_cfg.get("weight_decay", 1e-2)),
        "momentum": float(optimizer_cfg.get("momentum", 0.9)),
        "nesterov": bool(optimizer_cfg.get("nesterov", True)),
    }


def _normalize_scheduler_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map nested scheduler config to the shared helper format."""

    scheduler_cfg = dict(training_cfg.get("scheduler", {}))
    return {
        "enabled": bool(scheduler_cfg.get("enabled", False)),
        "name": str(scheduler_cfg.get("name", "cosine")),
        "min_lr": float(scheduler_cfg.get("min_lr", 1e-6)),
        "step_size": int(scheduler_cfg.get("step_size", 10)),
        "gamma": float(scheduler_cfg.get("gamma", 0.1)),
    }


def _build_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve all training output paths."""

    output_dir = Path(config["run"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    return {
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "best_checkpoint": checkpoints_dir / "best.pt",
        "last_checkpoint": checkpoints_dir / "last.pt",
        "config_resolved": output_dir / "config_resolved.yaml",
        "metrics_json": output_dir / "metrics.json",
        "summary_json": output_dir / "summary.json",
        "train_log_txt": output_dir / "train.log",
    }


def _build_wandb_logging_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the appearance config to the shared W&B helper schema."""

    wandb_cfg = dict(config.get("wandb", {}))
    return {
        "use_wandb": bool(wandb_cfg.get("enabled", False)),
        "entity_env": str(wandb_cfg.get("entity_env", "WANDB_ENTITY")),
        "project": str(wandb_cfg.get("project", "wlasl-appearance-i3d")),
        "run_name": str(wandb_cfg.get("run_name", config["run"]["name"])),
        "tags": list(wandb_cfg.get("tags", [])),
        "log_model": bool(wandb_cfg.get("log_model", True)),
    }


def _build_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the nested appearance config to the shared loss helper schema."""

    return {
        "train": {"loss": str(config["training"]["loss"]["name"]).strip().lower()},
        "label_smoothing": dict(config["training"].get("label_smoothing", {})),
    }


def _extract_logits(model_output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalize model outputs to plain logits."""

    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


def _normalize_topk_values(
    topk_values: list[int] | tuple[int, ...] | None,
    *,
    num_classes: int | None = None,
) -> tuple[int, ...]:
    """Normalize, deduplicate, and clamp configured top-k values."""

    raw_values = DEFAULT_TOPK if topk_values is None else topk_values
    normalized: list[int] = []
    seen: set[int] = set()
    class_cap = None if num_classes is None else max(1, int(num_classes))

    for value in raw_values:
        k = int(value)
        if k <= 0:
            continue
        if class_cap is not None:
            k = min(k, class_cap)
        if k in seen:
            continue
        seen.add(k)
        normalized.append(k)

    if not normalized:
        fallback = min(DEFAULT_TOPK[0], class_cap) if class_cap is not None else DEFAULT_TOPK[0]
        normalized.append(max(1, int(fallback)))
    return tuple(normalized)


def _resolve_topk_from_config(config: dict[str, Any]) -> tuple[int, ...]:
    """Resolve appearance top-k metrics from config with safe defaults."""

    num_classes = config.get("model", {}).get("num_classes")
    return _normalize_topk_values(config.get("eval", {}).get("topk", list(DEFAULT_TOPK)), num_classes=num_classes)


def _build_split_metric_payload(
    split_name: str,
    metrics: dict[str, float],
    *,
    topk: tuple[int, ...],
    include_loss: bool = True,
) -> dict[str, float]:
    """Build slash-formatted metric payloads such as ``train/top10``."""

    payload: dict[str, float] = {}
    if include_loss and "loss" in metrics:
        payload[f"{split_name}/loss"] = float(metrics["loss"])
    for k in topk:
        payload[f"{split_name}/top{k}"] = float(metrics.get(f"top{k}", 0.0))
    return payload


def _history_entry_from_flat_metrics(flat_metrics: dict[str, float], *, topk: tuple[int, ...]) -> dict[str, float]:
    """Convert slash-formatted metrics to underscore keys for ``metrics.json`` history."""

    history_entry = {
        "epoch": int(flat_metrics["epoch"]),
        "lr": float(flat_metrics["lr"]),
        "train_loss": float(flat_metrics["train/loss"]),
        "val_loss": float(flat_metrics["val/loss"]),
    }
    for split_name in ("train", "val"):
        for k in topk:
            history_entry[f"{split_name}_top{k}"] = float(flat_metrics.get(f"{split_name}/top{k}", 0.0))
    return history_entry


def _augment_summary_with_split_metrics(
    summary: dict[str, Any],
    *,
    split_name: str,
    metrics: dict[str, float],
    topk: tuple[int, ...],
) -> None:
    """Attach flat summary keys such as ``test_top10``."""

    if "loss" in metrics:
        summary[f"{split_name}_loss"] = float(metrics["loss"])
    for k in topk:
        summary[f"{split_name}_top{k}"] = float(metrics.get(f"top{k}", 0.0))


def _validate_batch_shape(batch_video: torch.Tensor, *, clip_len: int, input_size: int) -> None:
    """Validate one batched video tensor shape."""

    if batch_video.ndim != 5:
        raise ValueError(
            "Expected batched appearance tensors with shape (B, C, T, H, W), "
            f"got {tuple(batch_video.shape)}."
        )
    _, channels, time, height, width = batch_video.shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}.")
    if time != int(clip_len):
        raise ValueError(f"Expected clip_len={clip_len}, got {time}.")
    if height != int(input_size) or width != int(input_size):
        raise ValueError(
            f"Expected spatial size ({input_size}, {input_size}), got ({height}, {width})."
        )


def run_one_epoch(
    *,
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
    clip_len: int,
    input_size: int,
    optimizer=None,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    grad_clip_norm: float | None = None,
    accumulation_steps: int = 1,
    topk: tuple[int, ...] = DEFAULT_TOPK,
) -> dict[str, float]:
    """Run one training or evaluation epoch."""

    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    topk_meters = {f"top{k}": AverageMeter(f"top{k}") for k in topk}
    accumulation_steps = max(1, int(accumulation_steps))

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        video = batch["video"]
        labels = batch["labels"]
        _validate_batch_shape(video, clip_len=clip_len, input_size=input_size)
        video = video.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        batch_size = int(labels.shape[0])

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                output = model(video)
                logits = _extract_logits(output)
                loss = criterion(logits, labels)
                scaled_loss = loss / accumulation_steps

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                    if batch_index % accumulation_steps == 0:
                        if grad_clip_norm is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if batch_index % accumulation_steps == 0:
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        metrics = accuracy_topk(logits.detach(), labels.detach(), topk=topk)
        loss_meter.update(float(loss.item()), n=batch_size)
        for key, meter in topk_meters.items():
            meter.update(float(metrics[key]), n=batch_size)

    if is_train and len(loader) % accumulation_steps != 0:
        if scaler is not None and scaler.is_enabled():
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    epoch_metrics = {"loss": loss_meter.avg}
    for key, meter in topk_meters.items():
        epoch_metrics[key] = meter.avg
    return epoch_metrics


def _is_improved(current: float, best: float | None, *, mode: str, min_delta: float = 0.0) -> bool:
    """Decide whether one metric improved over the current best."""

    if best is None:
        return True
    if mode == "max":
        return current > best + float(min_delta)
    if mode == "min":
        return current < best - float(min_delta)
    raise ValueError(f"Unsupported mode {mode!r}. Expected 'max' or 'min'.")


def _log_epoch_summary(
    logger,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
    *,
    topk: tuple[int, ...],
) -> None:
    """Write one concise epoch summary to the logger."""

    message_parts = [
        f"Epoch {epoch}/{total_epochs}",
        f"train_loss={metrics['train/loss']:.4f}",
    ]
    message_parts.extend(
        f"train_top{k}={float(metrics.get(f'train/top{k}', 0.0)):.4f}" for k in topk
    )
    message_parts.append(f"val_loss={metrics['val/loss']:.4f}")
    message_parts.extend(
        f"val_top{k}={float(metrics.get(f'val/top{k}', 0.0)):.4f}" for k in topk
    )
    logger.info(" | ".join(message_parts))


def _write_training_outputs(
    *,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    history: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Persist resolved config, history, and summaries."""

    ensure_dir(output_paths["output_dir"])
    ensure_dir(output_paths["checkpoints_dir"])
    write_yaml(config, output_paths["config_resolved"])
    write_json({"history": history, "summary": metrics_summary}, output_paths["metrics_json"])
    write_json(summary, output_paths["summary_json"])


def run_training(config_path: Path, args: argparse.Namespace) -> int:
    """Train one appearance baseline and persist outputs/checkpoints."""

    resolved_config = resolve_training_config(config_path, args)
    output_paths = _build_output_paths(resolved_config)
    logger = setup_logger(
        "slr.branches.appearance.train",
        log_file=None if args.dry_run else output_paths["train_log_txt"],
    )
    device = select_device(str(resolved_config["training"]["device"]), logger=logger)
    clip_len = int(resolved_config["data"]["clip_len"])
    input_size = int(resolved_config["data"]["input_size"])
    run_name = str(resolved_config["run"]["name"])

    logger.info("Resolved run_name=%s device=%s", run_name, device)
    logger.info(
        "Loss: %s epsilon=%s",
        str(resolved_config["runtime"]["loss_type"]),
        float(resolved_config["runtime"]["label_smoothing_epsilon"]),
    )
    set_seed(int(resolved_config["run"]["seed"]))

    datasets = build_appearance_datasets(resolved_config)
    dataloaders = build_appearance_dataloaders(resolved_config, datasets, device=device)
    model = build_appearance_model(resolved_config["model"]).to(device)
    criterion = build_loss_from_config(_build_loss_config(resolved_config))

    logger.info(
        "Datasets | train=%s val=%s test=%s | model=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        model.__class__.__name__,
    )

    sample_batch = next(iter(dataloaders["train"]))
    _validate_batch_shape(sample_batch["video"], clip_len=clip_len, input_size=input_size)
    with torch.no_grad():
        dry_output = model(sample_batch["video"].to(device))
        dry_logits = _extract_logits(dry_output)
        dry_loss = criterion(dry_logits, sample_batch["labels"].to(device))

    if args.dry_run:
        logger.info(
            "Dry run successful | batch_shape=%s logits_shape=%s loss=%.4f output_dir=%s",
            tuple(sample_batch["video"].shape),
            tuple(dry_logits.shape),
            float(dry_loss.item()),
            output_paths["output_dir"],
        )
        return 0

    optimizer = build_optimizer(model.parameters(), _normalize_optimizer_config(resolved_config["training"]))
    scheduler = build_scheduler(
        optimizer,
        _normalize_scheduler_config(resolved_config["training"]),
        epochs=int(resolved_config["training"]["epochs"]),
    )

    amp_requested = bool(resolved_config["training"].get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and device.type != "cuda":
        logger.warning("AMP was requested but CUDA is unavailable; AMP has been disabled.")
    scaler = GradScaler(enabled=amp_enabled)

    logging_cfg = _build_wandb_logging_config(resolved_config)
    wandb_run = init_wandb_run(
        resolved_config=resolved_config,
        logging_cfg=logging_cfg,
        run_name=str(logging_cfg["run_name"]),
        logger=logger,
    )

    resume_path_text = resolved_config["training"].get("resume_checkpoint")
    best_metric: float | None = None
    best_epoch = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if resume_path_text:
        resume_path = Path(str(resume_path_text))
        if not resume_path.is_absolute():
            resume_path = resume_path.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        payload = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        best_metric = float(payload["best_metric"]) if payload.get("best_metric") is not None else None
        best_epoch = int(payload.get("epoch", 0))
        start_epoch = best_epoch + 1
        logger.info("Resumed from checkpoint=%s epoch=%s", resume_path.as_posix(), best_epoch)

    topk = _resolve_topk_from_config(resolved_config)
    accumulation_steps = int(resolved_config["training"].get("gradient_accumulation_steps", 1))
    grad_clip_norm = resolved_config["training"].get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    early_cfg = dict(resolved_config["training"].get("early_stopping", {}))
    early_best: float | None = None
    early_wait = 0
    stopped_epoch: int | None = None

    monitor_metric = str(resolved_config["training"].get("monitor_metric", "val_top5"))
    if monitor_metric.startswith("val_"):
        monitor_key = f"val/{monitor_metric.split('_', 1)[1]}"
    elif monitor_metric.startswith("val/"):
        monitor_key = monitor_metric
    else:
        monitor_key = f"val/{monitor_metric}"

    try:
        if args.eval_only:
            if not resume_path_text:
                raise ValueError("--eval-only requires --resume or training.resume_checkpoint.")
            test_metrics = run_one_epoch(
                model=model,
                loader=dataloaders["test"],
                criterion=criterion,
                device=device,
                clip_len=clip_len,
                input_size=input_size,
                amp_enabled=amp_enabled,
                accumulation_steps=1,
                topk=topk,
            )
            result = {
                "split": "test",
                "checkpoint": str(resume_path_text),
                "loss": float(test_metrics["loss"]),
            }
            for k in topk:
                result[f"top{k}"] = float(test_metrics.get(f"top{k}", 0.0))
            logger.info("Eval-only result: %s", json.dumps(result))
            print(json.dumps(result, indent=2))
            return 0

        total_epochs = int(resolved_config["training"]["epochs"])
        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=dataloaders["train"],
                criterion=criterion,
                device=device,
                clip_len=clip_len,
                input_size=input_size,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
                accumulation_steps=accumulation_steps,
                topk=topk,
            )
            val_metrics = run_one_epoch(
                model=model,
                loader=dataloaders["val"],
                criterion=criterion,
                device=device,
                clip_len=clip_len,
                input_size=input_size,
                amp_enabled=amp_enabled,
                accumulation_steps=1,
                topk=topk,
            )
            if scheduler is not None:
                scheduler.step()

            lr = float(optimizer.param_groups[0]["lr"])
            flat_metrics = {"epoch": epoch, "lr": lr}
            flat_metrics.update(_build_split_metric_payload("train", train_metrics, topk=topk))
            flat_metrics.update(_build_split_metric_payload("val", val_metrics, topk=topk))
            history.append(_history_entry_from_flat_metrics(flat_metrics, topk=topk))
            _log_epoch_summary(logger, epoch, total_epochs, flat_metrics, topk=topk)
            log_wandb_metrics(wandb_run, flat_metrics, step=epoch)

            current_metric = float(flat_metrics[monitor_key])
            if _is_improved(current_metric, best_metric, mode=str(resolved_config["run"]["monitor_mode"])):
                best_metric = current_metric
                best_epoch = epoch
                save_checkpoint(
                    output_paths["best_checkpoint"],
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    last_metrics=flat_metrics,
                    config=resolved_config,
                    keypoint_set="appearance",
                    num_classes=int(resolved_config["model"]["num_classes"]),
                    num_nodes=0,
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
                keypoint_set="appearance",
                num_classes=int(resolved_config["model"]["num_classes"]),
                num_nodes=0,
                model_name=str(resolved_config["model"]["name"]),
                class_id_to_gloss=datasets["train"].id_to_gloss,
            )

            if bool(early_cfg.get("enabled", False)):
                early_monitor = str(early_cfg.get("monitor", "val_top5"))
                if early_monitor.startswith("val_"):
                    early_key = f"val/{early_monitor.split('_', 1)[1]}"
                elif early_monitor.startswith("val/"):
                    early_key = early_monitor
                else:
                    early_key = f"val/{early_monitor}"
                current_early = float(flat_metrics[early_key])
                early_mode = str(early_cfg.get("mode", "max")).strip().lower()
                if _is_improved(
                    current_early,
                    early_best,
                    mode=early_mode,
                    min_delta=float(early_cfg.get("min_delta", 0.0)),
                ):
                    early_best = current_early
                    early_wait = 0
                else:
                    early_wait += 1
                    logger.info(
                        "Early stopping wait %s/%s | monitor=%s current=%.4f best=%.4f",
                        early_wait,
                        int(early_cfg.get("patience", 10)),
                        early_monitor,
                        current_early,
                        0.0 if early_best is None else early_best,
                    )
                    if early_wait >= int(early_cfg.get("patience", 10)):
                        stopped_epoch = epoch
                        logger.info("Early stopping triggered at epoch=%s", stopped_epoch)
                        break

        if best_epoch <= 0:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        load_checkpoint(output_paths["best_checkpoint"], model, map_location=device)
        test_split = str(resolved_config["training"].get("eval_split", "test")).strip().lower()
        test_metrics = run_one_epoch(
            model=model,
            loader=dataloaders[test_split],
            criterion=criterion,
            device=device,
            clip_len=clip_len,
            input_size=input_size,
            amp_enabled=amp_enabled,
            accumulation_steps=1,
            topk=topk,
        )
        test_payload = _build_split_metric_payload("test", test_metrics, topk=topk)
        log_wandb_metrics(wandb_run, test_payload, step=best_epoch)

        if wandb_run is not None and bool(logging_cfg.get("log_model", True)):
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
            "best_metric": None if best_metric is None else float(best_metric),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "eval_topk": list(topk),
        }
        _augment_summary_with_split_metrics(metrics_summary, split_name="test", metrics=test_metrics, topk=topk)
        final_train_metrics = {"loss": history[-1]["train_loss"]}
        final_val_metrics = {"loss": history[-1]["val_loss"]}
        for k in topk:
            final_train_metrics[f"top{k}"] = float(history[-1].get(f"train_top{k}", 0.0))
            final_val_metrics[f"top{k}"] = float(history[-1].get(f"val_top{k}", 0.0))
        _augment_summary_with_split_metrics(metrics_summary, split_name="final_train", metrics=final_train_metrics, topk=topk)
        _augment_summary_with_split_metrics(metrics_summary, split_name="final_val", metrics=final_val_metrics, topk=topk)
        summary = {
            "run_name": run_name,
            "branch": "appearance",
            "model_name": str(resolved_config["model"]["name"]),
            "i3d_variant": "I3D-style / Inception3D-like RGB stream",
            "output_dir": str(output_paths["output_dir"].as_posix()),
            "best_checkpoint": str(output_paths["best_checkpoint"].as_posix()),
            "last_checkpoint": str(output_paths["last_checkpoint"].as_posix()),
            "config_path": str(config_path.as_posix()),
            "dataset": {
                "subset": "nslt100",
                "num_classes": int(resolved_config["model"]["num_classes"]),
                "clip_len": clip_len,
                "input_size": input_size,
                "num_samples": {
                    "train": len(datasets["train"]),
                    "val": len(datasets["val"]),
                    "test": len(datasets["test"]),
                },
            },
            "model": {
                "in_channels": int(resolved_config["model"]["in_channels"]),
                "num_classes": int(resolved_config["model"]["num_classes"]),
                "dropout": float(resolved_config["model"]["dropout"]),
                "feature_dim": int(resolved_config["model"]["feature_dim"]),
                "freeze_backbone": bool(resolved_config["model"]["freeze_backbone"]),
                "return_features": bool(resolved_config["model"]["return_features"]),
                "pretrained_path": resolved_config["model"].get("pretrained_path"),
            },
            "training": {
                "epochs": int(resolved_config["training"]["epochs"]),
                "batch_size": int(resolved_config["training"]["batch_size"]),
                "gradient_accumulation_steps": int(accumulation_steps),
                "amp": bool(amp_enabled),
                "device": str(device),
                "stopped_epoch": None if stopped_epoch is None else int(stopped_epoch),
            },
            "metrics": metrics_summary,
            "eval": {"topk": list(topk)},
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        _write_training_outputs(
            config=resolved_config,
            output_paths=output_paths,
            history=history,
            metrics_summary=metrics_summary,
            summary=summary,
        )
        logger.info(
            "Training finished. Best epoch=%s %s",
            best_epoch,
            " ".join(
                f"test_top{k}={float(test_metrics.get(f'top{k}', 0.0)):.4f}" for k in topk
            ),
        )
        return 0
    finally:
        finish_wandb_run(wandb_run)


def run_evaluation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    package_root: Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one appearance checkpoint against one split."""

    raw_config = read_yaml(config_path)
    normalized = _normalize_training_config(raw_config, config_path=config_path)
    if package_root is not None:
        normalized["data"]["package_root"] = str(Path(package_root).as_posix())
    if batch_size is not None:
        normalized["training"]["batch_size"] = int(batch_size)
    if device_name is not None:
        normalized["training"]["device"] = str(device_name)
    config = _attach_loss_metadata(normalized)

    logger = setup_logger("slr.branches.appearance.evaluate")
    device = select_device(str(config["training"]["device"]), logger=logger)
    clip_len = int(config["data"]["clip_len"])
    input_size = int(config["data"]["input_size"])
    datasets = build_appearance_datasets(config)
    dataloaders = build_appearance_dataloaders(
        config,
        datasets,
        device=device,
        batch_size_override=batch_size,
    )
    model = build_appearance_model(config["model"]).to(device)
    criterion = build_loss_from_config(_build_loss_config(config))
    payload = load_checkpoint(checkpoint_path, model, map_location=device)
    topk = _resolve_topk_from_config(config)
    metrics = run_one_epoch(
        model=model,
        loader=dataloaders[split],
        criterion=criterion,
        device=device,
        clip_len=clip_len,
        input_size=input_size,
        amp_enabled=bool(config["training"].get("amp", False)) and device.type == "cuda",
        accumulation_steps=1,
        topk=topk,
    )

    result = {
        "split": split,
        "checkpoint": str(checkpoint_path.as_posix()),
        "epoch": int(payload.get("epoch", 0)),
        "loss": float(metrics["loss"]),
    }
    for k in topk:
        result[f"top{k}"] = float(metrics.get(f"top{k}", 0.0))
    logger.info(
        "Evaluation split=%s %s",
        split,
        " ".join(
            [f"loss={result['loss']:.4f}"] + [f"top{k}={float(result[f'top{k}']):.4f}" for k in topk]
        ),
    )
    return result


def main() -> int:
    """CLI entrypoint for appearance training."""

    parser = build_parser()
    args = parser.parse_args()
    return run_training(args.config, args)


def evaluate_main() -> int:
    """CLI entrypoint for appearance checkpoint evaluation."""

    parser = build_evaluate_parser()
    args = parser.parse_args()
    result = run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=str(args.split),
        package_root=args.package_root,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


__all__ = [
    "build_appearance_dataloaders",
    "build_appearance_datasets",
    "build_evaluate_parser",
    "build_parser",
    "evaluate_main",
    "main",
    "resolve_training_config",
    "run_evaluation",
    "run_one_epoch",
    "run_training",
]
