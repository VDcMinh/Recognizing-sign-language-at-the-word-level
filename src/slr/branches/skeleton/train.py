"""Training and evaluation entrypoints for skeleton baselines."""

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

from slr.branches.skeleton.dataset import SkeletonGraphDataset, skeleton_collate_fn
from slr.branches.skeleton.graph import SkeletonGraph
from slr.branches.skeleton.models import build_skeleton_model
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
    """Create the CLI parser for skeleton baseline training."""

    parser = argparse.ArgumentParser(
        description="Train a skeleton baseline on precomputed graph tensors."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override train.learning_rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override train.weight_decay.",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Override model.dropout.")
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override logging.project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override the W&B entity.",
    )
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
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val subset size.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test subset size.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, datasets, and model without running training.",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for local checkpoint evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a skeleton checkpoint against train/val/test graph tensors."
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional dataloader.batch_size override.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    return parser


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
    """Fill missing sections and defaults for skeleton baseline training."""

    resolved = copy.deepcopy(config)
    resolved["config_path"] = str(config_path.as_posix())
    resolved["project_root"] = str(Path.cwd().as_posix())

    experiment = resolved.setdefault("experiment", {})
    dataset = resolved.setdefault("dataset", {})
    dataloader = resolved.setdefault("dataloader", {})
    graph = resolved.setdefault("graph", {})
    model = resolved.setdefault("model", {})
    train_cfg = resolved.setdefault("train", {})
    scheduler = resolved.setdefault("scheduler", {})
    logging_cfg = resolved.setdefault("logging", {})
    runtime = resolved.setdefault("runtime", {})

    experiment.setdefault("name", "skeleton-baseline")
    experiment.setdefault("seed", 42)
    experiment.setdefault("output_root", "outputs/skeleton")
    experiment.setdefault("monitor_metric", "val/top1")
    experiment.setdefault("monitor_mode", "max")
    experiment.setdefault("save_every_epoch", False)

    dataset.setdefault("num_classes", 100)
    dataset.setdefault("return_metadata", True)
    dataset.setdefault("strict_shape_check", True)
    manifests = dataset.setdefault("manifests", {})
    for split in ("train", "val", "test"):
        manifests.setdefault(split, "")

    dataloader.setdefault("batch_size", 16)
    dataloader.setdefault("num_workers", 0)
    dataloader.setdefault("pin_memory", True)
    dataloader.setdefault("shuffle_train", True)

    graph.setdefault("strategy", "spatial")
    graph.setdefault("add_self_links", True)
    graph.setdefault("normalize_adjacency", True)

    model.setdefault("name", "simple_stgcn")
    model.setdefault("in_channels", int(dataset.get("expected_shape", [3])[0]))
    model.setdefault("num_nodes", int(dataset.get("expected_shape", [3, 150, 27, 1])[2]))
    model.setdefault("num_classes", int(dataset.get("num_classes", 100)))
    model.setdefault("hidden_channels", 64)
    model.setdefault("base_channels", 64)
    model.setdefault("stage_channels", [64, 64, 64, 128, 128, 256])
    model.setdefault("temporal_strides", [1, 1, 1, 2, 1, 2])
    model.setdefault("dropout", 0.5)

    train_cfg.setdefault("epochs", 30)
    train_cfg.setdefault("device", "auto")
    train_cfg.setdefault("optimizer", "adamw")
    train_cfg.setdefault("learning_rate", 1e-3)
    train_cfg.setdefault("weight_decay", 5e-4)
    train_cfg.setdefault("loss", "cross_entropy")
    train_cfg.setdefault("grad_clip_norm", None)
    train_cfg.setdefault("amp", False)

    scheduler.setdefault("enabled", False)
    scheduler.setdefault("name", "cosine")
    scheduler.setdefault("min_lr", 1e-6)

    logging_cfg.setdefault("use_wandb", False)
    logging_cfg.setdefault("entity_env", "WANDB_ENTITY")
    logging_cfg.setdefault("project", "wlasl-skeleton")
    logging_cfg.setdefault("run_name", experiment["name"])
    logging_cfg.setdefault("tags", [])
    logging_cfg.setdefault("log_model", True)

    runtime.setdefault("limit_train", None)
    runtime.setdefault("limit_val", None)
    runtime.setdefault("limit_test", None)
    return resolved


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
        resolved["dataloader"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        resolved["train"]["learning_rate"] = float(args.lr)
    if args.weight_decay is not None:
        resolved["train"]["weight_decay"] = float(args.weight_decay)
    if args.dropout is not None:
        resolved["model"]["dropout"] = float(args.dropout)
    if args.device is not None:
        resolved["train"]["device"] = str(args.device)
    if args.seed is not None:
        resolved["experiment"]["seed"] = int(args.seed)
    if args.output_root is not None:
        resolved["experiment"]["output_root"] = str(Path(args.output_root).as_posix())
    if args.num_workers is not None:
        resolved["dataloader"]["num_workers"] = int(args.num_workers)
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
    output_root = Path(resolved["experiment"]["output_root"])
    resolved["logging"]["run_name"] = str(resolved["logging"].get("run_name") or run_name)
    resolved["experiment"]["output_dir"] = str((output_root / run_name).as_posix())
    return resolved


def resolve_training_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load, normalize, and override one skeleton training config."""

    config = read_yaml(config_path)
    normalized = _normalize_training_config(config, config_path=config_path)
    resolved = apply_cli_overrides(normalized, args)
    resolved = _attach_loss_metadata(resolved)

    expected_shape = tuple(int(value) for value in resolved["dataset"]["expected_shape"])
    if len(expected_shape) != 4:
        raise ValueError("dataset.expected_shape must contain [C, T, V, M].")
    if expected_shape[0] != int(resolved["model"]["in_channels"]):
        raise ValueError(
            f"dataset.expected_shape[0]={expected_shape[0]} does not match model.in_channels="
            f"{resolved['model']['in_channels']}."
        )
    if expected_shape[2] != int(resolved["model"]["num_nodes"]):
        raise ValueError(
            f"dataset.expected_shape[2]={expected_shape[2]} does not match model.num_nodes="
            f"{resolved['model']['num_nodes']}."
        )
    if int(resolved["model"]["num_classes"]) != int(resolved["dataset"]["num_classes"]):
        raise ValueError("model.num_classes must match dataset.num_classes.")
    return resolved


def select_device(device_name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    requested = str(device_name).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(device_name)


def build_skeleton_datasets(config: dict[str, Any]) -> dict[str, SkeletonGraphDataset]:
    """Instantiate train/val/test datasets from one resolved config."""

    runtime_cfg = config.get("runtime", {})
    limits = {
        "train": runtime_cfg.get("limit_train"),
        "val": runtime_cfg.get("limit_val"),
        "test": runtime_cfg.get("limit_test"),
    }

    datasets: dict[str, SkeletonGraphDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = SkeletonGraphDataset.from_config(
            config,
            split=split,
            limit=limits.get(split),
        )
    return datasets


def build_skeleton_dataloaders(
    config: dict[str, Any],
    datasets: dict[str, SkeletonGraphDataset],
    *,
    device: torch.device,
    batch_size_override: int | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for each requested split."""

    dataloader_cfg = config["dataloader"]
    batch_size = int(batch_size_override or dataloader_cfg["batch_size"])
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda"
    shuffle_train = bool(dataloader_cfg.get("shuffle_train", True))

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=skeleton_collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=skeleton_collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=skeleton_collate_fn,
        ),
    }


def build_graph_and_model(config: dict[str, Any], *, device: torch.device):
    """Create the skeleton graph topology and the configured baseline model."""

    graph_cfg = config["graph"]
    graph = SkeletonGraph(
        layout=str(graph_cfg["layout"]),
        strategy=str(graph_cfg["strategy"]),
        normalize=bool(graph_cfg["normalize_adjacency"]),
        add_self_links=bool(graph_cfg["add_self_links"]),
    )
    model = build_skeleton_model(config["model"], graph).to(device)
    return graph, model


def _validate_batch_shape(batch_data: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    if batch_data.ndim != 5:
        raise ValueError(
            f"Expected batched graph tensors with shape (N, C, T, V, M), got {tuple(batch_data.shape)}."
        )
    actual = tuple(int(value) for value in batch_data.shape[1:])
    if actual != expected_shape:
        raise ValueError(
            f"Expected batch sample shape {expected_shape}, got {actual}."
        )


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
    """Same as :func:`run_one_epoch`, with explicit batch shape validation."""

    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    topk_meters = {f"top{k}": AverageMeter(f"top{k}") for k in DEFAULT_TOPK}

    for batch in loader:
        data = batch["data"]
        labels = batch["labels"]
        _validate_batch_shape(data, expected_shape)
        data = data.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        batch_size = int(labels.shape[0])

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                logits = model(data)
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

    epoch_metrics = {
        "loss": loss_meter.avg,
    }
    for key, meter in topk_meters.items():
        epoch_metrics[key] = meter.avg
    return epoch_metrics


def _is_improved(current: float, best: float | None, *, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"Unsupported monitor mode {mode!r}. Expected 'max' or 'min'.")


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
    }


def _write_training_outputs(
    *,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    epoch_records: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    ensure_dir(output_paths["output_dir"])
    ensure_dir(output_paths["checkpoints_dir"])
    write_yaml(config, output_paths["config_resolved"])
    write_json(metrics_summary, output_paths["metrics_json"])
    write_json(summary, output_paths["summary_json"])
    write_dataframe_csv(pd.DataFrame(epoch_records), output_paths["train_log_csv"])


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
    """Train a simple skeleton baseline and persist all required outputs."""

    resolved_config = resolve_training_config(config_path, args)
    run_name = str(resolved_config["experiment"]["name"])
    output_paths = _build_output_paths(resolved_config)
    logger = setup_logger("slr.branches.skeleton.train")
    device = select_device(str(resolved_config["train"]["device"]))
    expected_shape = tuple(int(value) for value in resolved_config["dataset"]["expected_shape"])

    logger.info("Resolved run_name=%s device=%s", run_name, device)
    logger.info(_format_loss_log(resolved_config))
    set_seed(int(resolved_config["experiment"]["seed"]))

    datasets = build_skeleton_datasets(resolved_config)
    dataloaders = build_skeleton_dataloaders(resolved_config, datasets, device=device)
    graph, model = build_graph_and_model(resolved_config, device=device)

    logger.info(
        "Datasets | train=%s val=%s test=%s | graph=%s | model=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        graph,
        model.__class__.__name__,
    )

    if args.dry_run:
        sample_batch = next(iter(dataloaders["train"]))
        _validate_batch_shape(sample_batch["data"], expected_shape)
        with torch.no_grad():
            logits = model(sample_batch["data"].to(device))
        logger.info(
            "Dry run successful | batch_shape=%s logits_shape=%s output_dir=%s",
            tuple(sample_batch["data"].shape),
            tuple(logits.shape),
            output_paths["output_dir"],
        )
        return 0

    ensure_dir(output_paths["output_dir"])
    ensure_dir(output_paths["checkpoints_dir"])
    file_logger = logger

    criterion = build_loss_from_config(resolved_config)
    optimizer = build_optimizer(model.parameters(), resolved_config["train"])
    scheduler = build_scheduler(
        optimizer,
        resolved_config["scheduler"],
        epochs=int(resolved_config["train"]["epochs"]),
    )

    amp_requested = bool(resolved_config["train"].get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and device.type != "cuda":
        file_logger.warning("AMP was requested but CUDA is unavailable; AMP has been disabled.")
    scaler = GradScaler(enabled=amp_enabled)

    wandb_run = init_wandb_run(
        resolved_config=resolved_config,
        logging_cfg=resolved_config["logging"],
        run_name=str(resolved_config["logging"]["run_name"]),
        logger=file_logger,
        cli_entity=args.wandb_entity,
    )

    best_metric: float | None = None
    best_epoch = 0
    best_row: dict[str, Any] | None = None
    epoch_records: list[dict[str, Any]] = []
    monitor_metric = str(resolved_config["experiment"]["monitor_metric"])
    monitor_mode = str(resolved_config["experiment"]["monitor_mode"]).strip().lower()

    try:
        total_epochs = int(resolved_config["train"]["epochs"])
        grad_clip_norm = _as_float_or_none(resolved_config["train"].get("grad_clip_norm"))

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
            _log_epoch_summary(file_logger, epoch, total_epochs, flat_metrics)
            log_wandb_metrics(wandb_run, flat_metrics, step=epoch)

            if monitor_metric not in flat_metrics:
                raise KeyError(
                    f"Monitor metric {monitor_metric!r} is missing from epoch metrics."
                )
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
                    keypoint_set=str(resolved_config["dataset"]["keypoint_set"]),
                    num_classes=int(resolved_config["dataset"]["num_classes"]),
                    num_nodes=int(resolved_config["model"]["num_nodes"]),
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
                keypoint_set=str(resolved_config["dataset"]["keypoint_set"]),
                num_classes=int(resolved_config["dataset"]["num_classes"]),
                num_nodes=int(resolved_config["model"]["num_nodes"]),
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
                    keypoint_set=str(resolved_config["dataset"]["keypoint_set"]),
                    num_classes=int(resolved_config["dataset"]["num_classes"]),
                    num_nodes=int(resolved_config["model"]["num_nodes"]),
                    model_name=str(resolved_config["model"]["name"]),
                    class_id_to_gloss=datasets["train"].id_to_gloss,
                )

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
        final_test_payload = {
            "test/loss": float(test_metrics["loss"]),
            "test/top1": float(test_metrics["top1"]),
            "test/top5": float(test_metrics["top5"]),
            "test/top10": float(test_metrics["top10"]),
        }
        log_wandb_metrics(wandb_run, final_test_payload, step=best_epoch)

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
            "label_smoothing_epsilon": float(
                resolved_config["runtime"]["label_smoothing_epsilon"]
            ),
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
        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = metrics_summary["best_epoch"]
            wandb_run.summary["best_val_top1"] = metrics_summary["best_val_top1"]
            wandb_run.summary["best_val_top5"] = metrics_summary["best_val_top5"]
            wandb_run.summary["best_val_top10"] = metrics_summary["best_val_top10"]
            wandb_run.summary["test_loss"] = metrics_summary["test_loss"]
            wandb_run.summary["test_top1"] = metrics_summary["test_top1"]
            wandb_run.summary["test_top5"] = metrics_summary["test_top5"]
            wandb_run.summary["test_top10"] = metrics_summary["test_top10"]
        summary = {
            "run_name": run_name,
            "loss_type": str(resolved_config["runtime"]["loss_type"]),
            "label_smoothing_epsilon": float(
                resolved_config["runtime"]["label_smoothing_epsilon"]
            ),
            "keypoint_set": str(resolved_config["dataset"]["keypoint_set"]),
            "model_name": str(resolved_config["model"]["name"]),
            "output_dir": str(output_paths["output_dir"].as_posix()),
            "best_checkpoint": str(output_paths["best_checkpoint"].as_posix()),
            "last_checkpoint": str(output_paths["last_checkpoint"].as_posix()),
            "config_path": str(config_path.as_posix()),
            "dataset": {
                "subset": str(resolved_config["dataset"]["subset"]),
                "num_classes": int(resolved_config["dataset"]["num_classes"]),
                "expected_shape": list(expected_shape),
                "num_samples": {
                    "train": len(datasets["train"]),
                    "val": len(datasets["val"]),
                    "test": len(datasets["test"]),
                },
            },
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        _write_training_outputs(
            config=resolved_config,
            output_paths=output_paths,
            epoch_records=epoch_records,
            metrics_summary=metrics_summary,
            summary=summary,
        )
        file_logger.info("Training finished. Best epoch=%s test_top1=%.4f", best_epoch, test_metrics["top1"])
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
        config["dataloader"]["batch_size"] = int(batch_size)
    if device_name is not None:
        config["train"]["device"] = str(device_name)
    config = _attach_loss_metadata(config)

    logger = setup_logger("slr.branches.skeleton.evaluate")
    device = select_device(str(config["train"]["device"]))
    expected_shape = tuple(int(value) for value in config["dataset"]["expected_shape"])
    datasets = build_skeleton_datasets(config)
    dataloaders = build_skeleton_dataloaders(
        config,
        datasets,
        device=device,
        batch_size_override=batch_size,
    )
    _, model = build_graph_and_model(config, device=device)
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

    output_dir_text = raw_config.get("experiment", {}).get("output_dir")
    if output_dir_text:
        output_dir = Path(output_dir_text)
        ensure_dir(output_dir)
        output_path = output_dir / f"eval_{split}_{checkpoint_path.stem}.json"
        write_json(result, output_path)
    return result


def main() -> int:
    """CLI entrypoint for skeleton baseline training."""

    parser = build_parser()
    args = parser.parse_args()
    return run_training(args.config, args)


def evaluate_main() -> int:
    """CLI entrypoint for skeleton checkpoint evaluation."""

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
