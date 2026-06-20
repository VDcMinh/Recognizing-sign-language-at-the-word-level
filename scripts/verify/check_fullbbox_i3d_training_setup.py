"""Smoke-test the FullBBox-I3D appearance training setup and write a report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from slr.branches.appearance.train import (
    build_appearance_dataloaders,
    build_appearance_datasets,
    build_appearance_model,
    resolve_training_config,
    run_training,
    select_device,
)
from slr.training.losses import build_loss_from_config
from slr.training.metrics import accuracy_topk
from slr.utils.io import read_json, write_json
from slr.utils.logging import setup_logger


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the FullBBox-I3D training smoke test."""

    parser = argparse.ArgumentParser(
        description="Smoke-test the FullBBox-I3D appearance training setup."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/appearance/nslt100/fullbbox_i3d_ce.yaml"),
        help="Appearance training config YAML.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Optional override for data.package_root.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/current/fullbbox_i3d/check_fullbbox_i3d_training_setup.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to the Markdown path with .json suffix.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        default=Path("outputs/appearance/fullbbox_i3d_nslt100_smoke"),
        help="Output directory used by the one-epoch smoke run.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device override for the smoke run.")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count for the smoke run.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the smoke run.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the smoke run.")
    parser.add_argument("--limit-train", type=int, default=1, help="Limit train samples for the smoke run.")
    parser.add_argument("--limit-val", type=int, default=1, help="Limit val samples for the smoke run.")
    parser.add_argument("--limit-test", type=int, default=1, help="Limit test samples for the smoke run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed override for the smoke run.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="fullbbox_i3d_nslt100_smoke",
        help="Run name override for the smoke run.",
    )
    return parser


def _training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Create the namespace expected by the appearance training entrypoint."""

    return argparse.Namespace(
        config=args.config,
        package_root=args.package_root,
        output_dir=args.smoke_output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=None,
        eval_only=False,
        run_name=args.run_name,
        seed=args.seed,
        no_wandb=True,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        dry_run=False,
    )


def _build_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the appearance config to the shared loss helper schema."""

    return {
        "train": {"loss": str(config["training"]["loss"]["name"]).strip().lower()},
        "label_smoothing": dict(config["training"].get("label_smoothing", {})),
    }


def _count_parameters(model) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def _resolve_output_path(path: Path, repo_root: Path) -> Path:
    """Resolve one possibly relative path under the repo root."""

    return path if path.is_absolute() else (repo_root / path).resolve()


def _normalize_topk_values(topk_values: Any) -> list[int]:
    """Normalize top-k values for reporting."""

    if topk_values is None:
        return [1, 5, 10]
    normalized: list[int] = []
    seen: set[int] = set()
    for value in topk_values:
        k = int(value)
        if k <= 0 or k in seen:
            continue
        seen.add(k)
        normalized.append(k)
    return normalized or [1, 5, 10]


def _build_markdown(summary: dict[str, Any]) -> str:
    """Render one Markdown report from the smoke-test summary."""

    precheck = summary["precheck"]
    smoke = summary["smoke_run"]
    outputs = smoke["artifacts"]
    lines: list[str] = []
    lines.append("# FullBBox-I3D Training Setup Smoke Test")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Config: `{summary['config_path']}`")
    lines.append(f"- Package root: `{summary['package_root']}`")
    lines.append(f"- Device used: `{precheck['device']}`")
    lines.append(f"- I3D variant: `{summary['i3d_variant']}`")
    lines.append("")
    lines.append("## Precheck")
    lines.append(f"- Configured eval.topk: `{precheck['configured_topk']}`")
    lines.append(f"- Split sizes: train={precheck['dataset_sizes']['train']}, val={precheck['dataset_sizes']['val']}, test={precheck['dataset_sizes']['test']}")
    lines.append(f"- Batch shape: `{precheck['batch_shape']}`")
    lines.append(f"- Batch dtype: `{precheck['batch_dtype']}`")
    lines.append(f"- Label shape: `{precheck['label_shape']}`")
    lines.append(f"- Logits shape: `{precheck['logits_shape']}`")
    lines.append(f"- Loss type: `{precheck['loss_type']}`")
    lines.append(f"- Initial loss: `{precheck['initial_loss']}`")
    for key, value in precheck.get("batch_accuracy", {}).items():
        lines.append(f"- Batch {key}: `{value}`")
    lines.append(
        f"- Model parameters: total={precheck['parameter_count']['total']}, "
        f"trainable={precheck['parameter_count']['trainable']}"
    )
    lines.append("")
    lines.append("## Smoke Run")
    lines.append(f"- Exit code: `{smoke['exit_code']}`")
    lines.append(f"- Duration seconds: `{smoke['duration_seconds']}`")
    lines.append(f"- Output dir: `{smoke['output_dir']}`")
    lines.append(f"- Best checkpoint exists: `{outputs['best_checkpoint_exists']}`")
    lines.append(f"- Last checkpoint exists: `{outputs['last_checkpoint_exists']}`")
    lines.append(f"- Metrics JSON exists: `{outputs['metrics_json_exists']}`")
    lines.append(f"- Summary JSON exists: `{outputs['summary_json_exists']}`")
    if smoke["metrics_summary"]:
        lines.append(
            f"- Best epoch: `{smoke['metrics_summary'].get('best_epoch')}` | "
            f"Best metric: `{smoke['metrics_summary'].get('best_metric')}`"
        )
        metric_parts = []
        for k in smoke.get("configured_topk", []):
            metric_parts.append(f"test_top{k}=`{smoke['metrics_summary'].get(f'test_top{k}')}`")
        if metric_parts:
            lines.append(f"- {' | '.join(metric_parts)}")
    lines.append("")
    lines.append("## Notes")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    if summary["issues"]:
        lines.append("## Issues")
        for issue in summary["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    else:
        lines.append("## Issues")
        lines.append("- None.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run the FullBBox-I3D appearance training smoke test."""

    args = build_parser().parse_args()
    repo_root = Path.cwd().resolve()
    output_report = _resolve_output_path(args.output_report, repo_root)
    output_json = _resolve_output_path(
        args.output_json if args.output_json is not None else args.output_report.with_suffix(".json"),
        repo_root,
    )
    logger = setup_logger("scripts.verify.check_fullbbox_i3d_training_setup")
    training_args = _training_namespace(args)

    status = "FAILED"
    issues: list[str] = []
    notes = [
        "This smoke test uses a one-epoch run with limited samples to validate the FullBBox-I3D pipeline end-to-end.",
        "The implemented backbone is reported as an I3D-style / Inception3D-like RGB stream, not a claim of canonical I3D reproduction.",
    ]

    try:
        precheck_started = time.perf_counter()
        resolved_config = resolve_training_config(args.config, training_args)
        device = select_device(str(resolved_config["training"]["device"]), logger=logger)
        configured_topk = _normalize_topk_values(resolved_config.get("eval", {}).get("topk"))
        datasets = build_appearance_datasets(resolved_config)
        dataloaders = build_appearance_dataloaders(resolved_config, datasets, device=device)
        sample_batch = next(iter(dataloaders["train"]))
        model = build_appearance_model(resolved_config["model"]).to(device)
        criterion = build_loss_from_config(_build_loss_config(resolved_config))

        with torch.no_grad():
            video = sample_batch["video"].to(device)
            labels = sample_batch["labels"].to(device)
            output = model(video)
            logits = output["logits"] if isinstance(output, dict) else output
            initial_loss = float(criterion(logits, labels).item())
            batch_accuracy = accuracy_topk(logits, labels, topk=tuple(configured_topk))

        precheck_summary = {
            "duration_seconds": round(time.perf_counter() - precheck_started, 3),
            "device": str(device),
            "configured_topk": list(configured_topk),
            "dataset_sizes": {
                "train": len(datasets["train"]),
                "val": len(datasets["val"]),
                "test": len(datasets["test"]),
            },
            "batch_shape": list(sample_batch["video"].shape),
            "batch_dtype": str(sample_batch["video"].dtype),
            "label_shape": list(sample_batch["labels"].shape),
            "logits_shape": list(logits.shape),
            "initial_loss": round(initial_loss, 6),
            "loss_type": str(resolved_config["runtime"]["loss_type"]),
            "batch_accuracy": {key: round(float(value), 6) for key, value in batch_accuracy.items()},
            "parameter_count": _count_parameters(model),
        }

        smoke_started = time.perf_counter()
        exit_code = int(run_training(args.config, training_args))
        smoke_duration = round(time.perf_counter() - smoke_started, 3)

        output_dir = _resolve_output_path(Path(training_args.output_dir), repo_root)
        best_checkpoint = output_dir / "checkpoints" / "best.pt"
        last_checkpoint = output_dir / "checkpoints" / "last.pt"
        metrics_json = output_dir / "metrics.json"
        summary_json = output_dir / "summary.json"

        metrics_payload = read_json(metrics_json) if metrics_json.exists() else {}
        summary_payload = read_json(summary_json) if summary_json.exists() else {}

        if exit_code == 0 and best_checkpoint.exists() and last_checkpoint.exists():
            status = "READY"
        else:
            status = "FAILED"
            if exit_code != 0:
                issues.append(f"Smoke training exited with code {exit_code}.")
            if not best_checkpoint.exists():
                issues.append(f"Missing best checkpoint: {best_checkpoint.as_posix()}")
            if not last_checkpoint.exists():
                issues.append(f"Missing last checkpoint: {last_checkpoint.as_posix()}")

        summary = {
            "status": status,
            "config_path": _resolve_output_path(args.config, repo_root).as_posix(),
            "package_root": Path(resolved_config["data"]["package_root"]).resolve().as_posix(),
            "i3d_variant": "I3D-style / Inception3D-like RGB stream",
            "precheck": precheck_summary,
            "smoke_run": {
                "exit_code": exit_code,
                "duration_seconds": smoke_duration,
                "output_dir": output_dir.as_posix(),
                "configured_topk": list(configured_topk),
                "metrics_summary": metrics_payload.get("summary", {}),
                "artifacts": {
                    "best_checkpoint_exists": best_checkpoint.exists(),
                    "last_checkpoint_exists": last_checkpoint.exists(),
                    "metrics_json_exists": metrics_json.exists(),
                    "summary_json_exists": summary_json.exists(),
                },
                "summary_payload": summary_payload,
            },
            "notes": notes,
            "issues": issues,
        }
    except Exception as exc:  # pragma: no cover - failure path is environment-dependent
        status = "FAILED"
        issues.append(f"{type(exc).__name__}: {exc}")
        summary = {
            "status": status,
            "config_path": _resolve_output_path(args.config, repo_root).as_posix(),
            "package_root": "" if args.package_root is None else _resolve_output_path(args.package_root, repo_root).as_posix(),
            "i3d_variant": "I3D-style / Inception3D-like RGB stream",
            "precheck": {},
            "smoke_run": {},
            "notes": notes,
            "issues": issues,
        }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_build_markdown(summary), encoding="utf-8")
    write_json(summary, output_json)

    print(json.dumps({"status": summary["status"], "report": output_report.as_posix()}, ensure_ascii=False))
    return 0 if summary["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
