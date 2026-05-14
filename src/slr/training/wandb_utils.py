"""Optional Weights & Biases integration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_wandb_entity(
    logging_cfg: dict[str, Any],
    *,
    cli_entity: str | None = None,
) -> str | None:
    """Resolve the W&B entity from CLI, config, or environment."""

    if cli_entity:
        return str(cli_entity).strip()
    config_entity = logging_cfg.get("entity")
    if config_entity:
        return str(config_entity).strip()
    entity_env = str(logging_cfg.get("entity_env", "WANDB_ENTITY")).strip()
    return os.getenv(entity_env)


def init_wandb_run(
    *,
    resolved_config: dict[str, Any],
    logging_cfg: dict[str, Any],
    run_name: str,
    logger,
    cli_entity: str | None = None,
):
    """Initialize an optional W&B run, or return ``None`` when disabled."""

    if not bool(logging_cfg.get("use_wandb", False)):
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; disabling W&B logging for this run.")
        return None

    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set; disabling W&B logging for this run.")
        return None

    entity = resolve_wandb_entity(logging_cfg, cli_entity=cli_entity)
    if not entity:
        logger.warning("W&B entity could not be resolved; disabling W&B logging for this run.")
        return None

    project = str(logging_cfg.get("project", "wlasl-skeleton")).strip()
    tags = [str(tag) for tag in logging_cfg.get("tags", [])]
    logger.info("Initializing W&B run project=%s entity=%s name=%s", project, entity, run_name)
    return wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=resolved_config,
        tags=tags,
    )


def log_wandb_metrics(run, metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log one metrics payload when W&B is active."""

    if run is None:
        return
    run.log(dict(metrics), step=step)


def log_wandb_model_artifact(
    run,
    checkpoint_path: str | Path,
    *,
    artifact_name: str,
    aliases: list[str] | None = None,
) -> None:
    """Upload one checkpoint file as a W&B model artifact."""

    if run is None:
        return

    try:
        import wandb
    except ImportError:  # pragma: no cover - guarded by init
        return

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")

    artifact = wandb.Artifact(str(artifact_name), type="model")
    artifact.add_file(str(checkpoint), name=checkpoint.name)
    run.log_artifact(artifact, aliases=aliases or ["best"])


def finish_wandb_run(run) -> None:
    """Finalize the W&B run if one exists."""

    if run is not None:
        run.finish()
