"""Unified evaluation CLI scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the generic evaluation CLI parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint on a selected split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Branch or experiment config file used for evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the resolved evaluation plan only.",
    )
    return parser


def run(
    config_path: Path,
    checkpoint: Path | None = None,
    split: str = "test",
    dry_run: bool = False,
) -> int:
    """Placeholder evaluation entrypoint."""

    LOGGER.info("Evaluation config: %s", config_path)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Split: %s", split)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info("Placeholder only: evaluation loop integration is pending.")
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
