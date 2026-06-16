"""Unified training CLI scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the generic training CLI parser."""

    parser = argparse.ArgumentParser(
        description="Launch training for a selected branch configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Branch or experiment config file to train from.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device placeholder, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the resolved training plan only.",
    )
    return parser


def run(
    config_path: Path,
    experiment_dir: Path,
    device: str = "cpu",
    dry_run: bool = False,
) -> int:
    """Placeholder training entrypoint."""

    LOGGER.info("Training config: %s", config_path)
    LOGGER.info("Experiment dir: %s", experiment_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info("Placeholder only: training loop integration is pending.")
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        config_path=args.config,
        experiment_dir=args.experiment_dir,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
