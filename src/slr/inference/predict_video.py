"""Placeholder video prediction CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the video prediction parser."""

    parser = argparse.ArgumentParser(
        description="Run inference on a single sign video."
    )
    parser.add_argument("--config", type=Path, required=True, help="Model config.")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path for prediction.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    LOGGER.info(
        "Prediction placeholder: config=%s video=%s checkpoint=%s",
        args.config,
        args.video,
        args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
