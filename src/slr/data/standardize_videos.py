"""Standardize WLASL videos into a shared normalized layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir, read_yaml
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for video standardization."""

    parser = argparse.ArgumentParser(
        description="Standardize WLASL videos by crop/resize/letterbox rules."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/standardize.yaml"),
        help="Path to the standardization configuration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without writing videos or frames.",
    )
    return parser


def run(config_path: Path, limit: int | None = None, dry_run: bool = False) -> int:
    """Placeholder execution entrypoint for video standardization."""

    config = read_yaml(config_path)
    standardize_cfg = config.get("standardize", {})
    output_root = Path(
        standardize_cfg.get("output_root", "data/datasets/WLASL/standardized")
    )

    LOGGER.info("Standardization config: %s", config_path)
    LOGGER.info("Input manifest: %s", standardize_cfg.get("input_manifest"))
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Limit: %s", limit)
    LOGGER.info("Dry run: %s", dry_run)

    if dry_run:
        return 0

    for subdir in ("videos", "frames", "manifests", "logs", "reports"):
        ensure_dir(output_root / subdir)

    LOGGER.info(
        "Placeholder only: video decoding, crop/resize, and letterboxing are pending."
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(config_path=args.config, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
