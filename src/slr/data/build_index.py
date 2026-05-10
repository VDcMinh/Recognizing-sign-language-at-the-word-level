"""Build WLASL index manifests from raw metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir, read_yaml
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for index building."""

    parser = argparse.ArgumentParser(
        description="Build index manifests from WLASL raw metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/index.yaml"),
        help="Path to the index preprocessing configuration.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset override, e.g. nslt100.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without writing manifests.",
    )
    return parser


def run(config_path: Path, subset: str | None = None, dry_run: bool = False) -> int:
    """Placeholder execution entrypoint for the index stage."""

    config = read_yaml(config_path)
    index_cfg = config.get("index", {})
    resolved_subset = subset or index_cfg.get("subset", "nslt100")
    output_dir = Path(index_cfg.get("output_dir", "data/datasets/WLASL/index"))
    reports_dir = Path(index_cfg.get("reports_dir", output_dir / "reports"))

    LOGGER.info("Index stage configuration loaded from %s", config_path)
    LOGGER.info("Subset: %s", resolved_subset)
    LOGGER.info("Raw metadata root: %s", "data/datasets/WLASL/raw/metadata")
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Reports directory: %s", reports_dir)

    if dry_run:
        LOGGER.info("Dry run enabled; no files will be written.")
        return 0

    ensure_dir(output_dir)
    ensure_dir(reports_dir)
    LOGGER.info("Placeholder only: WLASL metadata parsing is not implemented yet.")
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(config_path=args.config, subset=args.subset, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
