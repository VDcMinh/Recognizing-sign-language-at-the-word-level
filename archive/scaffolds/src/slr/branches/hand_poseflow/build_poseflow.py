"""Build pose-flow features from shared pose sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for pose-flow generation."""

    parser = argparse.ArgumentParser(
        description="Build pose-flow features for the hand-poseflow branch."
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("data/datasets/WLASL/pose/rtmw_l/wholebody_133"),
        help="Directory containing shared pose files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs/hand_poseflow/rtmw_l"),
        help="Output root for pose-flow features.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="selected_31",
        choices=["selected_31", "hands_only"],
        help="Pose-flow feature variant to build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without writing pose-flow artifacts.",
    )
    return parser


def run(
    pose_dir: Path,
    output_root: Path,
    variant: str = "selected_31",
    dry_run: bool = False,
) -> int:
    """Placeholder execution entrypoint for pose-flow generation."""

    LOGGER.info("Pose dir: %s", pose_dir)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Variant: %s", variant)
    LOGGER.info("Dry run: %s", dry_run)

    if dry_run:
        return 0

    ensure_dir(output_root / "poseflow" / variant)
    ensure_dir(output_root / "combined")
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "reports")
    LOGGER.info("Placeholder only: pose-flow generation is not implemented yet.")
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        pose_dir=args.pose_dir,
        output_root=args.output_root,
        variant=args.variant,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
