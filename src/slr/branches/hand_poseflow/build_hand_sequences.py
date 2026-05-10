"""Build left and right hand image sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for hand sequence extraction."""

    parser = argparse.ArgumentParser(
        description="Build hand image sequences for the hand-poseflow branch."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/datasets/WLASL/standardized/frames"),
        help="Directory containing standardized frames.",
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
        help="Output root for hand image sequences.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without writing hand sequences.",
    )
    return parser


def run(
    frames_dir: Path, pose_dir: Path, output_root: Path, dry_run: bool = False
) -> int:
    """Placeholder execution entrypoint for hand sequence extraction."""

    LOGGER.info("Frames dir: %s", frames_dir)
    LOGGER.info("Pose dir: %s", pose_dir)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Dry run: %s", dry_run)

    if dry_run:
        return 0

    ensure_dir(output_root / "hand_sequences" / "left_hand")
    ensure_dir(output_root / "hand_sequences" / "right_hand")
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "reports")
    LOGGER.info("Placeholder only: hand sequence generation is not implemented yet.")
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        frames_dir=args.frames_dir,
        pose_dir=args.pose_dir,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
