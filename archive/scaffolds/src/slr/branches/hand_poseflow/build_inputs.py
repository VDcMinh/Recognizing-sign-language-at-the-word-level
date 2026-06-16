"""Coordinate hand-sequence and pose-flow preprocessing for the third branch."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.branches.hand_poseflow.build_hand_sequences import run as run_hand_sequences
from slr.branches.hand_poseflow.build_poseflow import run as run_poseflow
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the hand-poseflow branch input stage."""

    parser = argparse.ArgumentParser(
        description="Build hand sequence and pose-flow inputs for the third branch."
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
        help="Output root for hand sequences and pose-flow features.",
    )
    parser.add_argument(
        "--poseflow-variant",
        type=str,
        default="selected_31",
        choices=["selected_31", "hands_only"],
        help="Pose-flow variant to materialize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing branch artifacts.",
    )
    return parser


def run(
    frames_dir: Path,
    pose_dir: Path,
    output_root: Path,
    poseflow_variant: str = "selected_31",
    dry_run: bool = False,
) -> int:
    """Run both branch preparation sub-stages in sequence."""

    LOGGER.info("Building hand sequence inputs.")
    run_hand_sequences(
        frames_dir=frames_dir,
        pose_dir=pose_dir,
        output_root=output_root,
        dry_run=dry_run,
    )
    LOGGER.info("Building pose-flow inputs.")
    run_poseflow(
        pose_dir=pose_dir,
        output_root=output_root,
        variant=poseflow_variant,
        dry_run=dry_run,
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        frames_dir=args.frames_dir,
        pose_dir=args.pose_dir,
        output_root=args.output_root,
        poseflow_variant=args.poseflow_variant,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
