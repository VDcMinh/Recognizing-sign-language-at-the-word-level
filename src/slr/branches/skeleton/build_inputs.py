"""Build skeleton branch inputs from shared RTMW-l pose outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for skeleton branch input building."""

    parser = argparse.ArgumentParser(
        description="Build skeleton branch inputs from shared RTMW-l pose files."
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("data/datasets/WLASL/pose/rtmw_l/wholebody_133"),
        help="Directory containing shared wholebody-133 pose files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l"),
        help="Output root for selected keypoints, normalized pose, and graph tensors.",
    )
    parser.add_argument(
        "--keypoint-set",
        type=str,
        default="selected_31",
        choices=["selected_27", "selected_31", "selected_49"],
        help="Keypoint subset to materialize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without writing branch inputs.",
    )
    return parser


def run(
    pose_dir: Path,
    output_root: Path,
    keypoint_set: str = "selected_31",
    dry_run: bool = False,
) -> int:
    """Placeholder execution entrypoint for skeleton branch preprocessing."""

    LOGGER.info("Pose dir: %s", pose_dir)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Keypoint set: %s", keypoint_set)
    LOGGER.info("Dry run: %s", dry_run)

    if dry_run:
        return 0

    for subdir in (
        keypoint_set,
        "normalized",
        "graph_tensors",
        "manifests",
        "reports",
    ):
        ensure_dir(output_root / subdir)

    LOGGER.info(
        "Placeholder only: selected keypoints, normalization, and graph tensor export are pending."
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        pose_dir=args.pose_dir,
        output_root=args.output_root,
        keypoint_set=args.keypoint_set,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
