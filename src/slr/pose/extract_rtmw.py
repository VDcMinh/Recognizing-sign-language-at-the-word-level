"""Extract shared RTMW-l whole-body pose from standardized WLASL inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from slr.utils.io import ensure_dir, read_yaml
from slr.utils.logging import get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for RTMW-l extraction."""

    parser = argparse.ArgumentParser(
        description="Extract RTMW-l wholebody-133 pose from standardized WLASL clips."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/pose_rtmw_l.yaml"),
        help="Path to the RTMW-l extraction configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device placeholder, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without running pose extraction.",
    )
    return parser


def run(config_path: Path, device: str = "cpu", dry_run: bool = False) -> int:
    """Placeholder execution entrypoint for RTMW-l extraction."""

    config = read_yaml(config_path)
    pose_cfg = config.get("pose", {})
    output_dir = Path(pose_cfg.get("output_dir", "data/datasets/WLASL/pose/rtmw_l"))

    LOGGER.info("Pose config: %s", config_path)
    LOGGER.info("Backend: %s", pose_cfg.get("backend", "rtmw_l"))
    LOGGER.info("Expected keypoints: %s", pose_cfg.get("expected_keypoints", 133))
    LOGGER.info("Input source: %s", pose_cfg.get("input_source"))
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output directory: %s", output_dir)

    if dry_run:
        LOGGER.info("Dry run enabled; no pose files will be created.")
        return 0

    ensure_dir(output_dir)
    LOGGER.info(
        "Placeholder only: integrate MMPose / RTMW-l config and checkpoint loading here."
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(config_path=args.config, device=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
