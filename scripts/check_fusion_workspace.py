"""Check the required NSLT100 late-fusion workspace files."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the required late-fusion checkpoints/configs exist."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Fusion workspace directory, for example artifacts/fusion/nslt100.",
    )
    return parser


def expected_paths(workspace: Path) -> list[Path]:
    return [
        workspace / "checkpoints" / "skeleton" / "best.pt",
        workspace / "checkpoints" / "regions" / "best.pt",
        workspace / "configs" / "skeleton" / "config_resolved.yaml",
        workspace / "configs" / "regions" / "config_resolved.yaml",
    ]


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace)
    missing = [path for path in expected_paths(workspace) if not path.exists()]

    if missing:
        print("Missing:")
        for path in missing:
            print(f"- {path.as_posix()}")
        print()
        print("Please copy your best checkpoints and resolved configs to the paths above.")
        return 1

    print(f"Fusion workspace is ready: {workspace.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
