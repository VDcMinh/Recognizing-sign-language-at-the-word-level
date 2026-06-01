"""Safely clean generated outputs for the local-image regions branch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from slr.utils.io import read_json, stringify_path


DEFAULT_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l")
ALLOWED_SUBSETS = ("nslt100", "nslt300", "nslt1000", "nslt2000")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Safely remove generated regions-branch outputs for one subset."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="nslt100",
        help="Subset name to clean. Defaults to nslt100.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Regions output root. Defaults to data/datasets/WLASL/branch_inputs/regions/rtmw_l.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the resolved targets. Without this flag the script performs a dry-run only.",
    )
    return parser


def _resolve_root(root: Path) -> Path:
    """Resolve the regions output root safely under the current repo."""

    resolved = root.resolve(strict=False)
    normalized = str(resolved).replace("\\", "/")
    if not normalized.endswith("data/datasets/WLASL/branch_inputs/regions/rtmw_l"):
        raise ValueError(f"Refusing to operate outside the regions output root: {resolved}")
    return resolved


def _metadata_path(root: Path) -> Path:
    """Return the metadata path for the regions root."""

    return root / "metadata.json"


def _build_targets(root: Path, subset: str) -> list[Path]:
    """Build the safe deletion target list for one subset."""

    targets = [
        root / "crops" / subset,
        root / "tensors" / subset,
        root / "previews" / subset,
        root / "manifests" / f"{subset}_train.csv",
        root / "manifests" / f"{subset}_val.csv",
        root / "manifests" / f"{subset}_test.csv",
        root / "reports" / f"{subset}_region_crop_quality_report.md",
        root / "reports" / f"{subset}_region_low_quality_samples.csv",
        root / "logs" / f"{subset}_build_regions.log",
    ]

    metadata_path = _metadata_path(root)
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
        except Exception:
            metadata = None
        if isinstance(metadata, dict) and str(metadata.get("subset", "")).strip() == subset:
            targets.append(metadata_path)
    return targets


def _validate_targets(root: Path, targets: list[Path]) -> list[Path]:
    """Validate that all delete targets stay under the allowed root."""

    normalized_root = root.resolve(strict=False)
    validated: list[Path] = []
    for target in targets:
        resolved_target = target.resolve(strict=False)
        if normalized_root not in resolved_target.parents and resolved_target != normalized_root:
            raise ValueError(f"Refusing to delete path outside the regions root: {resolved_target}")
        validated.append(resolved_target)
    return validated


def _delete_target(path: Path) -> None:
    """Delete one file or directory when it exists."""

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def main() -> int:
    """Run the cleanup entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    subset = str(args.subset).strip()
    if subset not in ALLOWED_SUBSETS:
        raise ValueError(f"Unsupported subset: {subset!r}. Expected one of {ALLOWED_SUBSETS}.")

    root = _resolve_root(args.root)
    targets = _validate_targets(root, _build_targets(root, subset))

    print("== Regions Cleanup ==")
    print(f"subset: {subset}")
    print(f"regions root: {stringify_path(root)}")
    print(f"mode: {'delete' if args.yes else 'dry-run'}")
    print()
    print("Targets:")
    for target in targets:
        exists_text = "exists" if target.exists() else "missing"
        print(f"- {stringify_path(target)} [{exists_text}]")

    if not args.yes:
        print()
        print("Dry-run only. Re-run with --yes to delete these targets.")
        return 0

    for target in targets:
        _delete_target(target)

    print()
    print("Cleanup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
