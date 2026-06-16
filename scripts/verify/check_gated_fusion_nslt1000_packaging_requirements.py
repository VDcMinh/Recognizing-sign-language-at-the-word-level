"""Check whether the repo is ready to package NSLT1000 gated fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slr.branches.fusion.package_support import (
    PACKAGE_NAME_DEFAULT,
    READY_STATUS,
    build_requirements_summary,
    format_bytes,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the requirement-check CLI."""

    parser = argparse.ArgumentParser(
        description="Check whether the repo is ready to package WLASL NSLT1000 gated fusion."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Project root to inspect.")
    parser.add_argument(
        "--skeleton-checkpoint",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/checkpoints/skeleton/best.pt"),
        help="Path to the validated NSLT1000 skeleton checkpoint.",
    )
    parser.add_argument(
        "--regions-checkpoint",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/checkpoints/regions/best.pt"),
        help="Path to the validated NSLT1000 regions checkpoint.",
    )
    parser.add_argument(
        "--skeleton-config",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/configs/skeleton/config_resolved.yaml"),
        help="Path to the validated NSLT1000 skeleton resolved config.",
    )
    parser.add_argument(
        "--regions-config",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/configs/regions/config_resolved.yaml"),
        help="Path to the validated NSLT1000 regions resolved config.",
    )
    parser.add_argument(
        "--skeleton-manifest-root",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests"),
        help="Directory containing the NSLT1000 selected_31 skeleton manifests.",
    )
    parser.add_argument(
        "--regions-manifest-root",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l_union/manifests"),
        help="Directory containing the NSLT1000 union regions manifests.",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=Path("configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml"),
        help="Path to the local NSLT1000 fusion train config.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--check-all-tensor-paths",
        action="store_true",
        help="Resolve every source tensor path instead of sampling a few paths per split.",
    )
    return parser


def resolve_path(path_value: Path, *, repo_root: Path) -> Path:
    """Resolve one possibly-relative path against repo_root."""

    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def print_summary(summary: dict[str, object], *, output_json: Path | None) -> None:
    """Print the key readiness information."""

    print(f"package_name_default: {PACKAGE_NAME_DEFAULT}")
    print(f"status: {summary['status']}")
    print(f"ready_to_execute: {summary['ready_to_execute']}")
    print(f"free_disk: {summary['disk']['free_human']}")
    print(f"estimated_total_source_size: {summary['size_estimate']['total_human']}")
    print(
        "estimated_additional_physical_size_copy_mode: "
        f"{summary['size_estimate']['estimated_additional_physical_human_copy_mode']}"
    )
    print(
        "fusion_smoke_logits_shape: "
        f"{summary['fusion_smoke'].get('logits_shape', '<not available>')}"
    )
    print(
        "fusion_smoke_loss: "
        f"{summary['fusion_smoke'].get('loss', '<not available>')}"
    )
    for split, split_summary in summary["alignment"]["splits"].items():
        print(
            f"{split}: matched={split_summary['matched_count']} "
            f"class_mismatch={split_summary['class_mismatch_count']} "
            f"gloss_mismatch={split_summary['gloss_mismatch_count']} "
            f"split_mismatch={split_summary['split_mismatch_count']}"
        )
    if summary["warnings"]:
        print("warnings:")
        for item in summary["warnings"]:
            print(f"- {item}")
    if summary["errors"]:
        print("errors:")
        for item in summary["errors"]:
            print(f"- {item}")
    if output_json is not None:
        print(f"output_json: {output_json.as_posix()}")
    print(str(summary["status"]).upper())


def main() -> int:
    """Run the requirement check."""

    args = build_parser().parse_args()
    repo_root = resolve_path(args.repo_root, repo_root=Path.cwd())
    skeleton_checkpoint = resolve_path(args.skeleton_checkpoint, repo_root=repo_root)
    regions_checkpoint = resolve_path(args.regions_checkpoint, repo_root=repo_root)
    skeleton_config = resolve_path(args.skeleton_config, repo_root=repo_root)
    regions_config = resolve_path(args.regions_config, repo_root=repo_root)
    skeleton_manifest_root = resolve_path(args.skeleton_manifest_root, repo_root=repo_root)
    regions_manifest_root = resolve_path(args.regions_manifest_root, repo_root=repo_root)
    fusion_config = resolve_path(args.fusion_config, repo_root=repo_root)
    output_json = (
        resolve_path(args.output_json, repo_root=repo_root)
        if args.output_json is not None
        else None
    )

    summary = build_requirements_summary(
        repo_root=repo_root,
        skeleton_checkpoint=skeleton_checkpoint,
        regions_checkpoint=regions_checkpoint,
        skeleton_config=skeleton_config,
        regions_config=regions_config,
        skeleton_manifest_root=skeleton_manifest_root,
        regions_manifest_root=regions_manifest_root,
        fusion_config=fusion_config,
        check_all_tensor_paths=bool(args.check_all_tensor_paths),
    )

    if output_json is not None:
        write_json(output_json, summary)
    print_summary(summary, output_json=output_json)
    return 0 if summary["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
