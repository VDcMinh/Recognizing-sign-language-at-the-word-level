"""Package a self-contained NSLT1000 gated-fusion Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from slr.branches.fusion.package_support import (
    PACKAGE_NAME_DEFAULT,
    READY_STATUS,
    SAMPLE_SPLITS,
    build_alignment,
    build_kaggle_root,
    build_package_metadata,
    build_requirements_summary,
    canonical_sample_id,
    copy_report_files,
    create_package_readme,
    ensure_within_root,
    estimate_source_sizes,
    format_bytes,
    get_free_disk_bytes,
    load_build_state,
    load_regions_manifest_set,
    load_skeleton_manifest_set,
    materialize_file,
    now_utc_iso,
    relative_package_path,
    resolve_regions_source_tensor,
    resolve_skeleton_source_tensor,
    rewrite_regions_manifest_row,
    rewrite_skeleton_manifest_row,
    save_build_state,
    verify_existing_file,
    write_json,
    write_manifest,
)
from slr.utils.io import read_yaml, write_yaml


VERIFIER_SOURCE = Path("scripts/verify/verify_gated_fusion_nslt1000_package.py")
LOCAL_FUSION_CONFIG_DEFAULT = Path("configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml")
KAGGLE_FUSION_CONFIG_TEMPLATE = Path("configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_kaggle_ce.yaml")


class PackagingError(RuntimeError):
    """Raised when packaging cannot continue safely."""


def build_parser() -> argparse.ArgumentParser:
    """Build the packaging CLI."""

    parser = argparse.ArgumentParser(
        description="Package a fully self-contained NSLT1000 gated-fusion Kaggle dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Project root to package from.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("packaging_outputs"),
        help="Directory where the package root will be created.",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=PACKAGE_NAME_DEFAULT,
        help="Package directory name.",
    )
    parser.add_argument(
        "--skeleton-checkpoint",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/checkpoints/skeleton/best.pt"),
        help="Path to the NSLT1000 skeleton checkpoint.",
    )
    parser.add_argument(
        "--regions-checkpoint",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/checkpoints/regions/best.pt"),
        help="Path to the NSLT1000 regions checkpoint.",
    )
    parser.add_argument(
        "--skeleton-config",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/configs/skeleton/config_resolved.yaml"),
        help="Path to the NSLT1000 skeleton resolved config.",
    )
    parser.add_argument(
        "--regions-config",
        type=Path,
        default=Path("artifacts/fusion/nslt1000/configs/regions/config_resolved.yaml"),
        help="Path to the NSLT1000 regions resolved config.",
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
        "--link-mode",
        choices=("hardlink", "copy", "auto"),
        default="auto",
        help="Tensor materialization mode.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted build.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing package root.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the build without materializing tensors.")
    parser.add_argument("--execute", action="store_true", help="Actually build the package.")
    parser.add_argument(
        "--verify-after-build",
        action="store_true",
        help="Run the packaged verifier after a successful build.",
    )
    parser.add_argument(
        "--write-plan",
        type=Path,
        default=None,
        help="Optional JSON path for the generated build plan.",
    )
    return parser


def resolve_path(path_value: Path, *, repo_root: Path) -> Path:
    """Resolve one possibly-relative path against repo_root."""

    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def create_packaged_kaggle_config(*, template_path: Path, package_name: str) -> dict[str, Any]:
    """Create the portable package config for the chosen package name."""

    config = read_yaml(template_path)
    kaggle_root = build_kaggle_root(package_name)

    config["experiment"] = dict(config.get("experiment", {}))
    config["experiment"]["output_root"] = "/kaggle/working/outputs/fusion"

    dataset_cfg = dict(config.get("dataset", {}))
    skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
    regions_cfg = dict(dataset_cfg.get("regions", {}))
    skeleton_cfg["data_root"] = f"{kaggle_root}/branch_inputs/skeleton/rtmw_l"
    skeleton_cfg["manifests"] = {
        split: f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt1000_selected_31_{split}.csv"
        for split in SAMPLE_SPLITS
    }
    regions_cfg["data_root"] = f"{kaggle_root}/branch_inputs/regions/rtmw_l"
    regions_cfg["manifests"] = {
        split: f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt1000_{split}.csv"
        for split in SAMPLE_SPLITS
    }
    dataset_cfg["skeleton"] = skeleton_cfg
    dataset_cfg["regions"] = regions_cfg
    config["dataset"] = dataset_cfg

    skeleton_branch_cfg = dict(config.get("skeleton_branch", {}))
    regions_branch_cfg = dict(config.get("regions_branch", {}))
    skeleton_branch_cfg["config_path"] = f"{kaggle_root}/configs/skeleton_config_resolved.yaml"
    skeleton_branch_cfg["checkpoint_path"] = f"{kaggle_root}/checkpoints/skeleton/best.pt"
    regions_branch_cfg["config_path"] = f"{kaggle_root}/configs/regions_config_resolved.yaml"
    regions_branch_cfg["checkpoint_path"] = f"{kaggle_root}/checkpoints/regions/best.pt"
    config["skeleton_branch"] = skeleton_branch_cfg
    config["regions_branch"] = regions_branch_cfg
    return config


def build_plan(
    *,
    repo_root: Path,
    package_root: Path,
    package_name: str,
    link_mode: str,
    requirements_summary: dict[str, Any],
    size_estimate: dict[str, Any],
) -> dict[str, Any]:
    """Build the dry-run plan payload."""

    free_bytes = get_free_disk_bytes(repo_root)
    total_bytes = int(size_estimate.get("total_bytes", 0))
    additional_physical = 0 if link_mode == "hardlink" else total_bytes
    if link_mode == "auto":
        additional_physical = total_bytes

    return {
        "package_name": package_name,
        "package_root": package_root.as_posix(),
        "source_paths": {
            "repo_root": repo_root.as_posix(),
            "skeleton_checkpoint": requirements_summary["paths"]["skeleton_checkpoint"],
            "regions_checkpoint": requirements_summary["paths"]["regions_checkpoint"],
            "skeleton_config": requirements_summary["paths"]["skeleton_config"],
            "regions_config": requirements_summary["paths"]["regions_config"],
            "skeleton_manifest_root": requirements_summary["paths"]["skeleton_manifest_root"],
            "regions_manifest_root": requirements_summary["paths"]["regions_manifest_root"],
            "skeleton_tensor_source_root": size_estimate.get("skeleton_source_root"),
            "regions_branch_root": size_estimate.get("regions_branch_root"),
        },
        "destination_paths": {
            "package_root": package_root.as_posix(),
            "checkpoints_root": (package_root / "checkpoints").as_posix(),
            "configs_root": (package_root / "configs").as_posix(),
            "skeleton_root": (package_root / "branch_inputs/skeleton/rtmw_l").as_posix(),
            "regions_root": (package_root / "branch_inputs/regions/rtmw_l").as_posix(),
            "verify_root": (package_root / "verify").as_posix(),
        },
        "split_counts": requirements_summary["split_counts"],
        "skeleton_tensor_count": requirements_summary["total_samples"],
        "regions_tensor_count": requirements_summary["total_samples"],
        "checkpoint_files": {
            "skeleton": "checkpoints/skeleton/best.pt",
            "regions": "checkpoints/regions/best.pt",
        },
        "config_files": {
            "fusion": "configs/gated_feature_fusion_nslt1000_kaggle.yaml",
            "skeleton": "configs/skeleton_config_resolved.yaml",
            "regions": "configs/regions_config_resolved.yaml",
        },
        "estimated_logical_size_bytes": total_bytes,
        "estimated_logical_size_human": format_bytes(total_bytes),
        "estimated_additional_physical_size_bytes": additional_physical,
        "estimated_additional_physical_size_human": format_bytes(additional_physical),
        "selected_link_mode": link_mode,
        "disk_free_bytes": free_bytes,
        "disk_free_human": format_bytes(free_bytes),
        "sample_examples": size_estimate.get("sample_examples", []),
        "warnings": list(requirements_summary.get("warnings", [])),
        "errors": list(requirements_summary.get("errors", [])),
        "ready_to_execute": requirements_summary.get("status") == READY_STATUS,
    }


def remove_existing_package(package_root: Path, *, output_root: Path) -> None:
    """Delete an existing package root safely when --overwrite is used."""

    ensure_within_root(package_root, root=output_root)
    if package_root.exists():
        shutil.rmtree(package_root)


def materialize_branch_tensors(
    *,
    branch_name: str,
    items: list[Any],
    repo_root: Path,
    package_root: Path,
    link_mode: str,
    resume: bool,
    build_state_path: Path,
    state: dict[str, Any],
) -> dict[str, int]:
    """Materialize one branch's tensors into the package root."""

    counters = {"hardlink": 0, "copy": 0, "skipped": 0, "failed": 0}
    for index, item in enumerate(items, start=1):
        canonical_id = item.canonical_sample_id
        split = item.split
        state_key = f"{branch_name}:{split}:{canonical_id}"
        destination_root = package_root / "branch_inputs" / branch_name / "rtmw_l" / "tensors" / "nslt1000" / split
        destination_path = destination_root / f"{canonical_id}.npz"

        if branch_name == "skeleton":
            source_path = resolve_skeleton_source_tensor(item.skeleton_row, repo_root=repo_root)
        else:
            source_path = resolve_regions_source_tensor(item.regions_row, repo_root=repo_root)

        if resume and verify_existing_file(destination_path):
            counters["skipped"] += 1
            state["samples"][state_key] = {
                "status": "skipped_existing",
                "source": source_path.as_posix(),
                "destination": destination_path.as_posix(),
            }
            continue

        if destination_path.exists():
            destination_path.unlink()

        try:
            method = materialize_file(source_path, destination_path, link_mode=link_mode)
            counters[method] += 1
            state["samples"][state_key] = {
                "status": method,
                "source": source_path.as_posix(),
                "destination": destination_path.as_posix(),
            }
        except Exception as exc:
            counters["failed"] += 1
            state["samples"][state_key] = {
                "status": "failed",
                "error": str(exc),
                "source": source_path.as_posix(),
                "destination": destination_path.as_posix(),
            }
            save_build_state(build_state_path, state)
            raise PackagingError(
                f"Failed to materialize {branch_name} tensor split={split} sample_id={canonical_id}: {exc}"
            ) from exc

        if index % 200 == 0:
            save_build_state(build_state_path, state)

    save_build_state(build_state_path, state)
    return counters


def write_packaged_manifests(
    *,
    alignment: dict[str, Any],
    skeleton_manifests: Any,
    regions_manifests: Any,
    package_root: Path,
) -> dict[str, int]:
    """Write the canonical package manifests for both branches."""

    counts: dict[str, int] = {}
    for split in SAMPLE_SPLITS:
        items = alignment["splits"][split]["items"]
        skeleton_rows = [
            rewrite_skeleton_manifest_row(
                item.skeleton_row,
                canonical_sample_id_value=item.canonical_sample_id,
                split=split,
            )
            for item in items
        ]
        regions_rows = [
            rewrite_regions_manifest_row(
                item.regions_row,
                canonical_sample_id_value=item.canonical_sample_id,
                split=split,
            )
            for item in items
        ]
        write_manifest(
            package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests" / f"nslt1000_selected_31_{split}.csv",
            skeleton_manifests.fieldnames_by_split[split],
            skeleton_rows,
        )
        write_manifest(
            package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests" / f"nslt1000_{split}.csv",
            regions_manifests.fieldnames_by_split[split],
            regions_rows,
        )
        counts[split] = len(items)
    return counts


def run_packaged_verifier(package_root: Path) -> dict[str, Any]:
    """Run the packaged verify script and return its JSON summary."""

    verify_script = package_root / "verify" / "verify_package.py"
    verify_summary_path = package_root / "verify" / "verify_summary.json"
    command = [
        sys.executable,
        str(verify_script.resolve()),
        "--package-root",
        str(package_root.resolve()),
        "--output-json",
        str(verify_summary_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(package_root.resolve()),
    )
    if completed.returncode != 0:
        raise PackagingError(
            "Packaged verifier failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    with verify_summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_plan(plan: dict[str, Any], *, plan_path: Path | None) -> None:
    """Print the key dry-run summary."""

    print(f"package_root: {plan['package_root']}")
    print(f"selected_link_mode: {plan['selected_link_mode']}")
    print(f"estimated_logical_size: {plan['estimated_logical_size_human']}")
    print(f"estimated_additional_physical_size: {plan['estimated_additional_physical_size_human']}")
    print(f"disk_free: {plan['disk_free_human']}")
    print(f"ready_to_execute: {plan['ready_to_execute']}")
    if plan["warnings"]:
        print("warnings:")
        for item in plan["warnings"]:
            print(f"- {item}")
    if plan["errors"]:
        print("errors:")
        for item in plan["errors"]:
            print(f"- {item}")
    if plan_path is not None:
        print(f"plan_path: {plan_path.as_posix()}")


def main() -> int:
    """Run the packaging flow or dry-run plan."""

    args = build_parser().parse_args()
    repo_root = resolve_path(args.repo_root, repo_root=Path.cwd())
    output_root = resolve_path(args.output_root, repo_root=repo_root)
    package_name = str(args.package_name).strip()
    if not package_name:
        raise PackagingError("package_name must not be empty")

    skeleton_checkpoint = resolve_path(args.skeleton_checkpoint, repo_root=repo_root)
    regions_checkpoint = resolve_path(args.regions_checkpoint, repo_root=repo_root)
    skeleton_config = resolve_path(args.skeleton_config, repo_root=repo_root)
    regions_config = resolve_path(args.regions_config, repo_root=repo_root)
    skeleton_manifest_root = resolve_path(args.skeleton_manifest_root, repo_root=repo_root)
    regions_manifest_root = resolve_path(args.regions_manifest_root, repo_root=repo_root)
    local_fusion_config = resolve_path(LOCAL_FUSION_CONFIG_DEFAULT, repo_root=repo_root)
    kaggle_config_template = resolve_path(KAGGLE_FUSION_CONFIG_TEMPLATE, repo_root=repo_root)
    verifier_source = resolve_path(VERIFIER_SOURCE, repo_root=repo_root)
    write_plan_path = (
        resolve_path(args.write_plan, repo_root=repo_root)
        if args.write_plan is not None
        else None
    )
    package_root = output_root / package_name
    build_state_path = package_root / "build_state.json"

    if args.execute and args.dry_run:
        raise PackagingError("--dry-run and --execute are mutually exclusive")
    if not args.execute:
        args.dry_run = True

    requirements_summary = build_requirements_summary(
        repo_root=repo_root,
        skeleton_checkpoint=skeleton_checkpoint,
        regions_checkpoint=regions_checkpoint,
        skeleton_config=skeleton_config,
        regions_config=regions_config,
        skeleton_manifest_root=skeleton_manifest_root,
        regions_manifest_root=regions_manifest_root,
        fusion_config=local_fusion_config,
        check_all_tensor_paths=True,
    )

    skeleton_manifests = load_skeleton_manifest_set(skeleton_manifest_root)
    regions_manifests = load_regions_manifest_set(regions_manifest_root)
    alignment = build_alignment(skeleton_manifests, regions_manifests)
    size_estimate = estimate_source_sizes(alignment, repo_root=repo_root)
    plan = build_plan(
        repo_root=repo_root,
        package_root=package_root,
        package_name=package_name,
        link_mode=str(args.link_mode),
        requirements_summary=requirements_summary,
        size_estimate=size_estimate,
    )

    if write_plan_path is not None:
        write_json(write_plan_path, plan)

    if args.dry_run:
        print_plan(plan, plan_path=write_plan_path)
        return 0 if requirements_summary["status"] == READY_STATUS else 1

    if requirements_summary["status"] != READY_STATUS:
        raise PackagingError("Packaging prerequisites are not READY. Run the requirement checker first.")

    output_root.mkdir(parents=True, exist_ok=True)
    if package_root.exists():
        if args.resume:
            pass
        elif args.overwrite:
            remove_existing_package(package_root, output_root=output_root)
        else:
            raise PackagingError(
                "Package root already exists. Re-run with --resume or --overwrite."
            )

    package_root.mkdir(parents=True, exist_ok=True)
    build_state = load_build_state(build_state_path)
    if not build_state.get("created_at"):
        build_state["created_at"] = now_utc_iso()
    build_state.setdefault("samples", {})

    (package_root / "checkpoints" / "skeleton").mkdir(parents=True, exist_ok=True)
    (package_root / "checkpoints" / "regions").mkdir(parents=True, exist_ok=True)
    (package_root / "configs").mkdir(parents=True, exist_ok=True)
    (package_root / "verify").mkdir(parents=True, exist_ok=True)

    shutil.copy2(skeleton_checkpoint, package_root / "checkpoints" / "skeleton" / "best.pt")
    shutil.copy2(regions_checkpoint, package_root / "checkpoints" / "regions" / "best.pt")
    shutil.copy2(skeleton_config, package_root / "configs" / "skeleton_config_resolved.yaml")
    shutil.copy2(regions_config, package_root / "configs" / "regions_config_resolved.yaml")

    packaged_kaggle_config = create_packaged_kaggle_config(
        template_path=kaggle_config_template,
        package_name=package_name,
    )
    write_yaml(packaged_kaggle_config, package_root / "configs" / "gated_feature_fusion_nslt1000_kaggle.yaml")

    copied_reports = copy_report_files(
        regions_manifest_root.parent / "reports",
        package_root / "branch_inputs" / "regions" / "rtmw_l" / "reports",
    )

    all_items = [item for split in SAMPLE_SPLITS for item in alignment["splits"][split]["items"]]
    skeleton_result = materialize_branch_tensors(
        branch_name="skeleton",
        items=all_items,
        repo_root=repo_root,
        package_root=package_root,
        link_mode=str(args.link_mode),
        resume=bool(args.resume),
        build_state_path=build_state_path,
        state=build_state,
    )
    regions_result = materialize_branch_tensors(
        branch_name="regions",
        items=all_items,
        repo_root=repo_root,
        package_root=package_root,
        link_mode=str(args.link_mode),
        resume=bool(args.resume),
        build_state_path=build_state_path,
        state=build_state,
    )

    counts = write_packaged_manifests(
        alignment=alignment,
        skeleton_manifests=skeleton_manifests,
        regions_manifests=regions_manifests,
        package_root=package_root,
    )
    metadata = build_package_metadata(
        package_name=package_name,
        created_at=now_utc_iso(),
        package_version="1.0.0",
        link_mode=str(args.link_mode),
    )
    metadata["reports"] = [relative_package_path(Path(path), package_root=package_root) for path in map(Path, copied_reports)]
    metadata["build_summary"] = {
        "skeleton": skeleton_result,
        "regions": regions_result,
        "manifest_counts": counts,
    }
    write_json(package_root / "metadata.json", metadata)
    (package_root / "README.md").write_text(create_package_readme(package_name), encoding="utf-8")
    shutil.copy2(verifier_source, package_root / "verify" / "verify_package.py")

    verify_summary: dict[str, Any] | None = None
    if args.verify_after_build:
        verify_summary = run_packaged_verifier(package_root)

    print(f"package_root: {package_root.as_posix()}")
    print(f"link_mode: {args.link_mode}")
    print(f"skeleton_result: {json.dumps(skeleton_result, ensure_ascii=False)}")
    print(f"regions_result: {json.dumps(regions_result, ensure_ascii=False)}")
    print(f"copied_reports: {len(copied_reports)}")
    print(f"verify_after_build: {'pass' if verify_summary is not None else 'not_run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
