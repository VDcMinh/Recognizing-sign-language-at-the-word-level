from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_DIR_INCLUDES = ("configs", "scripts", "src", "slr")
REPO_FILE_INCLUDES = ("pyproject.toml", "requirements.txt", "sitecustomize.py", "README.md")
REQUIRED_REPO_DIRS = ("configs", "scripts", "src")
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}
CHECKPOINT_SUFFIXES = {".pth", ".pt"}
CONFIG_SUFFIXES = {".py", ".yaml", ".yml"}


class BundleError(RuntimeError):
    """Raised when bundle validation or creation fails."""


@dataclass(frozen=True)
class ValidationResult:
    """Resolved source inputs for the Kaggle bundle."""

    project_root: Path
    subset: str
    standardized_root: Path
    frames_root: Path
    frame_files: tuple[Path, ...]
    manifest_files: tuple[Path, ...]
    checkpoint_dir: Path
    checkpoint_model_files: tuple[Path, ...]
    checkpoint_config_files: tuple[Path, ...]
    repo_dirs: tuple[Path, ...]
    repo_files: tuple[Path, ...]
    repo_bundle_files: tuple[tuple[Path, Path], ...]
    standardized_bundle_files: tuple[tuple[Path, Path], ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the Kaggle bundle utility."""

    parser = argparse.ArgumentParser(
        description="Prepare a Kaggle upload bundle for WLASL RTMW-l pose extraction."
    )
    parser.add_argument("--subset", default="nslt100", help="Standardized subset to package.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle_bundle"),
        help="Bundle output directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing configs/, scripts/, src/, data/, and checkpoints/.",
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=Path("data/datasets/WLASL/standardized"),
        help="Root directory for standardized WLASL data.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pose/rtmw_l"),
        help="Directory containing RTMW-l config/checkpoint files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing bundle in the output directory.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Copy repo/standardized trees instead of producing zip archives.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned actions without writing output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed file-level logging.",
    )
    return parser.parse_args()


def _resolve_under(base: Path, value: Path) -> Path:
    """Resolve an absolute or project-relative path."""

    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _is_within(path: Path, root: Path) -> bool:
    """Return True when path is equal to or nested under root."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_within_project(project_root: Path, path: Path, label: str) -> None:
    """Fail when a source path is not contained inside the project root."""

    if not _is_within(path, project_root):
        raise BundleError(f"{label} must stay inside project root: {path}")


def _path_text(path: Path) -> str:
    """Render a path with stable POSIX separators."""

    return path.as_posix()


def _relative_text(path: Path, root: Path) -> str:
    """Render a path relative to root using POSIX separators."""

    return path.relative_to(root).as_posix()


def _format_bytes(size: int) -> str:
    """Pretty-print a byte size."""

    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _iter_media_files(root: Path) -> list[Path]:
    """Collect image frame files under root."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    )


def _collect_manifest_files(subset: str, manifests_root: Path) -> tuple[Path, ...]:
    """Resolve required and optional standardized manifests."""

    required = (
        manifests_root / f"{subset}_train.csv",
        manifests_root / f"{subset}_val.csv",
        manifests_root / f"{subset}_test.csv",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(_path_text(path) for path in missing)
        raise BundleError(f"Missing required standardized manifest file(s): {missing_text}")

    optional_all = manifests_root / f"{subset}_all.csv"
    files = list(required)
    if optional_all.exists():
        files.append(optional_all)
    return tuple(files)


def _is_generated_repo_dir(name: str) -> bool:
    """Return True for generated output directories that should stay out of repo.zip."""

    return name.startswith("_") and name.endswith("_output")


def _should_skip_repo_dir(relative_dir: Path) -> bool:
    """Return True when a repo directory should be excluded from the bundle."""

    blocked_parts = {
        ".git",
        "data",
        "checkpoints",
        "experiments",
        "reports",
        "notebooks",
        "kaggle_bundle",
    }
    for part in relative_dir.parts:
        if part in blocked_parts:
            return True
        if part == "__pycache__":
            return True
        if part == ".venv" or part.startswith(".venv-"):
            return True
        if _is_generated_repo_dir(part):
            return True
    return False


def _should_skip_repo_file(relative_file: Path) -> bool:
    """Return True when a repo file should be excluded from the bundle."""

    if _should_skip_repo_dir(relative_file.parent):
        return True
    name = relative_file.name
    return name.endswith(".pyc") or name.endswith(".zip")


def _collect_repo_bundle_files(
    project_root: Path,
    repo_dirs: Iterable[Path],
    repo_files: Iterable[Path],
) -> tuple[tuple[Path, Path], ...]:
    """Collect repo files and archive names for bundle creation."""

    bundle_files: list[tuple[Path, Path]] = []

    for file_path in repo_files:
        relative_path = file_path.relative_to(project_root)
        if _should_skip_repo_file(relative_path):
            continue
        bundle_files.append((file_path, relative_path))

    for directory in repo_dirs:
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(project_root)
            if _should_skip_repo_file(relative_path):
                continue
            bundle_files.append((file_path, relative_path))

    deduped: dict[str, tuple[Path, Path]] = {}
    for source_path, archive_path in bundle_files:
        deduped[archive_path.as_posix()] = (source_path, archive_path)
    return tuple(sorted(deduped.values(), key=lambda item: item[1].as_posix()))


def _collect_standardized_bundle_files(
    project_root: Path,
    frame_files: Iterable[Path],
    manifest_files: Iterable[Path],
) -> tuple[tuple[Path, Path], ...]:
    """Collect standardized inputs and archive names for bundle creation."""

    bundle_files = []
    for source_path in list(frame_files) + list(manifest_files):
        archive_path = source_path.relative_to(project_root)
        bundle_files.append((source_path, archive_path))
    return tuple(sorted(bundle_files, key=lambda item: item[1].as_posix()))


def validate_inputs(
    project_root: Path,
    subset: str,
    standardized_root: Path,
    checkpoint_dir: Path,
) -> ValidationResult:
    """Validate required inputs and resolve bundle file lists."""

    if not project_root.exists():
        raise BundleError(f"Project root does not exist: {project_root}")

    frames_root = standardized_root / "frames" / subset
    manifests_root = standardized_root / "manifests"

    for label, path in (
        ("standardized root", standardized_root),
        ("frames root", frames_root),
        ("manifests root", manifests_root),
        ("checkpoint dir", checkpoint_dir),
    ):
        _require_within_project(project_root, path, label)
        if not path.exists():
            raise BundleError(f"Required path does not exist for {label}: {path}")

    repo_dirs = tuple(
        (project_root / name).resolve()
        for name in REPO_DIR_INCLUDES
        if (project_root / name).exists()
    )
    repo_files = tuple(
        (project_root / name).resolve()
        for name in REPO_FILE_INCLUDES
        if (project_root / name).exists()
    )
    missing_repo_dirs = [name for name in REQUIRED_REPO_DIRS if not (project_root / name).exists()]
    if missing_repo_dirs:
        raise BundleError(
            "Repo is missing required directory/directories: "
            + ", ".join(missing_repo_dirs)
        )

    frame_files = tuple(_iter_media_files(frames_root))
    if not frame_files:
        raise BundleError(
            f"No frame image files found under standardized frames root: {frames_root}"
        )

    manifest_files = _collect_manifest_files(subset, manifests_root)

    checkpoint_model_files = tuple(
        sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        )
    )
    checkpoint_config_files = tuple(
        sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES
        )
    )
    if not checkpoint_model_files:
        raise BundleError(
            f"No RTMW-l checkpoint file (.pth/.pt) found in checkpoint dir: {checkpoint_dir}"
        )
    if not checkpoint_config_files:
        raise BundleError(
            f"No RTMW-l config file (.py/.yaml/.yml) found in checkpoint dir: {checkpoint_dir}"
        )

    repo_bundle_files = _collect_repo_bundle_files(project_root, repo_dirs, repo_files)
    standardized_bundle_files = _collect_standardized_bundle_files(
        project_root, frame_files, manifest_files
    )

    return ValidationResult(
        project_root=project_root,
        subset=subset,
        standardized_root=standardized_root,
        frames_root=frames_root,
        frame_files=frame_files,
        manifest_files=manifest_files,
        checkpoint_dir=checkpoint_dir,
        checkpoint_model_files=checkpoint_model_files,
        checkpoint_config_files=checkpoint_config_files,
        repo_dirs=repo_dirs,
        repo_files=repo_files,
        repo_bundle_files=repo_bundle_files,
        standardized_bundle_files=standardized_bundle_files,
    )


def validate_output_dir(
    output_dir: Path,
    project_root: Path,
    standardized_root: Path,
    checkpoint_dir: Path,
) -> None:
    """Guard against destructive or source-overlapping output locations."""

    raw_root = (project_root / "data" / "datasets" / "WLASL" / "raw").resolve()

    if output_dir == project_root:
        raise BundleError("Output directory must not be the project root.")
    if _is_within(project_root, output_dir):
        raise BundleError(
            f"Output directory must not contain the project root: {output_dir}"
        )
    for label, blocked_root in (
        ("raw dataset root", raw_root),
        ("standardized root", standardized_root),
        ("checkpoint root", checkpoint_dir.parent.parent.resolve()),
        ("checkpoint dir", checkpoint_dir),
    ):
        if _is_within(output_dir, blocked_root):
            raise BundleError(f"Output directory must not be inside {label}: {output_dir}")


def clear_output_dir(output_dir: Path) -> None:
    """Delete only the contents inside the output directory."""

    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def ensure_parent(path: Path) -> None:
    """Create parent directories for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_zip(bundle_path: Path, bundle_files: Iterable[tuple[Path, Path]]) -> int:
    """Create a zip archive and return its final size."""

    ensure_parent(bundle_path)
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path, archive_path in bundle_files:
            archive.write(source_path, archive_path.as_posix())
    return bundle_path.stat().st_size


def copy_bundle_tree(destination_root: Path, bundle_files: Iterable[tuple[Path, Path]]) -> int:
    """Copy a set of files into destination_root while preserving relative paths."""

    copied_size = 0
    for source_path, relative_path in bundle_files:
        destination_path = destination_root / relative_path
        ensure_parent(destination_path)
        shutil.copy2(source_path, destination_path)
        copied_size += destination_path.stat().st_size
    return copied_size


def copy_checkpoint_files(
    bundle_root: Path,
    checkpoint_dir: Path,
    checkpoint_files: Iterable[Path],
) -> list[Path]:
    """Copy selected RTMW-l checkpoint/config files into the bundle."""

    destination_dir = bundle_root / "checkpoints" / "pose" / "rtmw_l"
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []
    for source_path in checkpoint_files:
        if source_path.parent != checkpoint_dir:
            continue
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_files.append(destination_path)
    return copied_files


def directory_size(root: Path) -> int:
    """Return the total size of all files below root."""

    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_manifest(
    args: argparse.Namespace,
    validation: ValidationResult,
    output_dir: Path,
    repo_artifact_name: str,
    standardized_artifact_name: str,
    copied_checkpoint_files: list[Path],
    total_size_bytes: int,
) -> dict[str, object]:
    """Build bundle metadata for MANIFEST.json."""

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subset": validation.subset,
        "source_project_root": _path_text(validation.project_root),
        "output_dir": _path_text(output_dir),
        "repo_zip": repo_artifact_name if not args.no_zip else None,
        "standardized_zip": standardized_artifact_name if not args.no_zip else None,
        "repo_tree": "repo" if args.no_zip else None,
        "standardized_tree": f"standardized_{validation.subset}" if args.no_zip else None,
        "checkpoint_files": [
            _relative_text(path, output_dir) for path in copied_checkpoint_files
        ],
        "standardized_manifest_files": [
            _relative_text(path, validation.project_root) for path in validation.manifest_files
        ],
        "frame_root": _relative_text(validation.frames_root, validation.project_root),
        "number_of_frame_files": len(validation.frame_files),
        "total_size_bytes": total_size_bytes,
        "notes": [
            "This bundle contains copied files only. No source files were moved, renamed, or modified.",
            "No raw WLASL videos are included.",
            "No generated pose outputs are included.",
            "The standardized bundle includes frames and manifests for the requested subset only.",
        ],
    }


def write_manifest(path: Path, data: dict[str, object]) -> None:
    """Write JSON manifest with stable formatting."""

    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_readme_text(
    subset: str,
    use_kaggle_config: bool,
    no_zip: bool,
) -> str:
    """Create README_KAGGLE_BUNDLE.md contents."""

    config_path = (
        "configs/preprocessing/pose_rtmw_l_kaggle.yaml"
        if use_kaggle_config
        else "configs/preprocessing/pose_rtmw_l.yaml"
    )
    repo_artifact = "repo/" if no_zip else "repo.zip"
    standardized_artifact = f"standardized_{subset}/" if no_zip else f"standardized_{subset}.zip"
    repo_unpack_step = (
        "5. Copy repo/ into:\n   /kaggle/working/Recognizing-sign-language-at-the-word-level"
        if no_zip
        else "5. Unzip repo.zip into:\n   /kaggle/working/Recognizing-sign-language-at-the-word-level"
    )
    standardized_unpack_step = (
        f"6. Copy standardized_{subset}/ into that project root."
        if no_zip
        else f"6. Unzip standardized_{subset}.zip into that project root."
    )

    return "\n".join(
        [
            f"# Kaggle Bundle for WLASL {subset} RTMW-l Pose Extraction",
            "",
            "This bundle contains copied data only. Original local data was not moved or modified.",
            "",
            "## Contents",
            "",
            f"- {repo_artifact}",
            f"- {standardized_artifact}",
            "- checkpoints/pose/rtmw_l/",
            "- MANIFEST.json",
            "",
            "## Kaggle usage",
            "",
            "1. Create a private Kaggle Dataset.",
            "2. Upload all files/folders in this bundle.",
            "3. Create a Kaggle Notebook with GPU.",
            "4. Add this dataset to the notebook.",
            repo_unpack_step,
            standardized_unpack_step,
            "7. Copy checkpoints/ into the project root if needed.",
            "8. Create/use a Kaggle config pointing output to:",
            "   /kaggle/working/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/pose/rtmw_l",
            "9. Run:",
            f"   python scripts/02_extract_pose_rtmw.py --config {config_path} --startup-only",
            f"   python scripts/02_extract_pose_rtmw.py --config {config_path} --smoke-one-frame",
            f"   python scripts/02_extract_pose_rtmw.py --config {config_path} --limit 1",
            "",
            "## Notes",
            "",
            "- This bundle does not include raw videos.",
            "- This bundle does not include generated pose outputs.",
            f"- This bundle is intended for {subset} only.",
            "- Keep Kaggle Dataset private unless dataset licensing allows sharing.",
            "",
        ]
    )


def write_readme(
    path: Path,
    subset: str,
    use_kaggle_config: bool,
    no_zip: bool,
) -> None:
    """Write the bundle README."""

    ensure_parent(path)
    path.write_text(
        build_readme_text(subset, use_kaggle_config, no_zip),
        encoding="utf-8",
    )


def print_validation_summary(
    args: argparse.Namespace,
    validation: ValidationResult,
    output_dir: Path,
) -> None:
    """Print high-level validation and plan details."""

    print(f"Project root: {_path_text(validation.project_root)}")
    print(f"Subset: {validation.subset}")
    print(f"Output dir: {_path_text(output_dir)}")
    print(f"Source standardized frames dir: {_path_text(validation.frames_root)}")
    print("Source manifests:")
    for manifest_path in validation.manifest_files:
        print(f"  - {_path_text(manifest_path)}")
    print(f"Checkpoint dir: {_path_text(validation.checkpoint_dir)}")
    print(f"Number of frame files: {len(validation.frame_files)}")
    print("Repo directories included:")
    for directory in validation.repo_dirs:
        print(f"  - {_relative_text(directory, validation.project_root)}/")
    if validation.repo_files:
        print("Repo files included:")
        for file_path in validation.repo_files:
            print(f"  - {_relative_text(file_path, validation.project_root)}")
    print(f"Repo bundle file count: {len(validation.repo_bundle_files)}")
    print(f"Standardized bundle file count: {len(validation.standardized_bundle_files)}")
    print(
        "Bundle mode: "
        + ("copy trees only (--no-zip)" if args.no_zip else "zip repo and standardized inputs")
    )


def print_verbose_file_list(title: str, bundle_files: Iterable[tuple[Path, Path]]) -> None:
    """Print planned bundle files when verbose logging is enabled."""

    print(title)
    for _, archive_path in bundle_files:
        print(f"  - {archive_path.as_posix()}")


def verify_zip_excludes(zip_path: Path, blocked_prefixes: tuple[str, ...]) -> None:
    """Check zip contents against blocked top-level prefixes."""

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
    for blocked_prefix in blocked_prefixes:
        if any(name == blocked_prefix or name.startswith(f"{blocked_prefix}/") for name in names):
            raise BundleError(f"Unexpected blocked path found in {zip_path.name}: {blocked_prefix}")


def verify_standardized_zip(zip_path: Path, subset: str) -> None:
    """Check standardized zip only contains the expected roots."""

    allowed_prefixes = (
        f"data/datasets/WLASL/standardized/frames/{subset}/",
        "data/datasets/WLASL/standardized/manifests/",
    )
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if not name.startswith(allowed_prefixes):
                raise BundleError(
                    f"Unexpected file in standardized bundle {zip_path.name}: {name}"
                )
            if name.lower().endswith(".npz"):
                raise BundleError(
                    f"Pose output file unexpectedly found in standardized bundle: {name}"
                )


def main() -> int:
    """Prepare the Kaggle bundle."""

    args = parse_args()
    project_root = _resolve_under(Path.cwd(), args.project_root)
    standardized_root = _resolve_under(project_root, args.standardized_root)
    checkpoint_dir = _resolve_under(project_root, args.checkpoint_dir)
    output_dir = _resolve_under(project_root, args.output_dir)

    try:
        validate_output_dir(output_dir, project_root, standardized_root, checkpoint_dir)
        validation = validate_inputs(
            project_root=project_root,
            subset=args.subset,
            standardized_root=standardized_root,
            checkpoint_dir=checkpoint_dir,
        )
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        print(
            "ERROR: Output directory already exists and is not empty. "
            f"Use --overwrite or choose a different output dir: {output_dir}",
            file=sys.stderr,
        )
        return 1

    print_validation_summary(args, validation, output_dir)
    if args.verbose:
        print_verbose_file_list("Planned repo bundle files:", validation.repo_bundle_files)
        print_verbose_file_list(
            "Planned standardized bundle files:", validation.standardized_bundle_files
        )
        checkpoint_plan = list(validation.checkpoint_model_files) + list(
            validation.checkpoint_config_files
        )
        print("Checkpoint/config files to copy:")
        for checkpoint_path in checkpoint_plan:
            print(f"  - {_path_text(checkpoint_path)}")

    repo_artifact_name = "repo.zip" if not args.no_zip else "repo"
    standardized_artifact_name = (
        f"standardized_{validation.subset}.zip"
        if not args.no_zip
        else f"standardized_{validation.subset}"
    )

    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_output_dir(output_dir)

    repo_artifact_path = output_dir / repo_artifact_name
    standardized_artifact_path = output_dir / standardized_artifact_name

    try:
        if args.no_zip:
            repo_size = copy_bundle_tree(repo_artifact_path, validation.repo_bundle_files)
            standardized_size = copy_bundle_tree(
                standardized_artifact_path, validation.standardized_bundle_files
            )
        else:
            repo_size = write_zip(repo_artifact_path, validation.repo_bundle_files)
            standardized_size = write_zip(
                standardized_artifact_path, validation.standardized_bundle_files
            )
            verify_zip_excludes(repo_artifact_path, ("data", "checkpoints", ".git", ".venv"))
            verify_standardized_zip(standardized_artifact_path, validation.subset)

        copied_checkpoint_files = copy_checkpoint_files(
            output_dir,
            validation.checkpoint_dir,
            list(validation.checkpoint_model_files) + list(validation.checkpoint_config_files),
        )

        readme_path = output_dir / "README_KAGGLE_BUNDLE.md"
        write_readme(
            readme_path,
            subset=validation.subset,
            use_kaggle_config=(
                project_root / "configs" / "preprocessing" / "pose_rtmw_l_kaggle.yaml"
            ).exists(),
            no_zip=args.no_zip,
        )

        manifest_path = output_dir / "MANIFEST.json"
        manifest = build_manifest(
            args=args,
            validation=validation,
            output_dir=output_dir,
            repo_artifact_name=repo_artifact_name,
            standardized_artifact_name=standardized_artifact_name,
            copied_checkpoint_files=copied_checkpoint_files,
            total_size_bytes=0,
        )
        total_bundle_size = directory_size(output_dir)
        manifest["total_size_bytes"] = total_bundle_size
        write_manifest(manifest_path, manifest)

        # Re-read JSON once so the script fails clearly if manifest output is malformed.
        json.loads(manifest_path.read_text(encoding="utf-8"))
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Bundle created:")
    print(f"{output_dir.name}/")
    print(f"|-- {repo_artifact_name}")
    print(f"|-- {standardized_artifact_name}")
    print("|-- checkpoints/pose/rtmw_l/")
    print("|-- MANIFEST.json")
    print("`-- README_KAGGLE_BUNDLE.md")
    print("")
    print(f"{repo_artifact_name} size: {_format_bytes(repo_size)}")
    print(f"{standardized_artifact_name} size: {_format_bytes(standardized_size)}")
    print("Copied checkpoint/config files:")
    for checkpoint_path in copied_checkpoint_files:
        print(f"  - {_relative_text(checkpoint_path, output_dir)}")
    print(f"Total bundle size: {_format_bytes(total_bundle_size)}")
    print("")
    print("Next:")
    print(f"1. Upload contents of {output_dir.name}/ to a private Kaggle Dataset.")
    if args.no_zip:
        print(
            "2. In Kaggle Notebook, copy repo/ into "
            "/kaggle/working/Recognizing-sign-language-at-the-word-level."
        )
        print("3. Copy standardized tree into the project root.")
    else:
        print(
            "2. In Kaggle Notebook, unzip repo.zip into "
            "/kaggle/working/Recognizing-sign-language-at-the-word-level."
        )
        print("3. Unzip standardized bundle into the project root.")
    print("4. Copy/use checkpoints under checkpoints/pose/rtmw_l.")
    print("5. Run pose extraction startup/smoke tests before full extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
