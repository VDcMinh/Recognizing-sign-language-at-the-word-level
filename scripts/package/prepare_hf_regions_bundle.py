from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from slr.branches.regions.dataset import RegionClipDataset
from slr.branches.regions.region_schema import (
    BBOX_SOURCE_NAMES,
    DEFAULT_TENSOR_SHAPE,
    REGION_NAMES,
    TENSOR_FORMAT,
)
from slr.data.manifests import REGION_NPZ_FIELDS
from slr.utils.io import ensure_dir, read_json, read_yaml, write_csv, write_json, write_text


DEFAULT_SUBSET = "nslt100"
DEFAULT_INPUT_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l")
DEFAULT_OUTPUT_DIR = Path("hf_regions_nslt100_bundle")
DEFAULT_CONFIG_MAP = {
    "nslt100": Path("configs/preprocessing/regions/region_crops_nslt100.yaml"),
    "nslt300": Path("configs/preprocessing/regions/region_crops_nslt300.yaml"),
}
SPLITS = ("train", "val", "test")


class BundleError(RuntimeError):
    """Raised when the regions HF bundle cannot be prepared safely."""


@dataclass(frozen=True)
class SplitSummary:
    """Validation summary for one sanitized split manifest."""

    split: str
    manifest_path: Path
    rows: int
    ok_count: int
    file_count: int
    sample_id_count: int
    class_id_min: int
    class_id_max: int
    class_id_nunique: int
    tensor_dir: Path


@dataclass(frozen=True)
class ValidationResult:
    """Resolved input roots and validated source files."""

    project_root: Path
    input_root: Path
    output_dir: Path
    subset: str
    include_previews: bool
    include_reports: bool
    create_zips: bool
    upload: bool
    repo_id: str | None
    private: bool
    config_path: Path | None
    source_metadata_path: Path
    source_metadata: dict[str, Any]
    manifests_root: Path
    tensors_root: Path
    reports_root: Path
    previews_root: Path
    split_summaries: tuple[SplitSummary, ...]
    total_samples: int
    num_classes: int
    source_total_size_bytes: int
    report_files: tuple[Path, ...]
    preview_files: tuple[Path, ...]
    include_logs: bool = False


def parse_args() -> argparse.Namespace:
    """Parse CLI options for HF regions bundle creation."""

    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face upload bundle for train-ready WLASL regions data."
    )
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET, help="Subset to package, e.g. nslt100.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing train-ready regions data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the bundle.",
    )
    parser.add_argument(
        "--include-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include preview JPGs and preview paths in sanitized manifests.",
    )
    parser.add_argument(
        "--include-reports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include preprocessing reports in the bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing output directory contents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing bundle files.",
    )
    parser.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create zip archives alongside the folder bundle.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the finished bundle to Hugging Face using huggingface_hub.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id, e.g. user/wlasl-nslt100-regions-rtmw-l.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private Hugging Face dataset repo.",
    )
    return parser.parse_args()


def _resolve_under(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _iter_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise BundleError(f"Missing required {label}: {path}")


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "class_id", "split", "tensor_path", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BundleError(f"Manifest {path.name} is missing required columns: {missing}")
    return frame


def _validate_split(
    *,
    input_root: Path,
    subset: str,
    split: str,
) -> SplitSummary:
    manifest_path = input_root / "manifests" / f"{subset}_{split}.csv"
    tensor_dir = input_root / "tensors" / subset / split
    _ensure_exists(manifest_path, f"{split} manifest")
    _ensure_exists(tensor_dir, f"{split} tensor directory")

    frame = _load_manifest(manifest_path)
    ok_mask = frame["status"].fillna("").astype(str).str.strip().str.lower().eq("ok")
    if not bool(ok_mask.all()):
        raise BundleError(f"Manifest {manifest_path.name} contains non-ok rows.")

    labels = pd.to_numeric(frame["class_id"], errors="coerce")
    if labels.isna().any():
        raise BundleError(f"Manifest {manifest_path.name} contains non-numeric class_id values.")
    labels = labels.astype(int)

    npz_files = tuple(sorted(tensor_dir.glob("*.npz")))
    if len(npz_files) != len(frame):
        raise BundleError(
            f"Tensor count mismatch for {split}: manifest rows={len(frame)} npz files={len(npz_files)}."
        )

    return SplitSummary(
        split=split,
        manifest_path=manifest_path,
        rows=int(len(frame)),
        ok_count=int(ok_mask.sum()),
        file_count=int(len(npz_files)),
        sample_id_count=int(frame["sample_id"].astype(str).nunique()),
        class_id_min=int(labels.min()),
        class_id_max=int(labels.max()),
        class_id_nunique=int(labels.nunique()),
        tensor_dir=tensor_dir,
    )


def _sum_file_sizes(paths: tuple[Path, ...]) -> int:
    return int(sum(path.stat().st_size for path in paths if path.exists()))


def validate_inputs(args: argparse.Namespace) -> ValidationResult:
    project_root = Path(__file__).resolve().parents[1]
    input_root = _resolve_under(project_root, args.input_root)
    output_dir = _resolve_under(project_root, args.output_dir)
    subset = str(args.subset).strip()

    _ensure_exists(input_root, "input root")
    manifests_root = input_root / "manifests"
    tensors_root = input_root / "tensors"
    reports_root = input_root / "reports"
    previews_root = input_root / "previews"
    source_metadata_path = input_root / "metadata.json"
    _ensure_exists(manifests_root, "manifests root")
    _ensure_exists(tensors_root / subset, f"tensors root for {subset}")
    _ensure_exists(source_metadata_path, "source metadata.json")

    split_summaries = tuple(_validate_split(input_root=input_root, subset=subset, split=split) for split in SPLITS)
    total_samples = int(sum(summary.rows for summary in split_summaries))
    all_classes = sorted({class_id for summary in split_summaries for class_id in range(summary.class_id_min, summary.class_id_max + 1)})
    class_ids = []
    for summary in split_summaries:
        frame = _load_manifest(summary.manifest_path)
        class_ids.extend(pd.to_numeric(frame["class_id"], errors="coerce").astype(int).tolist())
    unique_class_ids = sorted(set(class_ids))
    if not unique_class_ids:
        raise BundleError(f"No class ids found for subset {subset}.")
    contiguous = unique_class_ids == list(range(min(unique_class_ids), max(unique_class_ids) + 1))
    if not contiguous:
        raise BundleError(f"class_id values for {subset} are not contiguous: {unique_class_ids[:10]} ...")
    num_classes = int(len(unique_class_ids))

    report_files = ()
    if args.include_reports:
        report_files = tuple(sorted(path for path in reports_root.glob(f"{subset}_*") if path.is_file()))
        if not report_files:
            raise BundleError(f"No subset-specific report files found for {subset} under {reports_root}.")

    preview_files = ()
    if args.include_previews:
        preview_root = previews_root / subset
        _ensure_exists(preview_root, f"previews root for {subset}")
        preview_files = _iter_files(preview_root)
        if not preview_files:
            raise BundleError(f"No preview files found under {preview_root}.")

    source_metadata = read_json(source_metadata_path)
    source_total_size_bytes = 0
    source_total_size_bytes += _sum_file_sizes(_iter_files(tensors_root / subset))
    source_total_size_bytes += _sum_file_sizes(tuple(summary.manifest_path for summary in split_summaries))
    source_total_size_bytes += _sum_file_sizes(report_files)
    source_total_size_bytes += _sum_file_sizes(preview_files)
    source_total_size_bytes += int(source_metadata_path.stat().st_size)

    config_path = DEFAULT_CONFIG_MAP.get(subset)
    if config_path is not None:
        config_path = _resolve_under(project_root, config_path)
        if not config_path.exists():
            config_path = None

    return ValidationResult(
        project_root=project_root,
        input_root=input_root,
        output_dir=output_dir,
        subset=subset,
        include_previews=bool(args.include_previews),
        include_reports=bool(args.include_reports),
        create_zips=bool(args.zip),
        upload=bool(args.upload),
        repo_id=args.repo_id,
        private=bool(args.private),
        config_path=config_path,
        source_metadata_path=source_metadata_path,
        source_metadata=source_metadata,
        manifests_root=manifests_root,
        tensors_root=tensors_root,
        reports_root=reports_root,
        previews_root=previews_root,
        split_summaries=split_summaries,
        total_samples=total_samples,
        num_classes=num_classes,
        source_total_size_bytes=source_total_size_bytes,
        report_files=report_files,
        preview_files=preview_files,
    )


def _copy_tree(src: Path, dst: Path) -> int:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return len(_iter_files(dst))


def _copy_files(files: tuple[Path, ...], destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for source_path in files:
        shutil.copy2(source_path, destination_dir / source_path.name)


def _sanitize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _sanitize_manifest(
    *,
    source_path: Path,
    output_path: Path,
    subset: str,
    split: str,
    include_previews: bool,
) -> dict[str, Any]:
    frame = _load_manifest(source_path).copy()
    source_tensor_paths = frame["tensor_path"].fillna("").astype(str).str.strip()
    tensor_basenames = source_tensor_paths.map(lambda value: Path(value).name)
    if bool((tensor_basenames == "").any()):
        raise BundleError(f"Manifest {source_path.name} contains empty tensor_path values.")
    frame["tensor_path"] = tensor_basenames.map(lambda name: f"tensors/{subset}/{split}/{name}")
    if "preview_path" in frame.columns:
        if include_previews:
            source_preview_paths = frame["preview_path"].fillna("").astype(str).str.strip()
            frame["preview_path"] = source_preview_paths.map(
                lambda value: f"previews/{subset}/{split}/{Path(value).name}" if value else ""
            )
        else:
            frame["preview_path"] = ""
    if "crop_root" in frame.columns:
        frame["crop_root"] = frame["crop_root"].map(_sanitize_optional_text)
        frame["crop_root"] = ""

    write_csv(frame, output_path)
    return {
        "rows": int(len(frame)),
        "ok_count": int(frame["status"].fillna("").astype(str).str.strip().str.lower().eq("ok").sum()),
        "path": output_path,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip(zip_path: Path, root: Path, include_root: Path) -> int:
    files = _iter_files(include_root)
    if not files:
        raise BundleError(f"Cannot create empty zip archive: {zip_path.name}")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in files:
            archive.write(source_path, source_path.relative_to(root))
    return int(zip_path.stat().st_size)


def _build_metadata(validation: ValidationResult, bundle_root: Path) -> dict[str, Any]:
    config_data = read_yaml(validation.config_path) if validation.config_path is not None else {}
    input_cfg = config_data.get("input", {})
    subset = validation.subset
    metadata = {
        "dataset": "WLASL",
        "branch": "regions",
        "subset": subset,
        "pose_backend": str(input_cfg.get("pose_backend", validation.source_metadata.get("pose_backend", "rtmw_l"))),
        "pose_layout": str(input_cfg.get("pose_layout", validation.source_metadata.get("pose_layout", "wholebody_133"))),
        "tensor_format": TENSOR_FORMAT,
        "expected_shape": list(DEFAULT_TENSOR_SHAPE),
        "regions": list(REGION_NAMES),
        "num_classes": int(validation.num_classes),
        "total_samples": int(validation.total_samples),
        "splits": {summary.split: int(summary.rows) for summary in validation.split_summaries},
        "bundle_root": bundle_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derived_data": True,
        "includes_raw_videos": False,
        "includes_standardized_frames": False,
        "includes_previews": bool(validation.include_previews),
        "includes_reports": bool(validation.include_reports),
        "bbox_source_codes": {str(key): value for key, value in BBOX_SOURCE_NAMES.items()},
    }
    return metadata


def _build_dataset_card(validation: ValidationResult, bundle_root: Path) -> str:
    subset = validation.subset
    return "\n".join(
        [
            f"# WLASL {subset.upper()} Regions Local Image Dataset",
            "",
            "## Description",
            "This derived dataset contains local image tensors for word-level sign language recognition,",
            "including left hand, right hand, and face regions.",
            "",
            "## Source",
            "Built from WLASL standardized frames and RTMW-l wholebody_133 pose outputs.",
            "",
            "## Subset",
            subset,
            "",
            "## Tensor format",
            "data: (R, C, T, H, W) = (3, 3, 64, 112, 112)",
            "valid_mask: (R, T)",
            "bbox_source: (R, T)",
            "",
            "## Region order",
            "0: left_hand",
            "1: right_hand",
            "2: face",
            "",
            "## bbox_source encoding",
            "0: black_crop_failed",
            "1: current_keypoints",
            "2: previous_bbox_fallback",
            "",
            "## Files",
            f"- tensors/{subset}/...",
            "- manifests/...",
            "- metadata.json",
            "- reports/...",
            "",
            "## Usage",
            "Example command for training:",
            "",
            "```bash",
            "python scripts/train/train_regions.py \\",
            "  --config configs/train/regions/nslt100/face_hands/region_resnet18_gru_ce.yaml \\",
            f"  --data-root <bundle_root> \\",
            f"  --train-manifest <bundle_root>/manifests/{subset}_train.csv \\",
            f"  --val-manifest <bundle_root>/manifests/{subset}_val.csv \\",
            f"  --test-manifest <bundle_root>/manifests/{subset}_test.csv \\",
            "  --output-dir <output_dir> \\",
            f"  --run-name regions_cnn_gru_{subset}_ce_v1",
            "```",
            "",
            "## Notes",
            "- This bundle does not include original WLASL raw videos.",
            "- This bundle includes derived local-region tensors only.",
            "- Use according to WLASL licensing and usage constraints.",
            "- This is not an official WLASL release; it is a derived preprocessed tensor bundle.",
            "",
            f"Bundle root name: `{bundle_root.name}`",
        ]
    )


def _build_readme(validation: ValidationResult, bundle_root: Path) -> str:
    subset = validation.subset
    return "\n".join(
        [
            f"# WLASL {subset.upper()} Regions Local Image Dataset",
            "",
            "This bundle contains derived local-region tensors for training the `regions` branch",
            "of the Recognizing-sign-language-at-the-word-level project.",
            "",
            "## Description",
            "Each sample stores left hand, right hand, and face local image clips derived from",
            "standardized frames plus RTMW-l wholebody_133 pose keypoints.",
            "",
            "## Tensor format",
            "data: (R, C, T, H, W) = (3, 3, 64, 112, 112)",
            "valid_mask: (R, T)",
            "bbox_source: (R, T)",
            "",
            "## Region order",
            "0: left_hand",
            "1: right_hand",
            "2: face",
            "",
            "## Files",
            f"- tensors/{subset}/train/*.npz",
            f"- tensors/{subset}/val/*.npz",
            f"- tensors/{subset}/test/*.npz",
            f"- manifests/{subset}_train.csv",
            f"- manifests/{subset}_val.csv",
            f"- manifests/{subset}_test.csv",
            "- metadata.json",
            "- MANIFEST.json",
            "- dataset_card.md",
            "",
            "## Training example",
            "```bash",
            "python scripts/train/train_regions.py \\",
            "  --config configs/train/regions/nslt100/face_hands/region_resnet18_gru_ce.yaml \\",
            "  --data-root <bundle_root> \\",
            f"  --train-manifest <bundle_root>/manifests/{subset}_train.csv \\",
            f"  --val-manifest <bundle_root>/manifests/{subset}_val.csv \\",
            f"  --test-manifest <bundle_root>/manifests/{subset}_test.csv \\",
            "  --output-dir <output_dir> \\",
            f"  --run-name regions_cnn_gru_{subset}_ce_v1",
            "```",
            "",
            "## Notes",
            "- This bundle contains derived/preprocessed tensor data only.",
            "- It does not include raw WLASL videos.",
            "- It does not include standardized frames.",
            "- Use according to WLASL licensing and usage constraints.",
            "",
            f"Bundle root name: `{bundle_root.name}`",
        ]
    )


def _build_manifest_json(
    *,
    validation: ValidationResult,
    bundle_root: Path,
    checksum_path: str,
    split_file_counts: dict[str, int],
    zip_outputs: dict[str, int],
) -> dict[str, Any]:
    subset = validation.subset
    return {
        "dataset": "WLASL",
        "branch": "regions",
        "subset": subset,
        "pose_backend": "rtmw_l",
        "pose_layout": "wholebody_133",
        "tensor_format": TENSOR_FORMAT,
        "expected_shape": list(DEFAULT_TENSOR_SHAPE),
        "regions": list(REGION_NAMES),
        "splits": {
            split: {
                "manifest": f"manifests/{subset}_{split}.csv",
                "tensor_dir": f"tensors/{subset}/{split}",
                "samples": int(next(summary.rows for summary in validation.split_summaries if summary.split == split)),
                "files": int(split_file_counts[split]),
            }
            for split in SPLITS
        },
        "npz_keys": list(REGION_NPZ_FIELDS),
        "bbox_source_encoding": {str(key): value for key, value in BBOX_SOURCE_NAMES.items()},
        "sample_counts": {summary.split: int(summary.rows) for summary in validation.split_summaries},
        "file_counts": {
            "manifests": 3,
            "tensor_files": int(sum(split_file_counts.values())),
            "report_files": int(len(validation.report_files)) if validation.include_reports else 0,
            "preview_files": int(len(validation.preview_files)) if validation.include_previews else 0,
        },
        "total_size_bytes": int(_sum_file_sizes(_iter_files(bundle_root))),
        "checksum_path": checksum_path,
        "zip_files": {name: int(size) for name, size in zip_outputs.items()},
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise BundleError(
                f"Output directory already exists and is not empty: {output_dir}. Use --overwrite."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _upload_bundle(output_dir: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, LocalTokenNotFoundError
    except ImportError as exc:
        raise BundleError(
            "huggingface_hub is required for --upload. Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    try:
        api.whoami()
    except LocalTokenNotFoundError as exc:
        raise BundleError(
            "Hugging Face auth token was not found. Run `huggingface-cli login` or set HF_TOKEN."
        ) from exc
    except Exception as exc:
        raise BundleError(
            "Could not verify Hugging Face authentication. Run `huggingface-cli login` and try again."
        ) from exc

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
            commit_message=f"Upload {output_dir.name} bundle",
        )
    except HfHubHTTPError as exc:
        raise BundleError(f"Hugging Face upload failed for repo {repo_id!r}: {exc}") from exc


def main() -> int:
    args = parse_args()
    validation = validate_inputs(args)

    print(f"subset: {validation.subset}")
    print(f"input_root: {validation.input_root}")
    print(f"output_dir: {validation.output_dir}")
    for summary in validation.split_summaries:
        print(
            f"{summary.split}: rows={summary.rows} ok={summary.ok_count} "
            f"npz={summary.file_count} class_id_range={summary.class_id_min}..{summary.class_id_max}"
        )
    print(f"total_samples: {validation.total_samples}")
    print(f"num_classes: {validation.num_classes}")
    print(f"include_previews: {validation.include_previews}")
    print(f"include_reports: {validation.include_reports}")
    print(f"create_zips: {validation.create_zips}")
    print(f"source_payload_size: {_format_bytes(validation.source_total_size_bytes)}")

    if args.dry_run:
        print("dry-run: validation succeeded, no files were written.")
        return 0

    _validate_output_dir(validation.output_dir, overwrite=bool(args.overwrite))
    bundle_root = validation.output_dir

    tensors_bundle_root = bundle_root / "tensors" / validation.subset
    manifests_bundle_root = bundle_root / "manifests"
    checksums_root = bundle_root / "checksums"
    reports_bundle_root = bundle_root / "reports"
    previews_bundle_root = bundle_root / "previews" / validation.subset

    ensure_dir(tensors_bundle_root)
    ensure_dir(manifests_bundle_root)
    ensure_dir(checksums_root)

    split_file_counts: dict[str, int] = {}
    for summary in validation.split_summaries:
        copied = _copy_tree(summary.tensor_dir, tensors_bundle_root / summary.split)
        split_file_counts[summary.split] = int(copied)

    for summary in validation.split_summaries:
        _sanitize_manifest(
            source_path=summary.manifest_path,
            output_path=manifests_bundle_root / summary.manifest_path.name,
            subset=validation.subset,
            split=summary.split,
            include_previews=validation.include_previews,
        )

    if validation.include_reports:
        _copy_files(validation.report_files, reports_bundle_root)
    if validation.include_previews:
        for split in SPLITS:
            source_preview_dir = validation.previews_root / validation.subset / split
            target_preview_dir = previews_bundle_root / split
            _copy_tree(source_preview_dir, target_preview_dir)

    metadata = _build_metadata(validation, bundle_root)
    write_json(metadata, bundle_root / "metadata.json")
    write_text(_build_readme(validation, bundle_root), bundle_root / "README.md")
    write_text(_build_dataset_card(validation, bundle_root), bundle_root / "dataset_card.md")

    zip_outputs: dict[str, int] = {}
    if validation.create_zips:
        zip_outputs["tensors_nslt100.zip" if validation.subset == "nslt100" else f"tensors_{validation.subset}.zip"] = _write_zip(
            bundle_root / ("tensors_nslt100.zip" if validation.subset == "nslt100" else f"tensors_{validation.subset}.zip"),
            bundle_root,
            bundle_root / "tensors",
        )
        zip_outputs["manifests.zip"] = _write_zip(bundle_root / "manifests.zip", bundle_root, manifests_bundle_root)
        if validation.include_reports:
            zip_outputs["reports.zip"] = _write_zip(bundle_root / "reports.zip", bundle_root, reports_bundle_root)
        if validation.include_previews:
            preview_zip_name = "previews_nslt100.zip" if validation.subset == "nslt100" else f"previews_{validation.subset}.zip"
            zip_outputs[preview_zip_name] = _write_zip(bundle_root / preview_zip_name, bundle_root, bundle_root / "previews")

    manifest_json = _build_manifest_json(
        validation=validation,
        bundle_root=bundle_root,
        checksum_path="checksums/sha256_manifest.json",
        split_file_counts=split_file_counts,
        zip_outputs=zip_outputs,
    )
    write_json(manifest_json, bundle_root / "MANIFEST.json")

    checksum_targets = [
        bundle_root / "README.md",
        bundle_root / "dataset_card.md",
        bundle_root / "metadata.json",
        bundle_root / "MANIFEST.json",
    ]
    checksum_targets.extend(sorted(manifests_bundle_root.glob("*.csv")))
    if validation.include_reports:
        checksum_targets.extend(_iter_files(reports_bundle_root))
    for zip_name in sorted(zip_outputs):
        checksum_targets.append(bundle_root / zip_name)

    sha_manifest = {
        "bundle_root": bundle_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checksum_scope": [
            "README.md",
            "dataset_card.md",
            "metadata.json",
            "MANIFEST.json",
            "sanitized manifests",
            "reports (if included)",
            "zip files (if created)",
        ],
        "checksums": {
            str(path.relative_to(bundle_root)).replace("\\", "/"): _sha256_file(path)
            for path in checksum_targets
            if path.exists()
        },
    }
    write_json(sha_manifest, checksums_root / "sha256_manifest.json")

    if validation.upload:
        if not validation.repo_id:
            raise BundleError("--repo-id is required when --upload is enabled.")
        _upload_bundle(bundle_root, validation.repo_id, validation.private)

    total_bundle_size = _sum_file_sizes(_iter_files(bundle_root))
    print()
    print("bundle files created:")
    print(f"- {bundle_root / 'README.md'}")
    print(f"- {bundle_root / 'dataset_card.md'}")
    print(f"- {bundle_root / 'metadata.json'}")
    print(f"- {bundle_root / 'MANIFEST.json'}")
    print(f"- {bundle_root / 'checksums' / 'sha256_manifest.json'}")
    print(f"total bundle size: {_format_bytes(total_bundle_size)}")
    if zip_outputs:
        print("zip outputs:")
        for zip_name, size_bytes in zip_outputs.items():
            print(f"- {zip_name}: {_format_bytes(size_bytes)}")
    if validation.upload:
        print(f"uploaded repo: {validation.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
