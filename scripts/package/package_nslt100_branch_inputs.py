"""Package NSLT100 branch inputs for Kaggle upload."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package WLASL nslt100 skeleton and regions branch inputs for Kaggle."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Path to data/datasets/WLASL/branch_inputs.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory where the package folder will be created.")
    parser.add_argument("--subset", type=str, default="nslt100", help="Subset name to package.")
    parser.add_argument(
        "--package-name",
        type=str,
        default=None,
        help="Override the output folder and combined zip basename.",
    )
    parser.add_argument(
        "--skeleton-keypoint-set",
        type=str,
        default="selected_27",
        help="Skeleton keypoint set to package.",
    )
    parser.add_argument("--include-train", type=parse_bool, default=True, help="Include the train split.")
    parser.add_argument("--include-val", type=parse_bool, default=True, help="Include the val split.")
    parser.add_argument("--include-test", type=parse_bool, default=True, help="Include the test split.")
    parser.add_argument(
        "--include-region-reports",
        type=parse_bool,
        default=True,
        help="Copy lightweight subset-specific region reports.",
    )
    parser.add_argument("--zip", dest="create_zip", action="store_true", help="Create one combined zip archive.")
    parser.add_argument("--no-zip", dest="create_zip", action="store_false", help="Skip zip creation.")
    parser.set_defaults(create_zip=False)
    parser.add_argument(
        "--split-by-branch",
        action="store_true",
        help="Also create separate skeleton/regions zip archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing package folder and zip files under output-root.",
    )
    return parser


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {path.as_posix()}. "
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path.as_posix()}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def zip_directory(source_dir: Path, zip_path: Path, *, prefix: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(source_dir).as_posix()
            archive.write(path, arcname=f"{prefix}/{relative}")


def collect_splits(args: argparse.Namespace) -> list[str]:
    splits: list[str] = []
    if args.include_train:
        splits.append("train")
    if args.include_val:
        splits.append("val")
    if args.include_test:
        splits.append("test")
    if not splits:
        raise ValueError("At least one split must be included.")
    return splits


def resolve_skeleton_tensor_source(
    source_root: Path,
    *,
    keypoint_set: str,
    subset: str,
    split: str,
    row: dict[str, str],
) -> Path:
    raw = Path(row.get("graph_tensor_path", ""))
    if str(raw) and raw.exists():
        return raw
    sample_id = str(row.get("sample_id", "")).strip()
    return source_root / "skeleton" / "rtmw_l" / "graph_tensors" / keypoint_set / subset / split / f"{sample_id}.npz"


def resolve_regions_tensor_source(
    source_root: Path,
    *,
    subset: str,
    split: str,
    row: dict[str, str],
) -> Path:
    raw = Path(row.get("tensor_path", ""))
    if str(raw) and raw.exists():
        return raw
    sample_id = str(row.get("sample_id", "")).strip()
    return source_root / "regions" / "rtmw_l" / "tensors" / subset / split / f"{sample_id}.npz"


def package_skeleton_branch(
    *,
    source_root: Path,
    package_root: Path,
    subset: str,
    keypoint_set: str,
    splits: list[str],
) -> dict[str, Any]:
    branch_root = package_root / "skeleton" / "rtmw_l"
    manifests_dir = branch_root / "manifests"
    tensors_root = branch_root / "tensors" / subset
    branch_root.mkdir(parents=True, exist_ok=True)

    manifest_files: list[str] = []
    tensor_counts: dict[str, int] = {}
    sample_shape: list[int] | None = None

    for split in splits:
        manifest_name = f"{subset}_{keypoint_set}_{split}.csv"
        source_manifest = source_root / "skeleton" / "rtmw_l" / "manifests" / manifest_name
        fieldnames, rows = read_manifest(source_manifest)
        rewritten_rows: list[dict[str, str]] = []
        split_count = 0

        for row in rows:
            source_tensor = resolve_skeleton_tensor_source(
                source_root,
                keypoint_set=keypoint_set,
                subset=subset,
                split=split,
                row=row,
            )
            if not source_tensor.exists():
                raise FileNotFoundError(
                    f"Missing skeleton tensor for sample_id={row.get('sample_id')} split={split}: "
                    f"{source_tensor.as_posix()}"
                )
            destination_tensor = tensors_root / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["graph_tensor_path"] = f"tensors/{subset}/{split}/{source_tensor.name}"
            if "selected_path" in updated:
                updated["selected_path"] = ""
            if "normalized_path" in updated:
                updated["normalized_path"] = ""
            if "pose_path" in updated:
                updated["pose_path"] = ""
            rewritten_rows.append(updated)
            split_count += 1

        write_manifest(manifests_dir / manifest_name, fieldnames, rewritten_rows)
        manifest_files.append(f"manifests/{manifest_name}")
        tensor_counts[split] = split_count

        if rewritten_rows and sample_shape is None:
            import numpy as np

            sample_path = tensors_root / split / Path(rewritten_rows[0]["graph_tensor_path"]).name
            with np.load(sample_path, allow_pickle=False) as payload:
                array = payload["data"] if "data" in payload else payload["tensor"]
                sample_shape = [int(value) for value in array.shape]

    metadata = {
        "dataset": "WLASL",
        "subset": subset,
        "created_for": "skeleton_regions_late_fusion",
        "backend": "rtmw_l",
        "keypoint_set": keypoint_set,
        "expected_shape": sample_shape,
        "splits": splits,
        "manifest_files": manifest_files,
        "tensor_counts": tensor_counts,
        "graph_tensor_shape": sample_shape,
        "tensor_root": f"tensors/{subset}",
        "notes": [
            "This package includes graph tensors only.",
            "Non-required source paths were cleared in copied manifests.",
        ],
    }
    with (branch_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def package_regions_branch(
    *,
    source_root: Path,
    package_root: Path,
    subset: str,
    splits: list[str],
    include_reports: bool,
) -> dict[str, Any]:
    branch_root = package_root / "regions" / "rtmw_l"
    manifests_dir = branch_root / "manifests"
    tensors_root = branch_root / "tensors" / subset
    branch_root.mkdir(parents=True, exist_ok=True)

    manifest_files: list[str] = []
    tensor_counts: dict[str, int] = {}
    sample_shape: list[int] | None = None

    for split in splits:
        manifest_name = f"{subset}_{split}.csv"
        source_manifest = source_root / "regions" / "rtmw_l" / "manifests" / manifest_name
        fieldnames, rows = read_manifest(source_manifest)
        rewritten_rows: list[dict[str, str]] = []
        split_count = 0

        for row in rows:
            source_tensor = resolve_regions_tensor_source(
                source_root,
                subset=subset,
                split=split,
                row=row,
            )
            if not source_tensor.exists():
                raise FileNotFoundError(
                    f"Missing regions tensor for sample_id={row.get('sample_id')} split={split}: "
                    f"{source_tensor.as_posix()}"
                )
            destination_tensor = tensors_root / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["tensor_path"] = f"tensors/{subset}/{split}/{source_tensor.name}"
            if "preview_path" in updated:
                updated["preview_path"] = ""
            if "crop_root" in updated:
                updated["crop_root"] = ""
            rewritten_rows.append(updated)
            split_count += 1

        write_manifest(manifests_dir / manifest_name, fieldnames, rewritten_rows)
        manifest_files.append(f"manifests/{manifest_name}")
        tensor_counts[split] = split_count

        if rewritten_rows and sample_shape is None:
            import numpy as np

            sample_path = tensors_root / split / Path(rewritten_rows[0]["tensor_path"]).name
            with np.load(sample_path, allow_pickle=False) as payload:
                sample_shape = [int(value) for value in payload["data"].shape]

    copied_reports: list[str] = []
    if include_reports:
        reports_root = source_root / "regions" / "rtmw_l" / "reports"
        destination_reports = branch_root / "reports"
        for path in sorted(reports_root.glob(f"{subset}*")):
            if path.is_file():
                copy_file(path, destination_reports / path.name)
                copied_reports.append(f"reports/{path.name}")

    metadata = {
        "dataset": "WLASL",
        "subset": subset,
        "created_for": "skeleton_regions_late_fusion",
        "backend": "rtmw_l",
        "regions": ["left_hand", "right_hand", "face"],
        "expected_shape": sample_shape,
        "splits": splits,
        "manifest_files": manifest_files,
        "tensor_counts": tensor_counts,
        "tensor_shape": sample_shape,
        "tensor_root": f"tensors/{subset}",
        "copied_reports": copied_reports,
        "notes": [
            "This package excludes previews, crops, logs, and checkpoints.",
            "Copied manifests were rewritten to package-relative tensor paths.",
        ],
    }
    with (branch_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def write_package_readme(
    *,
    package_root: Path,
    package_name: str,
    subset: str,
    skeleton_keypoint_set: str,
) -> None:
    text = f"""# WLASL {subset} Branch Inputs

This dataset package contains the branch inputs required for NSLT100 Skeleton + Regions late fusion.

Included:

- skeleton `{skeleton_keypoint_set}` manifests and graph tensors
- regions manifests and tensors
- optional lightweight regions reports
- regions include `left_hand`, `right_hand`, `face`

Not included:

- raw videos
- training checkpoints
- W&B logs
- training outputs
- regions previews/crops

## Kaggle Usage

This package uses skeleton `{skeleton_keypoint_set}` and does not contain checkpoints or raw videos.

If you upload the extracted folder as a Kaggle Dataset, you can point the notebook directly to:

```python
from pathlib import Path

BRANCH_INPUTS_ROOT = Path("/kaggle/input/<dataset-slug>/{package_name}")

SKELETON_DATA_ROOT = str(BRANCH_INPUTS_ROOT / "skeleton/rtmw_l")
SKELETON_VAL_MANIFEST = str(BRANCH_INPUTS_ROOT / "skeleton/rtmw_l/manifests/{subset}_{skeleton_keypoint_set}_val.csv")
SKELETON_TEST_MANIFEST = str(BRANCH_INPUTS_ROOT / "skeleton/rtmw_l/manifests/{subset}_{skeleton_keypoint_set}_test.csv")

REGIONS_DATA_ROOT = str(BRANCH_INPUTS_ROOT / "regions/rtmw_l")
REGIONS_VAL_MANIFEST = str(BRANCH_INPUTS_ROOT / "regions/rtmw_l/manifests/{subset}_val.csv")
REGIONS_TEST_MANIFEST = str(BRANCH_INPUTS_ROOT / "regions/rtmw_l/manifests/{subset}_test.csv")
```

If you upload the zip file as a Kaggle Dataset, unzip it first inside the notebook:

```bash
unzip /kaggle/input/<dataset-name>/{package_name}.zip -d /kaggle/working/
```

Expected structure after unzip:

```text
/kaggle/working/{package_name}/
|- skeleton/rtmw_l/...
`- regions/rtmw_l/...
```
"""
    (package_root / "README.md").write_text(text, encoding="utf-8")


def write_package_metadata(
    *,
    package_root: Path,
    package_name: str,
    subset: str,
    skeleton_metadata: dict[str, Any],
    regions_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "subset": subset,
        "created_for": "skeleton_regions_late_fusion",
        "package_name": package_name,
        "branches": {
            "skeleton": {
                "backend": skeleton_metadata["backend"],
                "keypoint_set": skeleton_metadata["keypoint_set"],
                "expected_shape": skeleton_metadata["expected_shape"],
                "splits": skeleton_metadata["splits"],
                "manifest_files": skeleton_metadata["manifest_files"],
                "tensor_counts": skeleton_metadata["tensor_counts"],
            },
            "regions": {
                "backend": regions_metadata["backend"],
                "regions": regions_metadata["regions"],
                "expected_shape": regions_metadata["expected_shape"],
                "splits": regions_metadata["splits"],
                "manifest_files": regions_metadata["manifest_files"],
                "tensor_counts": regions_metadata["tensor_counts"],
            },
        },
        "notes": [
            "This bundle contains only the assets required to run skeleton + regions late fusion.",
            "Skeleton manifests were rewritten to use package-relative graph tensor paths.",
            "Regions manifests were rewritten to use package-relative tensor paths.",
        ],
    }
    with (package_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def summarize_package(package_root: Path) -> dict[str, Any]:
    files = [path for path in package_root.rglob("*") if path.is_file()]
    total_size = sum(path.stat().st_size for path in files)
    return {
        "package_root": package_root.as_posix(),
        "file_count": len(files),
        "total_size_bytes": total_size,
    }


def build_branch_zip(package_root: Path, *, subset: str, branch_name: str, output_root: Path) -> Path:
    branch_dir = package_root / branch_name
    zip_name = f"wlasl-{subset}-{branch_name}-branch-inputs.zip"
    zip_path = output_root / zip_name
    zip_directory(branch_dir, zip_path, prefix=f"wlasl-{subset}-{branch_name}-branch-inputs")
    return zip_path


def main() -> int:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    subset = str(args.subset).strip()
    skeleton_keypoint_set = str(args.skeleton_keypoint_set).strip()
    splits = collect_splits(args)

    package_name = str(args.package_name).strip() if args.package_name else f"wlasl-{subset}-branch-inputs"
    package_root = output_root / package_name
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(package_root, overwrite=bool(args.overwrite))

    skeleton_metadata = package_skeleton_branch(
        source_root=source_root,
        package_root=package_root,
        subset=subset,
        keypoint_set=skeleton_keypoint_set,
        splits=splits,
    )
    regions_metadata = package_regions_branch(
        source_root=source_root,
        package_root=package_root,
        subset=subset,
        splits=splits,
        include_reports=bool(args.include_region_reports),
    )
    write_package_readme(
        package_root=package_root,
        package_name=package_name,
        subset=subset,
        skeleton_keypoint_set=skeleton_keypoint_set,
    )
    metadata = write_package_metadata(
        package_root=package_root,
        package_name=package_name,
        subset=subset,
        skeleton_metadata=skeleton_metadata,
        regions_metadata=regions_metadata,
    )

    zip_paths: list[str] = []
    if args.create_zip:
        combined_zip = output_root / f"{package_name}.zip"
        zip_directory(package_root, combined_zip, prefix=package_name)
        zip_paths.append(combined_zip.as_posix())
    if args.split_by_branch:
        zip_paths.append(
            build_branch_zip(package_root, subset=subset, branch_name="skeleton", output_root=output_root).as_posix()
        )
        zip_paths.append(
            build_branch_zip(package_root, subset=subset, branch_name="regions", output_root=output_root).as_posix()
        )

    summary = summarize_package(package_root)
    summary["package_name"] = package_name
    summary["zip_paths"] = zip_paths
    summary["metadata"] = metadata
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
