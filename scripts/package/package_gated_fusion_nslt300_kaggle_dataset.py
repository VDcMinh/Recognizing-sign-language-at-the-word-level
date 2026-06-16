"""Build a Kaggle-ready NSLT300 gated-fusion dataset package."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required for this script. Run it with the project environment, for example: "
        ".\\.venv-rtmw310\\Scripts\\python.exe scripts/package/package_gated_fusion_nslt300_kaggle_dataset.py ..."
    ) from exc


SUBSET = "nslt300"
NUM_CLASSES = 300
PACKAGE_NAME_DEFAULT = "wlasl-nslt300-gated-fusion-ready"
KAGGLE_ROOT_TEMPLATE = "/kaggle/input/{package_name}/{package_name}"
SKELETON_EXPECTED_SHAPE = [3, 150, 31, 1]
REGIONS_EXPECTED_SHAPE = [3, 3, 64, 112, 112]
ACTIVE_REGIONS = ["left_hand", "right_hand", "face"]
REPORT_PATH_DEFAULT = Path("reports/packaging/gated_fusion_nslt300_kaggle_package_report.md")
REQUIREMENT_REPORT_PATH = Path("reports/packaging/gated_fusion_nslt300_requirement_check_report.md")
FUSION_CONFIG_PATH = Path("configs/train/fusion/gated_feature/nslt300/gated_feature_fusion_ce.yaml")
REQUIREMENT_SCRIPT_PATH = Path("scripts/verify/check_gated_fusion_nslt300_packaging_requirements.py")
REFERENCE_PATHS = [
    "scripts/package/package_gated_fusion_nslt100_kaggle_dataset.py",
    "scripts/verify/check_gated_fusion_setup.py",
    "configs/train/fusion/gated_feature/nslt100/gated_feature_fusion_ce.yaml",
    "configs/train/fusion/late_fusion/nslt100/skeleton_regions_late_fusion.yaml",
    "packaging_outputs/wlasl-nslt100-gated-fusion-ready",
    "reports/packaging/gated_fusion_nslt100_kaggle_package_report.md",
    "artifacts/fusion/nslt100",
]


class PackagingError(RuntimeError):
    """Raised when the package cannot be built safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Kaggle-ready package for WLASL NSLT300 gated feature fusion."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("packaging_outputs"),
        help="Directory where the package folder and zip will be created.",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=PACKAGE_NAME_DEFAULT,
        help="Name of the package folder and zip basename.",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=FUSION_CONFIG_PATH,
        help="Base gated-fusion NSLT300 train config.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH_DEFAULT,
        help="Markdown package report path written into the repo.",
    )
    parser.add_argument(
        "--requirement-report-path",
        type=Path,
        default=REQUIREMENT_REPORT_PATH,
        help="Markdown requirement-check report path written into the repo.",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=3,
        help="How many tensors per split the packaged verify script should load.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing package folder and zip before rebuilding.",
    )
    parser.add_argument(
        "--zip",
        dest="create_zip",
        action="store_true",
        help="Create a zip archive after verify passes.",
    )
    parser.add_argument(
        "--no-zip",
        dest="create_zip",
        action="store_false",
        help="Skip zip creation.",
    )
    parser.set_defaults(create_zip=False)
    return parser


def resolve_path(value: Path, *, root: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise PackagingError(f"Missing required {label}: {path.as_posix()}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise PackagingError(f"Expected a YAML mapping at {path.as_posix()}, got {type(payload)!r}.")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PackagingError(f"Manifest has no header: {path.as_posix()}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def summarize_directory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def zip_directory(source_dir: Path, zip_path: Path, *, prefix: str) -> int:
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for item in sorted(source_dir.rglob("*")):
            if item.is_dir():
                continue
            archive.write(item, arcname=f"{prefix}/{item.relative_to(source_dir).as_posix()}")
    return zip_path.stat().st_size


def load_requirement_module(project_root: Path):
    module_path = (project_root / REQUIREMENT_SCRIPT_PATH).resolve()
    ensure_exists(module_path, "NSLT300 requirement-check script")
    spec = importlib.util.spec_from_file_location("check_gated_fusion_nslt300_packaging_requirements", module_path)
    if spec is None or spec.loader is None:
        raise PackagingError(f"Could not load requirement-check module: {module_path.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_requirement_audit(
    *,
    project_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    module = load_requirement_module(project_root)
    pairing_min_coverage = float(getattr(module, "PAIRING_MIN_COVERAGE", 0.95))
    summary = module.build_summary(project_root, pairing_min_coverage=pairing_min_coverage)
    report_text = module.create_report(summary, pairing_min_coverage=pairing_min_coverage)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    summary["pairing_min_coverage"] = pairing_min_coverage
    summary["report_path"] = report_path.as_posix()
    return summary


def build_kaggle_root(package_name: str) -> str:
    return KAGGLE_ROOT_TEMPLATE.format(package_name=package_name)


def normalize_sample_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def resolve_skeleton_tensor_path(row: dict[str, str], tensor_root: Path, split: str) -> Path:
    raw_path = Path(row.get("graph_tensor_path", ""))
    if str(raw_path) and raw_path.exists():
        return raw_path
    sample_id = str(row.get("sample_id", "")).strip()
    candidate = tensor_root / split / f"{sample_id}.npz"
    if candidate.exists():
        return candidate
    raise PackagingError(
        f"Missing skeleton tensor for sample_id={sample_id} split={split}: {candidate.as_posix()}"
    )


def resolve_regions_tensor_path(row: dict[str, str], tensor_root: Path, split: str) -> Path:
    raw_path = Path(row.get("tensor_path", ""))
    if str(raw_path) and raw_path.exists():
        return raw_path
    sample_id = str(row.get("sample_id", "")).strip()
    candidate = tensor_root / split / f"{sample_id}.npz"
    if candidate.exists():
        return candidate
    raise PackagingError(
        f"Missing regions tensor for sample_id={sample_id} split={split}: {candidate.as_posix()}"
    )


def copy_skeleton_branch(
    *,
    manifest_dir: Path,
    tensor_root: Path,
    package_root: Path,
) -> dict[str, int]:
    destination_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l"
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        manifest_name = f"{SUBSET}_selected_31_{split}.csv"
        fieldnames, rows = read_manifest(manifest_dir / manifest_name)
        rewritten_rows: list[dict[str, str]] = []
        for row in rows:
            source_tensor = resolve_skeleton_tensor_path(row, tensor_root, split)
            destination_tensor = destination_root / "tensors" / SUBSET / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["graph_tensor_path"] = f"tensors/{SUBSET}/{split}/{source_tensor.name}"
            for key in ("pose_path", "selected_path", "normalized_path", "error_message"):
                if key in updated:
                    updated[key] = ""
            rewritten_rows.append(updated)

        write_manifest(destination_root / "manifests" / manifest_name, fieldnames, rewritten_rows)
        counts[split] = len(rewritten_rows)
    return counts


def copy_regions_branch(
    *,
    manifest_dir: Path,
    tensor_root: Path,
    reports_root: Path,
    package_root: Path,
) -> tuple[dict[str, int], list[str]]:
    destination_root = package_root / "branch_inputs" / "regions" / "rtmw_l"
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        manifest_name = f"{SUBSET}_{split}.csv"
        fieldnames, rows = read_manifest(manifest_dir / manifest_name)
        rewritten_rows: list[dict[str, str]] = []
        for row in rows:
            source_tensor = resolve_regions_tensor_path(row, tensor_root, split)
            destination_tensor = destination_root / "tensors" / SUBSET / split / source_tensor.name
            copy_file(source_tensor, destination_tensor)

            updated = dict(row)
            updated["tensor_path"] = f"tensors/{SUBSET}/{split}/{source_tensor.name}"
            for key in ("crop_root", "preview_path", "error_message"):
                if key in updated:
                    updated[key] = ""
            rewritten_rows.append(updated)

        write_manifest(destination_root / "manifests" / manifest_name, fieldnames, rewritten_rows)
        counts[split] = len(rewritten_rows)

    copied_reports: list[str] = []
    if reports_root.exists():
        for item in sorted(reports_root.rglob("*")):
            if item.is_dir():
                continue
            if SUBSET not in item.as_posix().lower():
                continue
            rel = item.relative_to(reports_root)
            copy_file(item, destination_root / "reports" / rel)
            copied_reports.append(f"reports/{rel.as_posix()}")
    return counts, copied_reports


def create_kaggle_config(
    *,
    base_config: dict[str, Any],
    package_name: str,
) -> dict[str, Any]:
    kaggle_root = build_kaggle_root(package_name)
    config = json.loads(json.dumps(base_config))

    config.setdefault("experiment", {})
    config["experiment"]["output_root"] = "/kaggle/working/outputs/fusion"

    config["dataset"] = {
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "skeleton": {
            "data_root": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l",
            "keypoint_set": "selected_31",
            "expected_shape": list(SKELETON_EXPECTED_SHAPE),
            "manifests": {
                "train": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_train.csv",
                "val": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_val.csv",
                "test": f"{kaggle_root}/branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_test.csv",
            },
            "return_metadata": True,
            "strict_shape_check": True,
        },
        "regions": {
            "data_root": f"{kaggle_root}/branch_inputs/regions/rtmw_l",
            "expected_shape": list(REGIONS_EXPECTED_SHAPE),
            "region_order": list(ACTIVE_REGIONS),
            "active_regions": list(ACTIVE_REGIONS),
            "manifests": {
                "train": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt300_train.csv",
                "val": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt300_val.csv",
                "test": f"{kaggle_root}/branch_inputs/regions/rtmw_l/manifests/nslt300_test.csv",
            },
            "normalize": {"type": "imagenet"},
            "return_metadata": True,
            "strict_shape_check": True,
        },
    }

    config.setdefault("skeleton_branch", {})
    config["skeleton_branch"]["config_path"] = f"{kaggle_root}/configs/skeleton_config_resolved.yaml"
    config["skeleton_branch"]["checkpoint_path"] = f"{kaggle_root}/checkpoints/skeleton/best.pt"
    config["skeleton_branch"]["graph"] = {
        "layout": "selected_31",
        "strategy": "spatial",
        "add_self_links": True,
        "normalize_adjacency": True,
    }
    config["skeleton_branch"]["model"] = {
        "name": "stgcnpp",
        "in_channels": 3,
        "num_nodes": 31,
        "num_classes": NUM_CLASSES,
        "base_channels": 64,
        "dropout": 0.5,
    }

    config.setdefault("regions_branch", {})
    config["regions_branch"]["config_path"] = f"{kaggle_root}/configs/regions_config_resolved.yaml"
    config["regions_branch"]["checkpoint_path"] = f"{kaggle_root}/checkpoints/regions/best.pt"
    config["regions_branch"]["model"] = {
        "name": "region_resnet18_gru",
        "num_classes": NUM_CLASSES,
        "num_regions": len(ACTIVE_REGIONS),
        "in_channels": 3,
        "clip_len": 64,
        "crop_size": 112,
        "pretrained": False,
        "freeze_encoder": True,
        "encoder_name": "resnet18",
        "encoder_feature_dim": 512,
        "gru_hidden_size": 128,
        "gru_num_layers": 1,
        "bidirectional": True,
        "dropout": 0.5,
        "fusion": "concat",
        "use_valid_mask": True,
    }

    config.setdefault("logging", {})
    config["logging"]["project"] = "wlasl-nslt300-gated-fusion"
    return config


def create_metadata(
    *,
    package_name: str,
    skeleton_counts: dict[str, int],
    regions_counts: dict[str, int],
    kaggle_config: dict[str, Any],
) -> dict[str, Any]:
    fusion_model = dict(kaggle_config.get("fusion_model", {}))
    return {
        "package_name": package_name,
        "subset": SUBSET,
        "num_classes": NUM_CLASSES,
        "purpose": "Kaggle-ready package for Gated Feature Fusion on NSLT300",
        "skeleton": {
            "checkpoint": "checkpoints/skeleton/best.pt",
            "config": "configs/skeleton_config_resolved.yaml",
            "keypoint_set": "selected_31",
            "expected_shape": list(SKELETON_EXPECTED_SHAPE),
            "counts": skeleton_counts,
        },
        "regions": {
            "checkpoint": "checkpoints/regions/best.pt",
            "config": "configs/regions_config_resolved.yaml",
            "active_regions": list(ACTIVE_REGIONS),
            "expected_shape": list(REGIONS_EXPECTED_SHAPE),
            "counts": regions_counts,
        },
        "fusion": {
            "config": "configs/gated_feature_fusion_nslt300_kaggle.yaml",
            "model": "GatedFeatureFusion",
            "hidden_dim": int(fusion_model.get("hidden_dim", 256)),
            "freeze_skeleton": bool(fusion_model.get("freeze_skeleton", True)),
            "freeze_regions": bool(fusion_model.get("freeze_regions", True)),
        },
    }


def create_readme(package_name: str) -> str:
    kaggle_root = build_kaggle_root(package_name)
    return textwrap.dedent(
        f"""
        # WLASL NSLT300 Gated Fusion Ready Dataset

        ## 1. Purpose
        This package is a Kaggle-ready bundle for training and evaluating Gated Feature Fusion on WLASL NSLT300.

        ## 2. What is included
        - Skeleton NSLT300 branch inputs using selected_31
        - Regions NSLT300 branch inputs using all regions: left_hand, right_hand, face
        - Skeleton NSLT300 `best.pt`
        - Regions NSLT300 `best.pt`
        - Skeleton `config_resolved.yaml`
        - Regions `config_resolved.yaml`
        - Gated Fusion NSLT300 Kaggle config
        - `README.md`
        - `metadata.json`
        - `verify/verify_package.py`
        - `verify/verify_summary.json`

        ## 3. Folder structure
        ```text
        {package_name}/
        |-- README.md
        |-- metadata.json
        |-- configs/
        |   |-- gated_feature_fusion_nslt300_kaggle.yaml
        |   |-- skeleton_config_resolved.yaml
        |   `-- regions_config_resolved.yaml
        |-- checkpoints/
        |   |-- skeleton/
        |   |   `-- best.pt
        |   `-- regions/
        |       `-- best.pt
        |-- branch_inputs/
        |   |-- skeleton/rtmw_l/...
        |   `-- regions/rtmw_l/...
        `-- verify/
            |-- verify_package.py
            `-- verify_summary.json
        ```

        ## 4. Skeleton branch details
        - subset: `nslt300`
        - keypoint_set: `selected_31`
        - expected_shape: `{tuple(SKELETON_EXPECTED_SHAPE)}`
        - manifests:
          - `branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_train.csv`
          - `branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_val.csv`
          - `branch_inputs/skeleton/rtmw_l/manifests/nslt300_selected_31_test.csv`

        ## 5. Regions branch details
        - subset: `nslt300`
        - active_regions: `left_hand`, `right_hand`, `face`
        - expected_shape: `{tuple(REGIONS_EXPECTED_SHAPE)}`
        - manifests:
          - `branch_inputs/regions/rtmw_l/manifests/nslt300_train.csv`
          - `branch_inputs/regions/rtmw_l/manifests/nslt300_val.csv`
          - `branch_inputs/regions/rtmw_l/manifests/nslt300_test.csv`

        ## 6. Checkpoint and config details
        - Skeleton checkpoint: `checkpoints/skeleton/best.pt`
        - Regions checkpoint: `checkpoints/regions/best.pt`
        - Skeleton config: `configs/skeleton_config_resolved.yaml`
        - Regions config: `configs/regions_config_resolved.yaml`
        - Fusion config: `configs/gated_feature_fusion_nslt300_kaggle.yaml`

        ## 7. How to upload to Kaggle
        Upload the package folder or the generated zip to a Kaggle Dataset, then attach that dataset to your notebook.
        The generated config assumes the dataset is mounted at:
        `{kaggle_root}`

        ## 8. How to verify
        ```bash
        python {kaggle_root}/verify/verify_package.py --package-root {kaggle_root}
        ```

        ## 9. How to train
        ```bash
        python scripts/train/train_gated_fusion.py \\
          --config {kaggle_root}/configs/gated_feature_fusion_nslt300_kaggle.yaml
        ```

        ## 10. How to evaluate
        Validation:
        ```bash
        python scripts/evaluate/evaluate_gated_fusion.py \\
          --config {kaggle_root}/configs/gated_feature_fusion_nslt300_kaggle.yaml \\
          --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt300-sel31-ce-regions/best.pt \\
          --split val
        ```

        Test:
        ```bash
        python scripts/evaluate/evaluate_gated_fusion.py \\
          --config {kaggle_root}/configs/gated_feature_fusion_nslt300_kaggle.yaml \\
          --checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt300-sel31-ce-regions/best.pt \\
          --split test
        ```

        ## 11. Important notes
        - `/kaggle/input` is read-only. Write outputs under `/kaggle/working`.
        - This package is for NSLT300 only.
        - Do not replace the packaged NSLT300 checkpoints or configs with NSLT100 assets.
        - If the Kaggle dataset slug differs from `{package_name}`, update the paths inside the Kaggle config.
        """
    ).strip() + "\n"


def create_verify_script() -> str:
    return textwrap.dedent(
        """
        \"\"\"Verify a Kaggle-ready NSLT300 gated-fusion package.\"\"\"

        from __future__ import annotations

        import argparse
        import ast
        import csv
        import json
        from pathlib import Path
        from typing import Any

        import numpy as np
        import torch
        import yaml


        SUBSET = "nslt300"
        NUM_CLASSES = 300
        SKELETON_EXPECTED_SHAPE = (3, 150, 31, 1)
        REGIONS_EXPECTED_SHAPE = (3, 3, 64, 112, 112)
        ACTIVE_REGIONS = ["left_hand", "right_hand", "face"]


        def build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(description="Verify the packaged NSLT300 gated-fusion dataset.")
            parser.add_argument("--package-root", type=Path, required=True)
            parser.add_argument("--sample-checks", type=int, default=3)
            return parser


        def read_yaml(path: Path) -> dict[str, Any]:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if not isinstance(payload, dict):
                raise ValueError(f"Expected YAML mapping: {path.as_posix()}")
            return payload


        def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Manifest has no header: {path.as_posix()}")
                rows = [{key: value or "" for key, value in row.items()} for row in reader]
            return list(reader.fieldnames), rows


        def parse_shape_literal(value: str) -> list[int] | None:
            if not str(value).strip():
                return None
            try:
                payload = ast.literal_eval(str(value))
            except (SyntaxError, ValueError):
                return None
            if isinstance(payload, (list, tuple)):
                return [int(item) for item in payload]
            return None


        def load_tensor_shape(path: Path) -> list[int]:
            if not path.exists():
                raise FileNotFoundError(f"Missing tensor: {path.as_posix()}")
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as payload:
                    for key in payload.files:
                        array = payload[key]
                        if hasattr(array, "shape"):
                            return [int(item) for item in array.shape]
                raise ValueError(f"No array found inside tensor file: {path.as_posix()}")
            if path.suffix.lower() == ".npy":
                array = np.load(path, allow_pickle=False)
                return [int(item) for item in array.shape]
            raise ValueError(f"Unsupported tensor suffix: {path.as_posix()}")


        def normalize_sample_id(value: Any) -> str:
            text = str(value or "").strip()
            if text.isdigit():
                return str(int(text))
            return text


        def get_state_dict(payload: Any) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None
            for key in ("model_state_dict", "state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if all(hasattr(value, "shape") for value in payload.values()):
                return payload
            return None


        def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> tuple[int | None, list[str]]:
            preferred_keys = [
                "classifier.weight",
                "fc.weight",
                "head.weight",
                "classifier.2.weight",
                "head.1.weight",
            ]
            evidence: list[str] = []
            for key in preferred_keys:
                tensor = state_dict.get(key)
                shape = getattr(tensor, "shape", None)
                if shape is None or len(shape) != 2:
                    continue
                evidence.append(f"{key}={list(shape)}")
                return int(shape[0]), evidence

            for key, tensor in state_dict.items():
                shape = getattr(tensor, "shape", None)
                if shape is None or len(shape) != 2:
                    continue
                out_features = int(shape[0])
                if out_features in (100, 300):
                    evidence.append(f"{key}={[int(item) for item in shape]}")
                    return out_features, evidence
            return None, evidence


        def verify_tensor_split(
            *,
            branch_name: str,
            branch_root: Path,
            manifest_name: str,
            tensor_key: str,
            expected_shape: tuple[int, ...],
            sample_checks: int,
        ) -> dict[str, Any]:
            manifest_path = branch_root / "manifests" / manifest_name
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path.as_posix()}")

            _fieldnames, rows = read_manifest(manifest_path)
            if not rows:
                raise ValueError(f"Manifest has no rows: {manifest_path.as_posix()}")

            sample_rows = rows[: max(1, min(sample_checks, len(rows)))]
            checked_shapes: list[list[int]] = []
            for row in sample_rows:
                relative_tensor = Path(str(row.get(tensor_key, "")).strip())
                tensor_path = branch_root / relative_tensor
                actual_shape = load_tensor_shape(tensor_path)
                if tuple(actual_shape) != expected_shape:
                    raise ValueError(
                        f"{branch_name} tensor shape mismatch for {tensor_path.as_posix()}: "
                        f"expected {expected_shape}, got {tuple(actual_shape)}"
                    )
                manifest_shape = parse_shape_literal(row.get("tensor_shape", ""))
                if manifest_shape is not None and tuple(manifest_shape) != expected_shape:
                    raise ValueError(
                        f"{branch_name} manifest tensor_shape mismatch in {manifest_path.as_posix()}: "
                        f"expected {expected_shape}, got {tuple(manifest_shape)}"
                    )
                checked_shapes.append(actual_shape)

            return {
                "manifest": manifest_path.as_posix(),
                "count": len(rows),
                "checked_samples": len(sample_rows),
                "sample_tensor_shapes": checked_shapes,
            }


        def verify_skeleton_checkpoint(path: Path) -> dict[str, Any]:
            payload = torch.load(path, map_location="cpu")
            state_dict = get_state_dict(payload)
            if state_dict is None:
                raise ValueError(f"Checkpoint does not expose a readable state dict: {path.as_posix()}")

            a_shape = list(getattr(state_dict.get("A"), "shape", []))
            bn_shape = list(getattr(state_dict.get("data_bn.weight"), "shape", []))
            if a_shape == [3, 27, 27] or bn_shape == [81]:
                raise ValueError("Checkpoint is selected_27, not selected_31.")
            if a_shape not in ([], [3, 31, 31]) and bn_shape not in ([], [93]):
                raise ValueError(
                    "Could not confirm selected_31 from checkpoint tensors "
                    f"(A={a_shape or '<missing>'}, data_bn.weight={bn_shape or '<missing>'})."
                )

            num_classes, evidence = infer_num_classes_from_state_dict(state_dict)
            if num_classes != NUM_CLASSES:
                raise ValueError(
                    f"Checkpoint classifier output is not {NUM_CLASSES} classes "
                    f"(detected={num_classes}, evidence={evidence or ['<none>']})."
                )
            return {
                "A_shape": a_shape or None,
                "data_bn_weight_shape": bn_shape or None,
                "num_classes": num_classes,
                "classifier_evidence": evidence,
            }


        def verify_regions_checkpoint(path: Path) -> dict[str, Any]:
            payload = torch.load(path, map_location="cpu")
            state_dict = get_state_dict(payload)
            if state_dict is None:
                raise ValueError(f"Checkpoint does not expose a readable state dict: {path.as_posix()}")

            num_classes, evidence = infer_num_classes_from_state_dict(state_dict)
            if num_classes != NUM_CLASSES:
                raise ValueError(
                    f"Checkpoint classifier output is not {NUM_CLASSES} classes "
                    f"(detected={num_classes}, evidence={evidence or ['<none>']})."
                )

            payload_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            dataset_cfg = payload_cfg.get("dataset", {}) if isinstance(payload_cfg, dict) else {}
            model_cfg = payload_cfg.get("model", {}) if isinstance(payload_cfg, dict) else {}
            model_name = str(model_cfg.get("name", "")).strip().lower()
            active_regions = list(dataset_cfg.get("active_regions", dataset_cfg.get("region_order", [])))
            normalized = [str(item).strip() for item in active_regions]

            if model_name and model_name != "region_resnet18_gru":
                raise ValueError(f"Checkpoint model is {model_name}, not region_resnet18_gru.")
            if normalized and normalized != ACTIVE_REGIONS:
                raise ValueError(
                    f"Checkpoint active_regions mismatch: expected {ACTIVE_REGIONS}, got {normalized}."
                )
            return {
                "num_classes": num_classes,
                "classifier_evidence": evidence,
                "model_name": model_name or None,
                "active_regions": normalized or None,
            }


        def verify_skeleton_config(path: Path) -> dict[str, Any]:
            config = read_yaml(path)
            dataset_cfg = dict(config.get("dataset", {}))
            graph_cfg = dict(config.get("graph", {}))
            model_cfg = dict(config.get("model", {}))
            if str(dataset_cfg.get("subset", "")).strip() != SUBSET:
                raise ValueError("Skeleton config dataset.subset must be nslt300.")
            if str(dataset_cfg.get("keypoint_set", "")).strip() != "selected_31":
                raise ValueError("Skeleton config keypoint_set must be selected_31.")
            if list(dataset_cfg.get("expected_shape", [])) != list(SKELETON_EXPECTED_SHAPE):
                raise ValueError("Skeleton config expected_shape mismatch.")
            if str(graph_cfg.get("layout", "")).strip() != "selected_31":
                raise ValueError("Skeleton config graph.layout must be selected_31.")
            if int(model_cfg.get("num_nodes", -1)) != 31:
                raise ValueError("Skeleton config model.num_nodes must be 31.")
            if int(model_cfg.get("num_classes", -1)) != NUM_CLASSES:
                raise ValueError("Skeleton config model.num_classes must be 300.")
            return {
                "subset": SUBSET,
                "keypoint_set": "selected_31",
                "num_nodes": 31,
                "num_classes": NUM_CLASSES,
            }


        def verify_regions_config(path: Path) -> dict[str, Any]:
            config = read_yaml(path)
            dataset_cfg = dict(config.get("dataset", {}))
            model_cfg = dict(config.get("model", {}))
            normalized = [str(item).strip() for item in dataset_cfg.get("active_regions", dataset_cfg.get("region_order", []))]
            if str(dataset_cfg.get("subset", "")).strip() != SUBSET:
                raise ValueError("Regions config dataset.subset must be nslt300.")
            if list(dataset_cfg.get("expected_shape", [])) != list(REGIONS_EXPECTED_SHAPE):
                raise ValueError("Regions config expected_shape mismatch.")
            if normalized != ACTIVE_REGIONS:
                raise ValueError("Regions config active_regions must be all-regions.")
            if str(model_cfg.get("name", "")).strip().lower() != "region_resnet18_gru":
                raise ValueError("Regions config model.name must be region_resnet18_gru.")
            if int(model_cfg.get("num_classes", -1)) != NUM_CLASSES:
                raise ValueError("Regions config model.num_classes must be 300.")
            return {
                "subset": SUBSET,
                "active_regions": normalized,
                "num_classes": NUM_CLASSES,
            }


        def verify_kaggle_config(path: Path, package_name: str) -> dict[str, Any]:
            config = read_yaml(path)
            kaggle_root = f"/kaggle/input/{package_name}/{package_name}"
            dataset_cfg = dict(config.get("dataset", {}))
            skeleton_cfg = dict(dataset_cfg.get("skeleton", {}))
            regions_cfg = dict(dataset_cfg.get("regions", {}))
            experiment_cfg = dict(config.get("experiment", {}))
            if str(dataset_cfg.get("subset", "")).strip() != SUBSET:
                raise ValueError("Kaggle config dataset.subset must be nslt300.")
            if int(dataset_cfg.get("num_classes", -1)) != NUM_CLASSES:
                raise ValueError("Kaggle config dataset.num_classes must be 300.")
            if skeleton_cfg.get("data_root") != f"{kaggle_root}/branch_inputs/skeleton/rtmw_l":
                raise ValueError("Kaggle config skeleton data_root is incorrect.")
            if regions_cfg.get("data_root") != f"{kaggle_root}/branch_inputs/regions/rtmw_l":
                raise ValueError("Kaggle config regions data_root is incorrect.")
            if config.get("skeleton_branch", {}).get("config_path") != f"{kaggle_root}/configs/skeleton_config_resolved.yaml":
                raise ValueError("Kaggle config skeleton config_path is incorrect.")
            if config.get("regions_branch", {}).get("config_path") != f"{kaggle_root}/configs/regions_config_resolved.yaml":
                raise ValueError("Kaggle config regions config_path is incorrect.")
            if config.get("skeleton_branch", {}).get("checkpoint_path") != f"{kaggle_root}/checkpoints/skeleton/best.pt":
                raise ValueError("Kaggle config skeleton checkpoint_path is incorrect.")
            if config.get("regions_branch", {}).get("checkpoint_path") != f"{kaggle_root}/checkpoints/regions/best.pt":
                raise ValueError("Kaggle config regions checkpoint_path is incorrect.")
            if experiment_cfg.get("output_root") != "/kaggle/working/outputs/fusion":
                raise ValueError("Kaggle config experiment.output_root is incorrect.")
            return {"paths_confirmed": True}


        def verify_pairing(package_root: Path) -> dict[str, Any]:
            skeleton_manifest_dir = package_root / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests"
            regions_manifest_dir = package_root / "branch_inputs" / "regions" / "rtmw_l" / "manifests"
            results: dict[str, Any] = {}

            for split in ("train", "val", "test"):
                _s_fields, skeleton_rows = read_manifest(skeleton_manifest_dir / f"nslt300_selected_31_{split}.csv")
                _r_fields, regions_rows = read_manifest(regions_manifest_dir / f"nslt300_{split}.csv")
                skeleton_map = {normalize_sample_id(row["sample_id"]): row for row in skeleton_rows}
                regions_map = {normalize_sample_id(row["sample_id"]): row for row in regions_rows}
                skeleton_ids = set(skeleton_map)
                regions_ids = set(regions_map)
                matched_ids = sorted(skeleton_ids & regions_ids)
                label_mismatch = sum(
                    1
                    for sample_id in matched_ids
                    if str(skeleton_map[sample_id]["class_id"]).strip() != str(regions_map[sample_id]["class_id"]).strip()
                )
                gloss_mismatch = 0
                if matched_ids and all("gloss" in row for row in skeleton_rows) and all("gloss" in row for row in regions_rows):
                    gloss_mismatch = sum(
                        1
                        for sample_id in matched_ids
                        if str(skeleton_map[sample_id].get("gloss", "")).strip()
                        != str(regions_map[sample_id].get("gloss", "")).strip()
                    )
                if label_mismatch != 0:
                    raise ValueError(f"Pairing label mismatch detected in split={split}: {label_mismatch}")
                if gloss_mismatch != 0:
                    raise ValueError(f"Pairing gloss mismatch detected in split={split}: {gloss_mismatch}")
                results[split] = {
                    "skeleton_count": len(skeleton_ids),
                    "regions_count": len(regions_ids),
                    "matched_count": len(matched_ids),
                    "missing_in_skeleton": len(regions_ids - skeleton_ids),
                    "missing_in_regions": len(skeleton_ids - regions_ids),
                    "label_mismatch": label_mismatch,
                    "gloss_mismatch": gloss_mismatch,
                }
            return results


        def verify_metadata(path: Path, package_name: str) -> dict[str, Any]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("package_name") != package_name:
                raise ValueError("metadata.package_name is incorrect.")
            if payload.get("subset") != SUBSET:
                raise ValueError("metadata.subset is incorrect.")
            if int(payload.get("num_classes", -1)) != NUM_CLASSES:
                raise ValueError("metadata.num_classes is incorrect.")
            return {
                "package_name": payload.get("package_name"),
                "subset": payload.get("subset"),
                "num_classes": payload.get("num_classes"),
            }


        def main() -> int:
            args = build_parser().parse_args()
            package_root = args.package_root.resolve()
            package_name = package_root.name

            skeleton_root = package_root / "branch_inputs" / "skeleton" / "rtmw_l"
            regions_root = package_root / "branch_inputs" / "regions" / "rtmw_l"
            skeleton_summary = {}
            regions_summary = {}
            for split in ("train", "val", "test"):
                skeleton_summary[split] = verify_tensor_split(
                    branch_name="skeleton",
                    branch_root=skeleton_root,
                    manifest_name=f"nslt300_selected_31_{split}.csv",
                    tensor_key="graph_tensor_path",
                    expected_shape=SKELETON_EXPECTED_SHAPE,
                    sample_checks=int(args.sample_checks),
                )
                regions_summary[split] = verify_tensor_split(
                    branch_name="regions",
                    branch_root=regions_root,
                    manifest_name=f"nslt300_{split}.csv",
                    tensor_key="tensor_path",
                    expected_shape=REGIONS_EXPECTED_SHAPE,
                    sample_checks=int(args.sample_checks),
                )

            skeleton_checkpoint_path = package_root / "checkpoints" / "skeleton" / "best.pt"
            regions_checkpoint_path = package_root / "checkpoints" / "regions" / "best.pt"
            skeleton_config_path = package_root / "configs" / "skeleton_config_resolved.yaml"
            regions_config_path = package_root / "configs" / "regions_config_resolved.yaml"
            kaggle_config_path = package_root / "configs" / "gated_feature_fusion_nslt300_kaggle.yaml"
            metadata_path = package_root / "metadata.json"
            readme_path = package_root / "README.md"

            for required in (
                skeleton_checkpoint_path,
                regions_checkpoint_path,
                skeleton_config_path,
                regions_config_path,
                kaggle_config_path,
                metadata_path,
                readme_path,
            ):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required packaged file: {required.as_posix()}")

            summary = {
                "status": "pass",
                "package_root": package_root.as_posix(),
                "skeleton": {
                    "splits": skeleton_summary,
                    "checkpoint": verify_skeleton_checkpoint(skeleton_checkpoint_path),
                    "config": verify_skeleton_config(skeleton_config_path),
                },
                "regions": {
                    "splits": regions_summary,
                    "checkpoint": verify_regions_checkpoint(regions_checkpoint_path),
                    "config": verify_regions_config(regions_config_path),
                },
                "kaggle_config": verify_kaggle_config(kaggle_config_path, package_name),
                "pairing": verify_pairing(package_root),
                "metadata": verify_metadata(metadata_path, package_name),
                "readme": {"exists": True},
            }
            summary_path = package_root / "verify" / "verify_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("VERIFY PASS")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip() + "\n"


def write_verify_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(create_verify_script(), encoding="utf-8")


def run_verify(package_root: Path, sample_checks: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str((package_root / "verify" / "verify_package.py").resolve()),
        "--package-root",
        str(package_root.resolve()),
        "--sample-checks",
        str(sample_checks),
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
            "Package verify failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    summary_path = package_root / "verify" / "verify_summary.json"
    ensure_exists(summary_path, "verify summary")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def existing_reference_paths(project_root: Path) -> list[str]:
    results: list[str] = []
    for relative in REFERENCE_PATHS:
        path = (project_root / relative).resolve()
        if path.exists():
            results.append(path.as_posix())
    return results


def build_train_command(package_name: str) -> str:
    kaggle_root = build_kaggle_root(package_name)
    return (
        "python scripts/train/train_gated_fusion.py "
        f"--config {kaggle_root}/configs/gated_feature_fusion_nslt300_kaggle.yaml"
    )


def build_eval_command(package_name: str, split: str) -> str:
    kaggle_root = build_kaggle_root(package_name)
    return (
        "python scripts/evaluate/evaluate_gated_fusion.py "
        f"--config {kaggle_root}/configs/gated_feature_fusion_nslt300_kaggle.yaml "
        "--checkpoint /kaggle/working/outputs/fusion/gated-fusion-nslt300-sel31-ce-regions/best.pt "
        f"--split {split}"
    )


def create_report(
    *,
    project_root: Path,
    package_name: str,
    package_root: Path,
    zip_path: Path | None,
    zip_size_bytes: int | None,
    deleted_package_dir: bool,
    deleted_zip: bool,
    summary: dict[str, Any],
    skeleton_counts: dict[str, int],
    regions_counts: dict[str, int],
    copied_region_reports: list[str],
    verify_summary: dict[str, Any],
    requirement_report_path: Path,
) -> str:
    package_summary = summarize_directory(package_root)
    references = existing_reference_paths(project_root)
    zip_path_text = zip_path.as_posix() if zip_path is not None else "not created"
    zip_size_text = format_bytes(zip_size_bytes or 0) if zip_size_bytes is not None else "not created"
    pairing = verify_summary["pairing"]
    skeleton_inputs = summary["skeleton_inputs"]
    regions_inputs = summary["regions_inputs"]
    region_reports_text = ", ".join(copied_region_reports) if copied_region_reports else "None"
    train_command = build_train_command(package_name)
    eval_val_command = build_eval_command(package_name, "val")
    eval_test_command = build_eval_command(package_name, "test")

    sections = [
        "# Gated Fusion NSLT300 Kaggle Package Report",
        "",
        "## 1. Muc tieu",
        "Tao package Kaggle-ready `wlasl-nslt300-gated-fusion-ready` de train va evaluate Gated Feature Fusion tren NSLT300.",
        "",
        "## 2. Boi canh",
        "Package duoc xay dung bang cach tham khao flow NSLT100 trong repo, nhung toan bo checkpoint/config/data duoc giu dung cho NSLT300.",
        f"- requirement report: `{requirement_report_path.as_posix()}`",
        f"- old package folder deleted with --clean: `{'yes' if deleted_package_dir else 'no'}`",
        f"- old zip deleted with --clean: `{'yes' if deleted_zip else 'no'}`",
        "",
        "## 3. Ket luan READY hay NOT READY",
        "Conclusion: READY",
        "",
        "## 4. Cac file NSLT100 da tham khao",
    ]
    if references:
        sections.extend(f"- {path}" for path in references)
    else:
        sections.append("- None found")

    sections.extend(
        [
            "",
            "## 5. Requirement check",
            f"- status: `{'READY' if summary['ready'] else 'NOT READY'}`",
            f"- gating config path: `{summary['gated_fusion_config']['path']}`",
            f"- created or refreshed by requirement check: `{'yes' if summary['gated_fusion_config'].get('created') else 'no'}`",
            f"- pairing minimum coverage: `{summary.get('pairing_min_coverage', 0.95):.3f}`",
            "",
            "## 6. Skeleton branch inputs",
            f"- source root used: `{skeleton_inputs['source_root']}`",
            f"- canonical tensor root: `{skeleton_inputs['canonical_tensor_root']}`",
            f"- fallback tensor root: `{skeleton_inputs['fallback_tensor_root']}`",
            f"- counts: train={skeleton_counts['train']}, val={skeleton_counts['val']}, test={skeleton_counts['test']}",
            f"- sample shape: `{tuple(SKELETON_EXPECTED_SHAPE)}`",
        ]
    )
    if skeleton_inputs["warnings"]:
        sections.extend(f"- warning: {item}" for item in skeleton_inputs["warnings"])

    sections.extend(
        [
            "",
            "## 7. Regions branch inputs",
            f"- source root used: `{regions_inputs['source_root']}`",
            f"- counts: train={regions_counts['train']}, val={regions_counts['val']}, test={regions_counts['test']}",
            f"- sample shape: `{tuple(REGIONS_EXPECTED_SHAPE)}`",
            f"- copied reports: {region_reports_text}",
            "",
            "## 8. Skeleton checkpoint",
            f"- selected checkpoint path: `{summary['selected_skeleton_checkpoint']}`",
            f"- packaged checkpoint path: `{(package_root / 'checkpoints' / 'skeleton' / 'best.pt').as_posix()}`",
            "",
            "## 9. Regions checkpoint",
            f"- selected checkpoint path: `{summary['selected_regions_checkpoint']}`",
            f"- packaged checkpoint path: `{(package_root / 'checkpoints' / 'regions' / 'best.pt').as_posix()}`",
            "",
            "## 10. Skeleton config",
            f"- selected config path: `{summary['selected_skeleton_config']}`",
            f"- packaged config path: `{(package_root / 'configs' / 'skeleton_config_resolved.yaml').as_posix()}`",
            "",
            "## 11. Regions config",
            f"- selected config path: `{summary['selected_regions_config']}`",
            f"- packaged config path: `{(package_root / 'configs' / 'regions_config_resolved.yaml').as_posix()}`",
            "",
            "## 12. Pairing check",
            f"- train matched_count: `{pairing['train']['matched_count']}`",
            f"- val matched_count: `{pairing['val']['matched_count']}`",
            f"- test matched_count: `{pairing['test']['matched_count']}`",
            f"- train label_mismatch: `{pairing['train']['label_mismatch']}`",
            f"- val label_mismatch: `{pairing['val']['label_mismatch']}`",
            f"- test label_mismatch: `{pairing['test']['label_mismatch']}`",
            f"- train gloss_mismatch: `{pairing['train'].get('gloss_mismatch', 0)}`",
            f"- val gloss_mismatch: `{pairing['val'].get('gloss_mismatch', 0)}`",
            f"- test gloss_mismatch: `{pairing['test'].get('gloss_mismatch', 0)}`",
            "",
            "## 13. Gated Fusion NSLT300 config da tao",
            f"- repo train config: `{(project_root / FUSION_CONFIG_PATH).resolve().as_posix()}`",
            f"- Kaggle config in package: `{(package_root / 'configs' / 'gated_feature_fusion_nslt300_kaggle.yaml').as_posix()}`",
            "",
            "## 14. Package structure neu READY",
            f"- package folder path: `{package_root.as_posix()}`",
            "- includes `README.md`, `metadata.json`, `configs/`, `checkpoints/`, `branch_inputs/`, `verify/`",
            f"- package file count: `{package_summary['file_count']}`",
            "",
            "## 15. Metadata",
            f"- metadata path: `{(package_root / 'metadata.json').as_posix()}`",
            "- metadata subset: `nslt300`",
            "- metadata num_classes: `300`",
            "",
            "## 16. Verify package",
            f"- verify result: `{str(verify_summary.get('status', '')).upper()}`",
            f"- verify summary path: `{(package_root / 'verify' / 'verify_summary.json').as_posix()}`",
            f"- verify script path: `{(package_root / 'verify' / 'verify_package.py').as_posix()}`",
            "",
            "## 17. Zip output neu co",
            f"- zip path: `{zip_path_text}`",
            f"- zip size: `{zip_size_text}`",
            "",
            "## 18. Cach upload Kaggle",
            "Upload the package folder or zip to a private Kaggle Dataset, then attach that dataset to your notebook.",
            "",
            "## 19. Cach train/evaluate tren Kaggle",
            f"- train: `{train_command}`",
            f"- evaluate val: `{eval_val_command}`",
            f"- evaluate test: `{eval_test_command}`",
            "",
            "## 20. Nhung gi khong dong goi",
            "- raw videos",
            "- W&B logs",
            "- intermediate checkpoints",
            "- old outputs",
            "- notebook cache",
            "- .git",
            "- __pycache__",
            "",
            "## 21. Luu y quan trong",
            "- `/kaggle/input` is read-only; write outputs to `/kaggle/working`.",
            "- Skeleton tensors were sourced from `graph_tensors/selected_31/nslt300` and repackaged into the expected `branch_inputs/skeleton/rtmw_l/tensors/nslt300/` layout.",
            "- This package is NSLT300-only; do not swap in NSLT100 checkpoints or configs.",
            "",
            "## 22. Ket luan",
            "Package verify passed, so the folder and zip are ready to upload to Kaggle for NSLT300 gated feature fusion.",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()

    output_root = resolve_path(args.output_root, root=project_root)
    fusion_config_path = resolve_path(args.fusion_config, root=project_root)
    report_path = resolve_path(args.report_path, root=project_root)
    requirement_report_path = resolve_path(args.requirement_report_path, root=project_root)
    package_name = str(args.package_name).strip()
    if not package_name:
        raise PackagingError("package_name must not be empty.")

    package_root = output_root / package_name
    zip_path = output_root / f"{package_name}.zip"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = run_requirement_audit(project_root=project_root, report_path=requirement_report_path)
    if not summary["ready"]:
        raise PackagingError(
            "Requirement check returned NOT READY. Packaging stopped. "
            f"See {requirement_report_path.as_posix()} for details."
        )

    deleted_package_dir = False
    deleted_zip = False
    if args.clean:
        deleted_package_dir = remove_path(package_root)
        deleted_zip = remove_path(zip_path)
    elif package_root.exists() or zip_path.exists():
        raise PackagingError(
            "Package output already exists. Re-run with --clean to delete the old package folder and zip first."
        )

    ensure_exists(fusion_config_path, "gated fusion NSLT300 config")
    base_fusion_config = read_yaml(fusion_config_path)

    skeleton_manifest_dir = project_root / "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests"
    regions_manifest_dir = project_root / "data/datasets/WLASL/branch_inputs/regions/rtmw_l/manifests"
    skeleton_tensor_root = Path(summary["skeleton_inputs"]["source_root"])
    regions_tensor_root = Path(summary["regions_inputs"]["source_root"])
    regions_reports_root = Path(summary["regions_inputs"]["branch_root"]) / "reports"

    package_root.mkdir(parents=True, exist_ok=False)

    skeleton_counts = copy_skeleton_branch(
        manifest_dir=skeleton_manifest_dir,
        tensor_root=skeleton_tensor_root,
        package_root=package_root,
    )
    regions_counts, copied_region_reports = copy_regions_branch(
        manifest_dir=regions_manifest_dir,
        tensor_root=regions_tensor_root,
        reports_root=regions_reports_root,
        package_root=package_root,
    )

    copy_file(Path(summary["selected_skeleton_checkpoint"]), package_root / "checkpoints" / "skeleton" / "best.pt")
    copy_file(Path(summary["selected_regions_checkpoint"]), package_root / "checkpoints" / "regions" / "best.pt")
    copy_file(Path(summary["selected_skeleton_config"]), package_root / "configs" / "skeleton_config_resolved.yaml")
    copy_file(Path(summary["selected_regions_config"]), package_root / "configs" / "regions_config_resolved.yaml")

    kaggle_config = create_kaggle_config(base_config=base_fusion_config, package_name=package_name)
    write_yaml(package_root / "configs" / "gated_feature_fusion_nslt300_kaggle.yaml", kaggle_config)
    write_json(
        package_root / "metadata.json",
        create_metadata(
            package_name=package_name,
            skeleton_counts=skeleton_counts,
            regions_counts=regions_counts,
            kaggle_config=kaggle_config,
        ),
    )
    (package_root / "README.md").write_text(create_readme(package_name), encoding="utf-8")
    write_verify_script(package_root / "verify" / "verify_package.py")

    verify_summary = run_verify(package_root, sample_checks=max(1, int(args.sample_checks)))

    zip_size_bytes: int | None = None
    if args.create_zip:
        zip_size_bytes = zip_directory(package_root, zip_path, prefix=package_name)

    report_text = create_report(
        project_root=project_root,
        package_name=package_name,
        package_root=package_root,
        zip_path=zip_path if args.create_zip else None,
        zip_size_bytes=zip_size_bytes,
        deleted_package_dir=deleted_package_dir,
        deleted_zip=deleted_zip,
        summary=summary,
        skeleton_counts=skeleton_counts,
        regions_counts=regions_counts,
        copied_region_reports=copied_region_reports,
        verify_summary=verify_summary,
        requirement_report_path=requirement_report_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    package_summary = summarize_directory(package_root)
    print(f"requirement report: {requirement_report_path.as_posix()}")
    print(f"selected skeleton checkpoint: {summary['selected_skeleton_checkpoint']}")
    print(f"selected regions checkpoint: {summary['selected_regions_checkpoint']}")
    print(f"selected skeleton config: {summary['selected_skeleton_config']}")
    print(f"selected regions config: {summary['selected_regions_config']}")
    print(f"package path: {package_root.as_posix()}")
    print(f"zip path: {zip_path.as_posix() if args.create_zip else 'not created'}")
    print(f"zip size: {format_bytes(zip_size_bytes or 0) if args.create_zip else 'not created'}")
    print(f"total file count: {package_summary['file_count']}")
    print(f"verify result: {str(verify_summary.get('status', '')).upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
