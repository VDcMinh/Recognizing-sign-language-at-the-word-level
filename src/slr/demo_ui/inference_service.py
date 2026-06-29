"""Inference service backing the local React demo UI."""

from __future__ import annotations

import csv
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from slr.branches.fusion.build import build_gated_feature_fusion_from_config
from slr.branches.fusion.train import load_fusion_checkpoint, select_device as select_fusion_device
from slr.branches.regions.build_crops import (
    load_config as load_regions_build_config,
    process_sample as process_regions_sample,
    resolve_paths as resolve_regions_paths,
)
from slr.branches.regions.models import build_region_model
from slr.branches.regions.transforms import apply_region_dataset_normalization
from slr.branches.skeleton.build_inputs import (
    load_config as load_skeleton_build_config,
    process_sample as process_skeleton_sample,
)
from slr.branches.skeleton.train import (
    build_graph_and_model,
    select_device as select_skeleton_device,
)
from slr.pose.extract_rtmw import (
    load_config as load_pose_config,
    process_sample as process_pose_sample,
    setup_pose_model,
)
from slr.registry import load_registry
from slr.registry.schema import LoadedRegistry, ModelRecord
from slr.training.checkpointing import load_checkpoint
from slr.utils.io import ensure_dir, read_json, read_yaml, write_json
from slr.data.standardize_videos import (
    load_config as load_standardize_config,
    standardize_one_sample,
)
from slr.demo_ui import settings


@dataclass
class PreparedSample:
    """Artifacts produced while preparing one uploaded video."""

    job_id: str
    branch: str
    subset: str
    upload_path: Path
    standardized: dict[str, Any]
    pose: dict[str, Any]
    skeleton: dict[str, Any]
    regions: dict[str, Any]


@dataclass
class ModelBundle:
    """Loaded model plus metadata required for inference."""

    branch: str
    subset: str
    record: ModelRecord
    model: torch.nn.Module
    device: torch.device
    label_map: dict[str, Any]
    top_k: int
    region_normalize_config: dict[str, Any] | None = None


def _resolve_project_path(project_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def _safe_name(filename: str) -> str:
    text = Path(filename or "upload.mp4").name.strip()
    return text or "upload.mp4"


def _relative_to_project(project_root: Path, path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return resolved.as_posix()


def _prediction_rows(probabilities: np.ndarray, label_map: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    id_to_gloss = dict(label_map.get("id_to_gloss", {}))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    rows: list[dict[str, Any]] = []
    for class_id in top_indices:
        rows.append(
            {
                "rank": int(len(rows) + 1),
                "class_id": int(class_id),
                "gloss": str(id_to_gloss.get(str(int(class_id)), f"class_{int(class_id)}")),
                "confidence": float(probabilities[int(class_id)]),
            }
        )
    return rows


class DemoInferenceService:
    """Local upload-to-prediction service for the React demo UI."""

    def __init__(self) -> None:
        self.project_root = Path.cwd().resolve()
        self.active_subset = settings.validate_active_subset()
        self.registry = self._load_registry()
        self._model_cache: dict[tuple[str, str], ModelBundle] = {}
        self._pose_runtime: tuple[Any | None, dict[str, Any]] | None = None
        self._lock = threading.Lock()
        ensure_dir(settings.RUNTIME_ROOT)

    def _load_registry(self) -> LoadedRegistry:
        registry_path = _resolve_project_path(self.project_root, str(settings.REGISTRY_PATH))
        if registry_path is None:
            raise FileNotFoundError("Registry path is not configured.")
        return load_registry(registry_path, validate=False)

    def _get_record(self, *, subset: str, branch: str) -> ModelRecord:
        for record in self.registry.models:
            if (
                record.identity.subset == subset
                and record.identity.branch == branch
                and record.identity.status == "ready"
            ):
                return record
        raise KeyError(f"No ready model found for subset={subset!r}, branch={branch!r}.")

    def _get_pose_backend(self) -> tuple[Any | None, dict[str, Any]]:
        with self._lock:
            if self._pose_runtime is not None:
                return self._pose_runtime
            config = load_pose_config(Path("configs/preprocessing/pose/pose_rtmw_l.yaml"), subset_override=self.active_subset)
            self._pose_runtime = setup_pose_model(config)
            return self._pose_runtime

    def describe(self) -> dict[str, Any]:
        branches: list[dict[str, Any]] = []
        for branch in settings.SUPPORTED_BRANCHES:
            record = self._get_record(subset=self.active_subset, branch=branch)
            branches.append(
                {
                    "id": record.identity.branch,
                    "label": record.identity.display_name,
                    "branch": record.identity.branch,
                    "registry_id": record.identity.id,
                    "status": record.identity.status,
                    "top_k": int(record.inference.get("top_k", settings.DEFAULT_TOP_K)),
                }
            )

        pose_backend, pose_state = self._get_pose_backend()
        return {
            "active_subset": self.active_subset,
            "supported_subsets": list(settings.SUPPORTED_SUBSETS),
            "branches": branches,
            "pose_backend_ready": bool(pose_backend is not None),
            "pose_backend_error": str(pose_state.get("error", "")),
        }

    def _build_job_dir(self) -> tuple[str, Path]:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        job_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        job_dir = ensure_dir(settings.RUNTIME_ROOT / job_id)
        return job_id, job_dir

    def _save_upload(self, *, job_dir: Path, filename: str, content: bytes) -> Path:
        safe_name = _safe_name(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in settings.SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension {extension!r}. "
                f"Expected one of {settings.SUPPORTED_VIDEO_EXTENSIONS}."
            )
        target = job_dir / "upload" / safe_name
        ensure_dir(target.parent)
        target.write_bytes(content)
        return target

    def _prepare_standardized_sample(self, *, upload_path: Path, subset: str, job_dir: Path) -> dict[str, Any]:
        config = load_standardize_config(
            Path(f"configs/preprocessing/standardize/standardize_{subset}.yaml"),
            subset_override=subset,
        )
        config["standardization"]["overwrite"] = True
        config["standardization"]["save_frames"] = True
        config["standardization"]["save_video"] = False

        sample_row = pd.Series(
            {
                "instance_uid": "ui:upload",
                "sample_id": "input",
                "video_id": "input",
                "gloss": "",
                "class_id": -1,
                "split": "demo",
                "video_path": str(upload_path),
                "bbox": "",
                "notes": "ui_upload",
                "is_present_locally": True,
            }
        )
        result = standardize_one_sample(
            row=sample_row,
            subset=subset,
            config=config,
            split_frames_root=job_dir / "standardized" / "frames" / subset / "demo",
            split_videos_root=job_dir / "standardized" / "videos" / subset / "demo",
            dry_run=False,
        )
        if str(result.get("status")) != "ok":
            raise RuntimeError(f"Standardization failed: {result.get('error_message') or result.get('status')}")
        return result

    def _prepare_pose_sample(self, *, standardized: dict[str, Any], subset: str, job_dir: Path) -> dict[str, Any]:
        config = load_pose_config(
            Path("configs/preprocessing/pose/pose_rtmw_l.yaml"),
            subset_override=subset,
        )
        inferencer, model_state = self._get_pose_backend()
        result = process_pose_sample(
            row=pd.Series(standardized),
            inferencer=inferencer,
            model_state=model_state,
            config=config,
            split_pose_root=job_dir / "pose" / "wholebody_133" / subset / "demo",
            dry_run=False,
        )
        if str(result.get("status")) != "ok":
            raise RuntimeError(f"Pose extraction failed: {result.get('error_message') or result.get('status')}")
        return result

    def _read_confidence_scale(self, subset: str) -> float:
        manifest_path = self.project_root / "data" / "datasets" / "WLASL" / "branch_inputs" / "skeleton" / "rtmw_l" / "manifests" / f"{subset}_selected_31_train.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Confidence-scale source manifest does not exist: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Confidence-scale source manifest is empty: {manifest_path}")
        return float(first_row["confidence_scale"])

    def _prepare_skeleton_sample(self, *, pose: dict[str, Any], subset: str, job_dir: Path) -> dict[str, Any]:
        config = load_skeleton_build_config(
            Path(f"configs/build_inputs/skeleton/{subset}/selected_31.yaml"),
            subset_override=subset,
        )
        config["output"]["selected_root"] = job_dir / "skeleton" / "selected_31"
        config["output"]["normalized_root"] = job_dir / "skeleton" / "normalized" / "selected_31"
        config["output"]["graph_tensor_root"] = job_dir / "skeleton" / "graph_tensors" / "selected_31"
        config["options"]["overwrite"] = True
        config["options"]["save_selected"] = False
        config["options"]["save_normalized"] = False
        config["options"]["save_graph_tensor"] = True
        confidence_scale = self._read_confidence_scale(subset)
        result, _ = process_skeleton_sample(
            row=pd.Series(pose),
            config=config,
            confidence_scale=confidence_scale,
            dry_run=False,
        )
        if str(result.get("status")) != "ok":
            raise RuntimeError(f"Skeleton input build failed: {result.get('error_message') or result.get('status')}")
        return result

    def _prepare_regions_sample(
        self,
        *,
        standardized: dict[str, Any],
        pose: dict[str, Any],
        subset: str,
        job_dir: Path,
    ) -> dict[str, Any]:
        config = load_regions_build_config(
            Path(f"configs/preprocessing/regions/region_crops_{subset}.yaml"),
            subset_override=subset,
        )
        config["input"]["standardized_frames_root"] = job_dir / "standardized" / "frames"
        config["input"]["pose_root"] = job_dir / "pose"
        config["input"]["pose_backend_root"] = job_dir / "pose"
        config["output"]["crops_root"] = job_dir / "regions" / "crops"
        config["output"]["tensors_root"] = job_dir / "regions" / "tensors"
        config["output"]["previews_root"] = job_dir / "regions" / "previews"
        config["output"]["manifests_root"] = job_dir / "regions" / "manifests"
        config["output"]["reports_root"] = job_dir / "regions" / "reports"
        config["output"]["logs_root"] = job_dir / "regions" / "logs"
        config["output"]["metadata_path"] = job_dir / "regions" / "metadata.json"
        config["options"]["overwrite"] = True
        config["options"]["save_crops"] = False
        config["options"]["save_tensors"] = True
        config["options"]["save_previews"] = True
        paths = resolve_regions_paths(config)
        result, _ = process_regions_sample(
            standardized_row=pd.Series(standardized),
            pose_row=pd.Series(pose),
            config=config,
            paths=paths,
            dry_run=False,
        )
        if str(result.get("status")) != "ok":
            raise RuntimeError(f"Region input build failed: {result.get('error_message') or result.get('status')}")
        return result

    def _prepare_sample(self, *, branch: str, filename: str, content: bytes) -> PreparedSample:
        subset = self.active_subset
        job_id, job_dir = self._build_job_dir()
        upload_path = self._save_upload(job_dir=job_dir, filename=filename, content=content)
        standardized = self._prepare_standardized_sample(upload_path=upload_path, subset=subset, job_dir=job_dir)
        pose = self._prepare_pose_sample(standardized=standardized, subset=subset, job_dir=job_dir)
        skeleton = self._prepare_skeleton_sample(pose=pose, subset=subset, job_dir=job_dir)
        regions = self._prepare_regions_sample(
            standardized=standardized,
            pose=pose,
            subset=subset,
            job_dir=job_dir,
        )
        metadata = {
            "job_id": job_id,
            "branch": branch,
            "subset": subset,
            "upload_path": _relative_to_project(self.project_root, upload_path),
            "standardized": standardized,
            "pose": pose,
            "skeleton": skeleton,
            "regions": regions,
        }
        write_json(metadata, job_dir / "job_summary.json")
        return PreparedSample(
            job_id=job_id,
            branch=branch,
            subset=subset,
            upload_path=upload_path,
            standardized=standardized,
            pose=pose,
            skeleton=skeleton,
            regions=regions,
        )

    def _load_label_map(self, record: ModelRecord) -> dict[str, Any]:
        class_map_path = _resolve_project_path(self.project_root, record.artifacts.class_map.local_path)
        if class_map_path is None:
            raise FileNotFoundError(f"Model {record.identity.id} is missing artifacts.class_map.")
        return dict(read_json(class_map_path))

    def _build_fusion_runtime_config(self, *, subset: str) -> dict[str, Any]:
        fusion_record = self._get_record(subset=subset, branch="fusion")
        skeleton_record = self._get_record(subset=subset, branch="skeleton")
        regions_record = self._get_record(subset=subset, branch="regions")
        return {
            "dataset": {
                "subset": subset,
                "num_classes": int(fusion_record.identity.num_classes),
                "skeleton": {
                    "keypoint_set": str(skeleton_record.input.get("keypoint_set", "selected_31")),
                    "expected_shape": list(skeleton_record.input.get("expected_shape", [3, 150, 31, 1])),
                },
                "regions": {
                    "expected_shape": list(regions_record.input.get("expected_shape", [3, 3, 64, 112, 112])),
                    "region_order": list(regions_record.input.get("active_regions", ["left_hand", "right_hand", "face"])),
                    "active_regions": list(regions_record.input.get("active_regions", ["left_hand", "right_hand", "face"])),
                    "normalize": {"type": "imagenet"},
                },
            },
            "runtime": {
                "device": "auto",
            },
            "train": {
                "device": "auto",
            },
            "skeleton_branch": {
                "config_path": str(_resolve_project_path(self.project_root, skeleton_record.artifacts.resolved_config.local_path)),
                "checkpoint_path": str(_resolve_project_path(self.project_root, skeleton_record.artifacts.checkpoint.local_path)),
                "graph": {
                    "layout": str(skeleton_record.input.get("keypoint_set", "selected_31")),
                    "strategy": "spatial",
                    "add_self_links": True,
                    "normalize_adjacency": True,
                },
            },
            "regions_branch": {
                "config_path": str(_resolve_project_path(self.project_root, regions_record.artifacts.resolved_config.local_path)),
                "checkpoint_path": str(_resolve_project_path(self.project_root, regions_record.artifacts.checkpoint.local_path)),
            },
            "fusion_model": {
                "name": str(fusion_record.model.get("name", "gated_feature_fusion")),
                "hidden_dim": int(fusion_record.model.get("hidden_dim", 256)),
                "proj_dropout": 0.2,
                "classifier_dropout": 0.5,
                "freeze_skeleton": bool(fusion_record.model.get("freeze_skeleton", True)),
                "freeze_regions": bool(fusion_record.model.get("freeze_regions", True)),
            },
        }

    def _load_model_bundle(self, *, branch: str, subset: str) -> ModelBundle:
        cache_key = (branch, subset)
        with self._lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached

        record = self._get_record(subset=subset, branch=branch)
        checkpoint_path = _resolve_project_path(self.project_root, record.artifacts.checkpoint.local_path)
        config_path = _resolve_project_path(self.project_root, record.artifacts.resolved_config.local_path)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist for {record.identity.id}.")
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Resolved config does not exist for {record.identity.id}.")

        if branch == "skeleton":
            device = select_skeleton_device("auto")
            config = read_yaml(config_path)
            _, model = build_graph_and_model(config, device=device)
            load_checkpoint(checkpoint_path, model, map_location=device)
            bundle = ModelBundle(
                branch=branch,
                subset=subset,
                record=record,
                model=model.eval(),
                device=device,
                label_map=self._load_label_map(record),
                top_k=int(record.inference.get("top_k", settings.DEFAULT_TOP_K)),
            )
        elif branch == "regions":
            from slr.branches.regions.train import select_device as select_regions_device

            class _NoopLogger:
                def info(self, *_args, **_kwargs) -> None:
                    return None

                def warning(self, *_args, **_kwargs) -> None:
                    return None

            device = select_regions_device("auto", logger=_NoopLogger())
            config = read_yaml(config_path)
            model_cfg = dict(config.get("model", {}))
            model_cfg["pretrained"] = False
            model = build_region_model(model_cfg).to(device)
            load_checkpoint(checkpoint_path, model, map_location=device)
            bundle = ModelBundle(
                branch=branch,
                subset=subset,
                record=record,
                model=model.eval(),
                device=device,
                label_map=self._load_label_map(record),
                top_k=int(record.inference.get("top_k", settings.DEFAULT_TOP_K)),
                region_normalize_config=dict(config.get("dataset", {}).get("normalize", {"type": "imagenet"})),
            )
        else:
            device = select_fusion_device("auto")
            runtime_config = self._build_fusion_runtime_config(subset=subset)
            model, _ = build_gated_feature_fusion_from_config(runtime_config, device=device)
            load_fusion_checkpoint(checkpoint_path, model, map_location=device)
            bundle = ModelBundle(
                branch=branch,
                subset=subset,
                record=record,
                model=model.eval(),
                device=device,
                label_map=self._load_label_map(record),
                top_k=int(record.inference.get("top_k", settings.DEFAULT_TOP_K)),
                region_normalize_config={"type": "imagenet"},
            )

        with self._lock:
            self._model_cache[cache_key] = bundle
        return bundle

    def _load_skeleton_tensor(self, sample: PreparedSample) -> torch.Tensor:
        tensor_path = Path(str(sample.skeleton["graph_tensor_path"]))
        with np.load(tensor_path, allow_pickle=False) as payload:
            data = np.asarray(payload["data"], dtype=np.float32)
        return torch.from_numpy(data).unsqueeze(0)

    def _load_regions_tensor(
        self,
        sample: PreparedSample,
        *,
        normalization_config: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tensor_path = Path(str(sample.regions["tensor_path"]))
        with np.load(tensor_path, allow_pickle=False) as payload:
            data = np.asarray(payload["data"], dtype=np.float32) / 255.0
            valid_mask = (
                np.asarray(payload["valid_mask"], dtype=np.float32)
                if "valid_mask" in payload
                else None
            )
        tensor = torch.from_numpy(data)
        tensor = apply_region_dataset_normalization(tensor, config=normalization_config)
        tensor = tensor.unsqueeze(0)
        valid_mask_tensor = None if valid_mask is None else torch.from_numpy(valid_mask).unsqueeze(0)
        return tensor, valid_mask_tensor

    def predict(self, *, branch: str, filename: str, content: bytes) -> dict[str, Any]:
        branch_name = str(branch).strip().lower()
        if branch_name not in settings.SUPPORTED_BRANCHES:
            raise ValueError(
                f"Unsupported branch {branch!r}. Expected one of {settings.SUPPORTED_BRANCHES}."
            )

        sample = self._prepare_sample(branch=branch_name, filename=filename, content=content)
        bundle = self._load_model_bundle(branch=branch_name, subset=sample.subset)

        with torch.no_grad():
            if branch_name == "skeleton":
                skeleton_tensor = self._load_skeleton_tensor(sample).to(bundle.device)
                logits = bundle.model(skeleton_tensor)
                extra: dict[str, Any] = {}
            elif branch_name == "regions":
                regions_tensor, valid_mask = self._load_regions_tensor(
                    sample,
                    normalization_config=bundle.region_normalize_config,
                )
                logits = bundle.model(
                    regions_tensor.to(bundle.device),
                    valid_mask=None if valid_mask is None else valid_mask.to(bundle.device),
                )
                extra = {}
            else:
                skeleton_tensor = self._load_skeleton_tensor(sample).to(bundle.device)
                regions_tensor, valid_mask = self._load_regions_tensor(
                    sample,
                    normalization_config=bundle.region_normalize_config,
                )
                logits, features = bundle.model(
                    skeleton_tensor,
                    regions_tensor.to(bundle.device),
                    return_features=True,
                    regions_valid_mask=None if valid_mask is None else valid_mask.to(bundle.device),
                )
                gate = features["gate"].detach().cpu()
                extra = {
                    "gate_mean": float(gate.mean().item()),
                    "gate_std": float(gate.std(unbiased=False).item()),
                }

        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        predictions = _prediction_rows(probabilities, bundle.label_map, bundle.top_k)
        top_prediction = predictions[0] if predictions else None

        return {
            "job_id": sample.job_id,
            "branch": branch_name,
            "branch_label": bundle.record.identity.display_name,
            "subset": sample.subset,
            "top_prediction": top_prediction,
            "predictions": predictions,
            "processing": {
                "upload_path": _relative_to_project(self.project_root, sample.upload_path),
                "frames_dir": _relative_to_project(self.project_root, sample.standardized.get("frames_dir")),
                "pose_path": _relative_to_project(self.project_root, sample.pose.get("pose_path")),
                "skeleton_tensor_path": _relative_to_project(self.project_root, sample.skeleton.get("graph_tensor_path")),
                "regions_tensor_path": _relative_to_project(self.project_root, sample.regions.get("tensor_path")),
                "regions_preview_path": _relative_to_project(self.project_root, sample.regions.get("preview_path")),
                "num_frames_standardized": int(sample.standardized.get("num_frames") or 0),
                "num_frames_pose": int(sample.pose.get("num_frames_pose") or 0),
            },
            "notes": {
                "standardized": str(sample.standardized.get("notes", "")),
                "pose": str(sample.pose.get("notes", "")),
                "skeleton": str(sample.skeleton.get("notes", "")),
                "regions": str(sample.regions.get("notes", "")),
            },
            "extra": extra,
        }
