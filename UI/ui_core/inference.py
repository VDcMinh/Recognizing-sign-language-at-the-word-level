from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from ui_core.model_registry import describe_model_assets, get_model_spec

from slr.branches.skeleton.graph import SkeletonGraph
from slr.branches.skeleton.models import build_skeleton_model
from slr.branches.skeleton.train import resolve_training_config, select_device
from slr.branches.skeleton.transforms import fix_sequence_length, to_graph_tensor_ctvm
from slr.pose.extract_rtmw import (
    DEFAULT_CONFIG_PATH as DEFAULT_POSE_CONFIG_PATH,
    extract_keypoints_from_result,
    load_config as load_pose_config,
    select_primary_person,
    setup_pose_model,
)
from slr.pose.keypoint_selection import select_keypoints
from slr.pose.keypoint_selection import get_keypoint_indices
from slr.pose.pose_normalization import (
    normalize_confidence,
    normalize_xy_to_minus1_1,
    sanitize_non_finite_keypoints,
)
from slr.training.checkpointing import load_checkpoint


UI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UI_ROOT.parent
CACHE_ROOT = UI_ROOT / "outputs" / "cache"

DEMO_MOCK_MODE = os.getenv("UI_DEMO_MOCK", "1").strip() != "0"
REAL_INFERENCE_ERROR = (
    "Real inference pipeline is not implemented yet. Please connect "
    "video-to-keypoint preprocessing and model inference."
)
FUSION_REAL_INFERENCE_ERROR = (
    "Real fusion inference is not implemented yet. Skeleton real inference is available, "
    "but Skeleton + Fusion still needs raw-video regions preprocessing and fusion execution."
)
FALLBACK_VOCABULARY = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "book",
    "drink",
    "school",
    "mother",
    "father",
]


class RealInferenceNotImplementedError(NotImplementedError):
    """Raised when the repo has not been connected to raw-video inference yet."""


def _default_skeleton_args() -> SimpleNamespace:
    return SimpleNamespace(
        run_name=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        dropout=None,
        device=None,
        seed=None,
        no_wandb=True,
        wandb_project=None,
        wandb_entity=None,
        output_root=None,
        num_workers=None,
        limit_train=None,
        limit_val=None,
        limit_test=None,
        dry_run=False,
    )


def _empty_keypoint_frame(num_keypoints: int = 133) -> np.ndarray:
    frame = np.zeros((num_keypoints, 3), dtype=np.float32)
    frame[:, :2] = np.nan
    return frame


def is_mock_mode_enabled() -> bool:
    return DEMO_MOCK_MODE


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _candidate_config_paths(model_name: str) -> list[Path]:
    spec = get_model_spec(model_name)
    paths: list[Path] = []
    if spec.config_path is not None:
        paths.append(spec.config_path)
    if spec.fallback_train_config is not None:
        paths.append(spec.fallback_train_config)
    return paths


def load_runtime_config(model_name: str) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    if spec.model_type == "skeleton":
        for path in _candidate_config_paths(model_name):
            if path.exists():
                return resolve_training_config(path, _default_skeleton_args())
        return {}

    for path in _candidate_config_paths(model_name):
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def load_class_vocabulary(model_name: str) -> list[str]:
    spec = get_model_spec(model_name)
    label_source = spec.label_source
    if not label_source.exists():
        return list(FALLBACK_VOCABULARY)

    label_map: dict[int, str] = {}
    with label_source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                class_id = int(row.get("class_id", "").strip())
            except ValueError:
                continue
            gloss = (row.get("gloss") or "").strip()
            if gloss and class_id not in label_map:
                label_map[class_id] = gloss

    if not label_map:
        return list(FALLBACK_VOCABULARY)
    return [label_map[index] for index in sorted(label_map)]


@lru_cache(maxsize=1)
def _load_pose_runtime() -> dict[str, Any]:
    pose_config_path = DEFAULT_POSE_CONFIG_PATH
    if not pose_config_path.is_absolute():
        pose_config_path = REPO_ROOT / pose_config_path

    with _pushd(REPO_ROOT):
        pose_config = load_pose_config(pose_config_path)
        inferencer, model_state = setup_pose_model(pose_config)
    if inferencer is None:
        raise RuntimeError(
            "Could not initialize RTMW-l pose inference. "
            f"Details: {model_state.get('error', 'unknown error')}"
        )
    return {
        "config": pose_config,
        "inferencer": inferencer,
        "model_state": model_state,
    }


@lru_cache(maxsize=2)
def _load_skeleton_runtime(model_name: str) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    if spec.model_type != "skeleton":
        raise RealInferenceNotImplementedError(FUSION_REAL_INFERENCE_ERROR)
    if spec.checkpoint_path is None or not spec.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {spec.checkpoint_path}"
        )

    config = load_runtime_config(model_name)
    if not config:
        raise FileNotFoundError("No usable skeleton config could be loaded for real inference.")

    device = select_device(str(config["train"]["device"]))
    graph_cfg = config["graph"]
    graph = SkeletonGraph(
        layout=str(graph_cfg["layout"]),
        strategy=str(graph_cfg["strategy"]),
        normalize=bool(graph_cfg["normalize_adjacency"]),
        add_self_links=bool(graph_cfg["add_self_links"]),
    )
    model = build_skeleton_model(config["model"], graph).to(device)
    checkpoint_payload = load_checkpoint(spec.checkpoint_path, model, map_location=device)
    model.eval()

    return {
        "config": config,
        "device": device,
        "graph": graph,
        "model": model,
        "checkpoint_payload": checkpoint_payload,
    }


def _labels_from_checkpoint_payload(
    checkpoint_payload: dict[str, Any],
    fallback_labels: list[str],
) -> list[str]:
    raw_map = checkpoint_payload.get("class_id_to_gloss", {})
    if not isinstance(raw_map, dict) or not raw_map:
        return fallback_labels

    normalized: dict[int, str] = {}
    for key, value in raw_map.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        gloss = str(value).strip()
        if gloss:
            normalized[class_id] = gloss
    if not normalized:
        return fallback_labels
    return [normalized[index] for index in sorted(normalized)]


def _load_confidence_scale(config: dict[str, Any]) -> float:
    manifests = config.get("dataset", {}).get("manifests", {})
    train_manifest = manifests.get("train")
    if not train_manifest:
        return 1.0

    manifest_path = Path(train_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        return 1.0

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                status = str(row.get("status", "")).strip().lower()
                scale = float(row.get("confidence_scale", ""))
            except (TypeError, ValueError):
                continue
            if status == "ok" and np.isfinite(scale) and scale > 0:
                return scale
    return 1.0


def _extract_resized_frames(
    video_path: str,
    *,
    target_num_frames: int,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(
        prefix="real_infer_frames_",
        dir=CACHE_ROOT,
    )
    frames_dir = Path(temp_dir.name)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        temp_dir.cleanup()
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    frame_paths: list[Path] = []
    frame_count = 0
    try:
        while frame_count < target_num_frames:
            success, frame = capture.read()
            if not success:
                break
            resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            frame_path = frames_dir / f"frame_{frame_count:04d}.jpg"
            if not cv2.imwrite(frame_path.as_posix(), resized):
                raise RuntimeError(f"Could not write temporary frame: {frame_path}")
            frame_paths.append(frame_path)
            frame_count += 1
    finally:
        capture.release()

    if not frame_paths:
        temp_dir.cleanup()
        raise RuntimeError("No readable frames were extracted from the uploaded video.")

    return {
        "temp_dir": temp_dir,
        "frames_dir": frames_dir,
        "frame_paths": frame_paths,
        "frame_count": frame_count,
        "output_width": output_width,
        "output_height": output_height,
    }


def build_skeleton_input_from_video(video_path: str, config: dict[str, Any]):
    if not config:
        raise RuntimeError("Skeleton config could not be loaded for real inference.")

    pose_runtime = _load_pose_runtime()
    pose_config = pose_runtime["config"]
    inferencer = pose_runtime["inferencer"]

    dataset_cfg = config["dataset"]
    sequence_target = int(dataset_cfg["expected_shape"][1])
    keypoint_set = str(dataset_cfg["keypoint_set"])
    num_persons = int(dataset_cfg["expected_shape"][3])
    output_width = int(pose_config["pose"]["input_width"])
    output_height = int(pose_config["pose"]["input_height"])
    confidence_scale = _load_confidence_scale(config)

    frame_payload = _extract_resized_frames(
        video_path,
        target_num_frames=sequence_target,
        output_width=output_width,
        output_height=output_height,
    )

    keypoint_frames: list[np.ndarray] = []
    try:
        inference_stream = inferencer(
            [str(path) for path in frame_payload["frame_paths"]],
            batch_size=int(pose_config["pose"]["batch_size"]),
            show=False,
            return_vis=False,
        )
        for batch_result in inference_stream:
            batch_predictions = batch_result.get("predictions", [])
            for instance_predictions in batch_predictions:
                primary_person, _ = select_primary_person(instance_predictions)
                if primary_person is None:
                    keypoint_frames.append(_empty_keypoint_frame())
                    continue
                pose_frame, _ = extract_keypoints_from_result(primary_person)
                keypoint_frames.append(pose_frame)
    except Exception:
        frame_payload["temp_dir"].cleanup()
        raise

    if not keypoint_frames:
        frame_payload["temp_dir"].cleanup()
        raise RuntimeError("Pose extraction returned no usable keypoint frames.")

    keypoints = np.stack(keypoint_frames, axis=0).astype(np.float32)
    selected = select_keypoints(keypoints, get_keypoint_indices(keypoint_set))
    normalized, _ = normalize_xy_to_minus1_1(
        selected,
        image_width=output_width,
        image_height=output_height,
        clip=True,
    )
    normalized = normalize_confidence(
        normalized,
        confidence_scale=confidence_scale,
        clip_min=0.0,
        clip_max=1.0,
    )
    sanitized, invalid_count = sanitize_non_finite_keypoints(normalized)
    fixed = fix_sequence_length(
        sanitized,
        target_num_frames=sequence_target,
        short_strategy="repeat",
        long_strategy="head",
    )
    graph_tensor = to_graph_tensor_ctvm(fixed, num_persons=num_persons)
    expected_shape = tuple(int(value) for value in dataset_cfg["expected_shape"])
    if tuple(graph_tensor.shape) != expected_shape:
        frame_payload["temp_dir"].cleanup()
        raise RuntimeError(
            f"Expected graph tensor shape {expected_shape}, got {tuple(graph_tensor.shape)}."
        )

    return {
        "graph_tensor": graph_tensor,
        "frame_count": int(frame_payload["frame_count"]),
        "keypoint_set": keypoint_set,
        "confidence_scale": float(confidence_scale),
        "invalid_value_count": int(invalid_count),
        "temp_dir": frame_payload["temp_dir"],
    }


def build_fusion_inputs_from_video(video_path: str, config: dict[str, Any]):
    _ = video_path, config
    raise RealInferenceNotImplementedError(FUSION_REAL_INFERENCE_ERROR)


def prepare_model_input(video_path: str, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    if is_mock_mode_enabled():
        return {
            "mode": "mock",
            "video_path": video_path,
            "model_type": spec.model_type,
            "runtime_mode": spec.runtime_mode,
        }

    if spec.model_type == "skeleton":
        payload = build_skeleton_input_from_video(video_path, config)
    elif spec.model_type == "fusion":
        payload = build_fusion_inputs_from_video(video_path, config)
    else:
        raise ValueError(f"Unsupported model type: {spec.model_type}")

    return {
        "mode": "real",
        "video_path": video_path,
        "payload": payload,
        "model_type": spec.model_type,
        "runtime_mode": spec.runtime_mode,
    }


def run_inference(video_path: str, model_name: str, model_input: dict[str, Any]) -> dict[str, Any]:
    labels = load_class_vocabulary(model_name)
    if is_mock_mode_enabled():
        return _run_mock_inference(video_path, model_name, labels=labels)
    return _run_real_inference(video_path, model_name, model_input, labels=labels)


def _run_mock_inference(video_path: str, model_name: str, *, labels: list[str]) -> dict[str, Any]:
    if not labels:
        labels = list(FALLBACK_VOCABULARY)
    seed = int.from_bytes(
        hashlib.sha256(f"{video_path}|{model_name}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0.0, scale=0.7, size=len(labels))
    primary_index = seed % len(labels)
    secondary_index = (primary_index + 7) % len(labels)
    tertiary_index = (primary_index + 19) % len(labels)
    logits[primary_index] += 4.8
    logits[secondary_index] += 2.3
    logits[tertiary_index] += 1.2
    return {
        "backend": "mock",
        "labels": labels,
        "logits": logits.astype(np.float32),
        "mock_reason": "Demo/mock mode is enabled for the UI.",
    }


def _run_real_inference(
    video_path: str,
    model_name: str,
    model_input: dict[str, Any],
    *,
    labels: list[str],
) -> dict[str, Any]:
    _ = video_path
    payload = model_input.get("payload", {})
    temp_dir = payload.get("temp_dir")
    try:
        runtime = _load_skeleton_runtime(model_name)
        resolved_labels = _labels_from_checkpoint_payload(
            runtime["checkpoint_payload"],
            labels,
        )
        graph_tensor = np.asarray(payload["graph_tensor"], dtype=np.float32)
        input_tensor = torch.as_tensor(graph_tensor, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(runtime["device"], non_blocking=runtime["device"].type == "cuda")

        with torch.inference_mode():
            logits = runtime["model"](input_tensor)

        logits_np = logits.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        if len(resolved_labels) != int(logits_np.shape[0]):
            resolved_labels = labels
        if len(resolved_labels) != int(logits_np.shape[0]):
            resolved_labels = [f"class_{index}" for index in range(int(logits_np.shape[0]))]

        return {
            "backend": "real_skeleton",
            "labels": resolved_labels,
            "logits": logits_np,
            "frame_count": int(payload.get("frame_count", 0)),
            "keypoint_set": str(payload.get("keypoint_set", "")),
            "confidence_scale": float(payload.get("confidence_scale", 1.0)),
            "invalid_value_count": int(payload.get("invalid_value_count", 0)),
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def convert_logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float32)
    shifted = array - np.max(array)
    exp = np.exp(shifted)
    denominator = np.sum(exp)
    if denominator <= 0:
        return np.zeros_like(array, dtype=np.float32)
    return exp / denominator


def get_topk_predictions(
    probabilities: np.ndarray,
    labels: list[str],
    *,
    k: int = 5,
) -> list[dict[str, float | str]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length.")

    limit = min(k, len(labels))
    ranked_indices = np.argsort(probabilities)[::-1][:limit]
    results: list[dict[str, float | str]] = []
    for index in ranked_indices:
        results.append(
            {
                "word": str(labels[int(index)]),
                "prob": float(probabilities[int(index)]),
            }
        )
    return results


def get_mock_or_checkpoint_warning(model_name: str) -> str | None:
    if is_mock_mode_enabled():
        assets = describe_model_assets(model_name)
        if assets["missing_checkpoints"]:
            return "Checkpoint not found for selected model. Running in demo mock mode."
        return "Demo/mock mode is enabled. Predictions are simulated for UI validation."
    return None


def get_required_asset_error(model_name: str) -> str | None:
    assets = describe_model_assets(model_name)
    if assets["missing_checkpoints"]:
        return "Checkpoint not found. Please place best.pt in the required folder."
    if assets["missing_configs"]:
        return "Config not found for selected model. Please place the resolved YAML file in the required folder."
    return None
