from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(SRC_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from slr.registry import RegistryValidationError, get_model_record, load_registry


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class RegistryLoaderTests(unittest.TestCase):
    def test_load_repo_registry(self) -> None:
        registry = load_registry(REPO_ROOT / "model_registry" / "registry.yaml")
        self.assertEqual(registry.registry_version, 1)
        self.assertEqual(len(registry.models), 3)
        self.assertIn("skeleton_nslt1000_sel31_v1", registry.models_by_id)
        self.assertEqual(
            get_model_record(registry, "regions_nslt1000_face_hands_v1").identity.branch,
            "regions",
        )

    def test_duplicate_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_root = root / "model_registry"
            _write_yaml(
                registry_root / "registry.yaml",
                {
                    "registry_version": 1,
                    "default_model": "duplicate_v1",
                    "models": [
                        {
                            "id": "duplicate_v1",
                            "display_name": "Duplicate A",
                            "branch": "skeleton",
                            "model_type": "dummy",
                            "status": "ready",
                            "registry_file": "models/a/model.yaml",
                        },
                        {
                            "id": "duplicate_v1",
                            "display_name": "Duplicate B",
                            "branch": "regions",
                            "model_type": "dummy",
                            "status": "ready",
                            "registry_file": "models/b/model.yaml",
                        },
                    ],
                },
            )
            _write_yaml(
                registry_root / "models" / "a" / "model.yaml",
                _minimal_model_payload(
                    model_id="duplicate_v1",
                    display_name="Duplicate A",
                    branch="skeleton",
                ),
            )
            _write_yaml(
                registry_root / "models" / "b" / "model.yaml",
                _minimal_model_payload(
                    model_id="duplicate_v1",
                    display_name="Duplicate B",
                    branch="regions",
                    class_path="slr.branches.regions.models.region_resnet18_gru.RegionResNet18GRU",
                    input_payload={
                        "type": "regions",
                        "active_regions": ["left_hand", "right_hand", "face"],
                        "num_frames": 64,
                        "image_size": 112,
                    },
                ),
            )
            with self.assertRaises(RegistryValidationError):
                load_registry(registry_root / "registry.yaml")

    def test_missing_config_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_root = root / "model_registry"
            _write_yaml(
                registry_root / "registry.yaml",
                {
                    "registry_version": 1,
                    "default_model": "skeleton_test_v1",
                    "models": [
                        {
                            "id": "skeleton_test_v1",
                            "display_name": "Skeleton Test",
                            "branch": "skeleton",
                            "model_type": "stgcnpp",
                            "status": "ready",
                            "registry_file": "models/skeleton/model.yaml",
                        }
                    ],
                },
            )
            payload = _minimal_model_payload(
                model_id="skeleton_test_v1",
                display_name="Skeleton Test",
                branch="skeleton",
            )
            payload["artifacts"]["resolved_config"]["local_path"] = "missing/config.yaml"
            _write_yaml(registry_root / "models" / "skeleton" / "model.yaml", payload)
            with self.assertRaises(RegistryValidationError):
                load_registry(registry_root / "registry.yaml")

    def test_missing_checkpoint_is_allowed_for_incomplete_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_root = root / "model_registry"
            config_path = root / "configs" / "fusion.yaml"
            class_map_path = root / "label_map.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("experiment: {}\n", encoding="utf-8")
            class_map_path.write_text("{}\n", encoding="utf-8")

            _write_yaml(
                registry_root / "registry.yaml",
                {
                    "registry_version": 1,
                    "default_model": "fusion_test_v1",
                    "models": [
                        {
                            "id": "fusion_test_v1",
                            "display_name": "Fusion Test",
                            "branch": "fusion",
                            "model_type": "gated_feature_fusion",
                            "status": "incomplete",
                            "registry_file": "models/fusion/model.yaml",
                        },
                        {
                            "id": "skeleton_ready_v1",
                            "display_name": "Skeleton Ready",
                            "branch": "skeleton",
                            "model_type": "stgcnpp",
                            "status": "ready",
                            "registry_file": "models/skeleton/model.yaml",
                        },
                        {
                            "id": "regions_ready_v1",
                            "display_name": "Regions Ready",
                            "branch": "regions",
                            "model_type": "region_resnet18_gru",
                            "status": "ready",
                            "registry_file": "models/regions/model.yaml",
                        },
                    ],
                },
            )
            _write_yaml(
                registry_root / "models" / "skeleton" / "model.yaml",
                _minimal_model_payload(
                    model_id="skeleton_ready_v1",
                    display_name="Skeleton Ready",
                    branch="skeleton",
                    config_local_path="configs/fusion.yaml",
                    checkpoint_local_path=None,
                    status="ready",
                    class_map_local_path="label_map.json",
                ),
            )
            _write_yaml(
                registry_root / "models" / "regions" / "model.yaml",
                _minimal_model_payload(
                    model_id="regions_ready_v1",
                    display_name="Regions Ready",
                    branch="regions",
                    class_path="slr.branches.regions.models.region_resnet18_gru.RegionResNet18GRU",
                    input_payload={
                        "type": "regions",
                        "active_regions": ["left_hand", "right_hand", "face"],
                        "num_frames": 64,
                        "image_size": 112,
                    },
                    config_local_path="configs/fusion.yaml",
                    checkpoint_local_path=None,
                    status="ready",
                    class_map_local_path="label_map.json",
                ),
            )
            payload = {
                "schema_version": 1,
                "identity": {
                    "id": "fusion_test_v1",
                    "display_name": "Fusion Test",
                    "description": "",
                    "branch": "fusion",
                    "task": "isolated_sign_language_recognition",
                    "subset": "nslt1000",
                    "num_classes": 1000,
                    "status": "incomplete",
                },
                "model": {
                    "name": "gated_feature_fusion",
                    "class_path": "slr.branches.fusion.models.gated_feature_fusion.GatedFeatureFusion",
                    "hidden_dim": 256,
                    "skeleton_registry_id": "skeleton_ready_v1",
                    "regions_registry_id": "regions_ready_v1",
                },
                "input": {"type": "fusion"},
                "artifacts": {
                    "checkpoint": {"local_path": None, "remote_type": None, "remote_uri": None, "sha256": None},
                    "class_map": {"local_path": "label_map.json", "remote_type": None, "remote_uri": None, "sha256": None},
                    "resolved_config": {"local_path": "configs/fusion.yaml", "remote_type": None, "remote_uri": None, "sha256": None},
                },
                "inference": {"device": "auto"},
                "ui": {"enabled": True},
            }
            _write_yaml(registry_root / "models" / "fusion" / "model.yaml", payload)
            registry = load_registry(registry_root / "registry.yaml")
            self.assertEqual(registry.default_model, "fusion_test_v1")


def _minimal_model_payload(
    *,
    model_id: str,
    display_name: str,
    branch: str,
    class_path: str = "slr.branches.skeleton.models.stgcnpp.STGCNPP",
    input_payload: dict | None = None,
    config_local_path: str = "configs/fusion.yaml",
    checkpoint_local_path: str | None = None,
    class_map_local_path: str = "label_map.json",
    status: str = "ready",
) -> dict:
    if input_payload is None:
        input_payload = {"type": "skeleton", "keypoint_set": "selected_31"}
    return {
        "schema_version": 1,
        "identity": {
            "id": model_id,
            "display_name": display_name,
            "description": "",
            "branch": branch,
            "task": "isolated_sign_language_recognition",
            "subset": "nslt1000",
            "num_classes": 1000,
            "status": status,
        },
        "model": {
            "name": "test_model",
            "class_path": class_path,
        },
        "input": input_payload,
        "artifacts": {
            "checkpoint": {
                "local_path": checkpoint_local_path,
                "remote_type": None,
                "remote_uri": "wandb://placeholder" if checkpoint_local_path is None else None,
                "sha256": None,
            },
            "class_map": {
                "local_path": class_map_local_path,
                "remote_type": None,
                "remote_uri": None,
                "sha256": None,
            },
            "resolved_config": {
                "local_path": config_local_path,
                "remote_type": None,
                "remote_uri": None,
                "sha256": None,
            },
        },
        "inference": {"device": "auto"},
        "ui": {"enabled": True},
    }


if __name__ == "__main__":
    unittest.main()
