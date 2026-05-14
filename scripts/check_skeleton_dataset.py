"""Sanity-check the skeleton graph tensor loader and topology."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slr.branches.skeleton.dataset import (
    SkeletonGraphDataset,
    load_skeleton_train_config,
    skeleton_collate_fn,
)
from slr.branches.skeleton.graph import SkeletonGraph


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check skeleton dataset loading and graph topology."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to one skeleton train config YAML file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Optional sample limit to keep the sanity check lightweight.",
    )
    return parser


def _format_shape(shape: tuple[int, ...]) -> str:
    """Render one tensor shape in a stable compact form."""

    return "(" + ", ".join(str(value) for value in shape) + ")"


def main() -> int:
    """Run the dataset and topology sanity checks."""

    parser = build_parser()
    args = parser.parse_args()

    config = load_skeleton_train_config(args.config)
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    graph_cfg = config["graph"]

    dataset = SkeletonGraphDataset.from_config(
        config,
        split=args.split,
        limit=args.limit,
    )

    class_ids = [record.class_id for record in dataset.records]
    print("== Dataset ==")
    print(f"config: {args.config}")
    print(f"manifest path: {dataset.manifest_path}")
    print(f"resolved samples: {len(dataset)}")
    print(f"keypoint_set: {dataset.keypoint_set}")
    print(f"expected_shape: {_format_shape(dataset.expected_shape)}")
    print(f"num_classes: {dataset.num_classes}")
    print(f"class_id min/max/nunique: {min(class_ids)} / {max(class_ids)} / {len(set(class_ids))}")
    print()

    sample_count = min(len(dataset), max(1, int(args.limit)))
    print("== Samples ==")
    for index in range(sample_count):
        sample = dataset[index]
        assert isinstance(sample, dict)
        sample_tensor = sample["data"]
        print(
            f"[{index}] sample_id={sample['sample_id']} gloss={sample['gloss']} "
            f"class_id={sample['class_id']} shape={tuple(sample_tensor.shape)} "
            f"min={float(sample_tensor.min()):.6f} max={float(sample_tensor.max()):.6f}"
        )
        assert tuple(sample_tensor.shape) == tuple(dataset.expected_shape)
        assert 0 <= int(sample["class_id"]) < int(dataset_cfg["num_classes"])
    print()

    batch_size = min(int(dataloader_cfg["batch_size"]), len(dataset))
    batch_size = max(1, batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(dataloader_cfg["pin_memory"]),
        collate_fn=skeleton_collate_fn,
    )
    batch = next(iter(loader))
    batch_data = batch["data"]
    batch_labels = batch["labels"]

    print("== Batch ==")
    print(f"batch data shape: {tuple(batch_data.shape)}")
    print(f"batch labels shape: {tuple(batch_labels.shape)}")
    print(f"batch labels: {batch_labels.tolist()}")
    print()

    graph = SkeletonGraph(
        layout=str(graph_cfg["layout"]),
        strategy=str(graph_cfg["strategy"]),
        normalize=bool(graph_cfg["normalize_adjacency"]),
        add_self_links=bool(graph_cfg["add_self_links"]),
    )
    adjacency = graph.adjacency()

    print("== Graph ==")
    print(f"layout: {graph.layout}")
    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {len(graph.edges)}")
    print(f"adjacency shape: {tuple(adjacency.shape)}")
    print(f"adjacency min/max: {float(adjacency.min()):.6f} / {float(adjacency.max()):.6f}")

    expected_v = int(dataset.expected_shape[2])
    assert batch_data.ndim == 5
    assert tuple(batch_data.shape[1:]) == tuple(dataset.expected_shape)
    assert tuple(batch_labels.shape) == (batch_data.shape[0],)
    assert graph.num_nodes == expected_v
    if graph.layout == "selected_27":
        assert adjacency.shape[-1] == 27 and adjacency.shape[-2] == 27
    if graph.layout == "selected_31":
        assert adjacency.shape[-1] == 31 and adjacency.shape[-2] == 31
    assert adjacency.shape[0] in {1, 3}

    print()
    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
