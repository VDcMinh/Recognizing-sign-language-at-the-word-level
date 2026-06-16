"""Materialize runtime Kaggle manifests from logical NSLT1000 union manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize runtime manifests from logical NSLT1000 incremental manifests."
    )
    parser.add_argument("--logical-manifest-root", type=Path, required=True)
    parser.add_argument("--nslt300-base-root", type=Path, required=True)
    parser.add_argument("--incremental-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _resolve_tensor_path(base_root: Path, incremental_root: Path, tensor_source: str, tensor_relpath: str) -> Path:
    rel = Path(str(tensor_relpath).replace("\\", "/"))
    if tensor_source == "nslt300_base":
        if str(rel).startswith("regions/rtmw_l/"):
            rel = Path(*rel.parts[2:])
        return (base_root / rel).resolve()
    if tensor_source == "nslt1000_incremental":
        if str(rel).startswith("regions/rtmw_l_incremental/"):
            rel = Path(*rel.parts[2:])
        return (incremental_root / rel).resolve()
    raise ValueError(f"Unsupported tensor_source: {tensor_source}")


def main() -> int:
    args = build_parser().parse_args()
    output_manifest_root = args.output_root / "manifests"
    output_manifest_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        logical_manifest = args.logical_manifest_root / f"nslt1000_{split}.csv"
        frame = pd.read_csv(logical_manifest)
        tensor_paths = []
        for _, row in frame.iterrows():
            resolved = _resolve_tensor_path(
                args.nslt300_base_root,
                args.incremental_root,
                str(row["tensor_source"]).strip(),
                str(row["tensor_relpath"]).strip(),
            )
            if not resolved.exists():
                raise FileNotFoundError(f"Resolved tensor path does not exist: {resolved}")
            tensor_paths.append(str(resolved).replace("\\", "/"))
        runtime = frame.copy()
        runtime["tensor_path"] = tensor_paths
        runtime["status"] = "ok"
        ordered_columns = [
            "sample_id",
            "video_id",
            "class_id",
            "gloss",
            "split",
            "tensor_path",
            "tensor_shape",
            "status",
            "region_order",
            "tensor_source",
        ]
        for column in ordered_columns:
            if column not in runtime.columns:
                runtime[column] = ""
        runtime.loc[:, ordered_columns].to_csv(output_manifest_root / f"nslt1000_{split}.csv", index=False, encoding="utf-8")
        summaries[split] = {"rows": int(len(runtime))}

    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(args.output_root).replace("\\", "/"),
                "splits": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
