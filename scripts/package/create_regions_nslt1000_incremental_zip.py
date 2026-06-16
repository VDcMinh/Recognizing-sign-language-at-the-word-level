"""Create a verified ZIP archive for the NSLT1000 incremental regions package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REQUIRED_RELATIVE_FILES = (
    "metadata.json",
    "README.md",
    "manifests/logical/nslt1000_train.csv",
    "manifests/logical/nslt1000_val.csv",
    "manifests/logical/nslt1000_test.csv",
    "scripts/materialize_regions_nslt1000_kaggle_manifests.py",
)


@dataclass(frozen=True)
class SourceStats:
    files: tuple[Path, ...]
    file_count: int
    npz_count: int
    source_bytes: int
    logical_rows: dict[str, int]
    metadata_counts: dict[str, int]
    free_bytes_before: int


@dataclass(frozen=True)
class ZipVerifyStats:
    file_count: int
    npz_count: int
    logical_rows: dict[str, int]
    metadata_counts: dict[str, int]
    bad_member: str | None
    root_prefix: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a ZIP_STORED archive for the NSLT1000 incremental regions package."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--expected-npz-count", type=int, required=True)
    parser.add_argument("--expected-file-count", type=int, required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-val-rows", type=int, required=True)
    parser.add_argument("--expected-test-rows", type=int, required=True)
    parser.add_argument("--minimum-free-margin-gb", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _format_gib(num_bytes: int) -> float:
    return num_bytes / float(1024 ** 3)


def _assert_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Path escaped package root: {path}") from exc


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def _extract_metadata_counts(metadata: dict) -> dict[str, int]:
    counts_container = metadata.get("counts")
    if isinstance(counts_container, dict):
        total_nslt1000 = counts_container.get("total_nslt1000")
        reused_nslt300 = counts_container.get("reused_nslt300")
        incremental_new = counts_container.get("incremental_new")
    else:
        total_nslt1000 = metadata.get("total_nslt1000")
        reused_nslt300 = metadata.get("reused_nslt300")
        incremental_new = metadata.get("incremental_new")
    counts = {
        "total_nslt1000": total_nslt1000,
        "reused_nslt300": reused_nslt300,
        "incremental_new": incremental_new,
    }
    missing = [key for key, value in counts.items() if not isinstance(value, int)]
    if missing:
        raise RuntimeError(f"metadata.json missing required count keys: {missing}")
    return counts


def _iter_source_files(package_root: Path, output_zip: Path, partial_zip: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    pending = [package_root]
    while pending:
        current = pending.pop()
        for path in sorted(current.iterdir()):
            if path.is_symlink():
                raise RuntimeError(f"Symlinks are not allowed in package source: {path}")
            if path.is_dir():
                resolved_dir = path.resolve(strict=True)
                _assert_within_root(resolved_dir, package_root)
                pending.append(resolved_dir)
                continue
            resolved = path.resolve(strict=True)
            _assert_within_root(resolved, package_root)
            if resolved == output_zip:
                continue
            if resolved == partial_zip:
                continue
            files.append(resolved)
    return tuple(sorted(files))


def _collect_source_stats(args: argparse.Namespace) -> SourceStats:
    package_root = args.package_root
    output_zip = args.output_zip
    partial_zip = Path(str(output_zip) + ".partial")
    for relative_path in REQUIRED_RELATIVE_FILES:
        full_path = package_root / relative_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing required source file: {full_path}")

    logical_rows = {
        "train": _count_csv_rows(package_root / "manifests" / "logical" / "nslt1000_train.csv"),
        "val": _count_csv_rows(package_root / "manifests" / "logical" / "nslt1000_val.csv"),
        "test": _count_csv_rows(package_root / "manifests" / "logical" / "nslt1000_test.csv"),
    }
    expected_rows = {
        "train": args.expected_train_rows,
        "val": args.expected_val_rows,
        "test": args.expected_test_rows,
    }
    for split, expected_value in expected_rows.items():
        actual_value = logical_rows[split]
        if actual_value != expected_value:
            raise RuntimeError(
                f"Logical manifest row count mismatch for {split}: "
                f"expected {expected_value}, got {actual_value}"
            )

    metadata_counts = _extract_metadata_counts(_read_json(package_root / "metadata.json"))
    files = _iter_source_files(package_root, output_zip, partial_zip)
    file_count = len(files)
    npz_count = sum(1 for path in files if path.suffix.lower() == ".npz")
    source_bytes = sum(path.stat().st_size for path in files)
    if file_count != args.expected_file_count:
        raise RuntimeError(f"Expected {args.expected_file_count} files, found {file_count}")
    if npz_count != args.expected_npz_count:
        raise RuntimeError(f"Expected {args.expected_npz_count} .npz files, found {npz_count}")

    disk_usage = shutil.disk_usage(package_root.anchor or package_root.drive or str(package_root))
    free_bytes_before = disk_usage.free
    minimum_free_bytes = source_bytes + int(args.minimum_free_margin_gb * (1024 ** 3))
    if free_bytes_before < minimum_free_bytes:
        raise RuntimeError(
            "Insufficient free disk space for ZIP creation: "
            f"free={free_bytes_before}, required={minimum_free_bytes}"
        )

    return SourceStats(
        files=files,
        file_count=file_count,
        npz_count=npz_count,
        source_bytes=source_bytes,
        logical_rows=logical_rows,
        metadata_counts=metadata_counts,
        free_bytes_before=free_bytes_before,
    )


def _safe_remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def _create_zip(
    package_root: Path,
    output_zip: Path,
    source_files: tuple[Path, ...],
    source_bytes: int,
) -> Path:
    partial_zip = Path(str(output_zip) + ".partial")
    partial_zip.parent.mkdir(parents=True, exist_ok=True)
    _safe_remove(partial_zip)
    root_name = package_root.name
    total_files = len(source_files)
    archived_files = 0
    archived_bytes = 0
    progress_interval = 500
    print(
        f"Creating ZIP_STORED archive at {partial_zip} "
        f"from {total_files} files ({source_bytes} bytes)"
    )
    try:
        with zipfile.ZipFile(
            partial_zip,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for path in source_files:
                relative_path = path.relative_to(package_root)
                arcname = str(PurePosixPath(root_name) / PurePosixPath(relative_path.as_posix()))
                archive.write(path, arcname=arcname)
                archived_files += 1
                archived_bytes += path.stat().st_size
                if archived_files % progress_interval == 0 or archived_files == total_files:
                    percentage = (archived_bytes / source_bytes * 100.0) if source_bytes else 100.0
                    print(f"files archived: {archived_files} / {total_files}")
                    print(f"bytes archived: {archived_bytes} / {source_bytes}")
                    print(f"percentage: {percentage:.2f}")
        return partial_zip
    except Exception:
        try:
            _safe_remove(partial_zip)
        except OSError as cleanup_error:
            print(f"Warning: failed to remove partial ZIP: {cleanup_error}", file=sys.stderr)
        raise


def _count_csv_rows_in_zip(archive: zipfile.ZipFile, member_name: str) -> int:
    with archive.open(member_name, "r") as raw_handle:
        wrapper = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
        try:
            reader = csv.DictReader(wrapper)
            return sum(1 for _ in reader)
        finally:
            wrapper.detach()


def _read_json_from_zip(archive: zipfile.ZipFile, member_name: str) -> dict:
    with archive.open(member_name, "r") as raw_handle:
        wrapper = io.TextIOWrapper(raw_handle, encoding="utf-8")
        try:
            data = json.load(wrapper)
        finally:
            wrapper.detach()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in ZIP member {member_name}")
    return data


def _verify_zip(args: argparse.Namespace, zip_path: Path) -> ZipVerifyStats:
    root_prefix = f"{args.package_root.name}/"
    required_members = {f"{root_prefix}{relative_path}" for relative_path in REQUIRED_RELATIVE_FILES}
    expected_rows = {
        "train": args.expected_train_rows,
        "val": args.expected_val_rows,
        "test": args.expected_test_rows,
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"CRC verification failed for ZIP member: {bad_member}")

        file_infos = [info for info in infos if not info.is_dir()]
        for info in file_infos:
            if not info.filename.startswith(root_prefix):
                raise RuntimeError(f"ZIP member outside required root folder: {info.filename}")

        member_names = {info.filename for info in file_infos}
        missing_members = sorted(required_members - member_names)
        if missing_members:
            raise RuntimeError(f"ZIP missing required members: {missing_members}")

        file_count = len(file_infos)
        npz_count = sum(1 for info in file_infos if info.filename.lower().endswith(".npz"))
        if file_count != args.expected_file_count:
            raise RuntimeError(
                f"ZIP file count mismatch: expected {args.expected_file_count}, got {file_count}"
            )
        if npz_count != args.expected_npz_count:
            raise RuntimeError(
                f"ZIP .npz count mismatch: expected {args.expected_npz_count}, got {npz_count}"
            )

        logical_rows = {}
        for split in ("train", "val", "test"):
            member_name = f"{root_prefix}manifests/logical/nslt1000_{split}.csv"
            logical_rows[split] = _count_csv_rows_in_zip(archive, member_name)
            if logical_rows[split] != expected_rows[split]:
                raise RuntimeError(
                    f"ZIP logical manifest row count mismatch for {split}: "
                    f"expected {expected_rows[split]}, got {logical_rows[split]}"
                )

        metadata_counts = _extract_metadata_counts(
            _read_json_from_zip(archive, f"{root_prefix}metadata.json")
        )

    expected_metadata_counts = {
        "total_nslt1000": 7232,
        "reused_nslt300": 2660,
        "incremental_new": 4572,
    }
    for key, expected_value in expected_metadata_counts.items():
        actual_value = metadata_counts[key]
        if actual_value != expected_value:
            raise RuntimeError(
                f"ZIP metadata count mismatch for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )

    return ZipVerifyStats(
        file_count=file_count,
        npz_count=npz_count,
        logical_rows=logical_rows,
        metadata_counts=metadata_counts,
        bad_member=bad_member,
        root_prefix=root_prefix,
    )


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_file(zip_path: Path, sha256_hex: str) -> Path:
    sha_path = Path(str(zip_path) + ".sha256.txt")
    line = f"{sha256_hex}  {zip_path.name}\n"
    sha_path.write_text(line, encoding="utf-8")
    return sha_path


def main() -> int:
    args = build_parser().parse_args()
    package_root = args.package_root.resolve()
    output_zip = args.output_zip.resolve()
    partial_zip = Path(str(output_zip) + ".partial")
    if not package_root.is_dir():
        raise NotADirectoryError(f"--package-root must be an existing directory: {package_root}")
    if output_zip.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output ZIP already exists: {output_zip}. Re-run with --overwrite to replace it safely."
        )

    source_stats = _collect_source_stats(
        argparse.Namespace(
            package_root=package_root,
            output_zip=output_zip,
            expected_npz_count=args.expected_npz_count,
            expected_file_count=args.expected_file_count,
            expected_train_rows=args.expected_train_rows,
            expected_val_rows=args.expected_val_rows,
            expected_test_rows=args.expected_test_rows,
            minimum_free_margin_gb=args.minimum_free_margin_gb,
        )
    )
    print(
        "Preflight source stats:",
        json.dumps(
            {
                "package_root": str(package_root),
                "output_zip": str(output_zip),
                "file_count": source_stats.file_count,
                "npz_count": source_stats.npz_count,
                "source_bytes": source_stats.source_bytes,
                "source_gib": round(_format_gib(source_stats.source_bytes), 6),
                "logical_rows": source_stats.logical_rows,
                "metadata_counts": source_stats.metadata_counts,
                "free_bytes_before": source_stats.free_bytes_before,
                "free_gib_before": round(_format_gib(source_stats.free_bytes_before), 6),
            },
            indent=2,
        ),
    )

    zip_created = _create_zip(
        package_root=package_root,
        output_zip=output_zip,
        source_files=source_stats.files,
        source_bytes=source_stats.source_bytes,
    )
    try:
        zip_verify_stats = _verify_zip(args, zip_created)
        if output_zip.exists() and not args.overwrite:
            raise FileExistsError(f"Output ZIP exists and overwrite is disabled: {output_zip}")
        os.replace(zip_created, output_zip)
        sha256_hex = _compute_sha256(output_zip)
        sha_path = _write_sha256_file(output_zip, sha256_hex)
    except Exception:
        try:
            _safe_remove(partial_zip)
        except OSError as cleanup_error:
            print(f"Warning: failed to remove partial ZIP after verification error: {cleanup_error}", file=sys.stderr)
        raise

    zip_size_bytes = output_zip.stat().st_size
    summary = {
        "status": "pass",
        "package_root": str(package_root),
        "output_zip": str(output_zip),
        "sha256_path": str(sha_path),
        "zip_size_bytes": zip_size_bytes,
        "zip_size_gib": round(_format_gib(zip_size_bytes), 6),
        "source_bytes": source_stats.source_bytes,
        "source_gib": round(_format_gib(source_stats.source_bytes), 6),
        "source_file_count": source_stats.file_count,
        "source_npz_count": source_stats.npz_count,
        "zip_file_count": zip_verify_stats.file_count,
        "zip_npz_count": zip_verify_stats.npz_count,
        "logical_rows": zip_verify_stats.logical_rows,
        "metadata_counts": zip_verify_stats.metadata_counts,
        "crc_test_pass": zip_verify_stats.bad_member is None,
        "root_prefix": zip_verify_stats.root_prefix,
        "sha256": sha256_hex,
        "compression_mode": "ZIP_STORED",
        "free_bytes_before": source_stats.free_bytes_before,
        "free_gib_before": round(_format_gib(source_stats.free_bytes_before), 6),
        "partial_zip": str(partial_zip),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
