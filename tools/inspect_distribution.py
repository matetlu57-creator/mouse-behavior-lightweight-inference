#!/usr/bin/env python3
"""Inspect built Python distributions for forbidden repository artifacts."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


FORBIDDEN_SUFFIXES = {
    ".avi",
    ".db",
    ".engine",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
    ".weights",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "results",
    "weights",
    "yolo_precompute",
}


def _archive_names(path: Path) -> Iterable[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            yield from archive.namelist()
        return
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            yield from archive.getnames()
        return
    raise ValueError(f"unsupported distribution artifact: {path.name}")


def inspect(directory: Path) -> list[str]:
    artifacts = sorted(
        path
        for path in directory.glob("*")
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not artifacts:
        return [f"no wheel or source distribution found in {directory}"]
    failures: list[str] = []
    for artifact in artifacts:
        if artifact.stat().st_size > 50 * 1024 * 1024:
            failures.append(f"distribution exceeds 50 MiB: {artifact.name}")
        for raw_name in _archive_names(artifact):
            path = PurePosixPath(raw_name.replace("\\", "/"))
            lowered_parts = {part.lower() for part in path.parts}
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                failures.append(f"forbidden artifact in {artifact.name}: {path}")
            if lowered_parts & FORBIDDEN_PARTS:
                failures.append(f"generated/private path in {artifact.name}: {path}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        print(f"ERROR: distribution directory does not exist: {directory}", file=sys.stderr)
        return 2
    try:
        failures = inspect(directory)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print("distribution inspection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
