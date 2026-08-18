#!/usr/bin/env python3
"""Check repository layout and reject tracked runtime artifacts."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".pt", ".pth",
    ".onnx", ".engine", ".weights", ".pkl", ".pickle", ".npz", ".npy",
    ".db", ".sqlite", ".sqlite3",
}
REQUIRED_PATHS = (
    "README.md",
    "pyproject.toml",
    "src/mouse_behavior",
    "scripts",
    "tests",
    "configs/default.yaml",
    "docs/index.md",
    "weights/README.md",
)


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def check(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")
    for relative in tracked_files(root):
        path = Path(relative)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"tracked runtime/data artifact: {relative}")
        lowered = relative.replace("\\", "/").lower()
        if any(part in lowered.split("/") for part in ("runs", "results", "yolo_precompute")):
            errors.append(f"tracked generated output/cache: {relative}")
        if lowered.startswith("outputs/") and lowered != "outputs/.gitkeep":
            errors.append(f"tracked generated output/cache: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("repository check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
