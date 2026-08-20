#!/usr/bin/env python3
"""Check repository layout and reject unsafe publication artifacts."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".weights",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".db",
    ".sqlite",
    ".sqlite3",
}
REQUIRED_PATHS = (
    ".editorconfig",
    ".github/CODEOWNERS",
    ".github/workflows/test.yml",
    ".pre-commit-config.yaml",
    ".quality-gate.toml",
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "src/mouse_behavior",
    "scripts",
    "scripts/run_quality.py",
    "scripts/validate_repository.py",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/regression",
    "tests/e2e",
    "configs/default.yaml",
    "docs/adr",
    "docs/index.md",
    "weights/README.md",
)

MAX_ORDINARY_GIT_BYTES = 50 * 1024 * 1024
FORBIDDEN_ROOT_DIRECTORIES = {
    "historical_v1.40_v1.41",
    "historical_v1.42.1",
    "original",
}
OBSOLETE_ROOT_FILES = {
    "ENGINEERING_REVIEW.md",
    "ENGINEERING_REVIEW_v1.43.md",
    "FINAL_VALIDATION_v1.43.txt",
    "MANIFEST_SHA256_v1.43.txt",
    "README_FIRST.md",
    "RUNBOOK.md",
    "TEST_RESULTS_v1.43.txt",
    "V1.43_STANDARD_BEHAVIOR_ENGINE.md",
    "fast_video_analysis_v124.yaml",
}
ALLOWED_ROOT_PYTHON = {
    "adaptive_arena_boundary.py",
    "annotation_website_export.py",
    "lightweight_behavior_inference.py",
    "lightweight_cache_behavior_analysis.py",
    "mask_trigger_controller.py",
    "mouse_chase_attack_extractor_base.py",
    "mouse_chase_attack_high_recall.py",
    "nvenc_video_writer.py",
    "standard_behavior_engine.py",
}
PRIVATE_KEY_PATTERN = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")
TOKEN_PATTERNS = (
    re.compile(rb"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)
CONFLICT_START = b"<" * 7 + b" "
CONFLICT_END = b">" * 7 + b" "
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


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
        absolute = root / path
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"tracked runtime/data artifact: {relative}")
        lowered = relative.replace("\\", "/").lower()
        if any(part in lowered.split("/") for part in ("runs", "results", "yolo_precompute")):
            errors.append(f"tracked generated output/cache: {relative}")
        if lowered.startswith("outputs/") and lowered != "outputs/.gitkeep":
            errors.append(f"tracked generated output/cache: {relative}")
        if path.parts and path.parts[0] in FORBIDDEN_ROOT_DIRECTORIES:
            errors.append(f"copied historical tree belongs in Git history: {relative}")
        if len(path.parts) == 1 and path.name in OBSOLETE_ROOT_FILES:
            errors.append(f"obsolete root file: {relative}")
        if (
            len(path.parts) == 1
            and path.suffix.lower() == ".py"
            and path.name not in ALLOWED_ROOT_PYTHON
        ):
            errors.append(f"new Python file must not live at repository root: {relative}")
        if absolute.is_file() and absolute.stat().st_size > MAX_ORDINARY_GIT_BYTES:
            errors.append(f"tracked file exceeds 50 MiB review gate: {relative}")
        if absolute.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            content = absolute.read_bytes()
            if CONFLICT_START in content or CONFLICT_END in content:
                errors.append(f"unresolved merge marker: {relative}")
            if PRIVATE_KEY_PATTERN.search(content) or any(
                pattern.search(content) for pattern in TOKEN_PATTERNS
            ):
                errors.append(f"possible secret in tracked file: {relative}")
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
