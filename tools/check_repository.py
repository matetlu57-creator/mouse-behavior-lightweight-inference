#!/usr/bin/env python3
"""Check repository layout and reject unsafe publication artifacts."""

from __future__ import annotations

import argparse
import hashlib
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
    "scripts/run_full_behavior_pipeline.py",
    "scripts/validate_repository.py",
    "src/mouse_behavior/full_pipeline",
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
ALLOWED_LFS_PATHS = frozenset({"weights/pose/best.pt"})
LFS_POINTER_VERSION = b"version https://git-lfs.github.com/spec/v1"
LFS_OID_PATTERN = re.compile(rb"^oid sha256:([0-9a-f]{64})$")
LFS_SIZE_PATTERN = re.compile(rb"^size ([0-9]+)$")
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
ALLOWED_ROOT_PYTHON: set[str] = set()
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


def parse_lfs_pointer(content: bytes) -> tuple[str, int] | None:
    """Return the SHA-256 and byte size from a valid Git LFS pointer."""

    lines = content.splitlines()
    if not lines or lines[0] != LFS_POINTER_VERSION:
        return None
    oid_match = next((match for line in lines if (match := LFS_OID_PATTERN.match(line))), None)
    size_match = next((match for line in lines if (match := LFS_SIZE_PATTERN.match(line))), None)
    if oid_match is None or size_match is None:
        return None
    try:
        size = int(size_match.group(1))
    except ValueError:
        return None
    return oid_match.group(1).decode("ascii"), size


def _git_blob(root: Path, relative: str) -> bytes:
    """Read a staged or committed Git blob without applying working-tree filters."""

    for spec in (f":{relative}", f"HEAD:{relative}"):
        result = subprocess.run(
            ["git", "-C", str(root), "show", spec],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
    return b""


def _has_lfs_filter(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "check-attr", "filter", "--", relative],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0 and result.stdout.rstrip().endswith(": filter: lfs")


def _worktree_matches_lfs_pointer(absolute: Path, pointer: tuple[str, int]) -> bool:
    if not absolute.is_file():
        return False
    with absolute.open("rb") as handle:
        sample = handle.read(4096)
    if parse_lfs_pointer(sample) is not None:
        return True

    expected_oid, expected_size = pointer
    digest = hashlib.sha256()
    size = 0
    with absolute.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size == expected_size and digest.hexdigest() == expected_oid


def is_allowed_lfs_artifact(root: Path, relative: str, absolute: Path) -> bool:
    """Allow only the registered model when it is a valid Git LFS object."""

    normalized = relative.replace("\\", "/")
    if normalized not in ALLOWED_LFS_PATHS or not _has_lfs_filter(root, relative):
        return False
    pointer = parse_lfs_pointer(_git_blob(root, relative))
    return pointer is not None and _worktree_matches_lfs_pointer(absolute, pointer)


def check(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")
    for relative in tracked_files(root):
        path = Path(relative)
        absolute = root / path
        allowed_lfs_artifact = is_allowed_lfs_artifact(root, relative, absolute)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not allowed_lfs_artifact:
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
        if (
            absolute.is_file()
            and absolute.stat().st_size > MAX_ORDINARY_GIT_BYTES
            and not allowed_lfs_artifact
        ):
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
