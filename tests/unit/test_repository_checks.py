from __future__ import annotations

from pathlib import Path

from tools.check_repository import (
    ALLOWED_LFS_PATHS,
    check,
    is_allowed_lfs_artifact,
    parse_lfs_pointer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "weights" / "pose" / "best.pt"
MODEL_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\noid sha256=not-a-valid-pointer\nsize 10\n"
)


def test_parse_lfs_pointer_requires_a_valid_sha256_oid() -> None:
    assert parse_lfs_pointer(MODEL_POINTER) is None
    assert parse_lfs_pointer(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 10\n"
    ) == ("a" * 64, 10)


def test_only_the_canonical_pose_weight_is_allowlisted() -> None:
    assert ALLOWED_LFS_PATHS == frozenset({"weights/pose/best.pt"})
    assert "weights/pose/other.pt" not in ALLOWED_LFS_PATHS


def test_repository_check_accepts_the_canonical_lfs_weight() -> None:
    errors = check(REPOSITORY_ROOT)
    assert not any("weights/pose/best.pt" in error for error in errors)
    assert is_allowed_lfs_artifact(REPOSITORY_ROOT, "weights/pose/best.pt", MODEL_PATH)
