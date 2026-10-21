from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

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


@pytest.mark.parametrize("relative", [".env", "config/.env.local", "credentials/client.pem",
    "credentials/signing.key", "data/private.csv", "data/animal/tracks.json"])
def test_validator_rejects_tracked_private_paths_without_secret_contents(tmp_path, relative):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic harmless fixture", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", relative], check=True)
    errors = check(tmp_path)
    assert any(relative in error and ("credential" in error or "private data" in error)
               for error in errors)


@pytest.mark.parametrize("relative,ignored", [
    (".env", True), ("config/.env.local", True), ("config/client.pem", True),
    ("config/signing.key", True), ("data/private.csv", True),
    ("data/nested/track.json", True), (".env.example", False), (".env.template", False),
    ("config/client.pem.template", False), ("data/README.md", False),
    ("data/.gitkeep", False), ("data/example.template.json", False),
])
def test_gitignore_keeps_credentials_and_root_data_private(tmp_path, relative, ignored):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((REPOSITORY_ROOT / ".gitignore").read_bytes())
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", "--", relative],
                            check=False)
    assert (result.returncode == 0) is ignored


def test_validator_accepts_documented_root_data_placeholders(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = ["data/README.md", "data/.gitkeep", "data/example.template.json", ".env.example"]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic template", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *paths], check=True)
    assert not any("credential" in error or "private data" in error for error in check(tmp_path))
