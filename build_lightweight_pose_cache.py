"""Backward-compatible CLI path for ``scripts/build_lightweight_pose_cache.py``."""

from __future__ import annotations

from _script_compat import load_script


_SCRIPT = load_script("build_lightweight_pose_cache.py", globals())


if __name__ == "__main__":
    raise SystemExit(_SCRIPT["main"]())
