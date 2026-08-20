"""Pose-cache model boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_pose_cache(
    video: str | Path,
    output: str | Path,
    model: str | Path,
    **kwargs: Any,
) -> Path:
    """Build a lightweight YOLO Pose cache through the reusable cache module."""

    from ..pose_cache import build_cache

    return build_cache(Path(video), Path(output), Path(model), **kwargs)
