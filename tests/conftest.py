from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def make_pose_payload(visual_track_id: int | None, x: float, y: float = 20.0) -> dict[str, Any]:
    points = np.asarray(
        [
            [x + 4.0, y],
            [x + 2.0, y - 2.0],
            [x + 2.0, y + 2.0],
            [x, y],
            [x, y - 2.0],
            [x, y + 2.0],
            [x - 4.0, y],
        ],
        dtype=np.float32,
    )
    return {
        "visual_track_id": visual_track_id,
        "keypoints_px": points,
        "keypoint_conf": np.full(7, 0.9, dtype=np.float32),
        "bbox_xyxy": np.asarray([x - 6.0, y - 5.0, x + 6.0, y + 5.0], dtype=np.float32),
        "box_conf": 0.95,
        "pose_quality": 0.9,
    }


@pytest.fixture
def write_pose_cache():
    def write(cache_dir: Path, records: list[dict[str, Any]]) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "yolo_results.000000.000000.pkl.gz"
        with gzip.open(path, "wb") as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_dir

    return write


@pytest.fixture
def pose_payload_factory():
    return make_pose_payload
