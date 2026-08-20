from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_lightweight():
    path = ROOT / "lightweight_behavior_inference.py"
    spec = importlib.util.spec_from_file_location("lightweight_behavior_clip_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_behavior_clip_extractor_writes_one_directory_per_behavior(tmp_path):
    lightweight = load_lightweight()
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((64, 64, 3), index, dtype=np.uint8))
    writer.release()

    events = pd.DataFrame(
        [
            {
                "behavior": "approach",
                "candidate_level": "",
                "start_frame": 2,
                "end_frame": 5,
            },
            {
                "behavior": "walking",
                "candidate_level": "extended",
                "start_frame": 15,
                "end_frame": 18,
            },
        ]
    )
    events_path = tmp_path / "lightweight_behavior_events.csv"
    events.to_csv(events_path, index=False)

    output = tmp_path / "behavior_clips"
    lightweight.extract_behavior_clips(
        source,
        events_path,
        output,
        behavior_names=["approach", "walking"],
        event_level="all",
        clip_seconds=0.5,
        max_clips_per_behavior=10,
    )

    assert len(list((output / "approach").glob("*.mp4"))) == 1
    assert len(list((output / "walking").glob("*.mp4"))) == 1
    with (output / "behavior_clip_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["total_clips"] == 2
    assert summary["clip_counts"] == {"approach": 1, "walking": 1}
