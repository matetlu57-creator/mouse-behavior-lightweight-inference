from __future__ import annotations

import importlib
import json

import cv2
import numpy as np
import pandas as pd


def load_lightweight():
    return importlib.import_module("mouse_behavior.lightweight_behavior_inference")


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
