"""Regression tests for merged all-behaviors heuristic pipeline and unlimited segment export."""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mouse_behavior.canonical_video_pipeline import (
    infer_all_behavior_events,
    run_all_behaviors_pipeline,
)
from mouse_behavior.evaluation.segment_export import _sha256
from mouse_behavior.visualization.rendering import (
    extract_behavior_clips,
    extract_four_class_clips,
)


def _create_tracked_source(tmp_path: Path, *, frame_count: int = 45, fps: float = 15.0):
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    assert writer.isOpened()
    for f in range(frame_count):
        writer.write(np.full((96, 160, 3), (f * 5) % 255, dtype=np.uint8))
    writer.release()

    cache = tmp_path / "cache"
    cache.mkdir()
    model = tmp_path / "model.pt"
    model.write_bytes(b"synthetic model fixture")

    rows = []
    for f in range(frame_count):
        # Track 1 stays stationary; Track 2 walks steadily across the arena
        x2 = 20.0 + f * 2.5
        dets = [
            dict(
                visual_track_id=1,
                keypoints_px=np.array(
                    [[35, 40], [30, 37], [30, 43], [25, 40], [13, 37], [13, 43], [5, 40]],
                    dtype=float,
                ),
                keypoint_conf=np.ones(7),
                bbox_xyxy=[2, 30, 40, 50],
                box_conf=0.9,
            ),
            dict(
                visual_track_id=2,
                keypoints_px=np.array(
                    [[x2 + 15, 70], [x2 + 10, 67], [x2 + 10, 73], [x2 + 5, 70], [x2 - 7, 67], [x2 - 7, 73], [x2 - 15, 70]],
                    dtype=float,
                ),
                keypoint_conf=np.ones(7),
                bbox_xyxy=[x2 - 18, 60, x2 + 18, 80],
                box_conf=0.9,
            ),
        ]
        rows.append(dict(frame=f, timestamp_seconds=f / fps, pose_detections=dets))

    with gzip.open(cache / "yolo_results.000000.000000.pkl.gz", "wb") as handle:
        pickle.dump(rows, handle)
    (cache / "yolo_results_status.json").write_text(
        json.dumps(
            dict(
                status="complete",
                video=str(video),
                total_frames=frame_count,
                next_frame=frame_count,
                chunk_count=1,
                fps=fps,
                trajectory_schema_version=1,
                tracking_backend="ultralytics",
                model=str(model),
            )
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "source_receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                source_video=dict(path=str(video), sha256=_sha256(video), fps=fps, frame_count=frame_count),
                model=dict(path=str(model), sha256=_sha256(model)),
                cache_directory=str(cache),
            )
        ),
        encoding="utf-8",
    )
    return video, cache, receipt


def test_run_all_behaviors_pipeline_enables_all_behaviors_and_fine_cropping(tmp_path: Path) -> None:
    video, cache, source_receipt = _create_tracked_source(tmp_path)
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    out = tmp_path / "all_behaviors_run"

    receipt = run_all_behaviors_pipeline(
        video,
        cache,
        out,
        config,
        mode="infer",
        source_receipt=source_receipt,
    )
    assert receipt["status"] == "complete"
    assert receipt["posture_mode"] == "calibrated"
    assert receipt["disabled_behaviors"] == []
    assert receipt["enable_fine_cropping"] is True
    assert receipt["max_events_per_behavior"] is None
    assert "mouse_behavior.behavior.segment_clipping" in json.loads(
        (out / "input_binding.json").read_text(encoding="utf-8")
    )["rule_source_sha256"]

    events = json.loads((out / "events.json").read_text(encoding="utf-8"))
    behaviors_seen = {ev["canonical_behavior"] for ev in events}
    assert "stationary" in behaviors_seen
    assert "locomotion" in behaviors_seen

    audit = receipt["inference_audit"]
    assert audit["enable_fine_cropping"] is True
    assert audit["max_events_per_behavior"] is None
    assert audit["segment_selection_audit"]["max_per_behavior"] is None


def test_infer_all_behavior_events_callable_directly(tmp_path: Path) -> None:
    from mouse_behavior.canonical_video_pipeline import TrackStore, bind_cache
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters

    video, cache, source_receipt = _create_tracked_source(tmp_path)
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite", source_receipt=source_receipt)
    events, audit = infer_all_behavior_events(
        TrackStore(tmp_path / "tracks.sqlite"),
        binding,
        scale,
        config_seed_parameters({}),
    )
    assert audit["max_events_per_behavior"] is None
    assert audit["enable_fine_cropping"] is True
    assert len(events) >= 2


def test_rendering_clip_extractors_have_no_default_200_clip_cap(tmp_path: Path) -> None:
    # Build a 250-frame synthetic video and 210 distinct behavior events (> old 200 cap)
    fps = 10.0
    total_frames = 250
    video = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert writer.isOpened()
    for i in range(total_frames):
        writer.write(np.full((32, 32, 3), i % 255, dtype=np.uint8))
    writer.release()

    rows = [
        {"behavior": "attack", "candidate_level": "strong", "start_frame": i, "end_frame": i}
        for i in range(215)
    ]
    events_csv = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(events_csv, index=False)

    out_beh = tmp_path / "beh_clips"
    extract_behavior_clips(
        video,
        events_csv,
        out_beh,
        behavior_names=["attack"],
        clip_seconds=0.1,
        min_start_interval_seconds=0.1,
    )
    beh_summary = json.loads((out_beh / "behavior_clip_summary.json").read_text(encoding="utf-8"))
    assert beh_summary["max_clips_per_behavior"] is None
    assert beh_summary["total_clips"] > 200

    # Four-class showcase extractor also has no default cap
    out_four = tmp_path / "four_clips"
    extract_four_class_clips(
        video,
        events_csv,
        out_four,
        clip_seconds=0.1,
        min_start_interval_seconds=0.1,
    )
    four_summary = json.loads((out_four / "four_class_clip_summary.json").read_text(encoding="utf-8"))
    assert four_summary["max_clips_per_class"] is None
