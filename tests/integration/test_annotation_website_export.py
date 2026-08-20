from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from annotation_website_export import (
    KEYPOINT_NAMES,
    build_annotations,
    export_complete_video_package,
)


def synthetic_tracks(frames: int = 6, mice: int = 4) -> dict[str, np.ndarray]:
    valid = np.ones((frames, mice), dtype=bool)
    centers = np.zeros((frames, mice, 2), dtype=float)
    centers[:, 0] = [10.0, 10.0]
    centers[:, 1] = [12.0, 10.0]
    centers[:, 2] = [11.0, 12.0]
    centers[:, 3] = [80.0, 80.0]
    keypoints = np.repeat(centers[:, :, None, :], len(KEYPOINT_NAMES), axis=2)
    boxes = np.zeros((frames, mice, 4), dtype=float)
    boxes[:, :, 0:2] = centers - 2.0
    boxes[:, :, 2:4] = centers + 2.0
    return {
        "valid": valid,
        "centers_px": centers,
        "keypoints_px": keypoints,
        "confidences": np.full((frames, mice, len(KEYPOINT_NAMES)), 0.9),
        "bboxes": boxes,
        "pose_quality": np.full((frames, mice), 0.85),
    }


def test_website_names_ranges_and_mouse_cardinality():
    tracks = synthetic_tracks()
    behavior_events = [
        {
            "behavior": "attack",
            "pair_key": "1_0",
            "actor_id": 1,
            "target_id": 0,
            "start_frame": 1,
            "end_frame": 2,
            "candidate_level": "strong",
            "peak_score": 0.9,
        },
        {
            "behavior": "stationary",
            "pair_key": "mouse_3",
            "actor_id": 3,
            "target_id": -1,
            "start_frame": 0,
            "end_frame": 5,
            "candidate_level": "extended",
            "peak_score": 0.8,
        },
        {
            "behavior": "huddle",
            "pair_key": "group",
            "actor_id": -1,
            "target_id": -1,
            "start_frame": 0,
            "end_frame": 5,
            "candidate_level": "extended",
            "peak_score": 0.8,
        },
        {
            "behavior": "isolation",
            "pair_key": "group",
            "actor_id": -1,
            "target_id": -1,
            "start_frame": 0,
            "end_frame": 5,
            "candidate_level": "extended",
            "peak_score": 0.8,
        },
    ]
    contact_events = [
        {
            "contact_type": "nose_head_and_nose_tail",
            "pair_key": "0_1",
            "contact_actor_id": 0,
            "contact_target_id": 1,
            "start_frame": 3,
            "end_frame": 3,
        }
    ]
    annotations, skipped = build_annotations(
        behavior_events,
        contact_events,
        tracks,
        fps=2.0,
        frame_count=6,
        huddle_distance_cm=5.0,
        cm_per_pixel=1.0,
    )

    assert not skipped
    by_behavior = {row["behavior"]: row for row in annotations}
    assert by_behavior["攻击行为"]["mouse_ids"] == [0, 1]
    assert by_behavior["攻击行为"]["start_time"] == 0.5
    assert by_behavior["攻击行为"]["end_time"] == 1.5
    assert by_behavior["静止"]["mouse_ids"] == [3]
    assert len(by_behavior["扎堆行为"]["mouse_ids"]) >= 2
    assert by_behavior["孤立行为"]["mouse_ids"] == [3]
    assert by_behavior["鼻头接触"]["mouse_ids"] == [0, 1]
    assert by_behavior["鼻尾接触"]["mouse_ids"] == [0, 1]


def test_complete_video_package_matches_full_video_contract(tmp_path: Path):
    source_video = tmp_path / "source.mov"
    source_video.write_bytes(b"test-video-placeholder")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    report = export_complete_video_package(
        source_video=source_video,
        output_dir=output_dir,
        behavior_events=[
            {
                "behavior": "approach",
                "pair_key": "0_1",
                "actor_id": 0,
                "target_id": 1,
                "start_frame": 0,
                "end_frame": 1,
                "candidate_level": "extended",
                "peak_score": 1.0,
            }
        ],
        contact_events=[],
        tracks=synthetic_tracks(frames=6),
        fps=30.0,
        frame_count=6,
        width=100,
        height=100,
        skeleton_edges=((0, 1), (0, 2)),
        cm_per_pixel=1.0,
        huddle_distance_cm=5.0,
        tracker_params={"expected_mice": 4},
    )
    video_dir = Path(report["video_directory"])
    assert sorted(path.name for path in video_dir.iterdir()) == [
        "annotations.json",
        "metadata.json",
        "tracks.jsonl",
        "video.mov",
    ]
    annotations = json.loads((video_dir / "annotations.json").read_text(encoding="utf-8"))
    metadata = json.loads((video_dir / "metadata.json").read_text(encoding="utf-8"))
    track_lines = (video_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    assert annotations["schema_version"] == "1.0"
    assert annotations["video_file"] == "video.mov"
    assert annotations["annotations"][0]["behavior"] == "接近"
    assert metadata["frame_count"] == len(track_lines) == 6
    assert metadata["source_relative"] == annotations["video_file"]
    assert metadata["keypoint_names"] == list(KEYPOINT_NAMES)
    for index, line in enumerate(track_lines):
        frame = json.loads(line)
        assert frame["frame_index"] == index
        assert frame["detection_count"] == len(frame["detections"])
        assert frame["schema_version"] == metadata["schema_version"]
        assert frame["video_id"] == metadata["video_id"]
