from __future__ import annotations

import json

import numpy as np

from mouse_behavior.annotation_website_export import _write_tracks_jsonl


def test_track_export_distinguishes_slot_visual_and_animal_ids(tmp_path):
    tracks = {
        "valid": np.asarray([[True]], dtype=bool),
        "bboxes": np.asarray([[[1.0, 2.0, 11.0, 12.0]]], dtype=float),
        "keypoints_px": np.zeros((1, 1, 7, 2), dtype=float),
        "confidences": np.full((1, 1, 7), 0.9, dtype=float),
        "pose_quality": np.asarray([[0.85]], dtype=float),
        "identity_confidence": np.asarray([[0.97]], dtype=float),
        "visual_track_ids": np.asarray([[42]], dtype=np.int64),
        "identity_labels": ("RFID-A",),
        "identity_source": "rfid_mtrack",
    }
    destination = tmp_path / "tracks.jsonl"

    count = _write_tracks_jsonl(
        destination,
        tracks,
        fps=30.0,
        frame_count=1,
        video_id="experiment",
        width=100,
        height=100,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    detection = payload["detections"][0]
    assert count == 1
    assert detection["track_id"] == 0
    assert detection["visual_track_id"] == 42
    assert detection["animal_id"] == "RFID-A"
    assert detection["identity_source"] == "rfid_mtrack"
    assert detection["identity_confidence"] == 0.97
