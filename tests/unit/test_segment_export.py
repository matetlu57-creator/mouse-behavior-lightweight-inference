"""Frame-exact dataset crops preserve inputs and both index coordinate systems."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.evaluation.exported_dataset import discover_samples
from mouse_behavior.evaluation.segment_export import export_segment


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _decode(path):
    capture = cv2.VideoCapture(str(path))
    values = []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                return values
            values.append(float(frame.mean()))
    finally:
        capture.release()


@pytest.fixture
def sample(tmp_path):
    sample_dir = tmp_path / "source" / "Approach" / "example"
    sample_dir.mkdir(parents=True)
    clip = sample_dir / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(8):
            writer.write(np.full((48, 64, 3), 20 + index * 25, dtype=np.uint8))
    finally:
        writer.release()
    _json(
        sample_dir / "annotation.json",
        {
            "behavior": "Approach",
            "mouse_ids": [1, 2],
            "participants": [
                {"role_name": "Approaching", "track_ids": [1]},
                {"role_name": "Approached", "track_ids": [2]},
            ],
            "confidence": "certain",
            "frame_range": {"start": 0, "end": 7},
            "time_range": {"start": 0.0, "end": 0.8},
        },
    )
    _json(
        sample_dir / "metadata.json",
        {
            "schema_version": "1.0",
            "clip": {
                "filename": "clip.mp4",
                "fps": 10.0,
                "frame_count": 8,
                "width": 64,
                "height": 48,
            },
            "tracks": {"frame_origin": 0, "time_origin": 0.0, "coordinate_system": "clip_pixels"},
            "pose": {"keypoint_names": ["nose"]},
            "source": {
                "video_filename": "recording.mp4",
                "start_frame": 100,
                "end_frame": 107,
                "start_time": 10.0,
                "end_time": 10.8,
                "crop_region": {"x": 4, "y": 5, "w": 64, "h": 48},
            },
        },
    )
    _json(
        sample_dir / "tracks.json",
        [
            {
                "frame": index,
                "time": index / 10,
                "source_frame": 100 + index,
                "detections": [
                    {"track_id": visual_id, "keypoints": [[index, visual_id, 1.0]]}
                    for visual_id in (1, 2, 9)
                ],
            }
            for index in range(8)
        ],
    )
    return discover_samples(tmp_path / "source")[0]


def test_export_is_actual_crop_and_discovery_roundtrip(sample, tmp_path):
    source_dir = Path(sample.sample_dir)
    before = _hashes(source_dir)
    output = tmp_path / "export" / "Static" / "segment"
    manifest = export_segment(
        sample,
        output,
        start_frame=2,
        end_frame=6,
        behavior="Static",
        mouse_ids=(2,),
        provenance={"decision": "manual", "score": 0.7,
                    "candidate": {"status": "posture_candidate_needs_review"}},
    )
    assert _hashes(source_dir) == before
    source_pixels = _decode(sample.clip_path)
    crop_pixels = _decode(output / "clip.mp4")
    assert len(crop_pixels) == 4
    assert crop_pixels == pytest.approx(source_pixels[2:6], abs=3.0)
    annotation = json.loads((output / "annotation.json").read_text())
    assert annotation["frame_range"] == {"start": 0, "end": 3}
    assert annotation["time_range"] == {"start": 0.0, "end": 0.4}
    assert annotation["mouse_ids"] == [2]
    assert "participants" not in annotation
    assert annotation["screening_status"] == "posture_candidate_needs_review"
    assert annotation["requires_review"] is True
    assert annotation["confidence"] == "screening_candidate"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"]["start_frame"] == 102
    assert metadata["source"]["end_frame"] == 105
    assert metadata["source"]["start_time"] == pytest.approx(10.2)
    assert metadata["source"]["end_time"] == pytest.approx(10.6)
    assert metadata["source"]["crop_region"]["x"] == 4
    assert metadata["clip"]["frame_count"] == 4
    assert metadata["pose"] == {"keypoint_names": ["nose"]}
    assert metadata["provenance"]["original_annotation"]["behavior"] == "Approach"
    assert metadata["provenance"]["original_metadata_source"]["start_frame"] == 100
    tracks = json.loads((output / "tracks.json").read_text())
    assert [entry["frame"] for entry in tracks] == [0, 1, 2, 3]
    assert [entry["time"] for entry in tracks] == [0.0, 0.1, 0.2, 0.3]
    assert [entry["original_clip_frame"] for entry in tracks] == [2, 3, 4, 5]
    assert [entry["source_frame"] for entry in tracks] == [102, 103, 104, 105]
    assert [d["track_id"] for d in tracks[0]["detections"]] == [1, 2, 9]
    original_tracks = json.loads(Path(sample.tracks_path).read_text())
    assert tracks[0]["detections"] == original_tracks[2]["detections"]
    assert manifest["source_clip_local_half_open"] == {"start": 2, "end": 6}
    assert manifest["recording_absolute_inclusive"] == {"start": 102, "end": 105}
    assert manifest["source_sha256"] == before
    assert manifest["output_sha256"] == _hashes(output)
    assert manifest["verification"]["frame_count"] == 4
    assert manifest["verification"]["bytes"] > 0
    assert manifest["verification"]["fps"] == pytest.approx(10.0)
    reloaded = discover_samples(tmp_path / "export")[0]
    assert (reloaded.behavior, reloaded.frame_start, reloaded.frame_end) == ("Static", 0, 3)
    assert reloaded.frame_count == 4
    assert reloaded.mouse_ids == (2,)


@pytest.mark.parametrize("start,end", [(-1, 4), (3, 3), (4, 3), (0, 9), (1.0, 4), (False, 4)])
def test_invalid_bounds_create_nothing(sample, tmp_path, start, end):
    target = tmp_path / "bad"
    with pytest.raises(ValueError, match="bounds"):
        export_segment(
            sample,
            target,
            start_frame=start,
            end_frame=end,
            behavior="Static",
            mouse_ids=(1,),
            provenance={},
        )
    assert not target.exists()


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_create_nothing(sample, tmp_path, fps):
    target = tmp_path / "bad"
    with pytest.raises(ValueError, match="FPS"):
        export_segment(
            replace(sample, fps=fps),
            target,
            start_frame=1,
            end_frame=3,
            behavior="Static",
            mouse_ids=(1,),
            provenance={},
        )
    assert not target.exists()


def test_refuses_existing_directory_and_preserves_contents(sample, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    protected = target / "user.txt"
    protected.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_segment(
            sample,
            target,
            start_frame=1,
            end_frame=3,
            behavior="Static",
            mouse_ids=(1,),
            provenance={},
        )
    assert protected.read_text() == "keep"


def test_failed_decode_cleans_only_new_directory(sample, tmp_path, monkeypatch):
    from mouse_behavior.evaluation import segment_export

    before = _hashes(Path(sample.sample_dir))
    target = tmp_path / "failed"

    def reject(*args):
        raise ValueError("injected decode failure")

    monkeypatch.setattr(segment_export, "_verify_video", reject)
    with pytest.raises(ValueError, match="decode failure"):
        export_segment(
            sample,
            target,
            start_frame=1,
            end_frame=3,
            behavior="Static",
            mouse_ids=(1,),
            provenance={},
        )
    assert not target.exists()
    assert _hashes(Path(sample.sample_dir)) == before


def test_single_frame_crop_drops_roles_for_changed_ids_but_retains_all_detections(sample, tmp_path):
    output = tmp_path / "one"
    export_segment(
        sample,
        output,
        start_frame=7,
        end_frame=8,
        behavior="Approach",
        mouse_ids=(1,),
        provenance={},
    )
    assert len(_decode(output / "clip.mp4")) == 1
    annotation = json.loads((output / "annotation.json").read_text())
    assert annotation["frame_range"] == {"start": 0, "end": 0}
    assert "participants" not in annotation
    tracks = json.loads((output / "tracks.json").read_text())
    assert [detection["track_id"] for detection in tracks[0]["detections"]] == [1, 2, 9]


@pytest.mark.parametrize("mouse_ids,inherit_roles", [((1, 2), False), ((9, 1, 2), True)])
def test_crop_only_inherits_role_evidence_for_exact_participant_set(
    sample, tmp_path, mouse_ids, inherit_roles,
):
    annotation_path = Path(sample.annotation_path)
    original = json.loads(annotation_path.read_text())
    original.update(
        mouse_ids=[1, 2, 9],
        participants=[
            {"role_name": "Approaching", "track_ids": [1]},
            {"role_name": "Approached", "track_ids": [2, 9]},
        ],
        actor_id=1,
        target_id=2,
        role_status="event_direction_with_measured_support",
        role_support_frame_intervals=[[0, 8]],
        role_support_frame_count=8,
    )
    _json(annotation_path, original)
    source_hashes = _hashes(Path(sample.sample_dir))
    output = tmp_path / "roles"
    export_segment(sample, output, start_frame=2, end_frame=6,
                   behavior="Approach", mouse_ids=mouse_ids, provenance={})

    cropped = json.loads((output / "annotation.json").read_text())
    role_fields = {"participants", "actor_id", "target_id", "role_status",
                   "role_support_frame_intervals", "role_support_frame_count"}
    if inherit_roles:
        assert cropped["participants"] == original["participants"]
        assert (cropped["actor_id"], cropped["target_id"]) == (1, 2)
        assert cropped["role_status"] == original["role_status"]
        assert cropped["role_support_frame_intervals"] == [[0, 4]]
        assert cropped["role_support_frame_count"] == 4
    else:
        assert not role_fields.intersection(cropped)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["provenance"]["original_annotation"] == original
    tracks = json.loads((output / "tracks.json").read_text())
    assert all([d["track_id"] for d in frame["detections"]] == [1, 2, 9] for frame in tracks)
    assert _hashes(Path(sample.sample_dir)) == source_hashes


def test_full_range_copies_video_and_still_verifies(sample, tmp_path, monkeypatch):
    from mouse_behavior.evaluation import segment_export

    def unexpected_transcode(*args, **kwargs):
        raise AssertionError("a full-range copy must not transcode")

    monkeypatch.setattr(segment_export.subprocess, "run", unexpected_transcode)
    manifest = export_segment(
        sample,
        tmp_path / "full",
        start_frame=0,
        end_frame=8,
        behavior="Approach",
        mouse_ids=(1, 2),
        provenance={},
    )
    assert manifest["source_sha256"]["clip.mp4"] == manifest["output_sha256"]["clip.mp4"]
    assert manifest["verification"]["frame_count"] == 8
    assert manifest["audio_preserved"] is True


def test_full_range_reencodes_rounded_container_fps_without_changing_frames(sample, tmp_path):
    from dataclasses import replace

    record = replace(sample, fps=10.01)
    manifest = export_segment(record, tmp_path / "fps_repaired", start_frame=0, end_frame=8,
                              behavior="Approach", mouse_ids=(1, 2), provenance={})
    assert manifest["verification"]["frame_count"] == 8
    assert manifest["verification"]["fps"] == pytest.approx(10.01, abs=.001)
    assert manifest["audio_preserved"] is False
    assert manifest["source_sha256"]["clip.mp4"] != manifest["output_sha256"]["clip.mp4"]
    assert len(_decode(tmp_path / "fps_repaired/clip.mp4")) == 8


def test_crop_recounts_merged_support_and_drops_unsupported_roles(sample, tmp_path):
    annotation_path = Path(sample.annotation_path)
    annotation = json.loads(annotation_path.read_text())
    annotation.update(actor_id=1, target_id=2, role_status="event_direction_with_measured_support",
                      role_support_frame_intervals=[[0, 2]], role_support_frame_count=2,
                      observed_all_participants_frame_intervals=[[0, 4], [2, 8]],
                      observed_all_participants_frame_count=10,
                      inferred_or_partially_unobserved_frame_count=99,
                      parent_candidate_frame_intervals=[[-2, 12]])
    annotation_path.write_text(json.dumps(annotation))
    export_segment(sample, tmp_path / "unsupported", start_frame=3, end_frame=6,
                   behavior="Approach", mouse_ids=(1, 2), provenance={})
    cropped = json.loads((tmp_path / "unsupported/annotation.json").read_text())
    assert cropped["role_support_frame_intervals"] == []
    assert cropped["role_support_frame_count"] == 0
    assert cropped["role_status"] == "unknown_in_cropped_interval"
    assert not {"actor_id", "target_id", "participants", "parent_candidate_frame_intervals"}.intersection(cropped)
    assert cropped["observed_all_participants_frame_intervals"] == [[0, 3]]
    assert cropped["observed_all_participants_frame_count"] == 3
    assert cropped["inferred_or_partially_unobserved_frame_count"] == 0


def test_crop_cannot_invent_visual_track_ids(sample, tmp_path):
    with pytest.raises(ValueError, match="existing source visual track IDs"):
        export_segment(sample, tmp_path / "invented", start_frame=0, end_frame=4,
                       behavior="Static", mouse_ids=(12345,), provenance={})
    assert not (tmp_path / "invented").exists()
