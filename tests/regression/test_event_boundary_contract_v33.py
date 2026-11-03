"""Source events and encoded playback expose one unambiguous frame contract."""
import csv
import json

import pytest

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.visualization.behavior_packages import _annotation, write_manifests


@pytest.mark.parametrize("behavior,frames,accepted", [
    ("approach", 30, False), ("approach", 31, True),
    ("together", 29, False), ("together", 30, True),
    ("huddle", 149, False), ("huddle", 150, True),
    ("social_clustering", 14, False), ("social_clustering", 15, True),
    ("isolation", 300, False), ("isolation", 301, True),
])
def test_context_never_changes_five_behavior_time_minima(behavior, frames, accepted):
    event = dict(canonical_behavior=behavior, start_frame=50, end_frame=50+frames)
    boundary = EventBoundary.for_event(event, dict(fps=30., frame_count=1000), 3.)
    assert boundary.as_dict()["event_frame_count"] == frames
    assert qualify_event_duration(event, 30.)["ok"] is accepted


def test_default_playback_is_exact_event_and_approach_never_has_trailing_context():
    event = dict(canonical_behavior="approach", start_frame=27, end_frame=61)
    info = dict(fps=30., frame_count=100)
    pure = EventBoundary.for_event(event, info).as_dict()
    assert pure["source_event_half_open"] == pure["playback_crop_half_open"] == {"start": 27, "end": 61}
    assert pure["event_frame_count"] == pure["playback_frame_count"] == 34
    review = EventBoundary.for_event(event, info, .25).as_dict()
    assert review["playback_crop_half_open"] == {"start": 19, "end": 61}
    assert review["event_offset_in_playback_half_open"] == {"start": 8, "end": 42}
    assert review["event_frame_count"] == 34 and review["playback_frame_count"] == 42
    assert review["context_after_frames"] == 0


def test_manifest_annotation_and_csv_preserve_identical_contract(tmp_path):
    event = dict(behavior="Approach", canonical_behavior="approach", event_id="approach_fixture",
                 mouse_ids=[1, 2], start_frame=27, end_frame=61)
    boundary = EventBoundary(27, 61, 19, 61, 30.).as_dict()
    annotation = _annotation(event, [], 42, 30., crop_start=19)
    row = dict(event, frame_count=42, **boundary)
    write_manifests(tmp_path, [row])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "2.0"
    for key, value in annotation["boundary"].items():
        assert manifest["clips"][0][key] == value
    with (tmp_path / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        flat = next(csv.DictReader(handle))
    assert json.loads(flat["source_event_half_open"]) == {"start": 27, "end": 61}
    assert int(flat["event_frame_count"]) == 34
    assert int(flat["playback_frame_count"]) == 42


@pytest.mark.parametrize("bounds", [(0, 1, -1, 1), (2, 2, 0, 4), (0, 5, 0, 4), (0.5, 2, 0, 2)])
def test_invalid_or_fractional_source_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        EventBoundary(*bounds, 30.)
