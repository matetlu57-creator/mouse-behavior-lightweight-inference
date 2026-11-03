"""Conflict review preserves V3 candidates outside contradictions with Static."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.review_reconciliation import defer_static_posture_conflicts
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters, LABEL_TO_CANONICAL
from mouse_behavior.visualization.behavior_packages import _metadata


def test_pipeline_applies_conflicts_after_global_v3_resolution(monkeypatch, tmp_path):
    import mouse_behavior.canonical_video_pipeline as pipeline
    from mouse_behavior.behavior import posture_adapter
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
    stationary = dict(event("stationary", 0, 60), behavior="Static")
    grooming = dict(event("grooming", 15, 30, v3=True), behavior="Grooming")
    monkeypatch.setattr(pipeline, "_infer_core_task", lambda spec: (
        [stationary], dict(core_start=0, core_end=60, track_ids=[1])))
    monkeypatch.setattr(posture_adapter, "resolve_posture_window_audits", lambda audits, fps: ([grooming], {}))
    monkeypatch.setattr(pipeline, "whole_range_isolation_audit", lambda *args: {"tracks": {}})
    events, audit = pipeline.infer_video_events(pipeline.TrackStore(tmp_path / "unused.sqlite"),
        {"source_video": {"fps": 30., "frame_count": 60}}, {}, HeuristicParameters(), posture_mode="v3_review")
    assert [(e["canonical_behavior"], e["start_frame"], e["end_frame"]) for e in events] == [
        ("posture_review", 15, 30), ("stationary", 30, 60)]
    assert audit["static_posture_conflicts"]["conflict_frame_count"] == 15


def event(name, start, end, mouse=1, v3=False):
    result = dict(event_id=f"{name}_{start}_{end}_{mouse}", canonical_behavior=name,
                  mouse_ids=[mouse], start_frame=start, end_frame=end,
                  measured_frame_intervals=[[start, end]], measured_frame_count=end-start)
    if v3:
        result["posture_engine"] = "v3_decoupled"
    return result


def test_split_both_labels_clip_support_and_keep_original_input():
    events = [event("stationary", 0, 40), event("grooming", 15, 25, v3=True),
              event("rearing", 28, 35, v3=True), event("approach", 0, 40, mouse=2)]
    original = deepcopy(events)
    formal, review, audit = defer_static_posture_conflicts(events, 10.)
    assert events == original
    assert [(e["start_frame"], e["end_frame"]) for e in formal if e["canonical_behavior"] == "stationary"] == [(0, 15)]
    assert [(e["start_frame"], e["end_frame"]) for e in review] == [(15, 25), (28, 35)]
    assert formal[0]["measured_frame_intervals"] == [[0, 15]]
    assert formal[0]["measured_frame_count"] == 15
    assert formal[0]["parent_candidate_event_id"] == events[0]["event_id"]
    assert formal[0]["event_id"] != events[0]["event_id"]
    assert next(e for e in formal if e["canonical_behavior"] == "approach") == events[-1]
    assert audit["conflict_frame_count"] == 17
    assert len(audit["duration_rejected_fragments"]) == 2
    assert all(e["review_queue_only"] and not e["biological_label_confirmed"] for e in review)
    assert review[0]["source_grooming"] and not review[0]["source_rearing"]
    assert defer_static_posture_conflicts(events, 10.) == (formal, review, audit)


def test_tiny_conflicts_remain_and_v3_fragments_use_point_three_seconds():
    events = [event("grooming", 0, 20, v3=True), event("stationary", 3, 4)]
    formal, review, _ = defer_static_posture_conflicts(events, 10.)
    assert [(e["start_frame"], e["end_frame"]) for e in formal] == [(0, 3), (4, 20)]
    assert review[0]["end_frame"] - review[0]["start_frame"] == 1
    assert qualify_event_duration(review[0], 10.)["ok"]
    assert review[0]["source_event_ids"] == sorted(e["event_id"] for e in events)


def test_union_overlaps_are_not_double_counted_and_ids_are_isolated():
    events = [event("stationary", 0, 30), event("grooming", 10, 20, v3=True),
              event("rearing", 15, 25, v3=True), event("grooming", 0, 30, mouse=2, v3=True)]
    formal, review, audit = defer_static_posture_conflicts(events, 10.)
    assert [(e["start_frame"], e["end_frame"]) for e in review] == [(10, 25)]
    assert review[0]["source_grooming"] and review[0]["source_rearing"]
    assert audit["conflict_frame_count"] == 15
    assert next(e for e in formal if e["mouse_ids"] == [2])["event_id"] == events[-1]["event_id"]


@pytest.mark.parametrize("name,frames,ok", [("PostureReview", 1, True), ("posture_review", 0, False),
    ("attack_review", 6, False), ("AttackReview", 7, True),
    ("dispersal_review", 50, False), ("DispersalReview", 51, True)])
def test_review_duration_types_are_optional_nonethogram_labels(name, frames, ok):
    assert qualify_event_duration(dict(behavior=name, start_frame=0, end_frame=frames), 10.)["ok"] is ok
    assert "PostureReview" not in LABEL_TO_CANONICAL


def test_trusted_dispersal_review_parameters():
    params = replace(HeuristicParameters(), dispersal_min_duration_s=6.)
    candidate = event("dispersal_review", 0, 51)
    candidate["required_s"] = .1
    assert not qualify_event_duration(candidate, 10., parameters=params)["ok"]


def test_optional_screening_stage_is_exported_in_metadata():
    binding = dict(source_video=dict(path="video.mp4", fps=10., width=160, height=90, sha256="source"),
                   cache_status_sha256="cache", chunk_sha256={}, screening_stage=dict(revision="v32-review"))
    assert _metadata(binding, {}, 0, 20, "test")["provenance"]["screening_stage"] == binding["screening_stage"]


@pytest.mark.parametrize("engine", [None, "legacy", "failed_strict_trial"])
def test_non_v3_candidates_cannot_change_static_or_enter_conflict_review(engine):
    legacy = dict(event("grooming", 5, 15), posture_engine=engine)
    events = [event("stationary", 0, 20), legacy]
    formal, review, audit = defer_static_posture_conflicts(events, 10.)
    assert formal == events
    assert not review and audit["conflict_frame_count"] == 0


def test_conflict_annotation_is_self_contained_and_keeps_source_coordinate_system():
    from mouse_behavior.visualization.behavior_packages import _annotation
    _, review, _ = defer_static_posture_conflicts(
        [event("stationary", 0, 40), event("grooming", 15, 25, v3=True)], 10.)
    result = _annotation(review[0], [], 20, 10., crop_start=10)
    assert result["frame_range"] == {"start": 5, "end": 14}
    assert result["source_support_source_frames"] == review[0]["source_support"]
    assert result["source_support_coordinate_system"] == "original_video_half_open_frames"
    assert result["review_queue_only"] and result["biological_label_confirmed"] is False
