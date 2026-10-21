"""Independent posture channels, explicit pixels and qualified export fragments."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mouse_behavior.behavior import posture_rules as rules
from mouse_behavior.behavior.posture_adapter import KEYPOINT_NAMES, posture_clip_from_rows, resolve_posture_window_audits
from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.canonical_video_pipeline import _directed_events, _infer_core_task


def clip_and_windows():
    clip = dict(points=np.ones((30, 7, 2)), boxes=np.ones((30, 4)), observed=np.ones(30, bool), fps=10., track_id=7)
    window = dict(start=0, end=30, center=15, coverage=1., body_relative_amplitude=.1,
                  posterior_excursion=.01, reversals_s=3., lateral_reversals_s=3., coherence=1.,
                  relative_motion_fraction=.5, spine_cosine=.99, back_fraction=.5,
                  neck_rear_excursion=.1, neck_length_change=.1, comotion=1., shape_to_translation_ratio=10.)
    return clip, [dict(window), dict(window, center=14)]


def test_each_parameter_change_preserves_entire_opposite_candidate_ledger():
    clip, windows = clip_and_windows()
    baseline = rules.predict(clip, windows, windows)
    assert baseline["candidates"]["Grooming"]["candidate_frame_count"] == 30
    assert baseline["candidates"]["Rearing"]["candidate_frame_count"] == 30
    g = rules.predict(clip, windows, windows, groom_rule=replace(rules.DEFAULT_GROOMING_CONFIG, min_relative_amplitude=10.))
    r = rules.predict(clip, windows, windows, rear_rule=replace(rules.DEFAULT_REARING_CONFIG, min_neck_excursion=10.))
    assert g["candidates"]["Rearing"] == baseline["candidates"]["Rearing"]
    assert r["candidates"]["Grooming"] == baseline["candidates"]["Grooming"]
    assert not g["candidates"]["Grooming"]["clip_candidate"]
    assert not r["candidates"]["Rearing"]["clip_candidate"]
    for mode in ("disabled", "prefer_grooming", "prefer_rearing"):
        changed = rules.predict(clip, windows, windows, resolver_mode=mode)
        assert changed["candidates"] == baseline["candidates"]


def test_resolution_requalifies_short_fragments_without_rewriting_raw_events(monkeypatch):
    clip, windows = clip_and_windows()
    # Repeated positive windows support all frames; a competing class occupies
    # the middle, leaving a .2-second Grooming fragment at the start.
    monkeypatch.setattr(rules, "measure", lambda *args, **kwargs: (windows, 30.))
    original = rules.predict
    def with_partial_conflict(*args, **kwargs):
        result = original(*args, **kwargs)
        result["resolution"]["frame_labels"] = ["Grooming"]*2 + ["Unknown"]*10 + ["Grooming"]*18
        return result
    monkeypatch.setattr(rules, "predict", with_partial_conflict)
    result = rules.infer_posture(clip)
    assert result["candidates"]["Grooming"]["events"][0]["start_frame"] == 0
    assert [(e["start_frame"], e["end_frame_exclusive"]) for e in result["resolution"]["events"]] == [(12, 30)]


def row(frame=0, confidence=1.):
    return dict(frame=frame, time=frame/10, detections=[dict(track_id=7, box=[0, 0, 10, 10],
                detection_confidence=confidence, keypoints=[[i, i+1, 1.] for i in range(7)])])


def test_adapter_preserves_missing_slots_track_ids_and_explicit_point_order():
    names = list(KEYPOINT_NAMES)
    names[1], names[2] = names[2], names[1]
    clip = posture_clip_from_rows([row(), row(2, .1)], 3, 10., 7, keypoint_names=names)
    assert clip["observed"].tolist() == [True, False, False]
    assert np.isnan(clip["points"][1:]).all()
    assert clip["points"][0, 1, 0] == 2
    assert np.isnan(posture_clip_from_rows([row()], 1, 10., 8)["points"]).all()
    with pytest.raises(ValueError, match="duplicate"):
        posture_clip_from_rows([row(), row()], 3, 10., 7)
    with pytest.raises(FrozenInstanceError):
        rules.DEFAULT_GROOMING_CONFIG.min_relative_amplitude = 2


def test_measured_role_reversal_cannot_publish_short_approach_fragments():
    actor = np.array([7]*9+[8])
    target = np.array([8]*9+[7])
    events = _directed_events(np.ones(10, bool), actor, target, [7, 8], "approach", np.ones(10, bool), fps=10.)
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 9)]
    assert all(qualify_event_duration(event, 10.)["ok"] for event in events)


def test_core_boundary_retains_only_a_real_complete_qualified_parent(monkeypatch):
    import mouse_behavior.canonical_video_pipeline as pipeline
    parent = dict(behavior="Isolation", canonical_behavior="isolation", mouse_ids=[7], actor_id=-1,
                  target_id=-1, start_frame=0, end_frame=110,
                  observed_all_participants_frame_intervals=[(0, 110)])
    monkeypatch.setattr(pipeline.TrackStore, "window", lambda *args: [])
    monkeypatch.setattr(pipeline, "infer_window_events", lambda *args, **kwargs: ([parent], {"track_ids": [7]}))
    events, _ = _infer_core_task(("unused.sqlite", 10., {}, None, 100, 120, 0, 120, "disabled"))
    assert len(events) == 1
    assert events[0]["start_frame"] == 0 and events[0]["end_frame"] == 110
    assert events[0]["retained_complete_parent_for_duration"] is True
    assert qualify_event_duration(events[0], 10.)["ok"]
    assert events[0]["observed_all_participants_frame_intervals"] == [(0, 110)]


@pytest.mark.parametrize("factory,kwargs", [(rules.TemporalConfig, {"minimum_positive_windows": True}),
    (rules.GroomingConfig, {"min_relative_amplitude": float("nan")}),
    (rules.RearingConfig, {"min_neck_excursion": -1})])
def test_posture_config_rejects_invalid_values(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_overlapping_contexts_resolve_same_id_conflicts_and_requalify_fragments():
    def window(offset, track, behavior, start, end):
        return dict(window_start=offset, posture_review={str(track): dict(profile="frozen", revision="test",
            candidates={name: dict(events=[dict(start_frame=start, end_frame_exclusive=end)] if name == behavior else [])
                        for name in ("Grooming", "Rearing")})})
    # Both standalone .4s parents become .2s after their disputed overlap is
    # removed, so neither can be exported. A different ID remains independent.
    windows = [window(0, 7, "Grooming", 18, 22), window(20, 7, "Rearing", 0, 4),
               window(20, 8, "Rearing", 0, 4)]
    events, audit = resolve_posture_window_audits(windows, 10.)
    assert [(event["mouse_ids"], event["start_frame"], event["end_frame"]) for event in events] == [([8], 20, 24)]
    assert audit["7"]["conflict_frame_intervals"] == [[20, 22]]
    assert len(audit["7"]["duration_rejected"]) == 2
    assert windows[0]["posture_review"]["7"]["candidates"]["Grooming"]["events"][0]["start_frame"] == 18
