"""Behavior phase, participant set and true duration must survive cropping."""
import numpy as np

from mouse_behavior.evaluation import group_event_rules as group
from tests.regression import test_heuristics_v16_audit as fixtures


def test_huddle_never_backfills_late_member_into_earlier_phase(monkeypatch):
    f = fixtures.features(300, 4)
    fixtures.set_centers(f, [[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
    f.valid[:60, 3] = False
    f.centers_cm[:60, 3] = np.nan
    fixtures.sync(f)
    monkeypatch.setattr(group, "_proposal_sets", lambda *args: [(0, 1, 2, 3)])
    candidates = group.group_rule_candidates(f, fixtures.confirmed_params(), (), behaviors=("huddle",))
    assert candidates, "eight seconds of supported four-member huddle is a positive control"
    for candidate in candidates:
        assert candidate.target_ids == tuple(map(int, f.track_ids))
        assert all(low >= 60 and (high-low)/f.fps >= 5. for low, high in candidate.frame_intervals)


def test_huddle_cannot_inherit_five_seconds_from_prejoin_members(monkeypatch):
    f = fixtures.features(180, 4)
    fixtures.set_centers(f, [[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
    f.valid[:70, 3] = False
    f.centers_cm[:70, 3] = np.nan
    fixtures.sync(f)
    monkeypatch.setattr(group, "_proposal_sets", lambda *args: [(0, 1, 2, 3)])
    candidates = group.group_rule_candidates(f, fixtures.confirmed_params(), (), behaviors=("huddle",))
    assert not candidates, "four-member phase lasts under 5 s; earlier trio time cannot qualify it"


def test_social_formation_core_cannot_export_only_its_dense_tail(monkeypatch, tmp_path):
    from mouse_behavior import canonical_video_pipeline as pipeline
    event = dict(behavior="Social clustering", canonical_behavior="social_clustering", mouse_ids=[1, 2, 3],
        start_frame=0, end_frame=90, canonical_support_frame_intervals=[[0, 40]])
    monkeypatch.setattr(pipeline.TrackStore, "window", lambda self, a, b: [])
    monkeypatch.setattr(pipeline, "infer_window_events", lambda *args, **kwargs: ([event], {}))
    events, _ = pipeline._infer_core_task((str(tmp_path / "unused.sqlite"), 30., {},
        fixtures.confirmed_params(), 60, 90, 0, 90, "disabled"))
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 90)]
    assert events[0]["canonical_support_frame_intervals"] == [(0, 40)]
    assert events[0]["retained_complete_parent_for_semantics"]


def test_approach_role_child_cannot_borrow_its_parent_endpoint():
    from mouse_behavior.canonical_video_pipeline import _directed_events
    from mouse_behavior.evaluation.approach_events import requalify_approach_event
    from mouse_behavior.evaluation.confirmed_rules import _open_approach_mask
    from tests.regression.test_continuation_policy_v16 import _approaching_pair
    f = _approaching_pair()
    p = fixtures.confirmed_params()
    mask = _open_approach_mask(f, 0, p)
    assert mask.all()
    ids = list(map(int, f.track_ids))
    actors = np.r_[np.full(40, ids[0]), np.full(20, ids[1])]
    targets = np.where(actors == ids[0], ids[1], ids[0])
    children = _directed_events(mask, actors, targets, ids, "approach", f.valid.all(axis=1),
        fps=f.fps, actor_centers={ids[i]: f.centers_cm[:, i] for i in range(2)}, min_actor_path_cm=10.)
    assert children and children[0]["end_frame"] == 40
    assert not requalify_approach_event(f, 0, children[0], p), "duration/path pass, but the new endpoint is still too far"


def test_isolation_retains_process_but_never_credits_missing_comparison_rows():
    from mouse_behavior.behavior.duration_policy import qualify_event_duration
    f = fixtures.features(480, 2)
    fixtures.set_centers(f, [[0., 0.], [30., 0.]])
    f.valid[100:115, 0] = False
    f.valid[220:235, 0] = False
    f.valid[340:355, 0] = False
    f.centers_cm[~f.valid] = np.nan
    fixtures.sync(f)
    p = fixtures.confirmed_params()
    candidate = next(c for c in group.isolation_candidates(f, p) if c.target_ids == (100,))
    assert candidate.frame_intervals == ((0, 480),)
    assert candidate.support_frame_intervals == ((0, 100), (115, 220), (235, 340), (355, 480))
    event = dict(canonical_behavior="isolation", start_frame=0, end_frame=480,
                 canonical_support_frame_intervals=candidate.support_frame_intervals)
    assert qualify_event_duration(event, f.fps)["ok"]
    event["canonical_support_frame_intervals"] = ((0, 300),)
    assert not qualify_event_duration(event, f.fps)["ok"], "exactly 10 s valid time is insufficient despite 16 s playback"
    f.centers_cm[400, 1] = [20., 0.]
    fixtures.sync(f)
    assert not any(c.target_ids == (100,) for c in group.isolation_candidates(f, p)), "a single measured near frame vetoes the whole process"
