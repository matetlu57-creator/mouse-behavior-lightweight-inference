"""Minimum durations qualify a process; they never prescribe its crop length."""
import numpy as np
from mouse_behavior.behavior.duration_policy import qualify_event_duration

from mouse_behavior import canonical_video_pipeline as pipeline
from mouse_behavior.evaluation import group_event_rules as group
from tests.regression import test_heuristics_v16_audit as fixtures


def test_huddle_short_unknown_gap_keeps_complete_eight_second_process(monkeypatch):
    f = fixtures.features(240, 3, 30.)
    fixtures.set_centers(f, [[0., 0.], [2., 0.], [0., 2.]])
    f.valid[90:96] = False
    f.centers_cm[90:96] = np.nan
    fixtures.sync(f)
    monkeypatch.setattr(group, "_proposal_sets", lambda *args: [(0, 1, 2)])
    before = f.valid.copy()
    candidates = group.group_rule_candidates(f, fixtures.confirmed_params(), (), behaviors=("huddle",))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.frame_intervals == ((0, 240),)
    assert candidate.support_frame_intervals == ((0, 90), (96, 240))
    assert candidate.evidence == 234 / 30.
    np.testing.assert_array_equal(f.valid, before)


def test_huddle_reports_all_complete_bouts_and_stops_at_measured_separation(monkeypatch):
    f = fixtures.features(570, 3, 30.)
    fixtures.set_centers(f, [[0., 0.], [2., 0.], [0., 2.]])
    f.centers_cm[240:300, 2] = [100., 100.]
    fixtures.sync(f)
    monkeypatch.setattr(group, "_proposal_sets", lambda *args: [(0, 1, 2)])
    candidates = group.group_rule_candidates(f, fixtures.confirmed_params(), (), behaviors=("huddle",))
    assert sorted(interval for c in candidates for interval in c.frame_intervals) == [(0, 240), (300, 570)]


def test_core_boundary_never_shortens_an_already_qualified_parent(monkeypatch, tmp_path):
    event = dict(behavior="Huddling", canonical_behavior="huddle", mouse_ids=[1, 2, 3],
        start_frame=100, end_frame=700, canonical_support_frame_intervals=[[100, 700]])
    monkeypatch.setattr(pipeline.TrackStore, "window", lambda self, a, b: [])
    monkeypatch.setattr(pipeline, "infer_window_events", lambda *a, **kw: ([event], {}))
    selected, _ = pipeline._infer_core_task((str(tmp_path / "unused.sqlite"), 30., {},
        fixtures.confirmed_params(), 300, 600, 0, 900, "disabled"))
    assert [(e["start_frame"], e["end_frame"]) for e in selected] == [(100, 700)]
    assert selected[0]["canonical_support_frame_intervals"] == [(100, 700)]


def test_same_ids_merge_complete_contexts_without_merging_different_members():
    def event(low, high, ids):
        return dict(behavior="Huddling", canonical_behavior="huddle", mouse_ids=ids,
                    start_frame=low, end_frame=high,
                    canonical_support_frame_intervals=[[low, high]])
    merged = pipeline.merge_events([event(100, 700, [1, 2, 3]), event(600, 900, [1, 2, 3]),
                                    event(600, 900, [1, 2, 4])])
    assert [(e["start_frame"], e["end_frame"], e["mouse_ids"]) for e in merged] == [
        (100, 900, [1, 2, 3]), (600, 900, [1, 2, 4])]


def test_unknown_time_cannot_supply_missing_huddle_duration():
    short = dict(canonical_behavior="huddle", start_frame=0, end_frame=240,
                 canonical_support_frame_intervals=[[0, 30], [210, 240]])
    assert not qualify_event_duration(short, 30.)["ok"]
    complete = dict(short, canonical_support_frame_intervals=[[0, 90], [96, 240]])
    result = qualify_event_duration(complete, 30.)
    assert result["ok"] and result["actual_s"] == 8.
    assert result["support_actual_s"] == 7.8


def test_huddle_cannot_borrow_connectivity_from_a_member_it_excludes(monkeypatch):
    f = fixtures.features(330, 4, 30.)
    fixtures.set_centers(f, [[0., 0.], [2., 0.], [0., 2.], [14.5, 0.]])
    f.valid[:150, 3] = False
    f.centers_cm[:150, 3] = np.nan
    f.centers_cm[150:] = [[0., 0.], [7., 0.], [22., 0.], [14.5, 0.]]
    fixtures.sync(f)
    fixtures.resync_kinematics(f)
    monkeypatch.setattr(group, "_proposal_sets", lambda *args: [(0, 1, 2, 3)])
    candidates = group.group_rule_candidates(f, fixtures.confirmed_params(), (), behaviors=("huddle",))
    assert len(candidates) == 1
    assert candidates[0].target_ids == tuple(map(int, f.track_ids[:3]))
    assert candidates[0].frame_intervals == ((0, 150),)
    assert candidates[0].support_frame_intervals == ((0, 150),)
    assert candidates[0].evidence == 5.
