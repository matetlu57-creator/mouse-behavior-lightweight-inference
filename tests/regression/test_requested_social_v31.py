"""Requested event qualification must use observed action, never crop padding."""
from dataclasses import replace

import numpy as np
import pytest

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.evaluation.confirmed_rules import _open_approach_mask, _qualify_observed_chase
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG as DEFAULT
from tests.regression import test_heuristics_v16_audit as audit


def closing_pair(count=31, step=.4):
    f = audit.features(count, 2, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.arange(count) * step
    f.centers_cm[:, 1, 0] = 22.
    audit.sync(f)
    audit.resync_kinematics(f)
    shape = np.array([[3.959, 0.], [2.5, .8], [2.5, -.8], [1., 0.],
                      [-2., 1.], [-2., -1.], [-3.959, 0.]])
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    return f


def test_requested_default_gates():
    assert DEFAULT.approach_min_duration_s == 0.3
    assert DEFAULT.approach_min_actor_path_cm == 0.0
    assert DEFAULT.attack_min_duration_s == 0.15
    assert DEFAULT.chasing_min_mean_speed_cm_s == 15.0


def test_policy_receipt_describes_the_executed_revision():
    from mouse_behavior.evaluation.confirmed_rules import policy_receipt

    receipt = policy_receipt()
    assert receipt["version"] == "confirmed_rules_v31"
    assert receipt["requested_event_minima"] == dict(approach_duration_s=0.3,
        approach_duration_comparator=">", approach_actor_path_cm=0.0,
        attack_duration_s=0.15, chase_each_mean_speed_cm_s=15.0)
    assert "15.0cm/s" in receipt["chasing_mean_speed_contract"]


@pytest.mark.parametrize("name,count,ok", [("approach", 9, False), ("approach", 10, True),
                                         ("attack", 4, False), ("attack", 5, True)])
def test_event_duration_excludes_context(name, count, ok):
    e = dict(canonical_behavior=name, start_frame=20, end_frame=20 + count)
    assert qualify_event_duration(e, 30.)["ok"] is ok


def test_actor_ten_cm_and_strict_one_second_are_both_needed():
    v31 = replace(DEFAULT, approach_min_duration_s=1., approach_min_actor_path_cm=10.)
    assert _open_approach_mask(closing_pair(), 0, config_seed_parameters({}), v31).any()
    assert not _open_approach_mask(closing_pair(30), 0, config_seed_parameters({}), v31).any()
    slow = closing_pair(60, .12)
    slow.centers_cm[:, 1, 0] -= 4.
    slow.collective_keypoints_cm[:, 1, :, 0] -= 4.
    audit.sync(slow)
    assert not _open_approach_mask(slow, 0, config_seed_parameters({}), v31).any()
    permissive = replace(DEFAULT, approach_min_actor_path_cm=0.)
    assert _open_approach_mask(slow, 0, config_seed_parameters({}), permissive).any()


def test_missing_actor_interval_cannot_supply_ten_cm_jump():
    v31 = replace(DEFAULT, approach_min_duration_s=1., approach_min_actor_path_cm=10.)
    f = closing_pair(60, .2)
    # Real observed adjacent travel is only 7.2cm after removal; endpoints
    # still span11.8cm. The missing jump must not satisfy the10cm gate.
    f.valid[20:40, 0] = False
    assert not _open_approach_mask(f, 0, config_seed_parameters({}), v31).any()


@pytest.mark.parametrize("speeds,positive", [((14., 20.), False), ((15., 15.), True)])
def test_each_chase_participant_must_reach_mean_thirty(speeds, positive):
    n, fps = 61, 30.
    centers = np.zeros((n, 2, 2))
    for slot, speed in enumerate(speeds):
        centers[:, slot, 0] = np.arange(n) * speed / fps
    mask = np.ones(n, bool)
    result = _qualify_observed_chase(mask, mask, centers, 0, 1, fps, DEFAULT)
    assert bool(result.any()) is positive


def test_role_defined_children_cannot_borrow_parent_actor_travel():
    from mouse_behavior.canonical_video_pipeline import _directed_events

    n = 100
    actors = np.r_[np.full(50, 100), np.full(50, 101)]
    targets = np.where(actors == 100, 101, 100)
    centers = {100: np.column_stack((np.arange(n) * .12, np.zeros(n))),
               101: np.column_stack((-np.arange(n) * .12, np.zeros(n)))}
    # Parent is3.33s and each actor's whole-parent path11.88cm; each real
    # role-defined child is1.67s but travels only5.88cm.
    mask = np.ones(n, bool)
    assert len(_directed_events(mask, actors, targets, [100, 101], "approach", mask, fps=30.)) == 2
    assert not _directed_events(mask, actors, targets, [100, 101], "approach", mask,
        fps=30., actor_centers=centers, min_actor_path_cm=10.)


def test_full_role_pipeline_rechecks_each_approach_child(tmp_path):
    from mouse_behavior.canonical_video_pipeline import infer_window_events, _directed_events
    from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks
    from tests.regression.test_social_arbitration_role_boundaries import _rows, _features, FPS, SCALE

    t = np.arange(120) / FPS
    rows = _rows(3.6 * t, 30. - 3.6 * t)
    for i, row in enumerate(rows):
        for point in row["detections"][1]["keypoints"]:
            point[0] = 2 * (30. - 3.6 * t[i]) / .1 - point[0]
        # Missing neck supplies no invented facing; each half has one real
        # measured actor. Both directions qualify at the parent-mask level.
        row["detections"][1 if i < 50 else 0]["keypoints"][3][2] = 0.
    f = _features(rows, tmp_path)
    p = config_seed_parameters({})
    roles = {}
    masks = pair_rule_masks(f, p, roles=roles)
    joint = f.valid.all(axis=1)
    old_children = _directed_events(masks["approach"][:, 0], roles["approach_actor"][:, 0],
        roles["approach_target"][:, 0], [100, 101], "approach", joint, fps=FPS)
    assert len(old_children) == 2
    assert all((e["end_frame"] - e["start_frame"]) / FPS > 0.3 for e in old_children)
    events, _ = infer_window_events(rows, FPS, SCALE, p)
    assert len([e for e in events if e["canonical_behavior"] == "approach"]) == 2


def test_chase_role_children_recheck_their_own_average():
    from mouse_behavior.canonical_video_pipeline import _directed_events

    n = 180
    steps = np.r_[np.full(90, 10.), np.full(89, 20.)] / 30.
    x = np.r_[0., np.cumsum(steps)]
    centers = {100: np.column_stack((x, np.zeros(n))),
               101: np.column_stack((x + 10., np.zeros(n)))}
    actors = np.r_[np.full(90, 100), np.full(90, 101)]
    targets = np.where(actors == 100, 101, 100)
    mask = np.ones(n, bool)
    events = _directed_events(mask, actors, targets, [100, 101], "chase", mask,
                              fps=30., actor_centers=centers)
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(90, 180)]


def test_merged_chases_cannot_double_count_the_fast_overlap(monkeypatch):
    from copy import deepcopy
    from pathlib import Path
    from mouse_behavior import canonical_video_pipeline as pipeline
    from tests.regression.test_social_arbitration_role_boundaries import _rows, SCALE

    speeds = np.full(149, 10.)
    speeds[59:89] = 30.
    x = np.r_[0., np.cumsum(speeds / 30.)]
    rows = _rows(x, x + 10.)

    class Store:
        path = Path("unused.sqlite")

        def window(self, start, end):
            result = deepcopy(rows[start:end])
            for i, row in enumerate(result):
                row["frame"] = i
            return result

        def rows(self, start, end):
            return iter(self.window(start, end))

    def core(spec):
        begin = spec[4]
        start, end = (0, 90) if begin == 0 else (60, 150)
        e = pipeline._event("chase", [100, 101], start, end, 100, 101)
        return [e], dict(core_start=begin, core_end=spec[5], track_ids=[100, 101])

    monkeypatch.setattr(pipeline, "_infer_core_task", core)
    events, audit_log = pipeline.infer_video_events(Store(),
        dict(source_video=dict(fps=30., frame_count=150)), SCALE, config_seed_parameters({}),
        core_seconds=3.)
    assert not events
    rejected = audit_log["merged_chase_motion_rejections"]
    assert len(rejected) == 1
    assert all(s["mean_speed_cm_s"] < 15. for s in rejected[0]["statistics"])


def test_inference_windows_cannot_hide_an_earlier_near_frame(monkeypatch, tmp_path):
    from pathlib import Path
    from mouse_behavior import canonical_video_pipeline as pipeline
    from tests.regression.test_social_arbitration_role_boundaries import _rows, _features, SCALE

    rows = _rows(np.zeros(400), np.r_[np.full(40, 5.), np.full(360, 30.)])

    class Store:
        path = Path("unused.sqlite")

        def rows(self, start, end):
            return iter(rows[start:end])

    params = config_seed_parameters({})
    binding = dict(source_video=dict(fps=30., frame_count=400))
    audit_log = pipeline.whole_range_isolation_audit(Store(), binding, SCALE, params)
    f = _features(rows, tmp_path)
    assert audit_log["tracks"]["100"]["minimum_cm"] == pytest.approx(float(f.nearest_distance_cm[:, 0].min()))
    assert audit_log["tracks"]["100"]["qualified"] is False

    def core(spec):
        return [pipeline._event("isolation", [100], 40, 400)], dict(
            core_start=0, core_end=400, track_ids=[100, 101])

    monkeypatch.setattr(pipeline, "_infer_core_task", core)
    events, _ = pipeline.infer_video_events(Store(), binding, SCALE, params)
    assert not events


def test_earlier_near_observation_vetoes_later_far_isolation():
    from mouse_behavior.evaluation.group_event_rules import isolation_candidates

    f = audit.features(400, 2, reference_body_cm=7.918)
    f.centers_cm[:, 1, 0] = 30.
    f.centers_cm[:40, 1, 0] = 5.
    audit.sync(f)
    assert not isolation_candidates(f, config_seed_parameters({}))
    f.centers_cm[:40, 1, 0] = 30.
    audit.sync(f)
    assert len(isolation_candidates(f, config_seed_parameters({}))) == 2


def test_isolation_near_frame_vetoes_and_missing_never_adds_time():
    from mouse_behavior.evaluation.group_event_rules import isolation_candidates

    f = audit.features(601, 2, reference_body_cm=7.918)
    f.centers_cm[:, 1, 0] = 30.
    f.centers_cm[300, 1, 0] = 20.  # strict >20; splits into exactly10s halves
    audit.sync(f)
    assert not isolation_candidates(f, config_seed_parameters({}))
    f = audit.features(301, 2, reference_body_cm=7.918)
    f.centers_cm[:, 1, 0] = 30.
    f.valid[150] = False
    audit.sync(f)
    assert not isolation_candidates(f, config_seed_parameters({}))


def test_together_contact_unknown_cannot_be_called_absent():
    from mouse_behavior.evaluation.confirmed_rules import together_contact_evidence

    f = audit.features(31, 2, reference_body_cm=7.918)
    for kind in ("head", "tail"):
        for direction in ("left_to_right", "right_to_left"):
            setattr(f, f"pair_nose_{kind}_{direction}_cm", np.full((31, 1), 30.))
    positive, known = together_contact_evidence(f, 0)
    assert known.all() and not positive.any()
    f.pair_nose_tail_right_to_left_cm[:] = np.nan
    positive, known = together_contact_evidence(f, 0)
    assert not known.any() and not positive.any()
    f.pair_nose_head_left_to_right_cm[:] = .1
    positive, known = together_contact_evidence(f, 0)
    assert positive.all() and not known.any()
