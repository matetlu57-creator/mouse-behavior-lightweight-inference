"""Counterexamples for frame support and independent temporal events."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression import test_group_evidence_seconds_v17 as group_fixtures
from tests.regression import test_heuristics_v18_behavior_specific as v18_fixtures
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.blind_top1 import Candidate, candidates_from_evidence


def _evaluator():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_active_dataset_v17.py"
    spec = importlib.util.spec_from_file_location("active_v19_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_old_three_argument_candidate_still_works():
    candidate = Candidate("dispersal", (1, 2), 2.0)
    assert candidate.evidence == 2.0
    assert candidate.frame_intervals is None


@pytest.mark.parametrize("intervals", [((4, 2),), ((-1, 3),), ((0, 4), (3, 6)), ((True, 3),)])
def test_candidate_rejects_malformed_frame_intervals(intervals):
    with pytest.raises(ValueError, match="interval"):
        Candidate("dispersal", (1, 2), 2.0, frame_intervals=intervals)


def test_frame_coverage_uses_union_and_validates_metadata_bounds():
    evaluator = _evaluator()
    candidate = Candidate("dispersal", (1, 2), 8.0, frame_intervals=((0, 10), (70, 80)))
    row = evaluator.group_candidate_coverage(candidate, 100, 10.0)
    assert row["coverage"] == pytest.approx(0.2), "a long gap must not become event evidence"
    assert row["legacy_evidence_coverage"] == pytest.approx(0.8)
    assert row["coverage_source"] == "event_frames"
    with pytest.raises(ValueError, match="interval"):
        evaluator.group_candidate_coverage(candidate, 75, 10.0)
    legacy = evaluator.group_candidate_coverage(Candidate("dispersal", (1, 2), 2.0), 100, 10.0)
    assert legacy["coverage_source"] == "legacy_evidence_proxy"


def test_public_group_outputs_retain_exact_process_frames():
    params = audit.confirmed_params()
    for fixture, behavior in [
        (group_fixtures._locomotion, "group_locomotion"),
        (lambda: group_fixtures._clustering(140), "social_clustering"),
        (lambda: group_fixtures._dispersal(180), "dispersal"),
        (group_fixtures._huddle_with_unknown_gaps, "huddle"),
    ]:
        f = fixture()
        candidates = group.group_rule_candidates(f, params, (), behaviors=(behavior,))
        assert candidates, behavior
        for c in candidates:
            assert c.frame_intervals, behavior
            assert all(0 <= lo < hi <= len(f.valid) for lo, hi in c.frame_intervals)
            if behavior == "dispersal":
                assert max(hi for lo, hi in c.frame_intervals) < round(9 * f.fps), (
                    "quiet tail is not a departure"
                )
            if behavior == "group_locomotion":
                assert c.support_frame_intervals, "simultaneous support differs from process span"


def test_three_target_group_and_frame_metadata_survive_prediction_json():
    # Blind target sets require three members; lower-level departure processes
    # may still detect two leavers from an initial group of at least three.
    prediction = {
        "group_scores": {"1,2,3": {"dispersal": 2.0}},
        "group_events": [
            {
                "behavior": "dispersal",
                "member_ids": [1, 2, 3],
                "evidence_seconds": 2.0,
                "frame_intervals": [[10, 30]],
                "support_frame_intervals": None,
            }
        ],
    }
    candidates = candidates_from_evidence(prediction)
    assert len(candidates) == 1
    assert candidates[0].target_ids == (1, 2, 3)
    assert candidates[0].frame_intervals == ((10, 30),)
    assert candidates[0].support_frame_intervals is None
    assert not candidates_from_evidence({"group_scores": {"1,2": {"dispersal": 2.0}}})
    assert not candidates_from_evidence({"group_scores": {"1": {"dispersal": 2.0}}})


def test_ranker_retains_group_frames_without_changing_seconds_features():
    from mouse_behavior.evaluation.calibrated_rule_ranker import build_candidate_table

    f = group_fixtures._locomotion()
    table = build_candidate_table(
        f, [audit.confirmed_params()], behavior_subset=("group_locomotion",)
    )
    candidate = next(c for c in table.candidates if c.evidence > 0)
    assert candidate.frame_intervals == ((11, 61),)
    row = table.candidates.index(candidate)
    assert table.values[row, table.feature_names.index("evidence_fraction")] == pytest.approx(
        5.0 / 10.0
    )


@pytest.mark.parametrize("support_frames,expected", [(27, False), (28, True)])
def test_fitted_group_sync_support_still_rejects_shorter_support(support_frames, expected):
    f = audit.features(120, 3)
    time = np.arange(120) / f.fps
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 20 * time + slot * 2.0
    f.centers_cm[:, 2, 0] = 20 * np.minimum(time, support_frames / f.fps) + 4.0
    audit.sync(f)
    audit.resync_kinematics(f)
    # This fixture counts measured inter-frame displacements; the harness
    # copies row1's speed to row0, which would otherwise add an extra bin.
    f.velocity_cm_s[0] = np.nan
    f.speed_cm_s[0] = np.nan
    cfg = group.DEFAULT_GROUP_HEURISTIC_CONFIG
    params = audit.confirmed_params(
        group_locomotion_max_distance_cm=cfg.locomotion_max_pair_distance_cm
    )
    result = group._synchronous_event(f, (0, 1, 2), f.velocity_cm_s, f.speed_cm_s, params, cfg)
    assert bool(result > 0) == expected


def test_social_priority_preserves_later_nonoverlapping_attack(monkeypatch):
    f = audit.features(120, 2)
    audit.set_centers(f, [[0.0, 0.0], [20.0, 0.0]])
    audit.sync(f)
    approach = np.zeros(120, bool)
    approach[:15] = True
    attack = np.zeros(120, bool)
    attack[60:78] = True
    monkeypatch.setattr(social, "_open_approach_mask", lambda *a, **k: approach)
    monkeypatch.setattr(social, "_attack_action_mask", lambda *a, **k: attack)
    monkeypatch.setattr(social, "_chase_mask", lambda *a, **k: np.zeros(120, bool))
    result = social.pair_rule_masks(f, audit.confirmed_params())
    np.testing.assert_array_equal(result["approach"][:, 0], approach)
    np.testing.assert_array_equal(result["attack"][:, 0], attack)
    assert not (result["approach"] & result["attack"]).any()


def test_attack_keeps_directed_deceleration_after_a_real_acceleration_trigger(monkeypatch):
    f = audit.features(30, 2)
    audit.set_centers(f, [[0.0, 0.0], [6.0, 0.0]])
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)
    f.pair_nose_tail_left_to_right_cm = np.full((30, 1), 50.0)
    f.pair_nose_tail_right_to_left_cm = np.full((30, 1), 50.0)

    def measured(positions, *a, **kw):
        velocity = np.zeros((30, 2))
        acceleration = np.zeros((30, 2))
        if positions[0, 0] < 3:
            velocity[5:23, 0] = 20.0
            acceleration[5:11, 0] = 150.0
            acceleration[11:23, 0] = -150.0
        return velocity, acceleration

    monkeypatch.setattr(social, "_local_quadratic_point_kinematics", measured)
    result = social._attack_action_mask(f, 0)
    assert result[5:23].all(), "a directed pounce does not end when acceleration turns negative"
    assert not result[:5].any() and not result[23:].any()


def test_center_trajectory_pursuit_does_not_require_an_unused_head_triangle():
    f = audit.features(90, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = np.arange(90) / f.fps * 60.0 + slot * 20.0
    audit.sync(f)
    audit.resync_kinematics(f)
    f.heading_unit = None
    f.pair_nose_head_left_to_right_cm = np.full((90, 1), np.nan)
    f.pair_nose_head_right_to_left_cm = np.full((90, 1), np.nan)
    assert social._chase_mask(f, 0).any()


def test_turning_leader_can_be_followed_without_instantaneous_parallel_velocity():
    f = audit.features(150, 2)
    time = np.arange(150) / f.fps
    for slot, delay in [(0, 0.3), (1, 0.0)]:
        shifted = time - delay
        f.centers_cm[:, slot, 0] = 60 * np.minimum(shifted, 2.0)
        f.centers_cm[:, slot, 1] = 60 * np.maximum(shifted - 2.0, 0.0)
    audit.sync(f)
    audit.resync_kinematics(f)
    # During the corner, follower moves east while leader already moves north.
    cosine = social._pursuit_direction_cosine(f, 0, 1, social.DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    corner = (time > 2.05) & (time < 2.25)
    assert (cosine[corner] > 0.9).all()
    assert social._chase_mask(f, 0)[corner].all()


@pytest.mark.parametrize("opposing", [False, True])
def test_parallel_side_by_side_or_opposing_motion_is_not_pursuit(opposing):
    f = audit.features(120, 2)
    time = np.arange(120) / f.fps
    f.centers_cm[:, 0, 0] = 20 * time
    f.centers_cm[:, 1, 0] = (-20 if opposing else 20) * time
    f.centers_cm[:, 1, 1] = 10.0
    audit.sync(f)
    audit.resync_kinematics(f)
    assert not social._chase_mask(f, 0).any()


def test_unknown_chase_gap_is_unlabeled_and_an_observed_stop_still_splits():
    f = audit.features(120, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 60 * np.arange(120) / f.fps + slot * 20.0
    audit.sync(f)
    audit.resync_kinematics(f)
    f.valid[40:46] = False
    dropout = social._chase_mask(f, 0)
    assert not dropout[40:46].any()
    assert dropout[:40].any() and dropout[46:].any()
    f.valid[40:46] = True
    f.speed_cm_s[40:46] = 0.0
    f.velocity_cm_s[40:46] = 0.0
    stopped = social._chase_mask(f, 0)
    assert not stopped[40:46].any()


def test_observed_return_to_initial_pile_ends_departure():
    f = v18_fixtures._departure()
    f.centers_cm[210:, 0] = f.centers_cm[0, 0]
    audit.sync(f)
    audit.resync_kinematics(f)
    phase = group.dispersal_phase(
        f.pair_distance_cm,
        np.ones(len(f.valid), bool),
        np.ones(len(f.valid), bool),
        audit.confirmed_params(),
        f.fps,
        centers_cm=f.centers_cm,
        member_valid=f.valid,
    )
    assert phase.any()
    assert not phase[210:].any()
