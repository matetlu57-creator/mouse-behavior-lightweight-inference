"""Executable GRP-8 evidence audit on the public v17 candidate/feature path.

Group evidence has distinct legitimate definitions: huddle counts compact rows,
clustering counts measured formation and confirmed dense state, locomotion counts its
motion-process span, isolation counts coobserved rows, and dispersal evaluates
an exact five-second interval. Equal units do not make these scores equivalent
biological quantities or independently calibrated ranking probabilities.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.regression import test_heuristics_v16_audit as harness
from mouse_behavior.evaluation.calibrated_rule_ranker import build_candidate_table
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_models import GROUP_BEHAVIORS
from mouse_behavior.evaluation.group_event_rules import group_rule_candidates, isolation_candidates


FPS = 10.0
IDS = (100, 101, 102)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parameters():
    """Load the shipped public configuration; no legacy parameter construction."""
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    result = config_seed_parameters(config)
    assert result.confirmed_rules_v1 > 0
    assert result.huddle_min_duration_s == 5.0
    return result


def _base(frames):
    f = harness.features(frames, 3, FPS, reference_body_cm=7.918)
    harness.set_centers(f, [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    return f


def _finish(f):
    f.centers_cm[~f.valid] = np.nan
    harness.sync(f)
    harness.resync_kinematics(f)
    return f


def _huddle_with_unknown_gaps():
    f = _base(120)
    f.valid[:] = False
    f.valid[10:40] = True
    f.valid[60:100] = True
    return _finish(f)


def _clustering(frames):
    f = _base(frames)
    x = np.full(frames, 15.0)
    x[10:35] = 15.0 - np.arange(25) * 0.5
    x[35:] = 3.0
    f.centers_cm[:, 2, 0] = x
    f.centers_cm[:, 2, 1] = 0.0
    # Measured directed snout/nearest-body distances supply actual region
    # geometry to the canonical fallback for an export without pose points.
    shape = f.pair_distance_cm.shape
    f.pair_nose_nearest_left_to_right_cm = np.full(shape, 50.0)
    f.pair_nose_nearest_right_to_left_cm = np.full(shape, 50.0)
    for edge, (_, right) in enumerate(zip(f.pair_i, f.pair_j, strict=True)):
        if right == 2:
            f.pair_nose_nearest_right_to_left_cm[:, edge] = np.maximum(0.0, x - 2.0)
    return _finish(f)


def _dispersal(frames):
    f = _base(frames)
    f.valid[:10] = False
    distance = np.zeros(frames)
    distance[20:81] = np.arange(61) * 0.3
    distance[81:] = 18.0
    f.centers_cm[:, 0, 0] -= distance
    f.centers_cm[:, 1, 0] += distance
    return _finish(f)


def _locomotion():
    f = _base(100)
    x = np.zeros(100)
    x[10:61] = np.arange(51) * 1.0  # 10cm/s exceeds public6.6875cm/s active floor.
    x[61:] = 50.0
    f.centers_cm[:, :, 0] += x[:, None]
    return _finish(f)


def _isolation():
    f = _base(120)
    harness.set_centers(f, [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    return _finish(f)


def _evidence(candidates, behavior, ids):
    matching = [c.evidence for c in candidates if c.behavior == behavior and c.target_ids == ids]
    assert matching, f"fixture failed to emit {behavior} on {ids}"
    return max(matching)


def _canonical_and_ranker(f, parameters):
    candidates = (*group_rule_candidates(f, parameters, ()), *isolation_candidates(f, parameters))
    table = build_candidate_table(f, [parameters], behavior_subset=GROUP_BEHAVIORS)
    return candidates, table


def _assert_evidence_and_fractions(f, candidates, table):
    duration = len(f.valid) / f.fps
    assert candidates, "adversary must emit real canonical group evidence"
    assert all(np.isfinite(c.evidence) and 0 <= c.evidence <= duration for c in candidates)
    assert all(np.isfinite(c.evidence) and 0 <= c.evidence <= duration for c in table.candidates)
    columns = [i for i, name in enumerate(table.feature_names)
               if name == "evidence_fraction"
               or (name.startswith("rule_") and name.endswith("_fraction"))]
    assert columns
    values = table.values[:, columns]
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert (values <= 1.0 + 1e-6).all()
    # Check the actual candidate-to-feature conversion, including the
    # aggregate evidence slot, rather than only checking column names.
    evidence_column = table.feature_names.index("evidence_fraction")
    for row, candidate in enumerate(table.candidates):
        assert table.values[row, evidence_column] == pytest.approx(
            candidate.evidence / duration, abs=1e-6)


def test_huddle_evidence_counts_compact_support_not_unknown_episode_envelope(parameters):
    f = _huddle_with_unknown_gaps()
    candidates, table = _canonical_and_ranker(f, parameters)
    # 30 + 40 observed compact rows. Envelope is rows10..99 (9s); clip is12s.
    expected = 70 / FPS
    assert _evidence(candidates, "huddle", IDS) == pytest.approx(expected)
    assert _evidence(table.candidates, "huddle", IDS) == pytest.approx(expected)
    _assert_evidence_and_fractions(f, candidates, table)


@pytest.mark.parametrize("frames", [60, 140])
def test_clustering_includes_measured_formation_prefix_before_dense_state(parameters, frames):
    f = _clustering(frames)
    candidates, table = _canonical_and_ranker(f, parameters)
    # The observed actor starts moving at11; onset anchor10 belongs to that
    # real adjacent step. V29 retrospectively confirms it after final density.
    assert np.flatnonzero(f.speed_cm_s[:, 2] > 0)[0] == 11
    assert not f.speed_cm_s[35:, 2].any()
    expected = (frames - 10) / FPS
    assert next(c for c in candidates if c.behavior == 'social_clustering'
                and c.target_ids == IDS).frame_intervals == ((10, frames),)
    assert _evidence(candidates, "social_clustering", IDS) == pytest.approx(expected)
    assert _evidence(table.candidates, "social_clustering", IDS) == pytest.approx(expected)
    _assert_evidence_and_fractions(f, candidates, table)


def test_dispersal_evidence_is_motion_interval_not_confirmation_or_remaining_clip_tail(parameters):
    evidence = []
    for frames in (90, 180):
        f = _dispersal(frames)
        candidates, table = _canonical_and_ranker(f, parameters)
        expected_ids = (100, 101)
        canonical = _evidence(candidates, "dispersal", expected_ids)
        assert _evidence(table.candidates, "dispersal", expected_ids) == pytest.approx(canonical)
        evidence.append(canonical)
        _assert_evidence_and_fractions(f, candidates, table)
    assert evidence[0] == pytest.approx(evidence[1]), "quiet clip tail changes dispersal evidence"
    # +5 s is confirmation only. Center-derived 0.3 s motion smoothing gives
    # support boundaries 2.1..8.2 s; elapsed interval excludes the final frame bin.
    assert evidence[0] == pytest.approx(6.1), "actual departure must exceed5s without counting quiet tail"


def test_locomotion_evidence_has_seconds_units_without_rank_boost(parameters):
    f = _locomotion()
    candidates, table = _canonical_and_ranker(f, parameters)
    # Fifty measured moving rows11..60 form the five-second process.
    assert _evidence(candidates, "group_locomotion", IDS) == pytest.approx(5.0)
    assert _evidence(table.candidates, "group_locomotion", IDS) == pytest.approx(5.0)
    _assert_evidence_and_fractions(f, candidates, table)


def test_isolation_evidence_is_coobserved_seconds_and_fraction_one(parameters):
    f = _isolation()
    candidates, table = _canonical_and_ranker(f, parameters)
    assert _evidence(candidates, "isolation", (100,)) == pytest.approx(12.0)
    assert _evidence(table.candidates, "isolation", (100,)) == pytest.approx(12.0)
    _assert_evidence_and_fractions(f, candidates, table)


@pytest.mark.parametrize("fixture", [
    _huddle_with_unknown_gaps, lambda: _clustering(60), lambda: _clustering(140),
    lambda: _dispersal(90), _locomotion, _isolation,
], ids=["huddle-unknown", "clustering-short", "clustering-long", "dispersal", "locomotion", "isolation"])
def test_all_emitted_group_seconds_and_actual_ranker_fractions_are_bounded(parameters, fixture):
    f = fixture()
    candidates, table = _canonical_and_ranker(f, parameters)
    _assert_evidence_and_fractions(f, candidates, table)


def test_huddle_supported_forward_hold_counts_compact_rows_without_raw_id_reappearance(parameters):
    # Held ID rows30..79 remain supported for the authorized5s. Both other
    # members remain observed and quiet at the same spatial pile throughout.
    short = _base(80)
    short.valid[30:, 2] = False
    short = _finish(short)
    short_candidates, short_table = _canonical_and_ranker(short, parameters)
    assert _evidence(short_candidates, "huddle", IDS) == pytest.approx(8.0)
    assert _evidence(short_table.candidates, "huddle", IDS) == pytest.approx(8.0)
    assert short.valid.all(axis=1).sum() / FPS == 3.0
    _assert_evidence_and_fractions(short, short_candidates, short_table)


def test_observed_huddle_separation_splits_support_instead_of_bridging_it(parameters):
    f = _base(120)
    f.centers_cm[40:60, 2] = [30.0, 30.0]
    f = _finish(f)
    candidates, table = _canonical_and_ranker(f, parameters)
    # Visible separation splits4s and6s compact intervals. Their sum10s is not
    # one huddle; the longer qualifying6s interval is the published evidence.
    assert _evidence(candidates, "huddle", IDS) == pytest.approx(6.0)
    assert _evidence(table.candidates, "huddle", IDS) == pytest.approx(6.0)
    _assert_evidence_and_fractions(f, candidates, table)


def test_dispersal_missing_exact_checkpoint_uses_a_later_measured_confirmation(parameters):
    f = _dispersal(90)
    f.valid[60, 1] = False
    f = _finish(f)
    assert f.valid[59, 1] and f.valid[61, 1]
    candidates, table = _canonical_and_ranker(f, parameters)
    assert any(c.behavior == "dispersal" and c.target_ids == (100, 101) for c in candidates)
    assert any(c.behavior == "dispersal" and c.target_ids == (100, 101) for c in table.candidates)


def test_dispersal_without_any_later_departed_pair_observation_is_unknown(parameters):
    f = _dispersal(90)
    f.valid[60:, 1] = False
    f = _finish(f)
    candidates, table = _canonical_and_ranker(f, parameters)
    assert not any(c.behavior == "dispersal" and c.evidence > 0 for c in candidates)
    assert not any(c.behavior == "dispersal" and c.evidence > 0 for c in table.candidates)


def test_outward_movement_without_formation_does_not_emit_clustering(parameters):
    f = _clustering(60)
    # Reverse the actual actor trajectory and directed-distance measurements:
    # it starts outside and recedes, never approaching or forming a group.
    x = 30.0 - f.centers_cm[:, 2, 0]
    f.centers_cm[:, 2, 0] = x
    for edge, (_, right) in enumerate(zip(f.pair_i, f.pair_j, strict=True)):
        if right == 2:
            f.pair_nose_nearest_right_to_left_cm[:, edge] = x - 2.0
    f = _finish(f)
    candidates, table = _canonical_and_ranker(f, parameters)
    assert not any(c.behavior == "social_clustering" and c.evidence > 0 for c in candidates)
    assert not any(c.behavior == "social_clustering" and c.evidence > 0 for c in table.candidates)
