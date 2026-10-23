"""Counterexamples for contact occupancy and observed formation travel."""
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG as cfg


def moving_pair(frames=90, speed=20.):
    f = audit.features(frames, 2)
    f.centers_cm[:, :, 0] = np.arange(frames)[:, None] / f.fps * speed
    f.centers_cm[:, 1, 0] += 5.
    audit.sync(f)
    audit.resync_kinematics(f)
    for kind in ('head', 'tail'):
        for direction in ('left_to_right', 'right_to_left'):
            setattr(f, f'pair_nose_{kind}_{direction}_cm', np.full((frames, 1), 30.))
    return f


def test_lower_priority_tail_cannot_prevent_whole_directed_chase_requalification(monkeypatch):
    f = moving_pair()
    f.pair_nose_tail_left_to_right_cm[:, 0] = .5

    def approach_mask(f, *args, retained_mask=None, **kwargs):
        raw = np.arange(len(f.valid)) < 75
        # Requalification may only retain frames not claimed by the restored chase.
        return raw if retained_mask is None else raw & retained_mask

    monkeypatch.setattr(social, '_open_approach_mask', approach_mask)
    monkeypatch.setattr(social, '_chase_mask', lambda f, *a, **k: np.ones(len(f.valid), bool))
    monkeypatch.setattr(social, '_attack_action_mask', lambda f, *a, **k: np.zeros(len(f.valid), bool))
    output = social.pair_rule_masks(f, audit.confirmed_params(), replace(cfg, chasing_min_mean_speed_cm_s=15.))
    assert output['chasing'][:, 0].all()
    assert not output['tail_contact'].any()
    assert not output['approach'].any()


def test_real_contact_in_fast_parallel_motion_without_pursuit_is_retained(monkeypatch):
    f = moving_pair()
    f.pair_nose_tail_left_to_right_cm[:, 0] = .5
    for helper in ('_open_approach_mask', '_chase_mask', '_attack_action_mask'):
        monkeypatch.setattr(social, helper, lambda f, *a, **k: np.zeros(len(f.valid), bool))
    output = social.pair_rule_masks(f, audit.confirmed_params())
    assert output['tail_contact'][:, 0].all(), 'speed alone is not proof of pursuit'


@pytest.mark.parametrize('kind', ['head', 'tail'])
def test_contact_onset_is_measured_and_cannot_paint_earlier_distant_frames(kind):
    measured = np.r_[np.full(10, 5.), np.full(20, .5)]
    custom = replace(cfg, contact_head_onset_distance_cm=2., contact_tail_onset_distance_cm=3.)
    qualified, _ = social._bridged_contact_bout_mask(
        strict=np.ones(30, bool), measured=measured, center=np.full(30, 7.),
        body_cm=7.918, fps=30., social_config=custom, kind=kind)
    assert not qualified[:10].any()
    assert qualified[10:].all()


@pytest.mark.parametrize('speed,expected', [(14.99, False), (15., True), (15.01, True)])
def test_each_chaser_has_fifteen_cm_s_observed_mean(speed, expected):
    f = moving_pair(speed=speed)
    known = np.ones(len(f.valid), bool)
    assert bool(social._qualify_observed_chase(
        known, known, f.centers_cm, 0, 1, f.fps, cfg).any()) is expected


@pytest.mark.parametrize('travel,expected', [(9.99, False), (10., False), (10.01, True)])
def test_clustering_actor_travel_is_strictly_greater_than_ten(travel, expected):
    f = audit.features(150, 3)
    f.centers_cm[:, 0] = [3., 0.]
    f.centers_cm[:61, 0, 0] = np.linspace(3. + travel, 3., 61)
    f.centers_cm[:, 1] = [0., 0.]
    f.centers_cm[:, 2] = [0., 2.]
    audit.sync(f)
    audit.resync_kinematics(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    result = group._clustering_process(f, (0, 1, 2), near, f.velocity_cm_s, f.speed_cm_s,
                                     group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert bool(result.frames) is expected
    if expected:
        assert result.frames[-1][1] == 150, 'restore the observed V26 dense-state window'


def test_tail_baseline_and_extension_keep_distinct_v26_durations():
    assert cfg.nose_contact_min_duration_s == .1
    assert cfg.nose_tail_min_duration_s == .5
    assert cfg.contact_tail_extension_min_duration_s == .15
    assert cfg.contact_head_onset_distance_cm != cfg.contact_tail_onset_distance_cm
    assert cfg.priority_order[-2:] == ('head_contact', 'tail_contact')


def test_derived_evaluation_selection_cannot_rewrite_or_duplicate_source_rows(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    from evaluate_active_dataset_v17 import validate_evaluation_selection
    original = [{'review_id': 'a', 'mouse_ids': [1]}, {'review_id': 'b', 'mouse_ids': [2]}]
    assert validate_evaluation_selection(original, original[1:]) == original[1:]
    for invalid in ([], [original[0], original[0]], [{'review_id': 'b', 'mouse_ids': [99]}],
                    [{'review_id': 'foreign', 'mouse_ids': [1]}]):
        with pytest.raises(ValueError):
            validate_evaluation_selection(original, invalid)


def test_contact_never_borrows_an_onset_deleted_by_higher_priority():
    support = np.ones(30, bool)
    selected = support.copy()
    selected[:5] = False
    onset = np.zeros(30, bool)
    onset[0] = True
    assert not social._measured_contact_bout_mask(selected, (support,), 30., .1, (onset,)).any()


def test_tail_broad_baseline_fragment_cannot_borrow_the_short_channel_minimum():
    joint = np.ones(10, bool)
    endpoints = {'head': (np.full(10, 10.), np.full(10, 10.)),
                 'tail': (np.full(10, 4.), np.full(10, 10.))}
    result = social._qualify_contact_channels(joint, joint, endpoints, 'tail', 30., cfg)
    assert not result.any(), 'broad tail needs .5s; short .15s requires the tighter distance'


def test_v28_cohort_receipt_binds_exact_thirty_and_exclusion_lineage(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    from evaluate_active_dataset_v17 import validate_v28_selection_contract, minimum_duration
    from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG
    source = [{'review_id': str(i), 'behavior': 'Isolation' if i < 39 else 'Locomotion'} for i in range(755)]
    derived = source[9:]
    selected = derived[:30]
    derived_path, proof_path = tmp_path/'derived.json', tmp_path/'proof.json'
    derived_path.write_text(json.dumps(derived), encoding='utf-8')
    proof_path.write_text(json.dumps({'source_manifest_sha256': 'original',
                                     'excluded': [{'sample_id': str(i)} for i in range(9)]}), encoding='utf-8')
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def protect(p, expected):
        assert sha(p) == expected
        return p
    contract = {'schema': 'v28_isolation_selection_contract', 'source_manifest_sha256': 'original',
                'selection_sha256': 'selected', 'selection_kind': 'isolation30',
                'derived_manifest_path': str(derived_path), 'derived_manifest_sha256': sha(derived_path),
                'exclusion_receipt_path': str(proof_path), 'exclusion_receipt_sha256': sha(proof_path)}
    assert validate_v28_selection_contract(source, selected, contract, 'original', 'selected', protect)['lineage_verified']
    for invalid in (selected[:-1], selected + [derived[30]], [derived[30]] + selected[1:]):
        with pytest.raises(ValueError):
            validate_v28_selection_contract(source, invalid, contract, 'original', 'selected', protect)
    with pytest.raises(ValueError):
        validate_v28_selection_contract(source, selected, contract, 'foreign-source', 'selected', protect)
    assert minimum_duration('nose_tail_contact', audit.confirmed_params(), DEFAULT_INDIVIDUAL_CONFIG) == .15
    assert cfg.contact_minimum_s('tail') == .5


def test_clustering_restores_common_group_center_geometry_without_extra_seed_radius():
    f = audit.features(180, 3)
    f.centers_cm[:, 0] = [12., 0.]
    f.centers_cm[:91, 0, 0] = np.linspace(25., 12., 91)
    f.centers_cm[:, 1] = [0., 0.]
    f.centers_cm[:, 2] = [0., 2.]
    audit.sync(f)
    audit.resync_kinematics(f)
    # Near edges can be supported by measured noses, as in the canonical
    # center<=10 OR nose<=4.0125 graph. State radius is about the shared center.
    near = np.ones(f.pair_distance_cm.shape, bool)
    process = group._clustering_process(f, (0, 1, 2), near, f.velocity_cm_s, f.speed_cm_s,
                                     group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert np.linalg.norm(f.centers_cm[-1, 0] - np.mean(f.centers_cm[-1, 1:], axis=0)) > 10.
    assert process.evidence > 0, 'do not impose a second radius about the recipients centroid'


@pytest.mark.parametrize('head,tail,expected', [(.6, .4, 'tail_contact'), (.4, .6, 'head_contact'),
                                              (.5, .5, 'head_contact')])
def test_contact_subtype_uses_closest_actual_endpoint_across_both_directions(monkeypatch, head, tail, expected):
    f = moving_pair(frames=12)
    f.pair_nose_head_right_to_left_cm[:, 0] = head
    f.pair_nose_tail_left_to_right_cm[:, 0] = tail
    for helper in ('_open_approach_mask', '_chase_mask', '_attack_action_mask'):
        monkeypatch.setattr(social, helper, lambda f, *a, **k: np.zeros(len(f.valid), bool))
    output = social.pair_rule_masks(f, audit.confirmed_params())
    assert output[expected][:, 0].all()
    assert not output['head_contact' if expected == 'tail_contact' else 'tail_contact'].any()


def test_unknown_competing_endpoint_cannot_veto_measured_contact():
    support = social._closest_contact_endpoint(np.array([.5, .5]),
                                              (np.array([np.nan, 1.]), np.array([np.nan, 2.])))
    assert support.all()
