"""Measured footprint proximity can rescue center-far action, never unknown rows."""
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as rules
from mouse_behavior.evaluation import social_event_geometry as geometry


def contact_fixture(frames=60):
    f = audit.features(frames, 2, reference_body_cm=7.918)
    audit.set_centers(f, [[0., 0.], [9., 0.]])
    audit.sync(f)
    left = [[4., 0.], [2.5, 1.], [2.5, -1.], [1., 0.], [-2., 1.], [-2., -1.], [-7., 0.]]
    right = [[4.5, 0.], [6., 1.], [6., -1.], [8., 0.], [11., 1.], [11., -1.], [17., 0.]]
    f.collective_keypoints_cm = np.tile(np.asarray([left, right]), (frames, 1, 1, 1))
    return f


def test_footprint_gap_is_symmetric_translation_invariant_and_ignores_tailroot():
    f = contact_fixture()
    gap = geometry.body_footprint_distances(f, 0, max_body_lengths=2.)
    np.testing.assert_allclose(gap, .5, atol=1e-5)
    f.collective_keypoints_cm[:, :, 6] = 1000.
    np.testing.assert_allclose(geometry.body_footprint_distances(f, 0, max_body_lengths=2.), gap)
    f.collective_keypoints_cm[:, :, :6] += [128., -63.]
    np.testing.assert_allclose(geometry.body_footprint_distances(f, 0, max_body_lengths=2.), gap, atol=1e-5)
    f.pair_i[:], f.pair_j[:] = 1, 0
    np.testing.assert_allclose(geometry.body_footprint_distances(f, 0, max_body_lengths=2.), gap, atol=1e-5)


def test_footprint_rejects_unknown_sparse_degenerate_and_implausibly_large_hulls():
    f = contact_fixture()
    f.valid[0, 0] = False
    f.collective_keypoints_cm[1, 0, 3:6] = np.nan
    f.collective_keypoints_cm[2, 0, :6, 1] = 0.
    f.collective_keypoints_cm[3, 0, 0, 0] = 1000.
    gap = geometry.body_footprint_distances(f, 0, max_body_lengths=2.)
    assert np.isnan(gap[:4]).all()
    assert np.isfinite(gap[4:]).all()
    with pytest.raises(ValueError):
        geometry.body_footprint_distances(f, 0, max_body_lengths=0.)
    for invalid in (np.nan, 0., -1.):
        f.reference_body_cm = invalid
        assert np.isnan(geometry.body_footprint_distances(f, 0, max_body_lengths=2.)).all()


def test_crossing_and_touching_footprints_have_zero_gap_without_vertex_containment():
    f = contact_fixture(2)
    horizontal = [[-3., -1.], [3., -1.], [3., 1.], [-3., 1.], [0., -1.], [0., 1.]]
    vertical = [[-1., -3.], [1., -3.], [1., 3.], [-1., 3.], [-1., 0.], [1., 0.]]
    f.collective_keypoints_cm[0, 0, :6] = horizontal
    f.collective_keypoints_cm[0, 1, :6] = vertical
    f.collective_keypoints_cm[1, 1, :6, 0] -= .5
    np.testing.assert_allclose(geometry.body_footprint_distances(f, 0, max_body_lengths=2.), 0., atol=1e-6)


def test_v23_contact_parameters_and_provenance_are_explicit():
    cfg = rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert cfg.attack_contact_surface_max_gap_cm == 1.1925119161605835
    assert cfg.attack_contact_max_footprint_body_lengths == 2.
    receipt = rules.policy_receipt()
    assert receipt['observed_contact_parameter_revision'] == '2026-10-18-v23-eight-loss-case-video-review'
    assert receipt['observed_contact_extension_requires_center_heavy_overlap'] is True
    assert receipt['observed_contact_extension_review_sha256'] == {
        'outputs/heuristics_v23_observed_contact_20261018/contact_extension_scope_review.json':
        '63956d8a984a5897abcdd34323add967976bb87aaddea6aca135615da6c8ee8b',
    }
    assert receipt['observed_contact_rescue_channel'].startswith('heavy_repeated_target_boxes_only')
    assert receipt['observed_contact_channel_review_sha256'] == {
        'outputs/heuristics_v23_observed_contact_20261018/contact_channel_review.json':
        '14878e96549a1623e1625ed2656b8c8ec0460a4e27ee55d8c6666bf427c63424',
    }
    assert receipt['independent_validation'] is False
    artifacts = receipt['observed_contact_fit_artifact_sha256']
    assert artifacts == {
        'outputs/heuristics_v23_observed_contact_20261018/observed_contact_fit.json':
        '1e23550454cd76f21ad983d3a550fe776c03a88845252897677656efc046fa8e',
    }
    for field in ('attack_contact_surface_max_gap_cm', 'attack_contact_max_footprint_body_lengths'):
        for invalid in (0., -1., np.nan):
            with pytest.raises(ValueError):
                replace(cfg, **{field: invalid})


def test_center_far_contact_does_not_expand_unvalidated_light_route(monkeypatch):
    f = contact_fixture()
    f.centers_cm[20:, 1, 0] += 10.
    f.collective_keypoints_cm[20:, 1, :, 0] += 10.
    audit.sync(f)
    def measured(points, *args, **kwargs):
        velocity = np.zeros((len(points), 2))
        acceleration = np.zeros_like(velocity)
        if points[0, 0] < 4.3:
            velocity[10:30, 0] = 15.
            acceleration[10:30, 0] = 150.
        return velocity, acceleration
    monkeypatch.setattr(rules, '_local_quadratic_point_kinematics', measured)
    channels = {}
    mask = rules._attack_action_mask(f, 0, channels=channels)
    assert not mask.any(), 'head motion plus footprint proximity cannot expand the light route'
    assert not channels['center_only_raw'].any()
    assert np.array_equal(channels['observed_contact_rescue'], mask)
    # The original center-near light detector remains available.
    f.centers_cm[:20, 1, 0] = 6.
    audit.sync(f)
    baseline = rules._attack_action_mask(f, 0)
    assert baseline[10:20].all()
    assert not baseline[:10].any() and not baseline[20:].any()


def test_contact_alone_and_bbox_overlap_alone_never_start_attack():
    f = contact_fixture()
    assert not rules._attack_action_mask(f, 0).any(), 'contact is not an attack trigger'
    f.collective_boxes_cm = np.ones((60, 2, 4))
    f.collective_keypoints_cm[:, 1, :, 0] += 20.
    assert not rules._attack_action_mask(f, 0).any(), 'coarse boxes are not body contact evidence'


def test_heavy_geometry_rescue_outputs_only_measured_rows_and_cannot_enlarge_missing_support():
    f = contact_fixture(90)
    shift = 20. * np.arange(90) / f.fps
    f.centers_cm[:, :, 0] += shift[:, None]
    f.collective_keypoints_cm[:, :, :, 0] += shift[:, None, None]
    # Later center-near cycles establish a real baseline heavy process;
    # earlier body-contact cycles may extend that same expanded process.
    f.centers_cm[45:, 1, 0] -= 3.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    for low, high in ((20, 25), (32, 37), (50, 55), (62, 67)):
        f.collective_boxes_cm[low:high, 1] = np.nan
        f.valid[low:high, 1] = False
    channels = {}
    mask = rules._attack_action_mask(f, 0, channels=channels)
    assert mask[25:32].all()
    assert channels['center_only_raw'].any()
    assert not ((mask & ~channels['center_only_raw']) & ~f.valid.all(axis=1)).any()
    assert not channels['observed_contact_rescue'][~f.valid.all(axis=1)].any()


def test_footprint_and_box_cycles_cannot_create_a_new_unanchored_heavy_process():
    f = contact_fixture(90)
    shift = 20. * np.arange(90) / f.fps
    f.centers_cm[:, :, 0] += shift[:, None]
    f.collective_keypoints_cm[:, :, :, 0] += shift[:, None, None]
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    for low, high in ((20, 25), (32, 37)):
        f.collective_boxes_cm[low:high, 1] = np.nan
        f.valid[low:high, 1] = False
    channels = {}
    mask = rules._attack_action_mask(f, 0, channels=channels)
    assert not channels['center_only_raw'].any()
    assert not mask.any(), 'new footprint cue only extends an established center-heavy process'
    assert np.array_equal(mask, channels['light_observed_acceleration'] | channels['heavy_repeated_target_boxes'])


def test_priority_cannot_make_previously_unqualified_unknown_attack_rows_positive(monkeypatch):
    f = contact_fixture(90)
    f.valid[10:12] = f.valid[21:23] = False
    def attack(features, edge, *args, channels=None, **kwargs):
        base = np.zeros(90, bool)
        base[10:23] = True
        mask = base.copy()
        mask[7:10] = mask[23:26] = True
        if channels is not None:
            channels['center_only_raw'] = base
        return mask
    approach = np.zeros(90, bool)
    approach[12:21] = True
    monkeypatch.setattr(rules, '_attack_action_mask', attack)
    monkeypatch.setattr(rules, '_open_approach_mask', lambda *a, **k: approach)
    monkeypatch.setattr(rules, '_chase_mask', lambda *a, **k: np.zeros(90, bool))
    actual = rules.pair_rule_masks(f, audit.confirmed_params())['attack'][:, 0]
    assert not actual[~f.valid.all(axis=1)].any()
    assert not actual.any(), 'after excluding unknown-only short tails, no >=0.15s bout remains'
