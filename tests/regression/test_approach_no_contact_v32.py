"""Approach geometry cannot include contact or borrow eligibility across unknowns."""
from dataclasses import replace

import numpy as np
import pytest

from mouse_behavior.evaluation.confirmed_rules import _open_approach_mask
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.social_event_geometry import observed_body_separation
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG as DEFAULT
from tests.regression.test_requested_social_v31 import closing_pair


def boxes_for(f):
    body = f.collective_keypoints_cm[:, :, :6]
    return np.concatenate((body.min(axis=2), body.max(axis=2)), axis=2)


V32 = replace(DEFAULT, approach_require_no_body_contact=True, approach_min_duration_s=1.0, approach_min_actor_path_cm=10.0)


def mask(f, config=V32):
    return _open_approach_mask(f, 0, config_seed_parameters({}), config)


def test_first_contact_keeps_only_independently_qualified_prefix():
    f = closing_pair(40)
    f.collective_boxes_cm = boxes_for(f)
    f.collective_boxes_cm[31:, 0] = f.collective_boxes_cm[31:, 1]
    result = mask(f)
    assert result[:31].all()
    assert not result[31:].any()


def test_all_contact_rejected_despite_closing_and_valid_endpoint():
    f = closing_pair()
    f.collective_boxes_cm = boxes_for(f)
    f.collective_boxes_cm[:, 0] = f.collective_boxes_cm[:, 1]
    assert not mask(f).any()


@pytest.mark.parametrize("boundary", [25, 30])
def test_precontact_prefix_must_requalify_own_path_and_strict_duration(boundary):
    f = closing_pair(40)
    f.collective_boxes_cm = boxes_for(f)
    f.collective_boxes_cm[boundary:, 0] = f.collective_boxes_cm[boundary:, 1]
    assert not mask(f).any()


def test_unknown_geometry_cannot_be_painted_or_bridge_approach():
    f = closing_pair()
    f.collective_keypoints_cm[15:17] = np.nan
    assert not mask(f).any()
    permissive = replace(V32, approach_min_duration_s=.1, approach_min_actor_path_cm=0.,
                         dist_approach_start_min_cm=12.)
    result = _open_approach_mask(f, 0, config_seed_parameters({}), permissive)
    assert not result[15:17].any()


def test_hull_contact_vetoes_disjoint_boxes_and_tail_does_not_define_body():
    f = closing_pair()
    f.collective_boxes_cm = boxes_for(f)
    assert observed_body_separation(f, 0, max_body_lengths=3.).all()
    f.collective_keypoints_cm[:, 0, 6] = f.collective_keypoints_cm[:, 1, 0]
    assert observed_body_separation(f, 0, max_body_lengths=3.).all()
    f.collective_keypoints_cm[:, 0, :6] = f.collective_keypoints_cm[:, 1, :6]
    assert not observed_body_separation(f, 0, max_body_lengths=3.).any()


def test_touching_rectangles_are_contact():
    f = closing_pair()
    f.collective_boxes_cm = boxes_for(f)
    f.collective_boxes_cm[:, 0, 2] = f.collective_boxes_cm[:, 1, 0]
    assert not mask(f).any()



def test_attack_footprint_guard_cannot_change_approach():
    f = closing_pair()
    baseline = mask(f)
    assert baseline.any()
    attack_changed = replace(V32, attack_contact_max_footprint_body_lengths=.01)
    assert np.array_equal(baseline, _open_approach_mask(
        f, 0, config_seed_parameters({}), attack_changed))
    own_changed = replace(V32, approach_max_footprint_body_lengths=.01)
    assert not _open_approach_mask(f, 0, config_seed_parameters({}), own_changed).any()


@pytest.mark.parametrize("bound", [0., -1., float("nan"), float("inf")])
def test_approach_footprint_bound_requires_finite_positive_value(bound):
    with pytest.raises(ValueError, match="approach_max_footprint_body_lengths"):
        replace(DEFAULT, approach_max_footprint_body_lengths=bound)
