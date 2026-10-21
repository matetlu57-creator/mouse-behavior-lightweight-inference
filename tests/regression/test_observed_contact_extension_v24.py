"""Subtype-specific short bouts add measured geometry without bridging unknowns."""
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression.test_observed_attack_contact_v23 import contact_fixture
from mouse_behavior.evaluation import confirmed_rules as rules

# Historical .3/.15s extension geometry fixtures remain reproducible.
# v27 live .1s channel minima have separate positive/negative regressions.
EXTENSION_TEST_CONFIG = replace(rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    nose_contact_min_duration_s=.5, contact_bridge_max_gap_s=.15,
    together_contact_veto_min_duration_s=.15,
    contact_head_extension_min_duration_s=.3, contact_tail_extension_min_duration_s=.15)


def test_v24_fit_and_subtype_defaults_are_explicit():
    receipt = rules.policy_receipt()
    config = rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert receipt['coherent_attack_parameter_revision'] == '2026-10-18-v24-coherent-attack-observed-subtype-contact'
    assert receipt['independent_validation'] is False
    assert receipt['attack_acceptance_default_coverage'] == .30
    assert receipt['other_behavior_acceptance_default_coverage'] == .50
    assert receipt['v24_fit_artifact_sha256'] == {
        'outputs/heuristics_v24_attack_contacts_20261018/fit_decision.json':
        '8077c1e0bc76d776fd17f04c0c4b1b43c86780aa93ec874ada7308f8f27b7753'}
    assert config.contact_head_extension_distance_cm == 2.2344581844846205
    assert config.contact_tail_extension_distance_cm == 3.127273859004017
    assert config.contact_head_extension_min_duration_s == .1
    assert config.contact_tail_extension_min_duration_s == .15


def endpoints_fixture(monkeypatch):
    f = contact_fixture()
    f.centers_cm[:, 1, 0] = 7.5
    audit.sync(f)
    head, tail = np.full(60, 20.), np.full(60, 20.)
    head[18:29], tail[18:29] = 1., 3.
    head[40:46], tail[40:46] = 3., 2.
    monkeypatch.setattr(rules, 'endpoint_distances',
        lambda features, edge, endpoint: (head if endpoint == 'head' else tail, np.full(60, 20.)))
    return f, head, tail


def test_short_head_and_tail_bouts_use_separate_observed_minima(monkeypatch):
    f, _, _ = endpoints_fixture(monkeypatch)
    before = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG, allow_contact_extension=False)
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG)
    assert not before['head_contact'].any() and not before['tail_contact'].any()
    assert actual['head_contact'][18:29, 0].all()
    assert actual['tail_contact'][40:46, 0].all()
    assert not (actual['head_contact'] & actual['tail_contact']).any()


def test_measured_negative_or_unknown_cannot_connect_short_tail_fragments(monkeypatch):
    f, _, tail = endpoints_fixture(monkeypatch)
    f.valid[42, 1] = False
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG)
    assert not actual['tail_contact'].any()
    f.valid[42, 1] = True
    tail[42] = 8.
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG)
    assert not actual['tail_contact'].any()


def test_extension_compares_endpoints_within_the_same_direction_and_obeys_priority(monkeypatch):
    f, head, tail = endpoints_fixture(monkeypatch)
    # Head distance qualifies, but this actor is still closer to the tail.
    head[18:29], tail[18:29] = 1., .5
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG)
    assert not actual['head_contact'][18:29].any()
    assert actual['tail_contact'][18:29].all()
    approach = np.zeros(60, bool)
    approach[18:29] = True
    monkeypatch.setattr(rules, '_open_approach_mask', lambda *a, **k: approach)
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), EXTENSION_TEST_CONFIG)
    assert actual['approach'][18:29].all()
    assert not actual['tail_contact'][18:29].any()


@pytest.mark.parametrize('field', ['contact_head_extension_distance_cm', 'contact_tail_extension_distance_cm',
    'contact_head_extension_min_duration_s', 'contact_tail_extension_min_duration_s'])
def test_extension_parameters_reject_invalid_values(field):
    for value in (0., -1., np.nan):
        with pytest.raises(ValueError):
            replace(rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG, **{field: value})
