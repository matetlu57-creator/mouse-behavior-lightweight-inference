"""Attack changes must not transfer historical unknown support between labels."""
import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression.test_observed_attack_contact_v23 import contact_fixture
from mouse_behavior.evaluation import confirmed_rules as rules


def ownership_fixture():
    """Supply real short-tail geometry; broad V23 footprints are not a close onset."""
    f = contact_fixture()
    f.collective_keypoints_cm = None
    for direction in ('left_to_right', 'right_to_left'):
        setattr(f, f'pair_nose_tail_{direction}_cm', np.full((60, 1), .5))
        setattr(f, f'pair_nose_head_{direction}_cm', np.full((60, 1), 20.))
    return f


def masks(f):
    shape = f.pair_distance_cm.shape
    result = {name: np.zeros(shape, bool) for name in
              ('approach', 'together', 'attack', 'chasing', 'head_contact', 'tail_contact',
               'long_contact_review', 'contact_start_context')}
    result.update(head_contact_seconds=np.zeros(shape[1]), tail_contact_seconds=np.zeros(shape[1]))
    return result


def test_unknown_cannot_transfer_from_old_head_to_new_tail():
    f = ownership_fixture()
    f.valid[20, 1] = False
    reference, actual = masks(f), masks(f)
    reference['head_contact'][10:40, 0] = True
    actual['tail_contact'][10:40, 0] = True
    rules._restrict_contact_unknown_support(actual, reference, f, rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert not actual['tail_contact'][20, 0]
    assert actual['tail_contact'][10:20, 0].all()
    assert actual['tail_contact'][21:40, 0].all()
    assert actual['tail_contact_seconds'][0] == pytest.approx(19 / f.fps)
    assert reference['head_contact'][20, 0]


def test_same_subtype_historical_unknown_support_is_retained():
    f = ownership_fixture()
    f.valid[20, 1] = False
    reference, actual = masks(f), masks(f)
    reference['tail_contact'][10:40, 0] = True
    actual['tail_contact'][10:40, 0] = True
    before = actual['tail_contact'].copy()
    rules._restrict_contact_unknown_support(actual, reference, f, rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    np.testing.assert_array_equal(actual['tail_contact'], before)


def test_removing_unknown_requalifies_each_fragment_instead_of_their_union():
    f = ownership_fixture()
    f.valid[20, 1] = False
    reference, actual = masks(f), masks(f)
    reference['head_contact'][18:23, 0] = True
    actual['tail_contact'][18:23, 0] = True
    rules._restrict_contact_unknown_support(actual, reference, f, rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert not actual['tail_contact'].any()
    assert actual['tail_contact_seconds'][0] == 0.


@pytest.mark.parametrize('extension', [False, True])
def test_public_path_uses_old_attack_canonical_reference_for_each_contact_subtype(monkeypatch, extension):
    f = contact_fixture()
    f.valid[20, 1] = False
    calls = []

    def base(features, params, config, *, allow_observed_attack_extension=True, diagnostics=None):
        assert features is f
        calls.append(allow_observed_attack_extension)
        result = masks(f)
        if allow_observed_attack_extension:
            if diagnostics is not None:
                diagnostics['new_coherent_candidate'] = True
            result['tail_contact'][18:23, 0] = True
        else:
            result['head_contact'][10:30, 0] = True
        return result

    monkeypatch.setattr(rules, '_base_pair_rule_masks', base)
    monkeypatch.setattr(rules, 'endpoint_distances',
                        lambda *args: (np.full(60, 100.), np.full(60, 100.)))
    actual = rules.pair_rule_masks(f, audit.confirmed_params(), allow_contact_extension=extension)
    # With v27's observed-support requalification, the extension rejects this
    # mocked geometry-less contact before unknowns need an old-path reference.
    assert calls == ([True] if extension else [True, False])
    assert not actual['tail_contact'].any()
    assert not actual['head_contact'].any()
