"""Formation process boundaries and observation-only continuity regressions."""
import numpy as np
import pytest
from dataclasses import replace

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.event_frames import interval_frame_count


def formation():
    f = audit.features(150, 3)
    f.centers_cm[:, 0] = [3., 0.]
    f.centers_cm[:11, 0] = [22., 0.]
    f.centers_cm[10:81, 0, 0] = np.linspace(22., 3., 71)
    f.centers_cm[:, 1] = [0., 0.]
    f.centers_cm[:, 2] = [0., 2.]
    return f


def predict(f):
    f.centers_cm[~f.valid] = np.nan
    audit.sync(f)
    audit.resync_kinematics(f)
    return group._clustering_process(
        f, tuple(range(len(f.track_ids))), f.pair_distance_cm <= 10.,
        f.velocity_cm_s, f.speed_cm_s, group.DEFAULT_GROUP_HEURISTIC_CONFIG,
        neighbor_cm=10., max_gap_s=5.)


def mask(f, result):
    output = np.zeros(len(f.valid), bool)
    for a, b in result.frames:
        output[a:b] = True
    assert interval_frame_count(result.frames, len(f.valid)) / f.fps == result.evidence
    return output


def test_formation_includes_measured_approach_but_not_idle_prefix():
    f = formation()
    result = predict(f)
    observed = mask(f, result)
    assert not observed[:10].any()
    assert observed[10:].all(), 'a confirmed formation includes its actual approach'


def test_confirmed_formation_survives_one_unknown_row_without_crediting_it():
    f = formation()
    f.valid[100, 2] = False
    observed = mask(f, predict(f))
    assert observed[99] and observed[101:].all()
    assert not observed[100], 'event memory supplies no missing frame'


def test_mutual_convergence_does_not_require_a_preexisting_dense_seed_pair():
    f = audit.features(150, 3)
    initial = np.array([[-18., 0.], [18., 0.], [0., 18.]])
    final = np.array([[-2., 0.], [2., 0.], [0., 2.]])
    fraction = np.minimum(np.arange(150) / 70., 1.)
    f.centers_cm[:] = initial + fraction[:, None, None] * (final - initial)
    observed = mask(f, predict(f))
    assert observed[:30].any(), 'count the actual convergence preceding compact state'
    assert observed[-30:].all()


def test_unobserved_teleport_is_never_a_formation_path():
    f = formation()
    f.centers_cm[:50, 0] = [22., 0.]
    f.centers_cm[50:, 0] = [3., 0.]
    f.valid[49:51, 0] = False
    assert not predict(f).frames


def test_rigid_translation_of_existing_pile_is_not_convergence():
    f = audit.features(150, 3)
    base = np.array([[0., 0.], [2., 0.], [0., 2.]])
    f.centers_cm[:] = base
    f.centers_cm[:, :, 0] += np.arange(150)[:, None] * 2.
    assert not predict(f).frames


def test_formation_that_never_stays_dense_is_not_emitted():
    f = formation()
    # The actor passes through in <0.5s and is then really far away.
    f.centers_cm[:70, 0] = [22., 0.]
    f.centers_cm[70:79, 0] = [3., 0.]
    f.centers_cm[79:, 0] = [22., 0.]
    assert not predict(f).frames


def test_short_dense_runs_cannot_borrow_stability_from_unknown_context():
    f = formation()
    # Five dense rows, three unknown rows, twelve dense rows: total support
    # exceeds .5s, but no observed run confirms a stable final state.
    f.valid[60:63, :] = False
    f.centers_cm[75:, 0] = [22., 0.]
    assert not predict(f).frames


@pytest.mark.parametrize('field', ['clustering_unknown_gap_s', 'clustering_motion_lookback_s',
                                  'clustering_min_toward_fraction', 'clustering_min_observed_fraction'])
@pytest.mark.parametrize('invalid', [0., -1., float('nan'), True])
def test_new_formation_settings_validate_bounds(field, invalid):
    with pytest.raises(ValueError):
        replace(group.DEFAULT_GROUP_HEURISTIC_CONFIG, **{field: invalid})


def test_current_policy_reports_contiguous_stability_and_retrospective_fit():
    from mouse_behavior.evaluation.confirmed_rules import policy_receipt
    receipt = policy_receipt()
    assert 'contiguous measured run>=0.5s' in receipt['group_rules']['social_clustering']
    assert receipt['clustering_fit_scope'].endswith('not_independent_validation')
