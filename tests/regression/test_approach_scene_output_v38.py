"""Approach output must keep third-mouse observations and requalify every cut."""
from copy import deepcopy
import numpy as np

from mouse_behavior.evaluation.approach_events import approach_scene_support, requalify_approach_event
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from tests.regression import test_heuristics_v16_audit as audit


def scene(count=60):
    f = audit.features(count, 4, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.arange(count)*.4
    f.centers_cm[:, 1] = [22., 0.]
    f.centers_cm[:, 2] = [100., 100.]
    f.centers_cm[:, 3] = [150., 100.]
    return sync(f)


def sync(f):
    audit.sync(f)
    audit.resync_kinematics(f)
    shape = np.array([[3.959, 0.], [2.5, .8], [2.5, -.8], [1., 0.],
                      [-2., 1.], [-2., -1.], [-3.959, 0.]])
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :]+shape
    body = f.collective_keypoints_cm[:, :, :6]
    f.collective_boxes_cm = np.concatenate([body.min(axis=2), body.max(axis=2)], axis=2)
    return f


def event(f):
    return dict(canonical_behavior='approach', behavior='Approach', mouse_ids=[100, 101],
                actor_id=100, target_id=101, event_id='parent', start_frame=0, end_frame=len(f.valid))


def qualified(f, guard=True):
    mask, _ = approach_scene_support(f, 0, 1)
    return requalify_approach_event(f, 0, event(f), config_seed_parameters({}),
                                  retained_mask=mask if guard else None)


def test_clear_scene_keeps_old_approach_and_does_not_mutate_features():
    f = scene()
    before = deepcopy(f.valid)
    assert qualified(f)
    assert qualified(f) == qualified(f, False)
    np.testing.assert_array_equal(f.valid, before)


def test_actor_contacting_third_mouse_vetoes_only_measured_contact_rows():
    f = scene()
    f.centers_cm[31:, 2] = f.centers_cm[31:, 0]
    sync(f)
    retained, reason = approach_scene_support(f, 0, 1)
    assert retained[:31].all() and not retained[31:].any()
    assert reason['actor_third_contact_or_unknown_frames'] == 29
    children = qualified(f)
    assert children and children[0]['end_frame'] <= 31
    assert children[0]['actor_observed_path_lower_bound_cm'] >= 10


def test_target_in_contact_with_third_mouse_is_not_an_independent_target():
    f = scene()
    f.centers_cm[:, 2] = f.centers_cm[:, 1]
    sync(f)
    retained, reason = approach_scene_support(f, 0, 1)
    assert not retained.any()
    assert reason['target_third_contact_or_unknown_frames'] == len(f.valid)
    assert not qualified(f)


def test_target_with_two_nearby_but_noncontacting_neighbors_is_crowded():
    f = scene()
    f.centers_cm[:, 2] = [22., 8.]
    f.centers_cm[:, 3] = [22., -8.]
    sync(f)
    retained, reason = approach_scene_support(f, 0, 1)
    assert reason['target_third_contact_or_unknown_frames'] == 0
    assert reason['target_crowd_frames'] == len(f.valid)
    assert not retained.any() and not qualified(f)


def test_actor_is_not_counted_as_another_crowd_member():
    f = scene()
    f.centers_cm[:, 2] = [22., 8.]
    sync(f)
    retained, reason = approach_scene_support(f, 0, 1)
    assert reason['target_crowd_frames'] == 0
    assert retained.all() and qualified(f)


def test_contact_cut_cannot_borrow_parent_duration_or_travel():
    f = scene()
    f.centers_cm[20:, 2] = f.centers_cm[20:, 0]
    sync(f)
    assert qualified(f, False)
    assert not qualified(f)


def test_tailroot_alone_does_not_establish_body_contact():
    f = scene()
    f.collective_keypoints_cm[:, 2, 6] = f.collective_keypoints_cm[:, 0, 0]
    retained, _ = approach_scene_support(f, 0, 1)
    assert retained.all()


def test_source_output_guard_loads_third_mice_instead_of_stripping_scene(monkeypatch):
    import json
    from mouse_behavior.evaluation import exported_dataset_features as loader
    from mouse_behavior.evaluation.approach_events import requalify_source_approach_event
    f = scene()
    f.centers_cm[:, 2] = f.centers_cm[:, 1]
    sync(f)
    rows = [dict(frame=i, detections=[dict(track_id=int(t)) for t in f.track_ids])
            for i in range(len(f.valid))]
    class Store:
        def window(self, start, end):
            return deepcopy(rows[start:end])
    def load(path, **kwargs):
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert {d['track_id'] for row in payload for d in row['detections']} == {100, 101, 102, 103}
        assert kwargs['max_tracks'] == 4
        return f
    monkeypatch.setattr(loader, 'load_exported_features', load)
    output = requalify_source_approach_event(Store(), event(f),
        dict(fps=30., frame_count=len(f.valid)), dict(cm_per_pixel=1., reference_body_cm=7.918),
        config_seed_parameters({}), scene_guard=True)
    assert not output
