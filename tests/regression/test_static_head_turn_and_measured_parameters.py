"""Head exploration is Static; body-size migration preserves physical units."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mouse_behavior.behavior import hierarchical_core as core
from mouse_behavior.behavior.collective_state import CollectiveRuleConfig
from mouse_behavior.behavior.ethogram import _extended_individual_and_group_events
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    HeuristicParameters, config_seed_parameters, load_exported_features, predict_features,
)
from mouse_behavior.preprocessing.kinematics import _kinematics
from mouse_behavior.preprocessing.pair_features import _interaction_radius, _pair_dataframe, _pair_metrics
from mouse_behavior.preprocessing.scale_resolution import resolve_scale


ROOT = Path(__file__).resolve().parents[2]


def _head_turns(translation_px=0.0):
    frames = 90
    base = np.array([[20., 50.], [28., 44.], [28., 56.], [50., 50.],
                     [78., 30.], [78., 70.], [120., 50.]])
    points = np.broadcast_to(base, (frames, 1, 7, 2)).copy()
    for frame in range(frames):
        angle = .7 * np.sin(frame * .6)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        points[frame, 0, :3] = (base[:3] - base[3]) @ rotation.T + base[3]
        points[frame, 0, :, 0] += frame * translation_px
    return points


@pytest.mark.parametrize("translation_px, expected", [(0., "stationary"), (12., "locomotion")])
def test_turning_head_does_not_replace_body_translation(tmp_path, translation_px, expected):
    points = _head_turns(translation_px)
    frames = len(points)
    valid = np.ones((frames, 1), dtype=bool)
    tracks = {
        "valid": valid, "pose_quality": np.ones_like(valid, dtype=float),
        "keypoints_px": points, "centers_px": points[:, :, :3].mean(axis=2),
        "body_lengths_px": np.linalg.norm(points[:, :, 0] - points[:, :, 6], axis=2),
    }
    kin = _kinematics(tracks, fps=30., reference_body_cm=10.7, cm_per_pixel=.1)
    # The nose moves rapidly even when the torso does not translate.
    assert np.quantile(kin["nose_speed"], .9) > 20.
    cfg = load_config(ROOT / "configs/default.yaml")
    events = _extended_individual_and_group_events(
        kin, pair_metrics={}, pair_i=np.array([], dtype=int), pair_j=np.array([], dtype=int),
        source_video=Path("head_turn.mp4"), source_fps=30., sample_stride=1, config=cfg,
    )
    individual = {e["behavior"] for e in events if e["behavior"] in {"stationary", "locomotion"}}
    assert expected in individual
    if translation_px == 0:
        assert individual == {"stationary"}
        assert np.max(kin["behavior_speed"]) == pytest.approx(0.)
        assert np.allclose(kin["centers_cm"], kin["centers_cm"][0])

    observations = [core.PoseObservation(7, i, i / 30., p[0]) for i, p in enumerate(points)]
    centers = np.array([core.compute_individual_geometry(o, 10.).center_cm for o in observations])
    distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    stats = core.MotionWindowStats(
        7, frames / 30., float(distances.mean() * 30.), float(distances.sum()),
        float(np.linalg.norm(centers[-1] - centers[0])), 1.,
        float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()),
    )
    thresholds = core.thresholds_from_config(cfg["hierarchical_heuristic_core"])
    assert core.detect_individual_motion(stats, thresholds.motion).label == (
        core.MotionLabel.STATIC.value if expected == "stationary" else core.MotionLabel.LOCOMOTION.value
    )

    payload = [{"frame": i, "detections": [{"track_id": 7,
                "keypoints": np.column_stack((p[0], np.ones(7))).tolist()}]}
               for i, p in enumerate(points)]
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    features = load_exported_features(path, fps=30.)
    prediction = predict_features(features, config_seed_parameters(cfg))
    assert prediction["identity_scores"]["7"][expected] > 0.
    if translation_px == 0:
        assert prediction["identity_scores"]["7"]["locomotion"] == 0.


def test_bounded_brief_body_excursion_stays_static_in_both_adapters(tmp_path):
    points = _head_turns()
    valid = np.ones((len(points), 1), dtype=bool)
    tracks = {
        "valid": valid,
        "pose_quality": np.ones_like(valid, dtype=float),
        "keypoints_px": points,
        "centers_px": points[:, :, :3].mean(axis=2),
        "body_lengths_px": np.linalg.norm(points[:, :, 0] - points[:, :, 6], axis=2),
    }
    kin = _kinematics(tracks, fps=30., reference_body_cm=10.7, cm_per_pixel=.1)
    excursion = np.array([.2, .5, .9, .5, .2])
    kin["centers_cm"][35:40, 0, 0] += excursion
    kin["behavior_speed"][35:40, 0] = 20.
    cfg = load_config(ROOT / "configs/default.yaml")

    events = _extended_individual_and_group_events(
        kin, pair_metrics={}, pair_i=np.array([], dtype=int), pair_j=np.array([], dtype=int),
        source_video=Path("brief_static_excursion.mp4"), source_fps=30., sample_stride=1,
        config=cfg,
    )
    individual = [event for event in events if event["behavior"] in {"stationary", "locomotion"}]
    assert {event["behavior"] for event in individual} == {"stationary"}
    assert individual[0]["start_frame"] == 0
    assert individual[0]["end_frame"] == len(points) - 1

    payload = [{"frame": index, "detections": [{
        "track_id": 7,
        "keypoints": np.column_stack((point[0], np.ones(7))).tolist(),
    }]} for index, point in enumerate(points)]
    track_path = tmp_path / "tracks.json"
    track_path.write_text(json.dumps(payload), encoding="utf-8")
    features = load_exported_features(track_path, fps=30.)
    modified_centers = features.centers_cm.copy()
    modified_centers[35:40, 0, 0] += excursion
    modified_speed = features.speed_cm_s.copy()
    modified_speed[35:40, 0] = 20.
    features = replace(features, centers_cm=modified_centers, speed_cm_s=modified_speed)
    temporal = predict_features(features, config_seed_parameters(cfg), temporal_only=True)

    assert temporal["individual_masks"]["stationary"][:, 0].all()
    assert not temporal["individual_masks"]["locomotion"][:, 0].any()


def test_measured_body_parameter_boundaries_and_unchanged_time():
    for relative in ("configs/default.yaml", "configs/profiles/beiyi.yaml"):
        cfg = load_config(ROOT / relative)
        thresholds = core.thresholds_from_config(cfg["hierarchical_heuristic_core"])
        params = config_seed_parameters(cfg)
        assert thresholds.motion.static_max_speed_cm_s == pytest.approx(.66875)
        assert thresholds.motion.moving_min_speed_cm_s == pytest.approx(.66875)
        assert thresholds.motion.static_max_center_radius_cm == pytest.approx(1.3375)
        assert thresholds.motion.static_max_transient_duration_s == pytest.approx(.2)
        assert thresholds.motion.static_max_transient_displacement_cm == pytest.approx(1.3375)
        # The legacy hierarchical adapter keeps its serialized thresholds;
        # current exported inference owns the approved 4 cm / 1 s policy.
        assert params.stationary_transient_max_duration_s == pytest.approx(1.)
        assert params.stationary_transient_max_displacement_cm == pytest.approx(4.)
        assert params.stationary_max_center_radius_cm == pytest.approx(4.)
        assert thresholds.social.together_min_distance_cm == pytest.approx(4.0125)
        assert thresholds.social.together_max_distance_cm == pytest.approx(7.)
        assert params.contact_distance_cm == pytest.approx(12.985665630296548)
        assert params.chase_min_speed_cm_s == pytest.approx(26.75)
        assert params.attack_mild_nose_accel_cm_s2 == pytest.approx(66.875)
        assert params.isolation_distance_cm == pytest.approx(20.)
        assert params.group_locomotion_max_distance_cm == pytest.approx(64.03693389892578)
        assert params.stationary_min_duration_s == 1.
        assert params.huddle_min_duration_s == 5.
        assert params.chase_min_duration_s == 1.
        assert params.chase_min_direction_similarity == .707
        assert params.huddle_width_multiplier == 2.
    assert HeuristicParameters().stationary_max_speed_cm_s == pytest.approx(.66875)
    assert HeuristicParameters().stationary_transient_max_duration_s == pytest.approx(1.)
    assert HeuristicParameters().stationary_transient_max_displacement_cm == pytest.approx(4.)
    assert core.MotionThresholds().static_max_speed_cm_s == pytest.approx(.66875)
    collective = CollectiveRuleConfig.from_mapping(None)
    assert collective.active_speed_cm_s == pytest.approx(4.0125)
    assert collective.low_motion_speed_cm_s == pytest.approx(13.375)
    assert collective.huddle_attack_speed_p90_max_cm_s == pytest.approx(60.1875)
    assert collective.huddle_attack_relative_motion_max_cm_s == pytest.approx(40.125)
    assert _interaction_radius({}) == pytest.approx(22.7375)


@pytest.mark.parametrize("width", [0., -1., float("nan"), float("inf")])
def test_invalid_explicit_width_fallback_is_rejected(width):
    with pytest.raises(ValueError, match="body_width_cm"):
        resolve_scale({"mode": "fixed", "cm_per_pixel": .1,
                       "assumed_mouse_body_length_cm": 10.7,
                       "assumed_mouse_body_width_cm": width})


def test_calibrated_width_and_parameters_are_not_scaled_twice():
    cfg = load_config(ROOT / "configs/default.yaml")
    resolved = resolve_scale(cfg["scale"], reference_body_px=100.)
    assert resolved.cm_per_pixel == pytest.approx(.07918)
    assert resolved.reference_width_cm == pytest.approx(3.068558)
    first = config_seed_parameters(cfg)
    second = config_seed_parameters(cfg)
    assert first == second
    assert first.stationary_max_speed_cm_s == pytest.approx(.66875)
    assert cfg["extended_behavior"]["individual"]["stationary_max_speed_cm_s"] == .66875
    assert cfg["extended_behavior"]["individual"]["stationary_transient_max_duration_seconds"] == 1.
    assert cfg["extended_behavior"]["individual"]["stationary_transient_max_displacement_cm"] == 4.
    assert cfg["extended_behavior"]["individual"]["stationary_max_center_radius_cm"] == 4.


def test_missing_hip_and_ear_width_uses_measured_fallback(tmp_path):
    points = np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.], [30., 0., 1.],
                       [0., 0., 0.], [0., 0., 0.], [100., 0., 1.]])
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps([{"frame": 0, "detections": [
        {"track_id": 7, "keypoints": points.tolist()}]}]), encoding="utf-8")
    features = load_exported_features(path, fps=30.)
    assert features.cm_per_pixel == pytest.approx(.07918)
    assert features.mouse_width_cm == pytest.approx(3.068558)
    assert features.body_width_ref_cm[0] == pytest.approx(3.068558)


@pytest.mark.parametrize("gap_cm, contact_expected", [(4.5, True), (5.5, False)])
def test_pair_contact_feature_boundary_uses_measured_body_basis(gap_cm, contact_expected):
    points = np.zeros((60, 2, 7, 2))
    points[:, :, 4:6, 0] = 50.
    points[:, :, 6, 0] = 100.
    points[:, 1, :, 1] = gap_cm / .1
    tracks = {"keypoints_px": points, "centers_px": points.mean(axis=2),
              "valid": np.ones((60, 2), dtype=bool),
              "pose_quality": np.ones((60, 2)), "body_lengths_px": np.full((60, 2), 100.)}
    kin = _kinematics(tracks, fps=30., reference_body_cm=10.7, cm_per_pixel=.1)
    metrics, pair_i, pair_j = _pair_metrics(kin, 30.)
    assert metrics["reference_body_cm"] == 10.7
    pair = _pair_dataframe(metrics, 0, 0, 1, pair_i, pair_j, 30., .1)
    np.testing.assert_allclose(pair["center_distance_body_lengths"], metrics["distance"][:, 0] / 10.7)
    assert pair["weak_contact"].all() == contact_expected
    assert not pair["strong_contact"].any()
    assert (metrics["repeated_contact"] > 0).any() == contact_expected
