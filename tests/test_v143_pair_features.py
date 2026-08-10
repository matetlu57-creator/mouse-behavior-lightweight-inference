from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_regression_helper():
    path = ROOT / "tests" / "regression_performance_test.py"
    spec = importlib.util.spec_from_file_location("v143_regression_helper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_main_and_base():
    helper = _load_regression_helper()
    _, base, _, main = helper.load_modules()
    return base, main


def _make_observation(base, frame: int, points: np.ndarray, center: np.ndarray, heading: np.ndarray):
    return base.MouseObservation(
        frame=frame,
        logical_id=1,
        raw_track_id=1,
        keypoints_px=np.asarray(points, dtype=float).copy(),
        keypoints_cm=np.asarray(points, dtype=float).copy(),
        keypoint_conf=np.ones(7, dtype=float),
        bbox_xyxy=np.array([-2.0, -1.0, 2.0, 1.0], dtype=float),
        box_conf=0.9,
        center_cm=np.asarray(center, dtype=float),
        head_cm=np.asarray(points[0], dtype=float),
        rear_cm=np.asarray(points[-1], dtype=float),
        heading=np.asarray(heading, dtype=float),
        velocity_cm_s=np.zeros(2, dtype=float),
        speed_cm_s=0.0,
        acceleration_cm_s2=0.0,
        angular_speed_deg_s=0.0,
        nose_speed_cm_s=0.0,
        body_length_cm=3.0,
        track_state="tracked",
    )


def test_body_frame_pose_is_translation_rotation_invariant():
    base, main = _load_main_and_base()
    raw = np.array([
        [2.0, 0.0], [1.5, -0.3], [1.5, 0.3], [1.0, 0.0],
        [0.0, -0.4], [0.0, 0.4], [-1.0, 0.0],
    ])
    first = _make_observation(base, 0, raw, np.zeros(2), np.array([1.0, 0.0]))
    theta = np.deg2rad(67.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.array([8.0, -4.0])
    rotated = raw @ rotation.T + translation
    second = _make_observation(base, 1, rotated, translation, np.array([np.cos(theta), np.sin(theta)]))
    pose_a, valid_a = main.PairFeatureComputer._body_frame_pose(first)
    pose_b, valid_b = main.PairFeatureComputer._body_frame_pose(second)
    np.testing.assert_array_equal(valid_a, valid_b)
    np.testing.assert_allclose(pose_a[valid_a], pose_b[valid_b], rtol=0.0, atol=1e-12)


def test_pose_deformation_ignores_global_motion_but_detects_internal_change():
    base, main = _load_main_and_base()
    with (ROOT / "mouse_chase_attack_config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    raw = np.array([
        [2.0, 0.0], [1.5, -0.3], [1.5, 0.3], [1.0, 0.0],
        [0.0, -0.4], [0.0, 0.4], [-1.0, 0.0],
    ])
    previous = _make_observation(base, 0, raw, np.zeros(2), np.array([1.0, 0.0]))
    history = base.ObservationHistory(max_frames=10)
    history.add(previous)
    computer = main.PairFeatureComputer(30.0, cfg)

    theta = np.deg2rad(40.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.array([4.0, 5.0])
    rigid_points = raw @ rotation.T + translation
    rigid = _make_observation(base, 1, rigid_points, translation, np.array([np.cos(theta), np.sin(theta)]))
    assert computer._pose_deformation(rigid, history) < 1e-10

    # Same global transform, but the nose changes relative to the body.
    deformed_local = raw.copy()
    deformed_local[0, 1] += 0.9
    deformed_points = deformed_local @ rotation.T + translation
    deformed = _make_observation(base, 2, deformed_points, translation, np.array([np.cos(theta), np.sin(theta)]))
    history2 = base.ObservationHistory(max_frames=10)
    history2.add(_make_observation(base, 1, rigid_points, translation, np.array([np.cos(theta), np.sin(theta)])))
    computer2 = main.PairFeatureComputer(30.0, cfg)
    assert computer2._pose_deformation(deformed, history2) > 0.05


def test_standard_reconciliation_preserves_intentional_ambiguous_attack_role(monkeypatch, tmp_path):
    _, main = _load_main_and_base()
    with (ROOT / "mouse_chase_attack_config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dummy = __import__("pandas").DataFrame({"frame": [10]})
    monkeypatch.setattr(main, "_load_pair_reconciliation_table", lambda _: dummy)
    monkeypatch.setattr(
        main,
        "_dominant_pair_for_event_kind",
        lambda *args, **kwargs: {
            "mouse_a_id": 1,
            "mouse_b_id": 2,
            "high_frame_count": 10,
            "rank": 1.0,
            "evidence_column": "selected_weak_attack_evidence",
            "high_threshold": 4.0,
        },
    )
    monkeypatch.setattr(main, "_event_window_role", lambda *args, **kwargs: (1, 2, 0.9))
    event = {
        "label_id": 2,
        "candidate_level": "weak",
        "pair_key": "1_2",
        "mouse_a_id": 1,
        "mouse_b_id": 2,
        "actor_id": -1,
        "target_id": -1,
        "start_frame": 10,
        "end_frame": 20,
        "start_time_s": 10 / 30.0,
        "end_time_s": 20 / 30.0,
    }
    out = main.reconcile_detected_event_pairs([event], tmp_path / "unused.csv", 100, 30.0, cfg)
    assert len(out) == 1
    assert out[0]["actor_id"] == -1
    assert out[0]["target_id"] == -1
    assert out[0]["pair_reconciliation_reason"] == "standard_behavior_role_preserved"


def test_standard_four_class_is_pure_composition_of_confirmed_behaviors():
    _, main = _load_main_and_base()
    with (ROOT / "mouse_chase_attack_config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # A confirmed standard combined event must stay label 3 even when the old
    # co-occurrence path gate would have rejected its short path.
    event = {
        "label_id": 3,
        "actor_id": 1,
        "target_id": 2,
        "actor_path_cm": 2.0,
        "target_path_cm": 2.0,
        "strict_chase": False,
    }
    result = main.classify_video_four_label([event], "weak", cfg)
    assert result["chase_present"] is True
    assert result["attack_present"] is True
    assert result["video_label_id"] == 3
