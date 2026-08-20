from __future__ import annotations

import copy
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_engine():
    path = ROOT / "standard_behavior_engine.py"
    spec = importlib.util.spec_from_file_location("standard_behavior_engine_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def config():
    with (ROOT / "mouse_chase_attack_config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def base_row(frame: int) -> dict:
    row = {
        "frame": frame,
        "time_s": frame / 30.0,
        "valid_pair": True,
        "mouse_a_id": 1,
        "mouse_b_id": 2,
        "mouse_a_track_state": "tracked",
        "mouse_b_track_state": "tracked",
        "identity_pair_quality": 1.0,
        "pose_pair_quality": 0.90,
        "center_distance_cm": 7.0,
        "pair_wall_jump_excluded": False,
        "selected_distance_drop_cm": 2.0,
        "selected_target_turn_angle_deg": 0.0,
        "selected_actor_speed_cm_s": 10.0,
        "selected_target_speed_cm_s": 10.0,
        "selected_actor_id": 1,
        "selected_target_id": 2,
        "selected_nose_body_distance_cm": 6.0,
        "trajectory_correlation": 0.95,
        "direction_similarity": 0.95,
        "pursuit_alignment": 0.90,
        "target_escape_alignment": 0.90,
        "actor_behind_target": True,
        "repeated_contact_count": 0,
    }
    for level in ("weak", "strong"):
        for suffix in (
            "strict_chase", "window_chase", "near_recovery_chase", "close_follow_chase",
            "strict_attack", "impulse_attack", "grapple_attack", "occlusion_overlap_attack",
        ):
            row[f"{level}_{suffix}"] = False
        row[f"{level}_final_chase"] = False
        row[f"{level}_final_attack"] = False

    # A -> B: clear pursuit. B -> A: same general motion but wrong causal role.
    directional = {
        "actor_speed_cm_s": 10.0,
        "target_speed_cm_s": 10.0,
        "actor_acceleration_cm_s2": 20.0,
        "target_acceleration_cm_s2": 20.0,
        "actor_nose_speed_cm_s": 12.0,
        "target_nose_speed_cm_s": 12.0,
        "actor_angular_speed_deg_s": 20.0,
        "target_angular_speed_deg_s": 20.0,
        "actor_body_length_cm": 8.0,
        "target_body_length_cm": 8.0,
        "center_distance_body_lengths": 0.875,
        "actor_head_relative_speed_cm_s": 2.0,
        "target_head_relative_speed_cm_s": 2.0,
        "direction_similarity": 0.95,
        "trajectory_correlation": 0.95,
        "target_turn_angle_deg": 0.0,
        "nose_body_distance_cm": 6.0,
    }
    for k, v in directional.items():
        row[f"a_to_b_{k}"] = v
        row[f"b_to_a_{k}"] = v
    row.update({
        "a_to_b_closing_speed_cm_s": 8.0,
        "a_to_b_pursuit_alignment": 0.90,
        "a_to_b_target_escape_alignment": 0.90,
        "a_to_b_behind_score": 0.90,
        "a_to_b_actor_behind_target": True,
        "b_to_a_closing_speed_cm_s": -8.0,
        "b_to_a_pursuit_alignment": -0.50,
        "b_to_a_target_escape_alignment": -0.50,
        "b_to_a_behind_score": -0.60,
        "b_to_a_actor_behind_target": False,
        "cluster_attack_hint": False,
        "cluster_detection_deficit": False,
        "cluster_merged_like": False,
        "cluster_overlap_iou": 0.0,
        "cluster_motion_bl_per_frame": 0.0,
        "cluster_active_frames": 0,
        "cluster_expected_count": 0,
        "cluster_observed_count": 0,
    })
    return row


def test_sustained_directional_chase_enters_fsm_and_resolves_role():
    engine = load_engine()
    cfg = config()
    df = pd.DataFrame([base_row(i) for i in range(30)])
    out = engine.apply_standard_behavior_engine(df, 30.0, cfg)
    assert bool(out["weak_standard_final_chase"].any())
    assert bool(out["strong_standard_final_chase"].any())
    active = out[out["weak_standard_final_chase"]]
    assert set(active["weak_standard_actor_id"].astype(int)) == {1}
    assert set(active["weak_standard_target_id"].astype(int)) == {2}
    assert float(active["weak_standard_role_confidence"].mean()) > 0.20


def test_selected_role_fallback_supports_legacy_pair_schema():
    engine = load_engine()
    cfg = config()
    rows = [base_row(i) for i in range(30)]
    for row in rows:
        row["selected_actor_id"] = 1
        row["selected_target_id"] = 2
        for key in list(row):
            if key.startswith(("a_to_b_", "b_to_a_")):
                row.pop(key)
    out = engine.apply_standard_behavior_engine(pd.DataFrame(rows), 30.0, cfg)
    assert bool(out["weak_standard_final_chase"].any())
    active = out[out["weak_standard_final_chase"]]
    assert set(active["weak_standard_actor_id"].astype(int)) == {1}
    assert set(active["weak_standard_target_id"].astype(int)) == {2}
    assert bool(out["weak_standard_chase_role_fallback"].any())


def test_low_quality_cannot_open_new_chase():
    engine = load_engine()
    cfg = config()
    rows = [base_row(i) for i in range(30)]
    for row in rows:
        row["pose_pair_quality"] = 0.05
    out = engine.apply_standard_behavior_engine(pd.DataFrame(rows), 30.0, cfg)
    assert not bool(out["weak_standard_final_chase"].any())
    assert not bool(out["strong_standard_final_chase"].any())


def test_attack_requires_causal_prepare_contact_reaction():
    engine = load_engine()
    cfg = config()
    rows = [base_row(i) for i in range(28)]
    # Remove chase-like target escape so attack is independently exercised.
    for row in rows:
        row["center_distance_cm"] = 5.0
        row["a_to_b_direction_similarity"] = 0.0
        row["b_to_a_direction_similarity"] = 0.0
        row["a_to_b_trajectory_correlation"] = 0.0
        row["b_to_a_trajectory_correlation"] = 0.0
        row["a_to_b_target_escape_alignment"] = 0.0
        row["b_to_a_target_escape_alignment"] = 0.0
        row["a_to_b_nose_body_distance_cm"] = 6.0
        row["b_to_a_nose_body_distance_cm"] = 6.0
        row["a_to_b_actor_speed_cm_s"] = 16.0
        row["a_to_b_target_speed_cm_s"] = 2.0
        row["a_to_b_actor_acceleration_cm_s2"] = 100.0
        row["a_to_b_actor_nose_speed_cm_s"] = 28.0
        row["a_to_b_actor_head_relative_speed_cm_s"] = 12.0
        row["a_to_b_closing_speed_cm_s"] = 12.0
        row["a_to_b_pursuit_alignment"] = 0.95
        row["b_to_a_pursuit_alignment"] = -0.8
        row["b_to_a_closing_speed_cm_s"] = -12.0
    # Initiation first.
    for i in range(3, 8):
        rows[i]["a_to_b_nose_body_distance_cm"] = 6.0
    # Then physical contact.
    for i in range(8, 14):
        rows[i]["a_to_b_nose_body_distance_cm"] = 1.8
        rows[i]["repeated_contact_count"] = 2
    # Then target escape/reaction while contact is still present.
    for i in range(11, 17):
        rows[i]["a_to_b_nose_body_distance_cm"] = 1.8
        rows[i]["a_to_b_target_speed_cm_s"] = 14.0
        rows[i]["a_to_b_target_escape_alignment"] = 0.95
        rows[i]["a_to_b_target_acceleration_cm_s2"] = 100.0
        rows[i]["a_to_b_target_turn_angle_deg"] = 65.0
    out = engine.apply_standard_behavior_engine(pd.DataFrame(rows), 30.0, cfg)
    assert bool(out["weak_standard_final_attack"].any())
    active = out[out["weak_standard_final_attack"]]
    assert "lunge_attack" in set(active["weak_standard_attack_subtype"])
    assert int(active["weak_standard_actor_id"].mode().iloc[0]) == 1


def test_contact_without_initiation_or_reaction_is_not_attack():
    engine = load_engine()
    cfg = config()
    rows = [base_row(i) for i in range(40)]
    for row in rows:
        row["center_distance_cm"] = 4.0
        row["a_to_b_nose_body_distance_cm"] = 1.8
        row["b_to_a_nose_body_distance_cm"] = 1.8
        row["a_to_b_actor_speed_cm_s"] = 1.0
        row["a_to_b_target_speed_cm_s"] = 1.0
        row["b_to_a_actor_speed_cm_s"] = 1.0
        row["b_to_a_target_speed_cm_s"] = 1.0
        row["a_to_b_closing_speed_cm_s"] = 0.0
        row["b_to_a_closing_speed_cm_s"] = 0.0
        row["a_to_b_pursuit_alignment"] = 0.0
        row["b_to_a_pursuit_alignment"] = 0.0
        row["a_to_b_target_escape_alignment"] = 0.0
        row["b_to_a_target_escape_alignment"] = 0.0
        row["a_to_b_actor_angular_speed_deg_s"] = 10.0
        row["a_to_b_target_angular_speed_deg_s"] = 10.0
        row["b_to_a_actor_angular_speed_deg_s"] = 10.0
        row["b_to_a_target_angular_speed_deg_s"] = 10.0
        row["repeated_contact_count"] = 1
    out = engine.apply_standard_behavior_engine(pd.DataFrame(rows), 30.0, cfg)
    assert not bool(out["weak_standard_final_attack"].any())
    assert not bool(out["strong_standard_final_attack"].any())


def test_inactive_rows_are_skipped_without_changing_fsm_events():
    engine = load_engine()
    cfg_skip = config()
    cfg_full = copy.deepcopy(cfg_skip)
    cfg_skip["standard_behavior_engine"]["skip_inactive_rows"] = True
    cfg_full["standard_behavior_engine"]["skip_inactive_rows"] = False

    rows = [base_row(i) for i in range(120)]
    for i in range(30, 90):
        rows[i]["valid_pair"] = False
    frame_df = pd.DataFrame(rows)

    skipped = engine.apply_standard_behavior_engine(frame_df, 30.0, cfg_skip)
    full = engine.apply_standard_behavior_engine(frame_df, 30.0, cfg_full)

    assert int(skipped["standard_behavior_compute_row"].sum()) == 60
    assert bool(full["standard_behavior_compute_row"].all())
    for level in ("weak", "strong"):
        for behavior in ("chase", "attack"):
            column = f"{level}_standard_final_{behavior}"
            np.testing.assert_array_equal(skipped[column], full[column])
        np.testing.assert_array_equal(
            skipped[f"{level}_standard_chase_state"],
            full[f"{level}_standard_chase_state"],
        )
        np.testing.assert_array_equal(
            skipped[f"{level}_standard_attack_state"],
            full[f"{level}_standard_attack_state"],
        )
        for column in (
            f"{level}_standard_chase_score",
            f"{level}_standard_attack_score",
            f"{level}_standard_role_confidence",
        ):
            np.testing.assert_allclose(
                skipped.loc[skipped["valid_pair"], column],
                full.loc[full["valid_pair"], column],
            )

    for level in ("weak", "strong"):
        assert engine.extract_standard_behavior_events(
            skipped, 30.0, level, pair_key="1_2"
        ) == engine.extract_standard_behavior_events(
            full, 30.0, level, pair_key="1_2"
        )


def test_provider_hint_is_not_skipped_when_pair_is_temporarily_invalid():
    engine = load_engine()
    cfg = config()
    cfg["standard_behavior_engine"]["skip_inactive_rows"] = True
    row = base_row(0)
    row["valid_pair"] = False
    row["weak_strict_attack"] = True

    out = engine.apply_standard_behavior_engine(pd.DataFrame([row]), 30.0, cfg)

    assert bool(out.loc[0, "standard_behavior_compute_candidate"])
    assert bool(out.loc[0, "standard_behavior_compute_row"])


def test_owned_dataframe_can_be_enriched_in_place_without_output_changes():
    engine = load_engine()
    cfg = config()
    source = pd.DataFrame([base_row(i) for i in range(30)])
    source_before = source.copy(deep=True)

    copied = engine.apply_standard_behavior_engine(source, 30.0, cfg)
    owned = source.copy(deep=True)
    inplace = engine.apply_standard_behavior_engine(
        owned,
        30.0,
        cfg,
        copy_input=False,
    )

    assert inplace is owned
    pd.testing.assert_frame_equal(inplace, copied)
    pd.testing.assert_frame_equal(source, source_before)


def test_standard_contact_type_retains_nose_body_fallback():
    engine = load_engine()
    row = base_row(0)
    row["a_to_b_nose_body_distance_cm"] = 2.0
    row["b_to_a_nose_body_distance_cm"] = 8.0
    row["a_to_b_nose_head_distance_cm"] = np.inf
    row["b_to_a_nose_head_distance_cm"] = np.inf
    row["a_to_b_nose_tail_distance_cm"] = np.inf
    row["b_to_a_nose_tail_distance_cm"] = np.inf

    out = engine.apply_standard_behavior_engine(
        pd.DataFrame([row]),
        30.0,
        config(),
    )

    assert out.loc[0, "weak_standard_contact_type"] == "nose_body"
    assert out.loc[0, "strong_standard_contact_type"] == "nose_body"


def test_invalid_timeline_gap_cannot_bridge_standard_fsm_events():
    engine = load_engine()
    rows = [base_row(i) for i in range(50)]
    for index in range(18, 32):
        rows[index]["valid_pair"] = False

    out = engine.apply_standard_behavior_engine(
        pd.DataFrame(rows),
        30.0,
        config(),
    )
    events = engine.extract_standard_behavior_events(
        out,
        30.0,
        "weak",
        pair_key="1_2",
    )

    assert set(out.loc[18:31, "weak_standard_chase_state"]) == {"IDLE"}
    assert all(
        not (event["start_frame"] < 18 and event["end_frame"] > 31)
        for event in events
    )


def test_ethogram_aggregates_chase_and_attack_independently():
    engine = load_engine()
    frames = np.arange(40)
    df = pd.DataFrame({
        "frame": frames,
        "weak_standard_final_chase": [True] * 40,
        "weak_standard_final_attack": [(12 <= i <= 21) for i in frames],
        "weak_standard_chase_actor_id": [1] * 40,
        "weak_standard_chase_target_id": [2] * 40,
        "weak_standard_attack_actor_id": [1 if 12 <= i <= 21 else -1 for i in frames],
        "weak_standard_attack_target_id": [2 if 12 <= i <= 21 else -1 for i in frames],
        "weak_standard_chase_score": np.linspace(0.7, 0.9, 40),
        "weak_standard_attack_score": [0.85 if 12 <= i <= 21 else 0.0 for i in frames],
        "weak_standard_chase_role_confidence": [0.6] * 40,
        "weak_standard_attack_role_confidence": [0.5 if 12 <= i <= 21 else 0.0 for i in frames],
        "weak_standard_behavior_confidence": [0.8] * 40,
        "weak_standard_attack_subtype": ["lunge_attack" if 12 <= i <= 21 else "none" for i in frames],
    })
    events = engine.extract_standard_behavior_events(df, 30.0, "weak", pair_key="1_2")
    chase = [e for e in events if e["behavior"] == "chase"]
    attack = [e for e in events if e["behavior"] == "attack"]
    assert len(chase) == 1
    assert (chase[0]["start_frame"], chase[0]["end_frame"]) == (0, 39)
    assert len(attack) == 1
    assert (attack[0]["start_frame"], attack[0]["end_frame"]) == (12, 21)
    assert attack[0]["subtype"] == "lunge_attack"
