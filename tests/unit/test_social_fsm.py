from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mouse_behavior.behavior.pair_analysis import _stitch_identity_bridged_events
from mouse_behavior.behavior.social_fsm import (
    _direction_ids,
    _extend_bouts_through_supported_trailing_gap,
    _retain_bouts_with_minimum_support,
    build_semantic_pair_signals,
)


def _pair_frame(frame_count: int = 30) -> pd.DataFrame:
    frame = np.arange(frame_count)
    result = pd.DataFrame(
        {
            "frame": frame,
            "valid_pair": True,
            "mouse_a_id": 1,
            "mouse_b_id": 2,
            "selected_actor_id": 1,
            "selected_target_id": 2,
            "center_distance_cm": np.full(frame_count, 100.0),
            "a_to_b_actor_behavior_speed_cm_s": np.full(frame_count, 8.0),
            "a_to_b_target_behavior_speed_cm_s": np.full(frame_count, 0.5),
            "a_to_b_pursuit_alignment": np.full(frame_count, 0.85),
            "a_to_b_target_escape_alignment": np.full(frame_count, 0.75),
            "a_to_b_direction_similarity": np.full(frame_count, 0.80),
            "a_to_b_target_turn_angle_deg": np.zeros(frame_count),
            "a_to_b_actor_speed_cm_s": np.full(frame_count, 8.0),
            "a_to_b_target_speed_cm_s": np.full(frame_count, 0.5),
            "a_to_b_actor_pose_deformation_energy": np.zeros(frame_count),
            "a_to_b_target_pose_deformation_energy": np.zeros(frame_count),
            "b_to_a_actor_behavior_speed_cm_s": np.full(frame_count, 0.5),
            "b_to_a_target_behavior_speed_cm_s": np.full(frame_count, 8.0),
            "b_to_a_pursuit_alignment": np.zeros(frame_count),
            "b_to_a_target_escape_alignment": np.zeros(frame_count),
            "b_to_a_direction_similarity": np.zeros(frame_count),
            "b_to_a_target_turn_angle_deg": np.zeros(frame_count),
            "b_to_a_actor_speed_cm_s": np.full(frame_count, 0.5),
            "b_to_a_target_speed_cm_s": np.full(frame_count, 8.0),
            "b_to_a_actor_pose_deformation_energy": np.zeros(frame_count),
            "b_to_a_target_pose_deformation_energy": np.zeros(frame_count),
            "a_to_b_nose_head_distance_cm": np.full(frame_count, 100.0),
            "b_to_a_nose_head_distance_cm": np.full(frame_count, 100.0),
            "a_to_b_nose_tail_distance_cm": np.full(frame_count, 100.0),
            "b_to_a_nose_tail_distance_cm": np.full(frame_count, 100.0),
            "a_to_b_nose_body_distance_cm": np.full(frame_count, 100.0),
            "b_to_a_nose_body_distance_cm": np.full(frame_count, 100.0),
        }
    )
    return result


def test_temporal_bridge_cannot_expand_sparse_pose_support_into_long_bout() -> None:
    mask = np.r_[np.ones(20, dtype=bool), np.zeros(2, dtype=bool), np.ones(10, dtype=bool)]
    support = np.r_[
        np.ones(8, dtype=bool),
        np.zeros(12, dtype=bool),
        np.zeros(2, dtype=bool),
        np.ones(8, dtype=bool),
        np.zeros(2, dtype=bool),
    ]

    filtered = _retain_bouts_with_minimum_support(
        mask,
        support,
        minimum_fraction=0.50,
    )

    assert not filtered[:20].any()
    assert not filtered[20:22].any()
    assert filtered[22:32].all()


def test_trailing_occlusion_bridge_requires_a_real_seed_and_stops_at_first_gap() -> None:
    mask = np.r_[np.ones(8, dtype=bool), np.zeros(12, dtype=bool)]
    support = np.r_[
        np.zeros(8, dtype=bool),
        np.ones(5, dtype=bool),
        np.zeros(1, dtype=bool),
        np.ones(6, dtype=bool),
    ]

    extended = _extend_bouts_through_supported_trailing_gap(
        mask,
        support,
        max_gap_frames=10,
        min_seed_frames=6,
    )
    rejected_sparse_seed = _extend_bouts_through_supported_trailing_gap(
        np.r_[np.ones(2, dtype=bool), np.zeros(18, dtype=bool)],
        support,
        max_gap_frames=10,
        min_seed_frames=6,
    )

    assert extended[:13].all()
    assert not extended[13:].any()
    assert rejected_sparse_seed.sum() == 2


def _social_config() -> dict[str, object]:
    return {
        "semantic_approach": {
            "min_relative_distance_drop": 0.03,
            "min_pursuit_alignment": 0.25,
            "min_speed_gap_ratio": 0.10,
            "hold_seconds": 0.20,
        },
        "semantic_chase": {
            "min_pursuit_alignment": 0.35,
            "min_target_escape_alignment": 0.20,
            "min_direction_similarity": 0.20,
            "min_combined_speed_cm_s": 3.0,
            "hold_seconds": 0.20,
        },
        "semantic_avoidance": {
            "near_distance_quantile": 0.35,
            "near_distance_multiplier": 1.50,
            "context_seconds": 3.0,
            "boundary_context_seconds": 0.50,
            "allow_clip_start_context": True,
            "min_target_escape_alignment": 0.35,
            "min_target_turn_angle_deg": 25.0,
            "turn_window_seconds": 0.50,
            "min_evader_speed_cm_s": 1.0,
            "min_relative_distance_increase": 0.03,
            "hold_seconds": 0.20,
        },
        "semantic_attack": {},
    }


def test_approach_uses_relative_closing_and_fast_mouse_as_actor() -> None:
    pair = _pair_frame()
    pair.loc[:2, "center_distance_cm"] = 100.0
    pair.loc[3:8, "center_distance_cm"] = [90, 80, 70, 60, 50, 40]

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=_social_config(),
        contact_config={"enabled": True},
    )

    assert signals["approach_mask"][3:].any()
    assert set(signals["approach_actor"][signals["approach_mask"]]) == {1}
    assert set(signals["approach_target"][signals["approach_mask"]]) == {2}


def test_approach_requires_near_arrival_and_a_mostly_stationary_target() -> None:
    pair = _pair_frame(40)
    pair["center_distance_cm"] = np.linspace(30.0, 18.0, len(pair))
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "max_current_distance_cm": 30.0,
        "max_final_distance_cm": 10.0,
        "arrival_context_seconds": 1.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_min_fraction": 0.60,
    }

    no_arrival = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )
    assert not no_arrival["approach_mask"].any()

    pair["center_distance_cm"] = np.linspace(16.0, 6.0, len(pair))
    pair["a_to_b_target_behavior_speed_cm_s"] = 4.0
    moving_target = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )
    assert not moving_target["approach_mask"].any()


def test_sparse_approach_evidence_is_not_expanded_to_a_long_state() -> None:
    pair = _pair_frame(50)
    pair["center_distance_cm"] = 16.0
    pair.loc[20:22, "center_distance_cm"] = [14.0, 12.0, 9.0]
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "max_current_distance_cm": 17.0,
        "max_final_distance_cm": 10.0,
        "arrival_context_seconds": 1.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_seconds": 0.75,
        "hold_min_fraction": 0.60,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["approach_mask"].any()


def test_confirmed_chase_suppresses_parallel_approach_state() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = np.linspace(16.0, 6.0, len(pair))
    pair["a_to_b_target_behavior_speed_cm_s"] = 4.0
    pair["a_to_b_target_speed_cm_s"] = 4.0
    pair["b_to_a_actor_behavior_speed_cm_s"] = 4.0
    pair["b_to_a_actor_speed_cm_s"] = 4.0
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "max_current_distance_cm": 17.0,
        "max_final_distance_cm": 10.0,
        "arrival_context_seconds": 2.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_min_fraction": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["chase_mask"].any()
    assert not np.any(signals["approach_mask"] & signals["chase_mask"])


def test_chase_role_uses_actor_behind_target_geometry() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = 8.0
    # Both directional branches pass every chase gate with equal non-geometric
    # evidence.  The only distinguishing cue is that mouse B is behind A.
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_behavior_speed_cm_s"] = 8.0
        pair[f"{prefix}_target_behavior_speed_cm_s"] = 6.0
        pair[f"{prefix}_actor_speed_cm_s"] = 8.0
        pair[f"{prefix}_target_speed_cm_s"] = 6.0
        pair[f"{prefix}_pursuit_alignment"] = 0.75
        pair[f"{prefix}_target_escape_alignment"] = 0.70
        pair[f"{prefix}_direction_similarity"] = 0.80
    pair["a_to_b_actor_behind_target"] = 0.0
    pair["b_to_a_actor_behind_target"] = 1.0

    baseline_config = _social_config()
    baseline_config["semantic_chase"] = {
        **baseline_config["semantic_chase"],
        "role_behind_weight": 0.0,
    }
    weighted_config = _social_config()
    weighted_config["semantic_chase"] = {
        **weighted_config["semantic_chase"],
        "role_behind_weight": 2.0,
    }
    baseline = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=baseline_config,
        contact_config={"enabled": True},
    )
    weighted = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=weighted_config,
        contact_config={"enabled": True},
    )

    for key in ("approach_mask", "chase_mask", "avoidance_mask", "attack_mask"):
        np.testing.assert_array_equal(baseline[key], weighted[key])
    mask = weighted["chase_mask"]
    assert mask.any()
    assert set(baseline["chase_actor"][mask]) == {1}
    assert set(weighted["chase_actor"][mask]) == {2}
    assert set(weighted["chase_target"][mask]) == {1}


def test_chase_role_is_not_reversed_by_late_turnaround() -> None:
    pair = _pair_frame(80)
    pair["center_distance_cm"] = 8.0
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_behavior_speed_cm_s"] = 8.0
        pair[f"{prefix}_target_behavior_speed_cm_s"] = 6.0
        pair[f"{prefix}_actor_speed_cm_s"] = 8.0
        pair[f"{prefix}_target_speed_cm_s"] = 6.0
        pair[f"{prefix}_pursuit_alignment"] = 0.75
        pair[f"{prefix}_target_escape_alignment"] = 0.70
        pair[f"{prefix}_direction_similarity"] = 0.80
    pair["a_to_b_actor_behind_target"] = 1.0
    pair["b_to_a_actor_behind_target"] = 0.0
    pair.loc[:19, "a_to_b_actor_behind_target"] = 0.0
    pair.loc[:19, "b_to_a_actor_behind_target"] = 1.0
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "role_behind_weight": 2.0,
        "role_initial_context_seconds": 2.0,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    mask = signals["chase_mask"]
    assert mask.any()
    assert set(signals["chase_actor"][mask]) == {2}
    assert set(signals["chase_target"][mask]) == {1}


def test_relative_arrival_and_contact_do_not_erase_a_supported_approach() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = np.linspace(32.0, 10.3, len(pair))
    pair["a_to_b_nose_head_distance_cm"] = 1.0
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "arrival_distance_quantile": 0.35,
        "arrival_distance_multiplier": 1.15,
        "arrival_context_seconds": 3.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_seconds": 0.75,
        "hold_min_fraction": 0.25,
        "suppress_during_contact": False,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True, "nose_head_distance_cm": 2.0},
    )

    assert signals["contact_mask"].all()
    assert signals["approach_mask"].any()
    assert set(signals["approach_actor"][signals["approach_mask"]]) == {1}


def test_confirmed_approach_persists_only_through_continuous_arrival_contact() -> None:
    pair = _pair_frame(50)
    pair["center_distance_cm"] = 20.0
    pair.loc[:14, "center_distance_cm"] = np.linspace(30.0, 8.0, 15)
    pair.loc[15:34, "center_distance_cm"] = 8.0
    pair.loc[35:, "center_distance_cm"] = 14.0
    pair["a_to_b_nose_head_distance_cm"] = 100.0
    pair.loc[12:34, "a_to_b_nose_head_distance_cm"] = 1.0
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "max_final_distance_cm": 10.0,
        "arrival_context_seconds": 1.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_seconds": 0.30,
        "hold_min_fraction": 0.25,
        "suppress_during_contact": False,
        "post_arrival_contact_hold_seconds": 3.0,
        "post_arrival_min_seed_seconds": 0.30,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True, "nose_head_distance_cm": 2.0},
    )

    assert signals["approach_mask"][12:35].all()
    assert not signals["approach_mask"][35:].any()


def test_stationary_target_does_not_turn_approach_into_chase() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = np.linspace(32.0, 8.0, len(pair))
    pair["a_to_b_target_behavior_speed_cm_s"] = 1.1
    pair["a_to_b_target_speed_cm_s"] = 1.1
    pair["b_to_a_actor_behavior_speed_cm_s"] = 1.1
    pair["b_to_a_actor_speed_cm_s"] = 1.1
    config = _social_config()
    config["semantic_approach"] = {
        **config["semantic_approach"],
        "arrival_distance_quantile": 0.35,
        "arrival_distance_multiplier": 1.15,
        "arrival_context_seconds": 3.0,
        "max_target_speed_fraction_of_actor": 0.35,
        "hold_min_fraction": 0.25,
        "suppress_during_contact": False,
    }
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "min_target_speed_fraction_of_actor": 0.15,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["approach_mask"].any()
    assert not signals["chase_mask"].any()


def test_chase_rejects_distant_parallel_motion_in_body_length_space() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = 8.0
    pair["center_distance_body_lengths"] = 4.0
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "max_distance_body_lengths": 2.5,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["chase_mask"].any()


def test_chase_speed_ratio_gate_can_bridge_a_documented_target_occlusion() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = 12.0
    pair["a_to_b_actor_behavior_speed_cm_s"] = 8.0
    pair["a_to_b_target_behavior_speed_cm_s"] = 1.1
    pair["a_to_b_target_bbox_observed"] = True
    pair.loc[45:, "a_to_b_target_bbox_observed"] = False
    pair["a_to_b_target_bbox_imputed"] = False
    pair.loc[45:, "a_to_b_target_bbox_imputed"] = True
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "min_actor_speed_cm_s": 3.0,
        "min_target_speed_cm_s": 1.0,
        "min_target_speed_fraction_of_actor": 0.15,
        "target_occlusion_observed_fraction": 0.85,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["chase_mask"].any()
    assert not signals["approach_mask"].any()


def test_chase_state_bridges_only_an_internal_bbox_supported_occlusion_gap() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = 12.0
    pair["a_to_b_actor_behavior_speed_cm_s"] = 8.0
    pair["a_to_b_target_behavior_speed_cm_s"] = 1.1
    pair["a_to_b_target_bbox_observed"] = True
    pair["a_to_b_target_bbox_imputed"] = False
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair.loc[20:29, "valid_pair"] = False
    pair.loc[20:29, "a_to_b_target_bbox_observed"] = False
    pair.loc[20:29, "a_to_b_target_bbox_imputed"] = True
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "min_actor_speed_cm_s": 3.0,
        "min_target_speed_cm_s": 1.0,
        "min_target_speed_fraction_of_actor": 0.15,
        "target_occlusion_observed_fraction": 0.85,
        "occlusion_fill_gap_seconds": 1.0,
        "occlusion_gap_min_imputed_fraction": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["chase_mask"][20:30].all()


def test_chase_state_extends_through_bounded_trailing_target_occlusion() -> None:
    pair = _pair_frame(40)
    pair["center_distance_cm"] = 12.0
    pair["a_to_b_actor_behavior_speed_cm_s"] = 8.0
    pair["a_to_b_target_behavior_speed_cm_s"] = 1.1
    pair["a_to_b_target_bbox_observed"] = True
    pair["a_to_b_target_bbox_imputed"] = False
    pair["a_to_b_actor_bbox_observed"] = True
    pair["a_to_b_actor_bbox_imputed"] = False
    pair["b_to_a_actor_bbox_observed"] = True
    pair["b_to_a_target_bbox_observed"] = True
    pair["b_to_a_actor_bbox_imputed"] = False
    pair["b_to_a_target_bbox_imputed"] = False
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair.loc[20:, "valid_pair"] = False
    pair.loc[20:, "a_to_b_target_bbox_observed"] = False
    pair.loc[20:, "a_to_b_target_bbox_imputed"] = True
    pair.loc[20:, "b_to_a_actor_bbox_observed"] = False
    pair.loc[20:, "b_to_a_actor_bbox_imputed"] = True
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "min_actor_speed_cm_s": 3.0,
        "min_target_speed_cm_s": 1.0,
        "min_target_speed_fraction_of_actor": 0.15,
        "target_occlusion_observed_fraction": 0.85,
        "occlusion_trailing_hold_seconds": 1.0,
        "occlusion_trailing_min_seed_seconds": 0.5,
        "min_pose_valid_fraction_per_bout": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["chase_mask"][:30].all()
    assert signals["chase_occlusion_bridge_mask"][20:30].all()
    assert not signals["chase_mask"][30:].any()


def test_chase_trailing_occlusion_allows_one_near_observed_transition_frame() -> None:
    pair = _pair_frame(40)
    pair["center_distance_cm"] = 12.0
    pair["bbox_center_distance_body_lengths"] = 1.0
    pair["a_to_b_actor_behavior_speed_cm_s"] = 8.0
    pair["a_to_b_target_behavior_speed_cm_s"] = 1.1
    pair["a_to_b_target_bbox_observed"] = True
    pair["a_to_b_target_bbox_imputed"] = False
    pair["a_to_b_actor_bbox_observed"] = True
    pair["a_to_b_actor_bbox_imputed"] = False
    pair["b_to_a_actor_bbox_observed"] = True
    pair["b_to_a_target_bbox_observed"] = True
    pair["b_to_a_actor_bbox_imputed"] = False
    pair["b_to_a_target_bbox_imputed"] = False
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    # Frame 20 is a near, fully observed but directionally neutral transition;
    # frame 21 begins the real single-target occlusion.
    pair.loc[20, "a_to_b_target_behavior_speed_cm_s"] = 0.0
    pair.loc[20, "a_to_b_pursuit_alignment"] = 0.0
    pair.loc[20, "a_to_b_target_escape_alignment"] = 0.0
    pair.loc[20, "a_to_b_direction_similarity"] = 0.0
    pair.loc[20, "bbox_center_distance_body_lengths"] = 2.8
    pair.loc[21:, "valid_pair"] = False
    pair.loc[21:, "a_to_b_target_bbox_observed"] = False
    pair.loc[21:, "a_to_b_target_bbox_imputed"] = True
    pair.loc[21:, "b_to_a_actor_bbox_observed"] = False
    pair.loc[21:, "b_to_a_actor_bbox_imputed"] = True
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "min_actor_speed_cm_s": 3.0,
        "min_target_speed_cm_s": 1.0,
        "min_target_speed_fraction_of_actor": 0.15,
        "target_occlusion_observed_fraction": 0.85,
        "hold_seconds": 0.10,
        "hold_min_fraction": 1.0,
        "occlusion_trailing_hold_seconds": 1.0,
        "occlusion_trailing_min_seed_seconds": 0.5,
        "occlusion_trailing_observed_bridge_frames": 1,
        "occlusion_trailing_max_distance_body_lengths": 3.0,
        "min_pose_valid_fraction_per_bout": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["chase_mask"][:30].all()
    assert signals["chase_occlusion_bridge_mask"][20:30].all()
    assert not signals["chase_mask"][30:].any()


def test_attack_bbox_validity_does_not_promote_sparse_chase_pose_support() -> None:
    pair = _pair_frame(40)
    pair["center_distance_cm"] = 12.0
    pair["valid_pair"] = False
    pair.loc[:4, "valid_pair"] = True
    pair.loc[35:, "valid_pair"] = True
    pair["a_to_b_target_bbox_observed"] = True
    pair["a_to_b_target_bbox_imputed"] = False
    pair.loc[5:34, "a_to_b_target_bbox_observed"] = False
    pair.loc[5:34, "a_to_b_target_bbox_imputed"] = True
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    config = _social_config()
    config["semantic_chase"] = {
        **config["semantic_chase"],
        "occlusion_fill_gap_seconds": 3.0,
        "occlusion_gap_min_imputed_fraction": 0.50,
        "min_pose_valid_fraction_per_bout": 0.50,
    }
    # Enabling the wider attack bbox state used to mutate the chase ``valid``
    # array in place, making this 25%-observed chase bout look 100% supported.
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "min_bbox_overlap_iou": 0.05,
        "max_bbox_center_distance_body_lengths": 1.35,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["chase_mask"].any()
    assert not signals["chase_occlusion_bridge_mask"].any()


def test_avoidance_requires_prior_approach_and_reports_evader_as_actor() -> None:
    pair = _pair_frame()
    pair.loc[:2, "center_distance_cm"] = 100.0
    pair.loc[3:14, "center_distance_cm"] = np.linspace(90.0, 30.0, 12)
    pair.loc[15:, "center_distance_cm"] = np.linspace(33.0, 75.0, len(pair) - 15)
    pair.loc[15:, "a_to_b_target_turn_angle_deg"] = 45.0
    pair.loc[15:, "a_to_b_target_speed_cm_s"] = 12.0

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=_social_config(),
        contact_config={"enabled": True},
    )

    mask = signals["avoidance_mask"]
    assert mask[15:].any()
    assert set(signals["avoidance_actor"][mask]) == {2}
    assert set(signals["avoidance_target"][mask]) == {1}
    assert not mask[:3].any()


def test_avoidance_recovers_when_approaching_mouse_occludes_and_target_turns_away() -> None:
    pair = _pair_frame(60)
    pair["center_distance_cm"] = 8.0
    pair["bbox_center_distance_body_lengths"] = 0.60
    pair["mouse_a_bbox_scale_px"] = 100.0
    pair["mouse_b_bbox_scale_px"] = 100.0
    pair["mouse_a_bbox_observed"] = True
    pair["mouse_b_bbox_observed"] = True
    pair["mouse_a_bbox_center_x_px"] = 100.0
    pair["mouse_a_bbox_center_y_px"] = 100.0
    pair["mouse_b_bbox_center_x_px"] = 150.0
    pair["mouse_b_bbox_center_y_px"] = 100.0
    pair.loc[10:, "valid_pair"] = False
    pair.loc[10:, "mouse_a_bbox_observed"] = False
    pair.loc[10:, "mouse_a_bbox_center_x_px"] = np.nan
    pair.loc[10:, "mouse_a_bbox_center_y_px"] = np.nan
    pair.loc[10:, "mouse_b_bbox_center_x_px"] = np.linspace(170.0, 500.0, 50)
    pair.loc[8:12, "a_to_b_target_turn_angle_deg"] = 60.0
    config = _social_config()
    config["semantic_avoidance"] = {
        **config["semantic_avoidance"],
        "occlusion_recovery_enabled": True,
        "occlusion_recovery_lookback_seconds": 1.0,
        "occlusion_recovery_confirmation_seconds": 1.0,
        "occlusion_recovery_state_seconds": 4.0,
        "occlusion_recovery_pre_context_seconds": 1.0,
        "occlusion_recovery_min_joint_frames": 4,
        "occlusion_recovery_min_pursuit_alignment": 0.35,
        "occlusion_recovery_min_pursuit_fraction": 0.30,
        "occlusion_recovery_min_turn_angle_deg": 45.0,
        "occlusion_recovery_max_start_distance_body_lengths": 1.25,
        "occlusion_recovery_min_distance_growth_body_lengths": 0.45,
        "occlusion_recovery_min_path_displacement_body_lengths": 0.45,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    mask = signals["avoidance_mask"]
    assert mask[:55].all()
    assert not mask[55:].any()
    assert set(signals["avoidance_actor"][mask]) == {2}
    assert set(signals["avoidance_target"][mask]) == {1}


def test_avoidance_occlusion_recovery_rejects_dropout_without_abrupt_turn() -> None:
    pair = _pair_frame(40)
    pair["bbox_center_distance_body_lengths"] = 0.60
    pair["mouse_a_bbox_scale_px"] = 100.0
    pair["mouse_b_bbox_scale_px"] = 100.0
    pair["mouse_a_bbox_observed"] = True
    pair["mouse_b_bbox_observed"] = True
    pair["mouse_a_bbox_center_x_px"] = 100.0
    pair["mouse_a_bbox_center_y_px"] = 100.0
    pair["mouse_b_bbox_center_x_px"] = 150.0
    pair["mouse_b_bbox_center_y_px"] = 100.0
    pair.loc[10:, "valid_pair"] = False
    pair.loc[10:, "mouse_a_bbox_observed"] = False
    pair.loc[10:, "mouse_b_bbox_center_x_px"] = np.linspace(170.0, 500.0, 30)
    config = _social_config()
    config["semantic_avoidance"] = {
        **config["semantic_avoidance"],
        "occlusion_recovery_enabled": True,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["avoidance_mask"].any()


def test_direction_ids_skip_invalid_leading_pair_rows() -> None:
    pair = _pair_frame(3)
    pair.loc[0, ["mouse_a_id", "mouse_b_id"]] = [np.nan, np.nan]

    assert _direction_ids(pair, "a_to_b") == (1, 2)
    assert _direction_ids(pair, "b_to_a") == (2, 1)


def test_attack_action_evidence_opens_a_temporal_contact_bout() -> None:
    pair = _pair_frame(90)
    pair.loc[:19, "center_distance_cm"] = 25.0
    pair.loc[20:64, "center_distance_cm"] = 2.0
    pair.loc[65:, "center_distance_cm"] = 8.0
    pair.loc[20:64, "a_to_b_nose_head_distance_cm"] = 2.0
    enriched = pd.DataFrame(
        {
            "weak_standard_attack_score": np.where(
                (np.arange(len(pair)) >= 25) & (np.arange(len(pair)) <= 45),
                0.85,
                0.0,
            ),
            "strong_standard_attack_score": np.zeros(len(pair)),
            "weak_standard_dynamic_attack_score": np.where(
                (np.arange(len(pair)) >= 25) & (np.arange(len(pair)) <= 45),
                0.85,
                0.0,
            ),
            "strong_standard_dynamic_attack_score": np.zeros(len(pair)),
        }
    )
    config = _social_config()
    config["semantic_attack"] = {
        "require_nose_head_contact": True,
        "min_attack_score": 0.68,
        "min_dynamic_score": 0.60,
        "min_raw_speed_cm_s": 8.0,
        "min_impact_pursuit_alignment": 0.45,
        "contact_support_seconds": 0.20,
        "min_contact_frames": 2,
        "trigger_support_seconds": 0.40,
        "min_trigger_frames": 2,
        "bout_context_seconds": 0.75,
    }

    signals = build_semantic_pair_signals(
        pair,
        enriched,
        fps=30.0,
        social_config=config,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 3.0,
            "nose_tail_distance_cm": 3.0,
        },
    )

    mask = signals["attack_mask"]
    assert mask[25:46].any()
    assert int(mask.sum()) >= 20


def test_bbox_motion_opens_attack_without_keypoint_contact() -> None:
    pair = _pair_frame(60)
    pair["bbox_pair_valid"] = True
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    pair.loc[20:44, "bbox_overlap_iou"] = 0.35
    pair.loc[20:44, "bbox_center_distance_body_lengths"] = 0.85
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_target_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_actor_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_target_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_actor_bbox_jump_score"] = 0.0
        pair[f"{prefix}_target_bbox_jump_score"] = 0.0
    pair.loc[25:38, "a_to_b_actor_bbox_speed_body_lengths_per_frame"] = 0.22
    pair.loc[25:38, "a_to_b_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.16
    pair.loc[25:38, "a_to_b_actor_bbox_jump_score"] = 0.35
    enriched = pd.DataFrame(index=pair.index)
    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "min_bbox_overlap_iou": 0.05,
        "max_bbox_center_distance_body_lengths": 1.35,
        "min_bbox_speed_body_lengths_per_frame": 0.10,
        "min_bbox_acceleration_body_lengths_per_frame2": 0.08,
        "min_bbox_jump_score": 0.20,
        "min_bbox_motion_frames": 2,
        "bbox_motion_support_seconds": 0.30,
        "bbox_contact_gap_seconds": 0.20,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
        "require_action_signature": True,
        "min_contact_frames": 2,
        "contact_support_seconds": 0.20,
        "trigger_support_seconds": 0.40,
        "min_trigger_frames": 2,
        "bout_context_seconds": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        enriched,
        fps=30.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["bbox_contact_mask"][20:45].any()
    assert signals["attack_mask"][25:39].any()
    assert not signals["contact_mask"].any()


def test_confirmed_attack_episode_bridges_only_bounded_close_pair_state() -> None:
    pair = _pair_frame(70)
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    pair.loc[10:49, "bbox_center_distance_body_lengths"] = 1.0
    pair.loc[20:29, "bbox_overlap_iou"] = 0.35
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_target_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_actor_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_target_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_actor_bbox_jump_score"] = 0.0
        pair[f"{prefix}_target_bbox_jump_score"] = 0.0
    pair.loc[20:29, "a_to_b_actor_bbox_speed_body_lengths_per_frame"] = 0.25
    pair.loc[20:29, "a_to_b_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.20
    pair.loc[20:29, "a_to_b_actor_bbox_jump_score"] = 0.35
    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "bbox_require_overlap": True,
        "min_bbox_overlap_iou": 0.20,
        "min_bbox_speed_body_lengths_per_frame": 0.10,
        "min_bbox_acceleration_body_lengths_per_frame2": 0.08,
        "min_bbox_jump_score": 0.20,
        "min_bbox_motion_frames": 2,
        "bbox_motion_support_seconds": 0.30,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
        "require_action_signature": True,
        "min_contact_frames": 2,
        "contact_support_seconds": 0.20,
        "trigger_support_seconds": 0.30,
        "min_trigger_frames": 2,
        "bout_context_seconds": 0.50,
        "state_hold_seconds": 2.00,
        "state_pre_hold_seconds": 1.00,
        "state_min_seed_seconds": 0.50,
        "state_max_distance_body_lengths": 1.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["attack_mask"][10:50].all()
    # Frame 30 is still part of the conservative direct-contact/context seed;
    # the state bridge starts at the first newly recovered frame.
    assert signals["attack_state_bridge_mask"][31:50].all()
    assert signals["attack_state_bridge_mask"][10:19].all()
    assert not signals["attack_state_bridge_mask"][19:31].any()
    assert not signals["attack_mask"][:10].any()
    assert not signals["attack_mask"][50:].any()


def test_close_pair_without_attack_seed_cannot_open_episode_state() -> None:
    pair = _pair_frame(40)
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 1.0
    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "state_hold_seconds": 2.0,
        "state_min_seed_seconds": 0.5,
        "state_max_distance_body_lengths": 1.5,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["attack_mask"].any()
    assert not signals["attack_state_bridge_mask"].any()


def _coherent_translation_attack_fixture(*, local_jitter: bool) -> tuple[pd.DataFrame, dict]:
    pair = _pair_frame(70)
    frame = np.arange(len(pair), dtype=float)
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair["bbox_pair_imputed"] = False
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    pair.loc[10:59, "bbox_overlap_iou"] = 0.35
    pair.loc[10:59, "bbox_center_distance_body_lengths"] = 0.85
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_target_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_actor_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_target_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_actor_bbox_jump_score"] = 0.0
        pair[f"{prefix}_target_bbox_jump_score"] = 0.0
    pair.loc[12:55, "a_to_b_actor_bbox_speed_body_lengths_per_frame"] = 0.22
    pair.loc[12:55, "a_to_b_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.16
    pair.loc[12:55, "a_to_b_actor_bbox_jump_score"] = 0.35

    if local_jitter:
        translation = np.where((frame.astype(int) % 2) == 0, -20.0, 20.0)
    else:
        translation = frame * 20.0
    pair["mouse_a_bbox_center_x_px"] = translation
    pair["mouse_a_bbox_center_y_px"] = 0.0
    pair["mouse_b_bbox_center_x_px"] = translation + 50.0
    pair["mouse_b_bbox_center_y_px"] = 0.0
    pair["mouse_a_bbox_scale_px"] = 100.0
    pair["mouse_b_bbox_scale_px"] = 100.0

    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "min_bbox_overlap_iou": 0.05,
        "max_bbox_center_distance_body_lengths": 1.35,
        "min_bbox_speed_body_lengths_per_frame": 0.10,
        "min_bbox_acceleration_body_lengths_per_frame2": 0.08,
        "min_bbox_jump_score": 0.20,
        "min_bbox_motion_frames": 2,
        "bbox_motion_support_seconds": 0.30,
        "bbox_contact_gap_seconds": 0.20,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
        "require_action_signature": True,
        "contact_support_seconds": 0.20,
        "min_contact_frames": 2,
        "trigger_support_seconds": 0.30,
        "min_trigger_frames": 2,
        "bout_context_seconds": 0.50,
        "reclassify_coherent_translation_as_chase": True,
        "coherent_translation_min_duration_seconds": 0.50,
        "coherent_translation_fill_gap_seconds": 0.20,
        "coherent_translation_min_net_body_lengths": 2.0,
        "coherent_translation_min_path_efficiency": 0.50,
        "coherent_translation_max_step_body_lengths": 1.0,
    }
    return pair, config


def test_coherently_translating_box_contact_is_chase_not_attack() -> None:
    pair, config = _coherent_translation_attack_fixture(local_jitter=False)

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    translated = signals["chase_bbox_translation_mask"]
    assert translated.any()
    assert signals["chase_mask"][translated].all()
    assert not signals["attack_mask"][translated].any()
    assert set(signals["chase_actor"][translated]) == {1}
    assert set(signals["chase_target"][translated]) == {2}


def test_local_box_jitter_remains_attack_instead_of_chase() -> None:
    pair, config = _coherent_translation_attack_fixture(local_jitter=True)

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["attack_mask"].any()
    assert not signals["chase_bbox_translation_mask"].any()


def test_provisional_attack_does_not_destroy_parallel_directional_chase() -> None:
    pair, config = _coherent_translation_attack_fixture(local_jitter=True)
    # Pose/trajectory evidence independently supports a directed chase while
    # the overlapping, rapidly changing boxes also open a provisional attack
    # state.  A later temporal/group gate may reject that attack, so the pair
    # FSM must preserve the chase channel as a valid fallback.
    pair["center_distance_cm"] = 8.0
    pair["a_to_b_target_behavior_speed_cm_s"] = 4.0
    pair["a_to_b_target_speed_cm_s"] = 4.0
    pair["b_to_a_actor_behavior_speed_cm_s"] = 4.0
    pair["b_to_a_actor_speed_cm_s"] = 4.0

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=10.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["attack_mask"].any()
    assert signals["chase_mask"].any()
    assert np.any(signals["attack_mask"] & signals["chase_mask"])


def test_bbox_motion_without_box_contact_does_not_open_attack() -> None:
    pair = _pair_frame(60)
    pair["bbox_pair_valid"] = True
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_bbox_speed_body_lengths_per_frame"] = 0.22
        pair[f"{prefix}_target_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.16
        pair[f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_actor_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_target_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_actor_bbox_jump_score"] = 0.35
        pair[f"{prefix}_target_bbox_jump_score"] = 0.0
    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "min_bbox_overlap_iou": 0.05,
        "max_bbox_center_distance_body_lengths": 1.35,
        "min_bbox_speed_body_lengths_per_frame": 0.10,
        "min_bbox_acceleration_body_lengths_per_frame2": 0.08,
        "min_bbox_jump_score": 0.20,
        "min_bbox_motion_frames": 2,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
        "require_action_signature": True,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=30.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert not signals["attack_mask"].any()


def test_bbox_attack_bridge_survives_short_target_occlusion() -> None:
    pair = _pair_frame(60)
    pair["bbox_pair_valid"] = True
    pair["bbox_pair_observed"] = True
    pair["bbox_pair_imputed"] = False
    pair["bbox_overlap_iou"] = 0.0
    pair["bbox_center_distance_body_lengths"] = 4.0
    pair.loc[20:44, "bbox_overlap_iou"] = 0.35
    pair.loc[20:44, "bbox_center_distance_body_lengths"] = 0.85
    # The target is absent from the normal pose track for a short interval,
    # but one real box remains visible and the box bridge is still valid.
    pair.loc[30:34, "valid_pair"] = False
    pair.loc[30:34, "bbox_pair_imputed"] = True
    for prefix in ("a_to_b", "b_to_a"):
        pair[f"{prefix}_actor_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_target_bbox_speed_body_lengths_per_frame"] = 0.0
        pair[f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2"] = 0.0
        pair[f"{prefix}_actor_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_target_bbox_area_change_ratio"] = 0.0
        pair[f"{prefix}_actor_bbox_jump_score"] = 0.0
        pair[f"{prefix}_target_bbox_jump_score"] = 0.0
    pair.loc[25:38, "a_to_b_actor_bbox_speed_body_lengths_per_frame"] = 0.22
    pair.loc[25:38, "a_to_b_actor_bbox_acceleration_body_lengths_per_frame2"] = 0.16
    pair.loc[25:38, "a_to_b_actor_bbox_jump_score"] = 0.35
    config = _social_config()
    config["semantic_attack"] = {
        "use_bbox_motion": True,
        "min_bbox_overlap_iou": 0.05,
        "max_bbox_center_distance_body_lengths": 1.35,
        "min_bbox_speed_body_lengths_per_frame": 0.10,
        "min_bbox_acceleration_body_lengths_per_frame2": 0.08,
        "min_bbox_jump_score": 0.20,
        "min_bbox_motion_frames": 2,
        "bbox_motion_support_seconds": 0.30,
        "min_raw_speed_cm_s": 99.0,
        "min_target_turn_angle_deg": 99.0,
        "min_pose_deformation": 99.0,
        "require_action_signature": True,
        "min_contact_frames": 2,
        "contact_support_seconds": 0.20,
        "trigger_support_seconds": 0.40,
        "min_trigger_frames": 2,
        "bout_context_seconds": 0.50,
    }

    signals = build_semantic_pair_signals(
        pair,
        pd.DataFrame(index=pair.index),
        fps=30.0,
        social_config=config,
        contact_config={"enabled": True},
    )

    assert signals["bbox_contact_mask"][30:35].all()
    assert signals["attack_mask"][30:35].any()
    assert not signals["contact_mask"][30:35].any()


def _event(
    behavior: str,
    pair_key: str,
    start: int,
    end: int,
    score: float,
) -> dict[str, object]:
    actor, target = (int(value) for value in pair_key.split("_"))
    return {
        "behavior": behavior,
        "event_scope": "pair",
        "event_recovery": f"semantic_{behavior}",
        "pair_key": pair_key,
        "actor_id": actor,
        "target_id": target,
        "start_frame": start,
        "end_frame": end,
        "core_start_frame": start,
        "core_end_frame": end,
        "analysis_start_frame": start,
        "analysis_end_frame": end,
        "peak_score": score,
        "mean_score": score,
        "source_video": str(Path("sample.mov")),
    }


def test_identity_bridge_does_not_transitively_merge_multiple_id_switches() -> None:
    events = [
        _event("attack", "2_6", 0, 10, 0.70),
        _event("attack", "2_9", 12, 20, 0.90),
        _event("attack", "4_9", 22, 30, 0.80),
    ]

    stitched = _stitch_identity_bridged_events(
        events,
        source_fps=10.0,
        config={
            "extended_behavior": {
                "social": {
                    "semantic_fsm": {
                        "identity_bridge_seconds": 0.20,
                        "max_identity_bridge_participants": 6,
                    }
                }
            }
        },
    )

    assert len(stitched) == 2
    assert stitched[0]["pair_key"] == "2_6|2_9"
    assert stitched[0]["participant_ids"] == [2, 6, 9]
    assert len(stitched[0]["role_trace"]) == 2
    assert stitched[0]["core_duration_s"] == 2.1
    assert stitched[1]["pair_key"] == "4_9"


def test_identity_bridge_does_not_join_unrelated_pairs() -> None:
    events = [
        _event("chase", "1_2", 0, 10, 0.80),
        _event("chase", "3_4", 12, 20, 0.90),
    ]

    stitched = _stitch_identity_bridged_events(
        events,
        source_fps=10.0,
        config={
            "extended_behavior": {"social": {"semantic_fsm": {"identity_bridge_seconds": 0.20}}}
        },
    )

    assert len(stitched) == 2


def test_identity_bridge_can_recover_short_avoidance_occlusion_when_enabled() -> None:
    events = [
        _event("avoidance", "1_2", 0, 10, 0.80),
        _event("avoidance", "1_3", 12, 20, 0.90),
    ]

    stitched = _stitch_identity_bridged_events(
        events,
        source_fps=10.0,
        config={
            "extended_behavior": {
                "social": {
                    "semantic_fsm": {
                        "identity_bridge_seconds": 0.20,
                        "bridge_avoidance": True,
                    }
                }
            }
        },
    )

    assert len(stitched) == 1
    assert stitched[0]["behavior"] == "avoidance"
    assert stitched[0]["pair_key"] == "1_2|1_3"
    assert stitched[0]["participant_ids"] == [1, 2, 3]
