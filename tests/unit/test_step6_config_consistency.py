"""Step 6-12 config consistency: every scale/definition block agrees with doc v4."""

from __future__ import annotations

from pathlib import Path

from mouse_behavior.config import load_config

REPO = Path(__file__).resolve().parents[2]


def _configs():
    return load_config(REPO / "configs" / "default.yaml"), load_config(
        REPO / "configs" / "profiles" / "beiyi.yaml"
    )


def test_default_config_carries_v4_social_values() -> None:
    default, beiyi = _configs()
    for cfg in (default, beiyi):
        social = cfg["hierarchical_heuristic_core"]["social"]
        assert social["together_max_distance_cm"] == 8.025
        assert social["together_min_distance_cm"] == 4.0125
        assert social["chasing_max_distance_cm"] == 16.05
        assert social["chasing_min_speed_cm_s"] == 26.75
        assert social["chasing_min_duration_s"] == 1.0
        assert social["min_heading_similarity"] == 0.707
        assert social["approach_min_closing_speed_cm_s"] == 2.675
        collective = cfg["hierarchical_heuristic_core"]["collective"]
        assert collective["huddling_min_duration_s"] == 5.0
        assert collective["social_clustering_min_duration_s"] == 0.0
        assert collective["dispersal_min_duration_s"] == 5.0
        assert collective["group_locomotion_max_neighbor_distance_cm"] == 33.4375
        assert collective["isolation_min_distance_cm"] == 26.75
        assert "huddle_max_high_motion_fraction" not in collective


def test_extended_blocks_match_v4_and_drop_speed_gates() -> None:
    default, beiyi = _configs()
    for cfg in (default, beiyi):
        social = cfg["extended_behavior"]["social"]
        group = cfg["extended_behavior"]["group"]
        assert social["together_max_distance_cm"] == 8.025
        assert social["together_min_distance_cm"] == 4.0125
        assert social["approach_min_end_distance_cm"] == 2.675
        assert social["approach_min_start_distance_cm"] == 6.6875
        assert social["approach_min_actor_speed_cm_s"] == 4.0125
        assert group["huddle_distance_cm"] == 8.025
        assert group["huddle_nose_distance_cm"] == 4.0125
        assert group["huddle_min_duration_seconds"] == 5.0
        assert group["isolation_distance_cm"] == 26.75
        assert group["group_locomotion"]["max_neighbor_distance_cm"] == 33.4375
        assert group["group_locomotion"]["min_moving_fraction"] == 0.6
        assert group["group_locomotion"]["min_speed_cm_s"] == 6.6875
        assert group["social_clustering"]["min_duration_seconds"] == 0.0
        assert group["dispersal"]["min_duration_seconds"] == 5.0
        # speed gates must stay commented out (absent from parsed yaml)
        assert "huddle_max_mean_speed_cm_s" not in group
        assert "together_max_individual_speed_cm_s" not in social