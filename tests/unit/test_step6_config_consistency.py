"""Compatibility YAML values and current authoritative rules remain explicit."""

from __future__ import annotations

from pathlib import Path

from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import config_seed_parameters
from mouse_behavior.evaluation.social_heuristic_config import SocialHeuristicConfig
from mouse_behavior.evaluation.group_heuristic_config import GroupHeuristicConfig

REPO = Path(__file__).resolve().parents[2]


def _configs():
    return load_config(REPO / "configs" / "default.yaml"), load_config(
        REPO / "configs" / "profiles" / "beiyi.yaml"
    )


def test_default_config_preserves_explicit_hierarchical_adapter_values() -> None:
    default, beiyi = _configs()
    for cfg in (default, beiyi):
        social = cfg["hierarchical_heuristic_core"]["social"]
        assert social["together_max_distance_cm"] == 7.0
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


def test_extended_blocks_use_current_values_and_authoritative_rule_configs() -> None:
    default, beiyi = _configs()
    for cfg in (default, beiyi):
        social = cfg["extended_behavior"]["social"]
        group = cfg["extended_behavior"]["group"]
        assert social["together_max_distance_cm"] == 7.0
        assert social["together_min_distance_cm"] == 4.0125
        assert social["approach_min_end_distance_cm"] == 2.675
        assert social["approach_min_start_distance_cm"] == 6.6875
        assert social["approach_min_actor_speed_cm_s"] == 4.0125
        assert group["huddle_distance_cm"] == 10.0
        assert group["huddle_nose_distance_cm"] == 4.0125
        assert group["huddle_min_duration_seconds"] == 5.0
        # The profile's historical mirror is retained for old receipts; the
        # confirmed loader below must still enforce the user-approved 20 cm.
        assert group["isolation_distance_cm"] == (
            20.0 if cfg is default else 5.520760536193848
        )
        assert group["group_locomotion"]["max_neighbor_distance_cm"] == 30.0
        assert group["group_locomotion"]["min_moving_fraction"] == 0.6
        assert group["group_locomotion"]["min_speed_cm_s"] == 6.6875
        assert group["social_clustering"]["min_duration_seconds"] == 0.0
        assert group["dispersal"]["min_duration_seconds"] == 5.0
        # speed gates must stay commented out (absent from parsed yaml)
        assert "huddle_max_mean_speed_cm_s" not in group
        assert "together_max_individual_speed_cm_s" not in social
        current = config_seed_parameters(cfg)
        assert current.isolation_distance_cm == 20.0
        assert current.group_locomotion_max_distance_cm == GroupHeuristicConfig().locomotion_max_pair_distance_cm
        # SocialHeuristicConfig is passed independently to prediction; the YAML
        # mirror retains compatibility values rather than overriding its fit.
        assert SocialHeuristicConfig().dist_together_max_cm == 5.318182945251465
