"""Unified public facade for 15-behavior heuristic recognition and segment clipping.

Attributes are resolved lazily via ``__getattr__`` so importing
``mouse_behavior.heuristics`` stays side-effect-free and avoids circular imports
between ``mouse_behavior.behavior`` and ``mouse_behavior.evaluation`` while
preserving exact object identity with each canonical defining module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_GROUP_HEURISTIC_CONFIG",
    "DEFAULT_INDIVIDUAL_CONFIG",
    "DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG",
    "DEFAULT_SOCIAL_HEURISTIC_CONFIG",
    "GroupHeuristicConfig",
    "HeuristicParameters",
    "InMemoryTrackStore",
    "IndividualHeuristicConfig",
    "IndividualRuleResult",
    "MAX_DURATION_S_BY_BEHAVIOR",
    "MIN_PARTICIPANTS_BY_BEHAVIOR",
    "SocialHeuristicConfig",
    "confirmed_profiles",
    "dispersal_phase",
    "endpoint_observations",
    "endpoints_fully_observed",
    "formation_masks",
    "group_rule_candidates",
    "individual_rule_masks",
    "is_process_boundary_closed",
    "isolation_candidates",
    "isolation_summaries",
    "max_duration_s_for_behavior",
    "min_participants_for_behavior",
    "pair_rule_masks",
    "pair_rule_roles",
    "policy_receipt",
    "predict_features",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_behavior_segments",
    "refine_chasing_segments",
    "refine_contact_and_together_segments",
    "refine_dispersal_segments",
    "refine_grooming_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
    "refine_locomotion_segments",
    "refine_rearing_segments",
    "refine_social_clustering_segments",
    "refine_static_segments",
    "select_behavior_segments",
    "social_profiles",
    "update_segment_bounds",
]

_EXPORT_SOURCES: dict[str, tuple[str, str]] = {
    # Configs and parameter models
    "IndividualHeuristicConfig": (
        "mouse_behavior.behavior.individual_rules",
        "IndividualHeuristicConfig",
    ),
    "DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG": (
        "mouse_behavior.behavior.individual_rules",
        "DEFAULT_INDIVIDUAL_CONFIG",
    ),
    "DEFAULT_INDIVIDUAL_CONFIG": (
        "mouse_behavior.behavior.individual_rules",
        "DEFAULT_INDIVIDUAL_CONFIG",
    ),
    "SocialHeuristicConfig": (
        "mouse_behavior.evaluation.social_heuristic_config",
        "SocialHeuristicConfig",
    ),
    "DEFAULT_SOCIAL_HEURISTIC_CONFIG": (
        "mouse_behavior.evaluation.social_heuristic_config",
        "DEFAULT_SOCIAL_HEURISTIC_CONFIG",
    ),
    "GroupHeuristicConfig": (
        "mouse_behavior.evaluation.group_heuristic_config",
        "GroupHeuristicConfig",
    ),
    "DEFAULT_GROUP_HEURISTIC_CONFIG": (
        "mouse_behavior.evaluation.group_heuristic_config",
        "DEFAULT_GROUP_HEURISTIC_CONFIG",
    ),
    "HeuristicParameters": (
        "mouse_behavior.evaluation.exported_dataset_models",
        "HeuristicParameters",
    ),
    # Heuristic recognition (individual, pair/social, group, prediction)
    "individual_rule_masks": (
        "mouse_behavior.behavior.individual_rules",
        "individual_rule_masks",
    ),
    "IndividualRuleResult": (
        "mouse_behavior.behavior.individual_rules",
        "IndividualRuleResult",
    ),
    "pair_rule_masks": (
        "mouse_behavior.evaluation.confirmed_rules",
        "pair_rule_masks",
    ),
    "pair_rule_roles": (
        "mouse_behavior.evaluation.confirmed_rules",
        "pair_rule_roles",
    ),
    "policy_receipt": (
        "mouse_behavior.evaluation.confirmed_rules",
        "policy_receipt",
    ),
    "confirmed_profiles": (
        "mouse_behavior.evaluation.confirmed_rules",
        "confirmed_profiles",
    ),
    "social_profiles": (
        "mouse_behavior.evaluation.confirmed_rules",
        "social_profiles",
    ),
    "group_rule_candidates": (
        "mouse_behavior.evaluation.group_event_rules",
        "group_rule_candidates",
    ),
    "isolation_candidates": (
        "mouse_behavior.evaluation.group_event_rules",
        "isolation_candidates",
    ),
    "isolation_summaries": (
        "mouse_behavior.evaluation.group_event_rules",
        "isolation_summaries",
    ),
    "dispersal_phase": (
        "mouse_behavior.evaluation.group_event_dispersal",
        "dispersal_phase",
    ),
    "formation_masks": (
        "mouse_behavior.evaluation.clustering_formation",
        "formation_masks",
    ),
    "predict_features": (
        "mouse_behavior.evaluation.exported_dataset_prediction",
        "predict_features",
    ),
    # Segment clipping and selection
    "refine_behavior_segments": (
        "mouse_behavior.behavior.segment_clipping",
        "refine_behavior_segments",
    ),
    "select_behavior_segments": (
        "mouse_behavior.behavior.segment_clipping",
        "select_behavior_segments",
    ),
    "InMemoryTrackStore": (
        "mouse_behavior.behavior.segment_clipping_common",
        "InMemoryTrackStore",
    ),
    "MIN_PARTICIPANTS_BY_BEHAVIOR": (
        "mouse_behavior.behavior.segment_clipping_common",
        "MIN_PARTICIPANTS_BY_BEHAVIOR",
    ),
    "MAX_DURATION_S_BY_BEHAVIOR": (
        "mouse_behavior.behavior.segment_clipping_common",
        "MAX_DURATION_S_BY_BEHAVIOR",
    ),
    "min_participants_for_behavior": (
        "mouse_behavior.behavior.segment_clipping_common",
        "min_participants_for_behavior",
    ),
    "max_duration_s_for_behavior": (
        "mouse_behavior.behavior.segment_clipping_common",
        "max_duration_s_for_behavior",
    ),
    "endpoint_observations": (
        "mouse_behavior.behavior.segment_clipping_common",
        "endpoint_observations",
    ),
    "endpoints_fully_observed": (
        "mouse_behavior.behavior.segment_clipping_common",
        "endpoints_fully_observed",
    ),
    "is_process_boundary_closed": (
        "mouse_behavior.behavior.segment_clipping_common",
        "is_process_boundary_closed",
    ),
    "update_segment_bounds": (
        "mouse_behavior.behavior.segment_clipping_common",
        "update_segment_bounds",
    ),
    "refine_static_segments": (
        "mouse_behavior.behavior.segment_clipping_individual",
        "refine_static_segments",
    ),
    "refine_locomotion_segments": (
        "mouse_behavior.behavior.segment_clipping_individual",
        "refine_locomotion_segments",
    ),
    "refine_rearing_segments": (
        "mouse_behavior.behavior.segment_clipping_individual",
        "refine_rearing_segments",
    ),
    "refine_grooming_segments": (
        "mouse_behavior.behavior.segment_clipping_individual",
        "refine_grooming_segments",
    ),
    "refine_approach_segments": (
        "mouse_behavior.behavior.segment_clipping_social",
        "refine_approach_segments",
    ),
    "refine_attack_segments": (
        "mouse_behavior.behavior.segment_clipping_social",
        "refine_attack_segments",
    ),
    "refine_chasing_segments": (
        "mouse_behavior.behavior.segment_clipping_social",
        "refine_chasing_segments",
    ),
    "refine_contact_and_together_segments": (
        "mouse_behavior.behavior.segment_clipping_social",
        "refine_contact_and_together_segments",
    ),
    "refine_huddle_segments": (
        "mouse_behavior.behavior.segment_clipping_group_motion",
        "refine_huddle_segments",
    ),
    "refine_social_clustering_segments": (
        "mouse_behavior.behavior.segment_clipping_group_clustering",
        "refine_social_clustering_segments",
    ),
    "refine_dispersal_segments": (
        "mouse_behavior.behavior.segment_clipping_group_dispersal",
        "refine_dispersal_segments",
    ),
    "refine_group_locomotion_segments": (
        "mouse_behavior.behavior.segment_clipping_group_motion",
        "refine_group_locomotion_segments",
    ),
    "refine_isolation_segments": (
        "mouse_behavior.behavior.segment_clipping_group_motion",
        "refine_isolation_segments",
    ),
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_SOURCES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
