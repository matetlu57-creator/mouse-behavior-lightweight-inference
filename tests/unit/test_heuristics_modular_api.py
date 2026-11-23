"""Unit tests for the unified heuristics facade and modularized submodules."""
from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

import mouse_behavior.heuristics as heuristics
from mouse_behavior.behavior import (
    individual_rules,
    segment_clipping,
    segment_clipping_common,
    segment_clipping_group,
    segment_clipping_group_clustering,
    segment_clipping_group_dispersal,
    segment_clipping_group_motion,
    segment_clipping_individual,
    segment_clipping_social,
)
from mouse_behavior.canonical_video_pipeline import rule_execution_binding
from mouse_behavior.evaluation import (
    clustering_formation,
    confirmed_rules,
    exported_dataset,
    exported_dataset_models,
    exported_dataset_prediction,
    group_event_clustering,
    group_event_dispersal,
    group_event_geometry,
    group_event_rules,
    group_heuristic_config,
    social_heuristic_config,
)


def test_heuristics_facade_is_lazy_and_side_effect_free() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)
    )
    probe = (
        "import sys, mouse_behavior.heuristics as h; "
        "heavy = ("
        "'numpy', "
        "'mouse_behavior.behavior.individual_rules', "
        "'mouse_behavior.behavior.segment_clipping', "
        "'mouse_behavior.evaluation.confirmed_rules', "
        "'mouse_behavior.evaluation.group_event_rules'"
        "); "
        "loaded = [m for m in heavy if m in sys.modules]; "
        "assert not loaded, f'eagerly imported: {loaded}'; "
        "assert 'IndividualHeuristicConfig' in h.__all__"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_heuristics_facade_exports_identity_with_canonical_modules() -> None:
    expected_mappings = {
        # Configs
        "IndividualHeuristicConfig": (individual_rules, "IndividualHeuristicConfig"),
        "DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG": (individual_rules, "DEFAULT_INDIVIDUAL_CONFIG"),
        "DEFAULT_INDIVIDUAL_CONFIG": (individual_rules, "DEFAULT_INDIVIDUAL_CONFIG"),
        "SocialHeuristicConfig": (social_heuristic_config, "SocialHeuristicConfig"),
        "DEFAULT_SOCIAL_HEURISTIC_CONFIG": (
            social_heuristic_config,
            "DEFAULT_SOCIAL_HEURISTIC_CONFIG",
        ),
        "GroupHeuristicConfig": (group_heuristic_config, "GroupHeuristicConfig"),
        "DEFAULT_GROUP_HEURISTIC_CONFIG": (
            group_heuristic_config,
            "DEFAULT_GROUP_HEURISTIC_CONFIG",
        ),
        "HeuristicParameters": (exported_dataset_models, "HeuristicParameters"),
        # Recognition
        "individual_rule_masks": (individual_rules, "individual_rule_masks"),
        "IndividualRuleResult": (individual_rules, "IndividualRuleResult"),
        "pair_rule_masks": (confirmed_rules, "pair_rule_masks"),
        "pair_rule_roles": (confirmed_rules, "pair_rule_roles"),
        "policy_receipt": (confirmed_rules, "policy_receipt"),
        "confirmed_profiles": (confirmed_rules, "confirmed_profiles"),
        "social_profiles": (confirmed_rules, "social_profiles"),
        "group_rule_candidates": (group_event_rules, "group_rule_candidates"),
        "isolation_candidates": (group_event_rules, "isolation_candidates"),
        "isolation_summaries": (group_event_rules, "isolation_summaries"),
        "dispersal_phase": (group_event_dispersal, "dispersal_phase"),
        "formation_masks": (clustering_formation, "formation_masks"),
        "predict_features": (exported_dataset_prediction, "predict_features"),
        # Clipping
        "refine_behavior_segments": (segment_clipping, "refine_behavior_segments"),
        "select_behavior_segments": (segment_clipping, "select_behavior_segments"),
        "InMemoryTrackStore": (segment_clipping_common, "InMemoryTrackStore"),
        "MIN_PARTICIPANTS_BY_BEHAVIOR": (segment_clipping_common, "MIN_PARTICIPANTS_BY_BEHAVIOR"),
        "MAX_DURATION_S_BY_BEHAVIOR": (segment_clipping_common, "MAX_DURATION_S_BY_BEHAVIOR"),
        "min_participants_for_behavior": (segment_clipping_common, "min_participants_for_behavior"),
        "max_duration_s_for_behavior": (segment_clipping_common, "max_duration_s_for_behavior"),
        "endpoint_observations": (segment_clipping_common, "endpoint_observations"),
        "endpoints_fully_observed": (segment_clipping_common, "endpoints_fully_observed"),
        "is_process_boundary_closed": (segment_clipping_common, "is_process_boundary_closed"),
        "update_segment_bounds": (segment_clipping_common, "update_segment_bounds"),
        "refine_static_segments": (segment_clipping_individual, "refine_static_segments"),
        "refine_locomotion_segments": (segment_clipping_individual, "refine_locomotion_segments"),
        "refine_rearing_segments": (segment_clipping_individual, "refine_rearing_segments"),
        "refine_grooming_segments": (segment_clipping_individual, "refine_grooming_segments"),
        "refine_approach_segments": (segment_clipping_social, "refine_approach_segments"),
        "refine_attack_segments": (segment_clipping_social, "refine_attack_segments"),
        "refine_chasing_segments": (segment_clipping_social, "refine_chasing_segments"),
        "refine_contact_and_together_segments": (
            segment_clipping_social,
            "refine_contact_and_together_segments",
        ),
        "refine_huddle_segments": (segment_clipping_group_motion, "refine_huddle_segments"),
        "refine_social_clustering_segments": (
            segment_clipping_group_clustering,
            "refine_social_clustering_segments",
        ),
        "refine_dispersal_segments": (
            segment_clipping_group_dispersal,
            "refine_dispersal_segments",
        ),
        "refine_group_locomotion_segments": (
            segment_clipping_group_motion,
            "refine_group_locomotion_segments",
        ),
        "refine_isolation_segments": (segment_clipping_group_motion, "refine_isolation_segments"),
    }

    assert set(expected_mappings) <= set(heuristics.__all__)
    assert set(heuristics.__all__) <= set(dir(heuristics))
    for public_name in heuristics.__all__:
        assert hasattr(heuristics, public_name)
    for public_name, (canonical_module, canonical_attr) in expected_mappings.items():
        assert getattr(heuristics, public_name) is getattr(canonical_module, canonical_attr)

    with pytest.raises(AttributeError):
        getattr(heuristics, "non_existent_heuristic_symbol")


def test_segment_clipping_group_facade_reexports_submodules() -> None:
    dispersal_exports = ("_recover_dispersal_origin_members", "refine_dispersal_segments")
    clustering_exports = (
        "_sc_occluded_members_verified",
        "_has_preexisting_scene_cluster_veto",
        "refine_social_clustering_segments",
    )
    motion_exports = (
        "refine_huddle_segments",
        "refine_group_locomotion_segments",
        "refine_isolation_segments",
    )

    for name in dispersal_exports:
        assert getattr(segment_clipping_group, name) is getattr(
            segment_clipping_group_dispersal, name
        )
    for name in clustering_exports:
        assert getattr(segment_clipping_group, name) is getattr(
            segment_clipping_group_clustering, name
        )
    for name in motion_exports:
        assert getattr(segment_clipping_group, name) is getattr(segment_clipping_group_motion, name)

    for public_name in (
        "refine_huddle_segments",
        "refine_social_clustering_segments",
        "refine_dispersal_segments",
        "refine_group_locomotion_segments",
        "refine_isolation_segments",
    ):
        assert getattr(segment_clipping, public_name) is getattr(
            segment_clipping_group, public_name
        )


def test_group_event_rules_reexports_submodules() -> None:
    geometry_exports = (
        "_GroupProcess",
        "group_separation",
        "_components",
        "_geometry",
        "_inside_pile_envelope",
        "_quiet_at",
        "_huddle_geometry",
        "_proposal_sets",
        "_window_positions",
        "_region_gap",
    )
    clustering_exports = (
        "_huddle_window",
        "_huddle_event",
        "_huddle_windows",
        "_is_parallel_group_locomotion",
        "_clustering_process",
        "_clustering_event",
        "_synchronous_process",
        "_synchronous_event",
        "_members_in_preexisting_scene_cluster",
    )
    dispersal_exports = (
        "_departure_origins",
        "_dispersal_endpoint_separated",
        "_departing_member_slots",
        "_dispersal_process",
        "dispersal_phase",
        "_prepare_departure_process",
        "_select_departure_for_checkpoint",
        "_departure_process_mask",
        "_initial_compact_components",
        "_observed_attempt_end",
        "_compact_origin_episodes",
        "_observed_departure_velocity",
    )

    for name in geometry_exports:
        assert getattr(group_event_rules, name) is getattr(group_event_geometry, name)
    for name in clustering_exports:
        assert getattr(group_event_rules, name) is getattr(group_event_clustering, name)
    for name in dispersal_exports:
        assert getattr(group_event_rules, name) is getattr(group_event_dispersal, name)


def test_exported_dataset_and_pipeline_provenance_cover_new_modules() -> None:
    source_files = exported_dataset.EXPORTED_DATASET_SOURCE_FILES
    assert tuple(source_files) == tuple(sorted(source_files))
    assert len(source_files) == len(set(source_files))

    expected_group_modules = {
        "src/mouse_behavior/evaluation/group_event_clustering.py",
        "src/mouse_behavior/evaluation/group_event_dispersal.py",
        "src/mouse_behavior/evaluation/group_event_geometry.py",
        "src/mouse_behavior/evaluation/group_event_rules.py",
    }
    assert expected_group_modules <= set(source_files)

    repo_root = Path(__file__).resolve().parents[2]
    for relative_path in source_files:
        assert (repo_root / relative_path).is_file()

    binding = rule_execution_binding(exported_dataset_models.HeuristicParameters())
    source_hashes = binding["rule_source_sha256"]
    expected_binding_modules = {
        "mouse_behavior.behavior.segment_clipping_group_clustering",
        "mouse_behavior.behavior.segment_clipping_group_dispersal",
        "mouse_behavior.behavior.segment_clipping_group_motion",
        "mouse_behavior.evaluation.group_event_clustering",
        "mouse_behavior.evaluation.group_event_dispersal",
        "mouse_behavior.evaluation.group_event_geometry",
        "mouse_behavior.evaluation.group_event_rules",
    }
    assert expected_binding_modules <= set(source_hashes)
    for module_name in expected_binding_modules:
        assert import_module(module_name) is not None
