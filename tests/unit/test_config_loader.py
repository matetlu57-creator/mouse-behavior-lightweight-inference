from __future__ import annotations

from pathlib import Path

import pytest

from mouse_behavior.config import load_config


def test_load_config_resolves_relative_extends_and_deep_merges(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "profiles" / "balanced.yaml"
    child.parent.mkdir()
    base.write_text(
        """\nengine:\n  enabled: true\n  thresholds:\n    enter: 2\n    exit: 1\nitems: [base]\n""",
        encoding="utf-8",
    )
    child.write_text(
        """\nextends: ../base.yaml\nengine:\n  thresholds:\n    enter: 3\nprofile: balanced\n""",
        encoding="utf-8",
    )

    result = load_config(child)

    assert result["engine"]["enabled"] is True
    assert result["engine"]["thresholds"] == {"enter": 3, "exit": 1}
    assert result["items"] == ["base"]
    assert result["profile"] == "balanced"


def test_load_config_rejects_inheritance_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="循环"):
        load_config(first)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_beiyi_profile_carries_document_duration_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "profiles" / "beiyi.yaml")

    individual = config["extended_behavior"]["individual"]
    social = config["extended_behavior"]["social"]
    group = config["extended_behavior"]["group"]
    semantic = social["semantic_fsm"]
    semantic_chase = semantic["semantic_chase"]
    semantic_attack = semantic["semantic_attack"]
    lightweight_tracking = config["lightweight_behavior_inference"]["tracking"]

    assert config["extended_behavior"]["unspecified_min_duration_seconds"] == 1.0
    assert individual["running_min_duration_seconds"] == 0.5
    assert individual["walking_min_duration_seconds"] == 1.0
    assert individual["stationary_min_duration_seconds"] == 1.0
    assert social["together_min_duration_seconds"] == 1.0
    assert social["approach_min_duration_seconds"] == 1.0
    assert social["avoidance_min_duration_seconds"] == 1.0
    assert social["attack_min_duration_seconds"] == 1.0
    assert social["chase_fallback"]["min_duration_seconds"] == 2.0
    assert semantic_chase["role_behind_weight"] == 2.0
    assert semantic_chase["role_initial_context_seconds"] == 2.0
    assert semantic_chase["min_directional_evidence_fraction_per_bout"] == 0.30
    assert semantic_chase["min_pose_valid_fraction_per_bout"] == 0.50
    assert semantic_attack["reclassify_coherent_translation_as_chase"] is True
    assert semantic_attack["coherent_translation_min_net_body_lengths"] == 2.0
    assert semantic_attack["coherent_translation_min_path_efficiency"] == 0.50
    assert lightweight_tracking["initial_min_detection_score"] == 0.50
    assert group["huddle_min_duration_seconds"] == 1.0
    assert group["huddle_distance_cm"] == 11.0
    assert group["huddle_max_pair_distance_body_lengths"] == 1.75
    assert group["huddle_min_cluster_density"] == 0.5
    assert group["huddle_density_mode"] == "local"
    assert group["huddle_local_neighbor_cap"] == 4
    assert group["huddle_body_length_cap_enabled"] is False
    assert group["huddle_resolve_attack_conflicts"] is True
    assert group["huddle_attack_independent_seconds"] == 2.0
    assert group["isolation_min_duration_seconds"] == 3.0
    assert config["contact_detection"]["nose_head_min_duration_seconds"] == 1.0
    assert config["contact_detection"]["nose_tail_min_duration_seconds"] == 0.5
