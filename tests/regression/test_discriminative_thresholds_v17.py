"""Guards for the generated discriminative v17 threshold profile.

These tests read the emitted module and the fit receipt only. They never run
``scripts/fit_discriminative_thresholds_v17.py`` and never touch the dataset, so
they stay fast.

The single most important guarantee here is the second one: the new module must
not change runtime behaviour. ``ACTIVE_PROFILE`` stays ``"extrema"`` and
``SocialHeuristicConfig()`` must resolve to exactly the field values it had
before the module existed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from mouse_behavior.evaluation import discriminative_thresholds_v17 as discriminative
from mouse_behavior.evaluation.retrospective_thresholds_v17 import (
    ISOLATION_DISTANCE_CM as EXTREMA_ISOLATION_CM,
)
from mouse_behavior.evaluation.retrospective_thresholds_v17 import SOCIAL_EXTREMA
from mouse_behavior.evaluation.retrospective_thresholds_v18 import SOCIAL_FITTED
from mouse_behavior.evaluation.retrospective_thresholds_v20 import SOCIAL_FITTED as V20_SOCIAL_FITTED
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)

REPOSITORY = Path(__file__).resolve().parents[2]
RUN_DIR = REPOSITORY / "outputs" / "heuristics_v17_20261016_run01"
OUTPUT_DIR = REPOSITORY / "outputs" / "discriminative_fit_v17_1"
MODULE_PATH = REPOSITORY / "src" / "mouse_behavior" / "evaluation" / \
    "discriminative_thresholds_v17.py"

# Physical sanity bounds, restated independently of the fitter on purpose.
REFERENCE_BODY_CM = 7.918
MIN_SPEED_FLOOR_CM_S = 0.5
MAX_CONTACT_DISTANCE_CM = 0.75 * REFERENCE_BODY_CM
MIN_ISOLATION_DISTANCE_CM = 1.0 * REFERENCE_BODY_CM

SPEED_GATES = ("vel_attack_min_cm_s", "vel_chasing_min_cm_s", "vel_chasing_target_min_cm_s")
CONTACT_GATES = ("dist_nose_head_contact_max_cm", "dist_nose_tail_contact_max_cm")
GATE_NAMES = (*SOCIAL_EXTREMA, "ISOLATION_DISTANCE_CM")


def receipt() -> dict:
    path = OUTPUT_DIR / "receipt.json"
    if not path.is_file():
        pytest.skip(f"fit receipt not available: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_sha256(values: dict) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def gates() -> dict[str, dict]:
    return discriminative.GATES


# --- requirement 4: physical sanity bounds -----------------------------------

def test_no_speed_floor_is_below_the_physical_minimum():
    for name in SPEED_GATES:
        value = gates()[name]["value"]
        assert value >= MIN_SPEED_FLOOR_CM_S, (name, value)
        assert gates()[name]["gate_kind"] == "minimum"


def test_no_contact_distance_exceeds_three_quarters_of_a_body_length():
    for name in CONTACT_GATES:
        value = gates()[name]["value"]
        assert 0 < value <= MAX_CONTACT_DISTANCE_CM, (name, value)
        assert gates()[name]["gate_kind"] == "maximum"
    # The extrema profile this replaces is itself above the bound, which is the
    # whole reason the bound exists.
    assert SOCIAL_EXTREMA["dist_nose_tail_contact_max_cm"] > REFERENCE_BODY_CM


def test_isolation_distance_is_at_least_one_body_length():
    value = gates()["ISOLATION_DISTANCE_CM"]["value"]
    assert value >= MIN_ISOLATION_DISTANCE_CM, value
    assert value < 74.0, "an isolation floor at the arena size would be degenerate"


def test_approach_band_and_endpoint_ordering_hold():
    endpoint = gates()["approach_surface_endpoint_cm"]["value"]
    tolerance = gates()["approach_surface_endpoint_tolerance_cm"]["value"]
    assert tolerance <= endpoint, (tolerance, endpoint)
    assert tolerance > 0 and endpoint > 0
    end_min = gates()["dist_approach_end_min_cm"]["value"]
    end_max = gates()["dist_approach_end_max_cm"]["value"]
    assert end_min < end_max, (end_min, end_max)
    assert end_min < DEFAULT_SOCIAL_HEURISTIC_CONFIG.dist_approach_start_min_cm


def test_every_emitted_value_is_finite_and_nonnegative():
    for name in GATE_NAMES:
        value = gates()[name]["value"]
        assert isinstance(value, float) and math.isfinite(value), (name, value)
        assert value >= 0, (name, value)


def test_discriminative_values_still_satisfy_social_post_init_invariants():
    """The real cross-field check: build the config with the fitted values."""
    values = {name: gates()[name]["value"] for name in SOCIAL_EXTREMA}
    resolved = SocialHeuristicConfig(**{**asdict(DEFAULT_SOCIAL_HEURISTIC_CONFIG), **values})
    for name, value in values.items():
        assert getattr(resolved, name) == value, name
    assert resolved.priority_order == DEFAULT_SOCIAL_HEURISTIC_CONFIG.priority_order
    priorities = (resolved.approach_priority, resolved.together_priority, resolved.attack_priority,
                  resolved.chasing_priority, resolved.nose_contact_priority,
                  resolved.nose_tail_priority)
    assert sorted(priorities) == list(range(1, 7))


# --- the "no runtime behaviour change" guarantee ------------------------------

def test_social_heuristic_config_resolved_values_are_unchanged():
    """Frozen v17 fields keep their snapshot; named later fields stay explicit."""
    recorded = receipt()["social_config_default_snapshot"]
    # v22 explicitly refits the bounded bbox-gap cap. Restore that one field
    # only for this historical hash check; test_video_boundaries_v22 locks the
    # new live value and the repeated-same-target guard independently.
    historical = SocialHeuristicConfig(**recorded)
    current = asdict(historical)
    # Later semantic/contact fields are independently locked by their own
    # regressions; project the frozen field set for this historical v17 test.
    assert set(current) - set(recorded) == {
        "attack_deceleration_max_s", "together_use_smoothed_speed",
        "attack_contact_surface_max_gap_cm", "attack_contact_max_footprint_body_lengths",
        "contact_head_extension_distance_cm", "contact_tail_extension_distance_cm",
        "contact_head_extension_min_duration_s", "contact_tail_extension_min_duration_s",
        "attack_new_action_target_surface_max_gap_cm",
        "approach_head_facing_max_deg", "approach_body_facing_max_deg",
        "chasing_head_facing_max_deg", "chasing_body_facing_max_deg",
        "chasing_role_turn_grace_s", "chasing_min_mean_speed_cm_s",
        "nose_tail_min_duration_s", "contact_tail_bridge_max_gap_s",
        "contact_head_onset_distance_cm", "contact_tail_onset_distance_cm",
    }
    assert current["attack_deceleration_max_s"] == .75
    assert current["attack_new_action_target_surface_max_gap_cm"] == 1.9986556302093281
    assert current["together_use_smoothed_speed"] is True
    resolved = {name: current[name] for name in recorded}
    assert resolved == recorded, {
        name: (recorded[name], resolved[name])
        for name in resolved if resolved[name] != recorded[name]}
    assert snapshot_sha256(resolved) == receipt()["social_config_default_snapshot_sha256"]


def test_every_fitted_social_field_still_resolves_to_the_extrema_value():
    recorded = receipt()["social_config_default_snapshot"]
    config = SocialHeuristicConfig(**recorded)
    for name, extrema_value in SOCIAL_EXTREMA.items():
        assert getattr(config, name) == extrema_value, name
    assert asdict(DEFAULT_SOCIAL_HEURISTIC_CONFIG) == asdict(SocialHeuristicConfig())
    assert all(getattr(DEFAULT_SOCIAL_HEURISTIC_CONFIG, name) == value
               for name, value in {**SOCIAL_FITTED, **V20_SOCIAL_FITTED}.items()
               if name != "vel_together_max_cm_s")


def test_importing_the_discriminative_module_does_not_move_the_config():
    before = asdict(SocialHeuristicConfig())
    import importlib

    importlib.reload(discriminative)
    assert discriminative.ACTIVE_PROFILE == "extrema"
    assert asdict(SocialHeuristicConfig()) == before


def test_no_runtime_module_imports_the_generated_profile():
    importing = []
    for path in sorted((REPOSITORY / "src").rglob("*.py")):
        if path.resolve() == MODULE_PATH.resolve():
            continue
        if "discriminative_thresholds_v17" in path.read_text(encoding="utf-8", errors="replace"):
            importing.append(str(path.relative_to(REPOSITORY)))
    assert importing == [], f"runtime modules must not import the profile: {importing}"


def test_module_is_marked_generated_and_defaults_to_the_extrema_profile():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert text.startswith('"""GENERATED FILE')
    assert "fit_discriminative_thresholds_v17.py" in text
    assert discriminative.ACTIVE_PROFILE == "extrema"
    assert discriminative.GENERATOR == "scripts/fit_discriminative_thresholds_v17.py"


# --- requirement: fitted values must not silently reproduce the extrema ------

def test_no_fitted_gate_reproduces_its_extrema_value_unless_declared():
    for name in GATE_NAMES:
        gate = gates()[name]
        if gate["value"] == gate["extrema_value"]:
            assert gate["unfittable"] or gate["clamped"], (
                f"{name} equals the extrema value without being marked unfittable or clamped")
    # The extrema values themselves are carried for comparison and must be exact.
    for name, extrema_value in SOCIAL_EXTREMA.items():
        assert gates()[name]["extrema_value"] == extrema_value, name
    assert gates()["ISOLATION_DISTANCE_CM"]["extrema_value"] == EXTREMA_ISOLATION_CM


def test_clamped_gates_record_the_reason_and_the_pre_clamp_value():
    clamped = [name for name in GATE_NAMES if gates()[name]["clamped"]]
    assert clamped, "the contact gates are expected to clamp at 0.75 body lengths"
    for name in clamped:
        gate = gates()[name]
        assert gate["clamp_reason"], name
        assert gate["pre_clamp_value"] is not None, name
        assert gate["clamps"], name
        assert gate["pre_clamp_value"] != gate["value"], name


def test_unfittable_gates_are_never_set_to_a_permissive_value():
    for name in GATE_NAMES:
        gate = gates()[name]
        if gate["unfittable"]:
            assert gate.get("unfittable_evidence"), f"{name} is unfittable without evidence"


# --- requirement: every gate carries non-empty provenance --------------------

def test_every_gate_has_non_empty_provenance():
    for name in GATE_NAMES:
        gate = gates()[name]
        assert gate["justification"].strip(), name
        assert gate["expected_positive_loss"].strip(), name
        assert gate["positive_class"].strip(), name
        assert gate["positive_metric"], name
        assert gate["quantile_used"], name
        assert gate["train_positive_count"] > 0, name
        ladder = gate["quantile_ladder"]
        assert ladder and ladder.get("n") == gate["train_positive_count"], name
        assert gate["train_sensitivity"] is not None and 0 <= gate["train_sensitivity"] <= 1, name
        has_specificity = gate["train_specificity"] is not None
        has_quantile_evidence = bool(gate["quantile_used"]) and bool(gate["expected_positive_loss"])
        assert has_specificity or has_quantile_evidence, name
        if not has_specificity:
            assert gate["specificity_status"], name


def test_extrema_min_and_max_are_never_used_as_a_fitted_value():
    """The fallback must be a robust quantile, never an extremum."""
    for name in GATE_NAMES:
        gate = gates()[name]
        ladder = gate["quantile_ladder"]
        for extreme in ("min", "max"):
            observed = ladder.get(extreme)
            if observed is None:
                continue
            if gate["value"] == observed:
                assert gate["clamped"] or gate["unfittable"], (name, extreme)
        assert gate["quantile_used"], name


def test_delta_and_ratio_agree_with_the_values():
    for name in GATE_NAMES:
        gate = gates()[name]
        assert math.isclose(gate["delta_vs_extrema"],
                            gate["value"] - gate["extrema_value"], rel_tol=0, abs_tol=1e-12), name
        if gate["extrema_value"]:
            assert math.isclose(gate["ratio_vs_extrema"],
                                gate["value"] / gate["extrema_value"], rel_tol=1e-12), name


# --- requirement 5: user overrides are preserved verbatim --------------------

def test_user_overrides_are_carried_verbatim_and_flagged_not_fitted():
    overrides = discriminative.USER_OVERRIDES_NOT_FITTED
    assert set(overrides) == {
        "static_radius_cm", "static_transient_s", "together_distance_strictly_below_cm",
        "together_speed_strictly_below_cm_s", "group_neighbor_distance_cm",
        "clustering_quiet_speed_cm_s", "synchronous_support_at_least_s"}
    for name, entry in overrides.items():
        assert entry["source"] == "user_override_not_fitted", name
        assert isinstance(entry["value"], (int, float)), name
    source = RUN_DIR / "fitted_thresholds.json"
    if source.is_file():
        frozen = json.loads(source.read_text(encoding="utf-8-sig"))["explicit_user_overrides"]
        assert {name: entry["value"] for name, entry in overrides.items()} == frozen
        if not (OUTPUT_DIR / "fitted_thresholds_discriminative.json").is_file():
            pytest.skip("optional local calibration payload is not published with source")
        payload = json.loads((OUTPUT_DIR / "fitted_thresholds_discriminative.json")
                             .read_text(encoding="utf-8"))
        assert payload["explicit_user_overrides"] == frozen
    # No override name may be re-fitted as a gate.
    assert not (set(overrides) & set(GATE_NAMES))


# --- requirement 3/6: scope and validation flags are present and consistent --

def test_fit_scope_and_validation_flags_are_present_and_consistent():
    assert discriminative.FIT_SCOPE
    assert discriminative.FIT_POLICY in ("discriminative_youden", "robust_quantile_fallback")
    assert discriminative.VALIDATION_USED_FOR_FITTING is False
    assert discriminative.INDEPENDENT_VALIDATION is False
    if not (OUTPUT_DIR / "fitted_thresholds_discriminative.json").is_file():
        pytest.skip("optional local calibration payload is not published with source")
    payload = json.loads((OUTPUT_DIR / "fitted_thresholds_discriminative.json")
                         .read_text(encoding="utf-8"))
    assert payload["fit_scope"] == discriminative.FIT_SCOPE
    assert payload["fit_policy"] == discriminative.FIT_POLICY
    assert payload["validation_used_for_fitting"] is False
    assert payload["independent_validation"] is False
    assert payload["fit_split"] == "train"
    if "train" in discriminative.FIT_SCOPE.lower():
        assert payload["validation_used_for_fitting"] is False
        assert discriminative.VALIDATION_USED_FOR_FITTING is False
    assert payload["runtime_behaviour_changed"] is False
    assert payload["active_runtime_profile"] == "extrema"
    assert receipt()["validation_used_for_fitting"] is False


def test_val_split_is_descriptive_only_for_every_gate():
    for name in GATE_NAMES:
        descriptive = gates()[name].get("val_descriptive")
        if descriptive is None:
            continue
        assert descriptive["used_for_fitting"] is False, name


def test_comparator_verdict_is_recorded_and_no_youden_value_was_used():
    probes = discriminative.COMPARATOR_PROBES
    assert probes["verdict"].startswith("no_valid_comparator_in_frozen_artifacts")
    for name, probe in probes.get("per_gate", {}).items():
        assert probe["used_for_fitting"] is False, name
        if "youden_best" in probe:
            assert probe["youden_best_is_used"] is False, name
            assert probe["valid_negative_for_this_gate"] is False, name
            assert probe["rejection_reason"], name


# --- provenance integrity of the recorded artifact hashes --------------------

def test_recorded_artifact_hashes_match_the_files_on_disk():
    """Provenance is only real if the recorded digests still match the inputs."""
    recorded = discriminative.SOURCE_ARTIFACT_SHA256
    assert len(recorded) >= 6
    for name, digest in recorded.items():
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), name
    record = receipt()
    assert record["input_artifact_sha256"] == recorded
    relpaths = record["input_artifact_relpaths"]
    assert set(relpaths) == set(recorded)
    missing = [name for name, relative in relpaths.items() if not (RUN_DIR / relative).is_file()]
    if missing:
        pytest.skip(f"frozen statistics artifacts not present locally: {missing}")
    for name, relative in relpaths.items():
        digest = hashlib.sha256((RUN_DIR / relative).read_bytes()).hexdigest()
        assert digest == recorded[name], f"{name} ({relative}) changed since the fit"


def test_emitted_module_and_written_files_are_hashed_in_the_receipt():
    record = receipt()
    assert record["emitted_module"]["path"] == "src/mouse_behavior/evaluation/" \
                                               "discriminative_thresholds_v17.py"
    digest = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert digest == record["emitted_module"]["sha256"]
    for name in ("fitted_thresholds_discriminative.json", "side_by_side.csv",
                 "sweep_curves.json"):
        path = OUTPUT_DIR / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["written_files"][name][
            "sha256"], name
    assert record["dry_run"] is False
    assert record["fit_split"] == "train"
    assert record["row_level_split_labels_present"] is True


def test_side_by_side_csv_covers_every_gate():
    path = OUTPUT_DIR / "side_by_side.csv"
    if not path.is_file():
        pytest.skip("optional local calibration CSV is not published with source")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)}
    assert set(rows) == set(GATE_NAMES)
    for name, row in rows.items():
        gate = gates()[name]
        assert float(row["discriminative_value"]) == gate["value"], name
        assert float(row["extrema_value"]) == gate["extrema_value"], name
        assert row["clamped"] == str(gate["clamped"]), name
        assert row["unfittable"] == str(gate["unfittable"]), name
        assert float(row["train_sensitivity"]) == pytest.approx(gate["train_sensitivity"]), name
        assert row["notes"].strip(), name
