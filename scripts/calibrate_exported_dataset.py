#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split, tune and validate the exported annotation dataset.

The script deliberately keeps dataset preparation and calibration outside the
video-inference CLI.  It consumes ``annotation.json``/``metadata.json``/
``tracks.json`` samples, excludes behavior folders with fewer than 50 clips,
holds out complete recording sessions, tunes the compact exported-track
heuristics on the training split, and evaluates the frozen parameters once on
the validation split.

Example::

    python scripts/calibrate_exported_dataset.py \
        --dataset-root PATH_TO_EXPORTED_DATASET
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import REPO_ROOT
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    ExportedFeatures,
    HeuristicParameters,
    SampleRecord,
    calibration_rank,
    calibration_objective,
    classify_target_ids,
    config_seed_parameters,
    discover_samples,
    evaluate_target_classifications,
    filter_behaviors,
    group_rule_calibration_rank,
    group_rule_parameter_tie_break,
    load_exported_features,
    minimum_behavior_accuracy,
    predict_features,
    split_by_recording_session,
    strict_top1_regressions,
    training_target_reached,
    write_split_manifest,
)


LOGGER = logging.getLogger(__name__)


# The search is intentionally small and interpretable.  Each value corresponds
# to a named distance/speed/duration gate in the repository's extended
# ethogram, while the exported adapter evaluates all candidates from one
# feature extraction pass.
TUNING_GRID: dict[str, tuple[float, ...]] = {
    # Durations come first because one-frame gates otherwise make many speed
    # and distance thresholds observationally equivalent during refinement.
    "stationary_min_duration_s": (1.0,),
    "stationary_max_speed_cm_s": (2.5, 4.0, 6.0, 8.0, 12.0, 20.0),
    "walking_min_duration_s": (1.0,),
    "walking_max_stop_gap_s": (0.0, 0.15, 0.25, 0.5, 1.0),
    "walking_min_speed_cm_s": (0.0, 0.5, 1.0, 2.0, 4.0),
    "walking_max_speed_cm_s": (18.0, 24.0, 40.0, 80.0, 160.0),
    "together_max_distance_cm": (6.0, 8.0, 12.0, 20.0, 40.0),
    "together_max_individual_speed_cm_s": (8.0, 10.0, 12.0),
    "pair_min_duration_s": (0.0, 0.05, 0.1, 0.2, 0.4),
    "contact_distance_cm": (2.0, 3.0, 5.0, 8.0, 12.0, 20.0),
    "attack_min_speed_cm_s": (0.0, 1.0, 2.0, 4.0, 8.0),
    "attack_vs_approach_multiplier": (1.0, 1.25, 1.5, 2.0),
    "attack_endpoint_distance_cm": (8.0, 12.0, 16.0),
    "attack_reacquisition_max_gap_s": (1.0, 3.0, 5.0, 8.0),
    "together_max_combined_speed_cm_s": (28.0, 50.0, 100.0, 250.0),
    "approach_terminal_distance_cm": (8.0, 10.0, 12.0, 14.0, 17.0, 20.0),
    "approach_window_s": (0.1, 0.3, 0.5, 1.0),
    "approach_min_distance_drop_cm": (-2.0, 0.0, 0.5, 1.5, 3.0),
    "approach_min_closing_speed_cm_s": (-2.0, 0.0, 0.5, 2.0, 4.0),
    "group_min_duration_s": (0.1, 0.3, 0.6, 1.0),
    # Huddling is a low-motion resting state. Tune the short detector-gap
    # tolerance separately from the isolation confirmation time.
    "huddle_distance_cm": (8.0, 9.0, 11.0, 13.0),
    "huddle_max_mean_speed_cm_s": (2.0, 3.0, 4.0, 5.0),
    "huddle_min_duration_s": (1.0, 1.5, 2.0),
    "huddle_max_gap_s": (0.0, 0.15, 0.5, 1.0, 3.0, 5.0),
    # Isolation uses distance to the nearest visible companion. Preserve the
    # 10 s behavior minimum while searching the empirically supported cutoff.
    "isolation_distance_cm": (25.0, 30.0, 35.0, 40.0, 50.0),
    "isolation_min_duration_s": (10.0,),
    # Social clustering is the transition into a stable formation. Keep the
    # five-second semantic minimum while tuning the formation window/drop.
    "clustering_window_s": (1.5, 2.0, 3.0, 4.0),
    "clustering_max_gap_s": (0.0, 0.15, 0.25, 0.5),
    "clustering_max_distance_cm": (25.0, 30.0, 35.0),
    "clustering_initial_max_distance_cm": (16.0, 20.0, 24.0, 30.0),
    "clustering_min_nearest_neighbor_drop_cm": (0.5, 1.0, 1.5, 2.0),
    "clustering_min_duration_s": (5.0, 6.0, 8.0),
    "clustering_min_mean_speed_cm_s": (5.0, 6.0, 8.0),
    "clustering_min_moving_member_fraction": (0.5, 0.67, 1.0),
}

TARGETED_GROUP_RULE_GRID: dict[str, tuple[float, ...]] = {
    "huddle_max_mean_speed_cm_s": (3.0, 4.0, 5.0),
    # Keep the one-body-width huddle geometry and require a participant to be
    # observed for at least some of the episode; broad cores can absorb a
    # neighboring group and turn a formation into an apparent Huddle.
    "huddle_min_member_support_fraction": (0.1, 0.25, 0.5),
    "clustering_initial_max_distance_cm": (8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 30.0),
    "clustering_min_mean_speed_cm_s": (5.0, 6.0, 8.0),
    "clustering_min_moving_member_fraction": (0.5, 0.67, 1.0),
}

# This search starts from a frozen historical parameter set.  Its only
# adjustable fields belong to the two collective behaviors requested by the
# user; Isolation and every individual/pair behavior are protected outcomes.
TWO_COLLECTIVE_RULE_GRID: dict[str, tuple[float, ...]] = {
    # A sustained group can already be close at the clip boundary. Search the
    # movement alternative on training only; zero keeps the historical rule.
    "clustering_motion_max_distance_cm": (0.0, 12.0, 15.0, 20.0),
    "clustering_motion_min_displacement_cm": (4.0, 6.0, 8.0),
    "clustering_formation_score_bonus": (1.0, 1.25, 1.5, 2.0, 2.5),
    "clustering_min_member_support_fraction": (0.0, 0.1, 0.25, 0.4, 0.6),
    "clustering_max_distance_cm": (12.0, 16.0, 20.0, 24.0, 30.0),
    "clustering_max_gap_s": (0.0, 0.1, 0.25, 0.5),
    "clustering_min_nearest_neighbor_drop_cm": (0.5, 1.0, 2.0, 3.0),
    "huddle_min_member_support_fraction": (0.0, 0.1, 0.25, 0.4, 0.6),
    "huddle_distance_cm": (5.0, 6.0, 7.0, 8.0),
    "huddle_width_multiplier": (2.0, 2.5, 3.0),
    "clustering_initial_max_distance_cm": (16.0, 20.0, 24.0, 28.0),
    "clustering_min_mean_speed_cm_s": (2.0, 3.0, 4.0),
    "clustering_min_moving_member_fraction": (1.0 / 3.0, 0.5, 2.0 / 3.0),
}

GROUP_RULE_BEHAVIORS = {"huddle", "isolation", "social_clustering"}


LoadedFeature = tuple[SampleRecord, ExportedFeatures]


def _parse_validation_sessions(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_feature_cache(
    records: Sequence[SampleRecord],
    *,
    max_tracks: int,
) -> list[LoadedFeature]:
    """Load label-free track features once for repeated training evaluations."""

    loaded: list[LoadedFeature] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        features = load_exported_features(
            record.tracks_path,
            fps=record.fps,
            max_tracks=max_tracks,
        )
        loaded.append((record, features))
        if index == 1 or index % 100 == 0 or index == total:
            LOGGER.info("loaded features %d/%d", index, total)
    return loaded


def _evaluate_loaded_candidates(
    loaded: Sequence[LoadedFeature],
    candidates: Sequence[HeuristicParameters],
    *,
    canonical_behaviors: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate parameter candidates without reparsing tracks for every round."""

    selected = (
        [item for item in loaded if item[0].canonical_behavior in canonical_behaviors]
        if canonical_behaviors is not None
        else loaded
    )
    rows: list[list[tuple[SampleRecord, Mapping[str, Any]]]] = [[] for _ in candidates]
    total = len(selected)
    for index, (record, features) in enumerate(selected, start=1):
        for candidate_index, candidate in enumerate(candidates):
            prediction = predict_features(features, candidate)
            rows[candidate_index].append((record, classify_target_ids(record, prediction)))
        if index == 1 or index % 100 == 0 or index == total:
            LOGGER.info("evaluated candidates %d/%d", index, total)
    return [evaluate_target_classifications(candidate_rows) for candidate_rows in rows]


def _prediction_rows_loaded(
    loaded: Sequence[LoadedFeature],
    parameters: HeuristicParameters,
    *,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (record, features) in enumerate(loaded, start=1):
        # This is the blind-inference boundary: neither the behavior label nor
        # annotation mouse_ids are passed to predict_features.
        prediction = predict_features(features, parameters)
        target_result = classify_target_ids(record, prediction)
        rows.append(
            {
                "split": split,
                "sample_id": record.sample_id,
                "behavior": record.behavior,
                "canonical_behavior": record.canonical_behavior,
                "recording_session": record.recording_session,
                "target_ids": ",".join(str(value) for value in record.mouse_ids),
                "target_layer": target_result["target_layer"],
                "target_ids_available": bool(target_result["target_ids_available"]),
                "target_id_hit": bool(target_result["target_hit"]),
                "predicted_behavior": target_result["predicted_behavior"] or "",
                "predicted_target_ids": ",".join(
                    str(value) for value in target_result["predicted_target_ids"]
                ),
                "behavior_top1_correct": bool(target_result["behavior_correct"]),
                "correct": bool(target_result["target_hit"]),
                "strict_top1_correct": bool(target_result["correct"]),
                "exact_ids_correct": bool(target_result["exact_ids_correct"]),
                "candidate_scores": json.dumps(
                    target_result["candidate_scores"], ensure_ascii=False, sort_keys=True
                ),
                "group_scores": json.dumps(
                    prediction["group_scores"], ensure_ascii=False, sort_keys=True
                ),
                "all_detected_behaviors": ",".join(prediction["detected"]),
                "track_selection_truncated": bool(prediction["track_selection_truncated"]),
                "selected_track_count": int(prediction["selected_track_count"]),
            }
        )
        if index == 1 or index % 100 == 0 or index == len(loaded):
            LOGGER.info("predictions[%s] %d/%d", split, index, len(loaded))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_initial_parameters(
    path: Path | None, config_parameters: HeuristicParameters
) -> HeuristicParameters:
    if path is None:
        return config_parameters
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"initial parameter file must contain a JSON object: {path}")
    known = set(config_parameters.as_dict())
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown initial parameters in {path}: {unknown}")
    return replace(
        config_parameters,
        **{name: float(value) for name, value in payload.items()},
    )


def _bootstrap_candidates(initial: HeuristicParameters) -> list[HeuristicParameters]:
    """Build an accuracy ladder before constrained coordinate refinement."""

    moderate = replace(
        initial,
        stationary_max_speed_cm_s=8.0,
        walking_min_speed_cm_s=0.5,
        walking_max_speed_cm_s=80.0,
        together_max_distance_cm=20.0,
        together_max_combined_speed_cm_s=100.0,
        pair_max_distance_cm=40.0,
        approach_min_distance_drop_cm=0.0,
        approach_min_closing_speed_cm_s=0.0,
        contact_distance_cm=8.0,
        attack_min_speed_cm_s=1.0,
        huddle_distance_cm=11.0,
        huddle_max_mean_speed_cm_s=5.0,
        huddle_min_duration_s=1.0,
        huddle_max_gap_s=3.0,
        attack_endpoint_distance_cm=12.0,
        attack_reacquisition_max_gap_s=5.0,
        walking_max_stop_gap_s=0.5,
        isolation_distance_cm=35.0,
        isolation_min_duration_s=10.0,
        clustering_max_distance_cm=30.0,
        clustering_min_nearest_neighbor_drop_cm=1.0,
        stationary_min_duration_s=1.0,
        walking_min_duration_s=1.0,
        pair_min_duration_s=0.05,
        group_min_duration_s=0.1,
        approach_window_s=0.1,
        clustering_window_s=2.0,
        clustering_min_duration_s=5.0,
        clustering_min_mean_speed_cm_s=5.0,
    )
    sensitive = replace(
        moderate,
        stationary_max_speed_cm_s=8.0,
        walking_min_speed_cm_s=2.0,
        walking_max_speed_cm_s=40.0,
        together_max_distance_cm=40.0,
        together_max_combined_speed_cm_s=100.0,
        pair_max_distance_cm=40.0,
        approach_min_distance_drop_cm=0.0,
        approach_min_closing_speed_cm_s=0.0,
        contact_distance_cm=8.0,
        attack_min_speed_cm_s=4.0,
        huddle_distance_cm=13.0,
        huddle_max_mean_speed_cm_s=5.0,
        huddle_min_duration_s=1.0,
        huddle_max_gap_s=5.0,
        attack_endpoint_distance_cm=16.0,
        attack_reacquisition_max_gap_s=8.0,
        walking_max_stop_gap_s=0.5,
        isolation_distance_cm=30.0,
        isolation_min_duration_s=10.0,
        clustering_max_distance_cm=40.0,
        clustering_min_nearest_neighbor_drop_cm=0.5,
        stationary_min_duration_s=1.0,
        walking_min_duration_s=1.0,
        pair_min_duration_s=0.0,
        group_min_duration_s=0.0,
        approach_window_s=0.1,
        clustering_window_s=3.0,
        clustering_min_duration_s=5.0,
        clustering_min_mean_speed_cm_s=5.0,
    )
    candidates: list[HeuristicParameters] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in (initial, moderate, sensitive):
        key = tuple(sorted(candidate.as_dict().items()))
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _coordinate_candidates(
    current: HeuristicParameters,
    parameter_name: str,
    values: Sequence[float],
) -> list[HeuristicParameters]:
    candidates = [current]
    for value in values:
        candidate = replace(current, **{parameter_name: float(value)})
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _non_group_regressions(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    current_rows = dict(metrics.get("per_behavior", {}))
    baseline_rows = dict(baseline.get("per_behavior", {}))
    regressions: list[str] = []
    for behavior, baseline_values in baseline_rows.items():
        if behavior in GROUP_RULE_BEHAVIORS or int(baseline_values.get("support", 0)) == 0:
            continue
        current_values = current_rows.get(behavior, {})
        if int(current_values.get("strict_top1_correct", 0)) < int(
            baseline_values.get("strict_top1_correct", 0)
        ):
            regressions.append(behavior)
    return sorted(regressions)


def _group_rule_rank(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[float, ...]:
    regressions = _non_group_regressions(metrics, baseline)
    regressions.extend(
        strict_top1_regressions(metrics, baseline)
    )
    regressions = sorted(set(regressions))
    if regressions:
        return (0.0, -float(len(regressions)))
    macro_f1, macro_strict = group_rule_calibration_rank(metrics)
    # A broad group rule can increase target-hit recall while turning
    # Huddling/Isolation clips into Social-clustering false positives. Rank
    # macro F1 first so false positives are penalized before maximizing hits.
    return (
        1.0,
        macro_f1,
        macro_strict,
        float(metrics.get("macro_f1", 0.0)),
        float(metrics.get("strict_target_id_accuracy", 0.0)),
    )


def _together_speed_rank(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[float, ...]:
    """Tune Together's low-motion ceiling without trading away another class."""

    regressions = strict_top1_regressions(
        metrics, baseline, excluded_behaviors={"together"}
    )
    together = dict(metrics.get("per_behavior", {}).get("together", {}))
    return (
        0.0 if regressions else 1.0,
        -float(len(regressions)),
        float(together.get("strict_top1_correct", 0)),
        float(together.get("f1", 0.0)),
    )


def _tune_group_rules(
    train_cache: Sequence[LoadedFeature],
    initial: HeuristicParameters,
    *,
    max_passes: int,
    target_accuracy: float,
) -> tuple[HeuristicParameters, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Tune low-motion/collective gates on train clips with class-wise guards."""

    seed = replace(
        initial,
        stationary_min_duration_s=max(initial.stationary_min_duration_s, 1.0),
        walking_min_duration_s=max(initial.walking_min_duration_s, 1.0),
        huddle_min_duration_s=max(initial.huddle_min_duration_s, 1.0),
        isolation_min_duration_s=max(initial.isolation_min_duration_s, 10.0),
        clustering_min_duration_s=max(initial.clustering_min_duration_s, 5.0),
    )
    baseline = _evaluate_loaded_candidates(train_cache, [seed])[0]
    group_train_cache = [
        item for item in train_cache if item[0].canonical_behavior in GROUP_RULE_BEHAVIORS
    ]
    baseline_focus = _evaluate_loaded_candidates(
        group_train_cache, [seed], canonical_behaviors=GROUP_RULE_BEHAVIORS
    )[0]
    current = seed
    current_focus_metrics = baseline_focus
    history: list[dict[str, Any]] = []
    for pass_index in range(1, max_passes + 1):
        changed = False
        for parameter_name, values in TARGETED_GROUP_RULE_GRID.items():
            candidates = _coordinate_candidates(current, parameter_name, values)
            metrics_list = _evaluate_loaded_candidates(
                group_train_cache,
                candidates,
                canonical_behaviors=GROUP_RULE_BEHAVIORS,
            )
            ranks = [_group_rule_rank(metrics, baseline_focus) for metrics in metrics_list]
            candidate_keys = [
                (
                    ranks[index],
                    group_rule_parameter_tie_break(
                        parameter_name,
                        float(getattr(candidates[index], parameter_name)),
                        index,
                    ),
                )
                for index in range(len(candidates))
            ]
            candidate_index = max(
                range(len(candidates)), key=lambda index: candidate_keys[index]
            )
            for candidate, metrics in zip(candidates, metrics_list):
                row = _history_row(
                    stage=f"group_rules_pass_{pass_index}",
                    parameter=parameter_name,
                    value=getattr(candidate, parameter_name),
                    metrics=metrics,
                    target_accuracy=target_accuracy,
                )
                row["non_group_regressions"] = "unchanged by focused candidate path"
                row["group_rule_rank"] = json.dumps(
                    _group_rule_rank(metrics, baseline_focus)
                )
                history.append(row)
            if candidate_keys[candidate_index] > candidate_keys[0]:
                current = candidates[candidate_index]
                current_focus_metrics = metrics_list[candidate_index]
                changed = True
                LOGGER.info(
                    "group rule tuned %s=%s focused strict=%.4f regressions=%s",
                    parameter_name,
                    getattr(current, parameter_name),
                    _group_rule_rank(current_focus_metrics, baseline_focus)[1],
                    "none (other layers are not scored in this search)",
                )
        if not changed:
            LOGGER.info("group-rule tuning converged after pass %d", pass_index)
            break
    current_metrics = _evaluate_loaded_candidates(train_cache, [current])[0]
    regressions = _non_group_regressions(current_metrics, baseline)
    if regressions:
        LOGGER.error("focused rule search changed non-group scores: %s", regressions)
        return seed, baseline, history, baseline

    # Together remains a pair-layer event, but the user-defined low-motion
    # gate belongs in this focused run. Select it on the full training split
    # and reject any setting that reduces another behavior's strict Top-1.
    together_baseline = current_metrics
    together_candidates = _coordinate_candidates(
        current,
        "together_max_individual_speed_cm_s",
        (8.0, 10.0, 12.0),
    )
    together_metrics = _evaluate_loaded_candidates(train_cache, together_candidates)
    together_ranks = [
        _together_speed_rank(candidate_metrics, together_baseline)
        for candidate_metrics in together_metrics
    ]
    for candidate, metrics in zip(together_candidates, together_metrics):
        row = _history_row(
            stage="together_low_motion_guarded",
            parameter="together_max_individual_speed_cm_s",
            value=candidate.together_max_individual_speed_cm_s,
            metrics=metrics,
            target_accuracy=target_accuracy,
        )
        row["protected_behavior_regressions"] = json.dumps(
            strict_top1_regressions(
                metrics, together_baseline, excluded_behaviors={"together"}
            )
        )
        row["together_speed_rank"] = json.dumps(
            _together_speed_rank(metrics, together_baseline)
        )
        history.append(row)
    best_together_index = max(
        range(len(together_candidates)),
        key=lambda index: together_ranks[index],
    )
    if together_ranks[best_together_index][0] > 0.0:
        current = together_candidates[best_together_index]
        current_metrics = together_metrics[best_together_index]
        LOGGER.info(
            "together low-motion speed tuned to %.1f cm/s; other strict Top-1 classes unchanged",
            current.together_max_individual_speed_cm_s,
        )
    return current, current_metrics, history, baseline


def _two_collective_rank(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[float, ...]:
    """Prefer strict ID-set hits while protecting all other labeled classes."""

    focus = ("huddle", "social_clustering")
    rows = metrics["per_behavior"]
    base_rows = baseline["per_behavior"]
    changed_protected = [
        name for name, row in base_rows.items()
        if name not in focus
        and int(row["support"]) > 0
        and int(row["strict_top1_correct"])
        != int(rows[name]["strict_top1_correct"])
    ]
    losses = sum(
        int(rows[name]["strict_top1_correct"] < base_rows[name]["strict_top1_correct"])
        for name in focus
    )
    if changed_protected or losses:
        return (0.0, -float(len(changed_protected) + losses))
    hits = sum(int(rows[name]["strict_top1_correct"]) for name in focus)
    minimum = min(float(rows[name]["strict_top1_accuracy"]) for name in focus)
    f1 = sum(float(rows[name]["f1"]) for name in focus) / len(focus)
    return (1.0, float(hits), minimum, f1)


def _tune_two_collective_rules(
    train_cache: Sequence[LoadedFeature],
    initial: HeuristicParameters,
    *,
    max_passes: int,
    target_accuracy: float,
    parameter_names: Sequence[str] | None = None,
) -> tuple[HeuristicParameters, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Tune only Huddling/Clustering on training clips, guarding Isolation."""

    baseline = _evaluate_loaded_candidates(train_cache, [initial])[0]
    focused_cache = [
        item for item in train_cache
        if item[0].canonical_behavior in {"huddle", "social_clustering", "isolation"}
    ]
    focused_baseline = _evaluate_loaded_candidates(focused_cache, [initial])[0]
    protected_cache = [
        item for item in train_cache
        if item[0].canonical_behavior not in {"huddle", "social_clustering"}
    ]

    def protected_results(parameters: HeuristicParameters) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                result["predicted_behavior"],
                tuple(result["predicted_target_ids"]),
                result["target_hit"],
                result["correct"],
            )
            for record, features in protected_cache
            for result in [classify_target_ids(record, predict_features(features, parameters))]
        )

    baseline_protected = protected_results(initial)
    current = initial
    history: list[dict[str, Any]] = []
    for pass_index in range(1, max_passes + 1):
        changed = False
        for name in (parameter_names or tuple(TWO_COLLECTIVE_RULE_GRID)):
            values = TWO_COLLECTIVE_RULE_GRID[name]
            candidates = _coordinate_candidates(current, name, values)
            metrics_list = _evaluate_loaded_candidates(focused_cache, candidates)
            ranks = [_two_collective_rank(value, focused_baseline) for value in metrics_list]
            best = 0
            ranked_indices = sorted(
                range(1, len(candidates)),
                key=lambda index: (ranks[index], -index),
                reverse=True,
            )
            for index in ranked_indices:
                if ranks[index] <= ranks[0]:
                    break
                if protected_results(candidates[index]) == baseline_protected:
                    best = index
                    break
            for candidate, metrics in zip(candidates, metrics_list):
                row = _history_row(
                    stage=f"two_collective_pass_{pass_index}",
                    parameter=name,
                    value=getattr(candidate, name),
                    metrics=metrics,
                    target_accuracy=target_accuracy,
                )
                row["protected_behavior_regressions"] = json.dumps(
                    strict_top1_regressions(
                        metrics, focused_baseline,
                        excluded_behaviors={"huddle", "social_clustering"},
                    )
                )
                row["two_collective_rank"] = json.dumps(
                    _two_collective_rank(metrics, focused_baseline)
                )
                history.append(row)
            if best != 0 and ranks[best] > ranks[0]:
                current = candidates[best]
                changed = True
                LOGGER.info("two-collective tuned %s=%s rank=%s", name, getattr(current, name), ranks[best])
        if not changed:
            break
    current_metrics = _evaluate_loaded_candidates(train_cache, [current])[0]
    protected = strict_top1_regressions(
        current_metrics, baseline,
        excluded_behaviors={"huddle", "social_clustering"},
    )
    changed_counts = [
        name for name, row in baseline["per_behavior"].items()
        if name not in {"huddle", "social_clustering"}
        and int(row["strict_top1_correct"])
        != int(current_metrics["per_behavior"][name]["strict_top1_correct"])
    ]
    if protected or changed_counts or protected_results(current) != baseline_protected:
        raise RuntimeError(
            "two-collective tuning changed protected training predictions: "
            f"{sorted(set(protected + changed_counts))}"
        )
    return current, current_metrics, history, baseline


def _history_row(
    *,
    stage: str,
    parameter: str,
    value: str | float,
    metrics: Mapping[str, Any],
    target_accuracy: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "parameter": parameter,
        "value": value,
        "target_reached": training_target_reached(metrics, target_accuracy),
        "minimum_behavior_accuracy": minimum_behavior_accuracy(metrics),
        "target_id_accuracy": metrics["target_id_accuracy"],
        "macro_target_id_accuracy": metrics["macro_target_id_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "strict_target_id_accuracy": metrics["strict_target_id_accuracy"],
        "objective": calibration_objective(metrics),
        "protected_behavior_regressions": "",
        "together_speed_rank": "",
        "per_behavior_accuracy": json.dumps(
            {
                name: details["accuracy"]
                for name, details in metrics["per_behavior"].items()
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _write_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    seed_parameters: HeuristicParameters,
    tuned_parameters: HeuristicParameters,
    training_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any] | None,
    best_objective: float,
    max_tracks: int,
    target_train_accuracy: float,
    target_validation_accuracy: float,
    validation_was_previously_observed: bool,
) -> None:
    training_met = training_target_reached(training_metrics, target_train_accuracy)
    lines = [
        "# Exported behavior dataset calibration",
        "",
        "## Protocol",
        "",
        "- Samples with behavior count < 50 were excluded from the experiment manifest.",
        "- Split unit: complete recording session; no source session appears in both sets.",
        f"- Training samples: {manifest['split']['train_count']} ({manifest['split']['train_fraction']:.3f})",
        f"- Validation samples: {manifest['split']['validation_count']} ({manifest['split']['validation_fraction']:.3f})",
        f"- Training acceptance: aggregate and every-behavior target-ID accuracy >= {target_train_accuracy:.1%}.",
        f"- Training acceptance result: {'PASS' if training_met else 'FAIL'}.",
        "- Parameter search used training samples only; validation parameters were frozen before evaluation.",
        f"- At most {max_tracks} most persistent visual IDs were evaluated per clip; this run retained every annotated validation target ID.",
        "- Blind inference receives only tracks, clip FPS, and the IDs present in tracks.json; labels and annotated target IDs are joined only after prediction.",
        "- Scoring is target-conditioned only after trajectory-only predictions are frozen: individual labels rank the target ID's individual behaviors, pair labels rank that exact pair, and group labels rank one event candidate containing every annotated member ID.",
        "- Isolation competes with collective group events that include its target ID; membership in a detected Huddling or Social-clustering event suppresses the contradictory Isolation candidate for that target.",
        "- The exported IDs are clip-local visual IDs. They are not an mTrack/RFID animal identity map.",
        *(
            [
                "- This validation partition was inspected by an earlier calibration run, so it is diagnostic rather than a pristine final holdout. A new untouched test set is required for an unbiased final estimate."
            ]
            if validation_was_previously_observed
            else []
        ),
        "",
        "## Metrics",
        "",
        "`target_id_accuracy` asks whether a candidate for the labeled behavior contains the annotated IDs; "
        "group candidates may include additional IDs. `strict_target_id_accuracy` requires the labeled behavior "
        "to rank first within its layer and the predicted ID set to exactly equal the annotation. "
        "The primary target-ID metric is recall-like rather than conventional mutually-exclusive multiclass accuracy. "
        "After the hard accuracy constraint is met, macro F1 penalizes rules that fire on target IDs belonging to other labels.",
        "",
        "| Split | Samples | Target-ID accuracy | Macro target-ID accuracy | Minimum behavior accuracy | Macro F1 | Strict Top-1 | Exact ID set | Prediction coverage | ID availability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| train | {training_metrics['n_samples']} | {training_metrics['target_id_accuracy']:.3f} | "
        f"{training_metrics['macro_target_id_accuracy']:.3f} | {minimum_behavior_accuracy(training_metrics):.3f} | "
        f"{training_metrics['macro_f1']:.3f} | "
        f"{training_metrics['strict_target_id_accuracy']:.3f} | "
        f"{training_metrics['exact_target_id_set_accuracy']:.3f} | "
        f"{training_metrics['prediction_coverage']:.3f} | {training_metrics['target_id_availability_rate']:.3f} |",
    ]
    if validation_metrics is not None:
        validation_met = training_target_reached(
            validation_metrics, target_validation_accuracy
        )
        lines.append(
            f"| validation | {validation_metrics['n_samples']} | {validation_metrics['target_id_accuracy']:.3f} | "
            f"{validation_metrics['macro_target_id_accuracy']:.3f} | {minimum_behavior_accuracy(validation_metrics):.3f} | "
            f"{validation_metrics['macro_f1']:.3f} | {validation_metrics['strict_target_id_accuracy']:.3f} | "
            f"{validation_metrics['exact_target_id_set_accuracy']:.3f} | "
            f"{validation_metrics['prediction_coverage']:.3f} | {validation_metrics['target_id_availability_rate']:.3f} |"
        )
        lines.extend(
            [
                "",
                f"Validation expectation (aggregate and every behavior >= {target_validation_accuracy:.1%}): "
                f"**{'PASS' if validation_met else 'FAIL'}**.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Validation was not run because the frozen training parameters did not meet the training acceptance gate.",
            ]
        )

    for split_name, metrics in (
        ("Training", training_metrics),
        ("Validation", validation_metrics),
    ):
        if metrics is None:
            continue
        lines.extend(
            [
                "",
                f"## {split_name} per behavior",
                "",
                "| Behavior | Support | Correct | Accuracy | Precision | F1 | Strict Top-1 | Strict accuracy | Exact ID set |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for behavior, values in metrics["per_behavior"].items():
            lines.append(
                f"| {behavior} | {values['support']} | {values['correct']} | {values['accuracy']:.3f} | "
                f"{values['precision']:.3f} | {values['f1']:.3f} | "
                f"{values['strict_top1_correct']} | {values['strict_top1_accuracy']:.3f} | "
                f"{values['exact_id_set_accuracy']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            f"- Training objective: `{best_objective:.6f}`",
            "- Seed parameters:",
            "```json",
            json.dumps(seed_parameters.as_dict(), ensure_ascii=False, indent=2),
            "```",
            "- Tuned parameters:",
            "```json",
            json.dumps(tuned_parameters.as_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
            "Generated files include the split manifest, tuning history, frozen parameters, metrics, and per-sample predictions. Validation is run only after parameters are frozen from training data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "exported_dataset_calibration",
    )
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument(
        "--initial-parameters",
        type=Path,
        default=None,
        help="Optional tuned-parameter JSON used as the starting candidate.",
    )
    parser.add_argument("--target-train-accuracy", type=float, default=0.95)
    parser.add_argument("--target-validation-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--validation-was-previously-observed",
        action="store_true",
        help="Mark validation results as diagnostic when this holdout was inspected before.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=2,
        help="Maximum constrained coordinate-refinement passes over the training set.",
    )
    rule_mode = parser.add_mutually_exclusive_group()
    rule_mode.add_argument(
        "--group-rules-only",
        action="store_true",
        help="Tune Together and collective group gates on training clips; always evaluate the frozen result.",
    )
    rule_mode.add_argument(
        "--two-group-only",
        action="store_true",
        help="Tune only Huddling and Social clustering; protect all other behavior counts.",
    )
    parser.add_argument(
        "--two-group-parameters",
        nargs="+",
        choices=tuple(TWO_COLLECTIVE_RULE_GRID),
        help="Optional subset of the two-group parameter grid for a focused pass.",
    )
    parser.add_argument(
        "--validation-sessions",
        default=None,
        help="Comma-separated source session filenames; omit to use the audited 4-session holdout.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.max_tracks < 2:
        raise ValueError("--max-tracks must be at least 2")
    if args.max_passes < 0:
        raise ValueError("--max-passes cannot be negative")
    if args.two_group_parameters and not args.two_group_only:
        raise ValueError("--two-group-parameters requires --two-group-only")
    for name, value in (
        ("--target-train-accuracy", args.target_train_accuracy),
        ("--target-validation-accuracy", args.target_validation_accuracy),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = discover_samples(args.dataset_root)
    retained, counts, excluded_counts = filter_behaviors(all_records, minimum_count=50)
    train_records, validation_records = split_by_recording_session(
        retained,
        validation_sessions=_parse_validation_sessions(args.validation_sessions),
    )
    split_dir = args.output_dir / "dataset_split"
    manifest = write_split_manifest(
        split_dir,
        all_records=retained,
        train_records=train_records,
        validation_records=validation_records,
        counts={behavior: counts[behavior] for behavior in sorted(counts) if behavior not in excluded_counts},
        excluded_counts=excluded_counts,
    )
    LOGGER.info(
        "discovered=%d retained=%d excluded=%d train=%d validation=%d",
        len(all_records),
        len(retained),
        sum(excluded_counts.values()),
        len(train_records),
        len(validation_records),
    )

    config = load_config(args.config)
    config_seed = config_seed_parameters(config)
    seed = _load_initial_parameters(args.initial_parameters, config_seed)
    train_cache = _load_feature_cache(train_records, max_tracks=args.max_tracks)
    history: list[dict[str, Any]] = []
    if args.two_group_only:
        tuned, training_metrics, history, baseline_metrics = _tune_two_collective_rules(
            train_cache,
            seed,
            max_passes=args.max_passes,
            target_accuracy=args.target_train_accuracy,
            parameter_names=args.two_group_parameters,
        )
        LOGGER.info(
            "two-group baseline strict=%.4f tuned strict=%.4f",
            baseline_metrics["strict_target_id_accuracy"],
            training_metrics["strict_target_id_accuracy"],
        )
    elif args.group_rules_only:
        tuned, training_metrics, history, baseline_metrics = _tune_group_rules(
            train_cache,
            seed,
            max_passes=args.max_passes,
            target_accuracy=args.target_train_accuracy,
        )
        LOGGER.info(
            "group-rule search baseline strict=%.4f tuned strict=%.4f non-group regressions=%s",
            baseline_metrics["strict_target_id_accuracy"],
            training_metrics["strict_target_id_accuracy"],
            _non_group_regressions(training_metrics, baseline_metrics),
        )
    else:
        bootstrap_candidates = _bootstrap_candidates(seed)
        bootstrap_metrics = _evaluate_loaded_candidates(train_cache, bootstrap_candidates)
        for index, (candidate, metrics) in enumerate(
            zip(bootstrap_candidates, bootstrap_metrics)
        ):
            history.append(
                _history_row(
                    stage=f"bootstrap_{index}",
                    parameter="all",
                    value=json.dumps(candidate.as_dict(), sort_keys=True),
                    metrics=metrics,
                    target_accuracy=args.target_train_accuracy,
                )
            )
            LOGGER.info(
                "bootstrap=%d overall=%.4f macro=%.4f minimum=%.4f f1=%.4f target=%s",
                index,
                metrics["target_id_accuracy"],
                metrics["macro_target_id_accuracy"],
                minimum_behavior_accuracy(metrics),
                metrics["macro_f1"],
                training_target_reached(metrics, args.target_train_accuracy),
            )
        best_index = max(
            range(len(bootstrap_candidates)),
            key=lambda index: (
                calibration_rank(bootstrap_metrics[index], args.target_train_accuracy),
                -index,
            ),
        )
        current = bootstrap_candidates[best_index]
        current_metrics = bootstrap_metrics[best_index]
        current_rank = calibration_rank(current_metrics, args.target_train_accuracy)

        for pass_index in range(1, args.max_passes + 1):
            changed = False
            for parameter_name, values in TUNING_GRID.items():
                candidates = _coordinate_candidates(current, parameter_name, values)
                metrics_list = _evaluate_loaded_candidates(train_cache, candidates)
                candidate_index = max(
                    range(len(candidates)),
                    key=lambda index: (
                        calibration_rank(metrics_list[index], args.target_train_accuracy),
                        -index,
                    ),
                )
                for candidate, metrics in zip(candidates, metrics_list):
                    history.append(
                        _history_row(
                            stage=f"coordinate_pass_{pass_index}",
                            parameter=parameter_name,
                            value=getattr(candidate, parameter_name),
                            metrics=metrics,
                            target_accuracy=args.target_train_accuracy,
                        )
                    )
                candidate_rank = calibration_rank(
                    metrics_list[candidate_index], args.target_train_accuracy
                )
                if candidate_rank > current_rank:
                    current = candidates[candidate_index]
                    current_metrics = metrics_list[candidate_index]
                    current_rank = candidate_rank
                    changed = True
                    LOGGER.info(
                        "pass=%d tuned %s=%s overall=%.4f minimum=%.4f f1=%.4f",
                        pass_index,
                        parameter_name,
                        getattr(current, parameter_name),
                        current_metrics["target_id_accuracy"],
                        minimum_behavior_accuracy(current_metrics),
                        current_metrics["macro_f1"],
                    )
                _write_csv(args.output_dir / "tuning_history.csv", history)
            if not changed:
                LOGGER.info("coordinate refinement converged after pass %d", pass_index)
                break

        tuned = current
        training_metrics = current_metrics
    best_objective = calibration_objective(training_metrics)
    _write_csv(args.output_dir / "tuning_history.csv", history)
    (args.output_dir / "tuned_parameters.json").write_text(
        json.dumps(tuned.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "train_predictions.csv",
        _prediction_rows_loaded(train_cache, tuned, split="train"),
    )
    training_met = training_target_reached(
        training_metrics, args.target_train_accuracy
    )
    if not training_met and not (args.group_rules_only or args.two_group_only):
        _write_report(
            args.output_dir / "calibration_report.md",
            manifest=manifest,
            seed_parameters=seed,
            tuned_parameters=tuned,
            training_metrics=training_metrics,
            validation_metrics=None,
            best_objective=best_objective,
            max_tracks=args.max_tracks,
            target_train_accuracy=args.target_train_accuracy,
            target_validation_accuracy=args.target_validation_accuracy,
            validation_was_previously_observed=args.validation_was_previously_observed,
        )
        LOGGER.error(
            "training gate failed: overall=%.4f minimum_behavior=%.4f; validation was not run",
            training_metrics["target_id_accuracy"],
            minimum_behavior_accuracy(training_metrics),
        )
        return 2

    del train_cache
    gc.collect()
    LOGGER.info(
        "parameters frozen before validation; training accuracy gate %s",
        "passed" if training_met else "not reached, validating diagnostic outcome",
    )
    validation_cache = _load_feature_cache(
        validation_records, max_tracks=args.max_tracks
    )
    validation_metrics = _evaluate_loaded_candidates(validation_cache, [tuned])[0]
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "validation_predictions.csv",
        _prediction_rows_loaded(validation_cache, tuned, split="validation"),
    )
    _write_report(
        args.output_dir / "calibration_report.md",
        manifest=manifest,
        seed_parameters=seed,
        tuned_parameters=tuned,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        best_objective=best_objective,
        max_tracks=args.max_tracks,
        target_train_accuracy=args.target_train_accuracy,
        target_validation_accuracy=args.target_validation_accuracy,
        validation_was_previously_observed=args.validation_was_previously_observed,
    )
    LOGGER.info("training metrics: %s", json.dumps(training_metrics, ensure_ascii=False))
    LOGGER.info("validation metrics: %s", json.dumps(validation_metrics, ensure_ascii=False))
    if not training_target_reached(
        validation_metrics, args.target_validation_accuracy
    ):
        LOGGER.warning(
            "validation expectation missed: overall=%.4f minimum_behavior=%.4f",
            validation_metrics["target_id_accuracy"],
            minimum_behavior_accuracy(validation_metrics),
        )
    LOGGER.info("calibration output=%s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
