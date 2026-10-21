"""Train-only heuristic search with recording-session outer holdouts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .blind_top1 import (
    Candidate,
    LabeledClip,
    Top1Prediction,
    candidates_from_evidence,
    evaluate_top1,
    hierarchical_candidate_indices,
    select_top1,
)
from .exported_dataset import (
    CANONICAL_BEHAVIORS,
    ExportedFeatures,
    HeuristicParameters,
    predict_features,
)


LOGGER = logging.getLogger(__name__)
WEIGHT_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
# The old recall-oriented search permitted zero durations and arena-wide
# contact distances.  These candidates stay within interpretable rule ranges.
RULE_GRID: dict[str, tuple[float, float]] = {
    # The document-defined group distances/durations are fixed policy, not
    # trainable shortcuts. Speed-based group cutoffs are likewise excluded.
    "stationary_max_speed_cm_s": (2.0, 8.0),
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    # "walking_min_speed_cm_s": (2.0, 6.0),
    "moving_min_speed_cm_s": (0.2, 3.0),
    "pair_min_duration_s": (0.08, 0.3),
    "attack_min_speed_cm_s": (4.0, 12.0),
    "approach_min_distance_drop_cm": (0.5, 3.0),
}


@dataclass(frozen=True)
class LoadedClip:
    label: LabeledClip
    features: ExportedFeatures


@dataclass(frozen=True)
class FittedRules:
    parameters: HeuristicParameters
    weights: dict[str, float]
    metrics: dict[str, Any]
    search_history: tuple[dict[str, Any], ...]


def session_folds(records: Sequence[LabeledClip]) -> list[tuple[str, list[int], list[int]]]:
    """Never put excerpts from one original recording on both fold sides."""

    sessions = sorted({record.recording_session for record in records})
    if len(sessions) < 2:
        raise ValueError("at least two independent recording sessions are required")
    folds = []
    for session in sessions:
        train = [index for index, record in enumerate(records) if record.recording_session != session]
        held_out = [index for index, record in enumerate(records) if record.recording_session == session]
        folds.append((session, train, held_out))
    return folds


def evidence_rows(
    loaded: Sequence[LoadedClip], parameters: HeuristicParameters
) -> list[tuple[LabeledClip, tuple[Candidate, ...]]]:
    return [
        (item.label, candidates_from_evidence(predict_features(item.features, parameters)))
        for item in loaded
    ]


def predict_rows(
    rows: Sequence[tuple[LabeledClip, Sequence[Candidate]]], weights: Mapping[str, float]
) -> list[tuple[LabeledClip, Top1Prediction]]:
    return [(label, select_top1(candidates, weights)) for label, candidates in rows]


def candidate_oracle_metrics(
    rows: Sequence[tuple[LabeledClip, Sequence[Candidate]]],
    *, require_positive_evidence: bool = False,
) -> dict[str, Any]:
    """Upper bound for ranking at fixed rule thresholds, on training labels."""

    per_behavior: dict[str, dict[str, int]] = {}
    for label, candidates in rows:
        selectable = [candidates[index] for index in hierarchical_candidate_indices(candidates)
                      if not require_positive_evidence or candidates[index].evidence > 0]
        values = per_behavior.setdefault(
            label.behavior, {"support": 0, "target_candidate": 0, "behavior_candidate": 0,
                             "id_candidate": 0}
        )
        values["support"] += 1
        values["target_candidate"] += int(any(
            item.behavior == label.behavior
            and tuple(sorted(label.target_ids)) == tuple(sorted(item.target_ids))
            for item in selectable
        ))
        values["behavior_candidate"] += int(any(item.behavior == label.behavior for item in selectable))
        values["id_candidate"] += int(any(
            set(label.target_ids).issubset(set(item.target_ids)) for item in selectable
        ))
    total = len(rows)
    target_candidate = sum(value["target_candidate"] for value in per_behavior.values())
    return {
        "n_samples": total,
        "target_candidate_rate": target_candidate / total if total else 0.0,
        "primary_metric": "strict_exact_selectable_candidate_coverage",
        "requires_positive_evidence": require_positive_evidence,
        "enumerated_exact_candidate_rate": sum(any(
            c.behavior == label.behavior and tuple(sorted(c.target_ids)) == tuple(sorted(label.target_ids))
            for c in candidates) for label, candidates in rows) / total if total else 0.0,
        "positive_exact_candidate_rate": sum(any(
            c.evidence > 0 and c.behavior == label.behavior
            and tuple(sorted(c.target_ids)) == tuple(sorted(label.target_ids))
            for c in candidates) for label, candidates in rows) / total if total else 0.0,
        "per_behavior": {
            name: {**value, "target_candidate_rate": value["target_candidate"] / value["support"]}
            for name, value in sorted(per_behavior.items())
        },
    }


def _rank(metrics: Mapping[str, Any], weights: Mapping[str, float]) -> tuple[float, ...]:
    # Macro accuracy is primary so frequent behaviors cannot hide minority
    # failure.  At equal accuracy, prefer a less extreme ranking calibration.
    complexity = sum(abs(math.log2(float(value))) for value in weights.values())
    return (
        float(metrics["macro_joint_top1_accuracy"]),
        float(metrics["joint_top1_accuracy"]),
        -complexity,
    )


def fit_weights(
    rows: Sequence[tuple[LabeledClip, Sequence[Candidate]]],
    *,
    initial: Mapping[str, float] | None = None,
    passes: int = 2,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Calibrate cross-behavior evidence on training labels only."""

    weights = {behavior: float((initial or {}).get(behavior, 1.0)) for behavior in CANONICAL_BEHAVIORS}
    metrics = evaluate_top1(predict_rows(rows, weights))
    current_rank = _rank(metrics, weights)
    for _ in range(passes):
        changed = False
        # Fix one reference weight to eliminate equivalent global rescalings.
        for behavior in CANONICAL_BEHAVIORS:
            if behavior == "stationary":
                continue
            best_weight = weights[behavior]
            best_metrics = metrics
            best_rank = current_rank
            for weight in WEIGHT_GRID:
                if weight == weights[behavior]:
                    continue
                trial = dict(weights)
                trial[behavior] = weight
                trial_metrics = evaluate_top1(predict_rows(rows, trial))
                trial_rank = _rank(trial_metrics, trial)
                if trial_rank > best_rank:
                    best_weight, best_metrics, best_rank = weight, trial_metrics, trial_rank
            if best_weight != weights[behavior]:
                weights[behavior] = best_weight
                metrics, current_rank = best_metrics, best_rank
                changed = True
        if not changed:
            break
    return weights, metrics


def fit_rules(
    loaded: Sequence[LoadedClip],
    *,
    seed: HeuristicParameters,
    max_rule_trials: int = 24,
    weight_passes: int = 2,
) -> FittedRules:
    if not loaded:
        raise ValueError("cannot tune an empty training fold")
    if max_rule_trials < 0:
        raise ValueError("max_rule_trials must be nonnegative")
    current = seed
    rows = evidence_rows(loaded, current)
    weights, metrics = fit_weights(rows, passes=weight_passes)
    current_rank = _rank(metrics, weights)
    history: list[dict[str, Any]] = [
        {"parameter": "seed", "value": None, "joint_top1": metrics["joint_top1_accuracy"],
         "macro_joint_top1": metrics["macro_joint_top1_accuracy"], "accepted": True}
    ]
    trials = 0
    for name, options in RULE_GRID.items():
        for value in options:
            if trials >= max_rule_trials:
                break
            if getattr(current, name) == value:
                continue
            candidate = replace(current, **{name: value})
            trials += 1
            candidate_rows = evidence_rows(loaded, candidate)
            candidate_metrics = evaluate_top1(predict_rows(candidate_rows, weights))
            candidate_rank = _rank(candidate_metrics, weights)
            accepted = candidate_rank > current_rank
            history.append(
                {"parameter": name, "value": value,
                 "joint_top1": candidate_metrics["joint_top1_accuracy"],
                 "macro_joint_top1": candidate_metrics["macro_joint_top1_accuracy"],
                 "accepted": accepted}
            )
            if accepted:
                current, rows, metrics, current_rank = (
                    candidate, candidate_rows, candidate_metrics, candidate_rank
                )
            LOGGER.info(
                "rule trial %d/%d %s=%s accepted=%s joint=%.4f macro=%.4f",
                trials, max_rule_trials, name, value, accepted,
                candidate_metrics["joint_top1_accuracy"],
                candidate_metrics["macro_joint_top1_accuracy"],
            )
        if trials >= max_rule_trials:
            break
    weights, metrics = fit_weights(rows, initial=weights, passes=weight_passes)
    return FittedRules(current, weights, metrics, tuple(history))
