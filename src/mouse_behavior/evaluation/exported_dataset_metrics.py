"""Target-ID classification metrics and calibration ranking helpers."""

from __future__ import annotations

from mouse_behavior.evaluation.exported_dataset_models import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    PAIR_BEHAVIORS,
    SampleRecord,
)
from mouse_behavior.evaluation.exported_dataset_temporal import _pair_key

from collections import Counter
from typing import Any, Mapping, Sequence
import numpy as np

def classify_target_ids(
    record: SampleRecord,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the annotated ID set after label-free inference has completed."""

    target = record.canonical_behavior
    target_ids = tuple(sorted(int(value) for value in record.mouse_ids))
    identity_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("identity_scores", {})).items()
    }
    pair_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("pair_scores", {})).items()
    }
    group_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("group_scores", {})).items()
    }
    selected_ids = {
        int(value)
        for value in prediction.get(
            "track_ids", [int(key) for key in identity_scores if str(key).lstrip("-").isdigit()]
        )
    }
    candidate_target_ids: dict[str, tuple[int, ...]] = {}

    def record_candidate(behavior: str, score: float, ids: tuple[int, ...]) -> None:
        previous = float(candidate_scores.get(behavior, 0.0))
        previous_ids = candidate_target_ids.get(behavior, ())
        if score > previous or (
            score == previous and score > 0.0 and (not previous_ids or len(ids) < len(previous_ids))
        ):
            candidate_scores[behavior] = float(score)
            candidate_target_ids[behavior] = ids

    if target in INDIVIDUAL_BEHAVIORS:
        if len(target_ids) != 1:
            raise ValueError(
                f"{record.behavior} requires exactly one target ID: {record.sample_id}"
            )
        key = str(target_ids[0])
        available = key in identity_scores
        candidate_behaviors = INDIVIDUAL_BEHAVIORS
        candidate_scores = {
            behavior: float(identity_scores.get(key, {}).get(behavior, 0.0))
            for behavior in candidate_behaviors
        }
        candidate_target_ids.update({behavior: target_ids for behavior in candidate_behaviors})
    elif target in PAIR_BEHAVIORS:
        if len(target_ids) != 2:
            raise ValueError(
                f"{record.behavior} requires exactly two target IDs: {record.sample_id}"
            )
        key = _pair_key(*target_ids)
        available = all(track_id in selected_ids for track_id in target_ids)
        if not selected_ids:
            available = key in pair_scores
        candidate_behaviors = PAIR_BEHAVIORS
        candidate_scores = {
            behavior: float(pair_scores.get(key, {}).get(behavior, 0.0))
            for behavior in candidate_behaviors
        }
        candidate_target_ids.update({behavior: target_ids for behavior in candidate_behaviors})
    elif target == "isolation":
        if len(target_ids) != 1:
            raise ValueError(f"Isolation requires one target ID: {record.sample_id}")
        key = str(target_ids[0])
        available = key in identity_scores
        candidate_behaviors = GROUP_BEHAVIORS
        candidate_scores = {behavior: 0.0 for behavior in candidate_behaviors}
        candidate_scores["isolation"] = float(identity_scores.get(key, {}).get("isolation", 0.0))
        candidate_target_ids["isolation"] = target_ids
        has_collective_candidate = False
        for raw_ids, scores in group_scores.items():
            try:
                group_ids = tuple(sorted(int(value) for value in raw_ids.split(",") if value))
            except ValueError:
                continue
            if len(group_ids) < 3 or target_ids[0] not in group_ids:
                continue
            for behavior in GROUP_BEHAVIORS:
                if behavior not in {"huddle", "social_clustering"}:
                    continue
                score = float(scores.get(behavior, 0.0))
                if score > 0.0:
                    has_collective_candidate = True
                    record_candidate(behavior, score, group_ids)
        if has_collective_candidate:
            candidate_scores["isolation"] = 0.0
    elif target in GROUP_BEHAVIORS:
        if len(target_ids) < 3:
            raise ValueError(
                f"{record.behavior} requires at least three target IDs: {record.sample_id}"
            )
        available = all(str(track_id) in identity_scores for track_id in target_ids)
        candidate_behaviors = GROUP_BEHAVIORS
        candidate_scores = {behavior: 0.0 for behavior in candidate_behaviors}
        if "group_scores" in prediction:
            target_id_set = set(target_ids)
            for raw_ids, scores in group_scores.items():
                try:
                    group_ids = tuple(sorted(int(value) for value in raw_ids.split(",") if value))
                except ValueError:
                    continue
                if len(group_ids) < 3 or not target_id_set.issubset(group_ids):
                    continue
                for behavior in candidate_behaviors:
                    record_candidate(behavior, float(scores.get(behavior, 0.0)), group_ids)
        else:
            # Read older cached predictions without event-level group candidates.
            candidate_scores = {
                behavior: min(
                    (
                        float(identity_scores.get(str(track_id), {}).get(behavior, 0.0))
                        for track_id in target_ids
                    ),
                    default=0.0,
                )
                for behavior in candidate_behaviors
            }
            candidate_target_ids.update({behavior: target_ids for behavior in candidate_behaviors})
    else:
        raise ValueError(f"unsupported canonical behavior {target!r}: {record.sample_id}")

    ranked = sorted(
        candidate_behaviors, key=lambda behavior: (-candidate_scores[behavior], behavior)
    )
    predicted = ranked[0] if ranked and candidate_scores[ranked[0]] > 0.0 else None
    predicted_ids = candidate_target_ids.get(predicted, ()) if predicted is not None else ()
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction.
    # The compatible_* fields stay in the schema as strict mirrors so frozen
    # exports and downstream CSV consumers keep their columns; they no longer
    # carry any Together-vs-contact credit.
    compatible_labels: list[str] = []
    compatible_correct = bool(
        predicted == target
    ) and tuple(sorted(predicted_ids)) == target_ids
    return {
        "target_layer": (
            "individual"
            if target in INDIVIDUAL_BEHAVIORS
            else "social"
            if target in PAIR_BEHAVIORS
            else "group"
        ),
        "target_ids": list(target_ids),
        "target_ids_available": bool(available),
        "candidate_scores": candidate_scores,
        "predicted_behavior": predicted,
        "predicted_target_ids": list(predicted_ids),
        "exact_ids_correct": tuple(sorted(predicted_ids)) == target_ids,
        "target_hit": candidate_scores.get(target, 0.0) > 0.0,
        "behavior_correct": predicted == target,
        "compatible_labels": compatible_labels,
        "compatible_credit": "",
        "compatible_correct": compatible_correct,
        # A matching group label with additional visual IDs is not a joint
        # behavior-and-ID hit, even when it contains every annotated member.
        "correct": predicted == target and tuple(sorted(predicted_ids)) == target_ids,
    }


def evaluate_predictions(
    rows: Sequence[tuple[SampleRecord, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Calculate strict behavior accuracy for the IDs named by each annotation."""

    classified = [(record, classify_target_ids(record, prediction)) for record, prediction in rows]
    return evaluate_target_classifications(classified)


def evaluate_target_classifications(
    rows: Sequence[tuple[SampleRecord, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate already-classified target-ID results without retaining full pair tensors."""

    per_behavior: dict[str, dict[str, int]] = {
        behavior: {
            "support": 0,
            "correct": 0,
            "strict_top1_correct": 0,
            "compatible_top1_correct": 0,
            "exact_id_set_correct": 0,
            "predicted": 0,
            "false_positive": 0,
        }
        for behavior in CANONICAL_BEHAVIORS
    }
    correct = 0
    strict_correct = 0
    covered = 0
    available = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for record, result in rows:
        target = record.canonical_behavior
        per_behavior.setdefault(
            target,
            {
                "support": 0,
                "correct": 0,
                "strict_top1_correct": 0,
                "compatible_top1_correct": 0,
                "exact_id_set_correct": 0,
                "predicted": 0,
                "false_positive": 0,
            },
        )["support"] += 1
        detected = {
            str(behavior)
            for behavior, score in dict(result.get("candidate_scores", {})).items()
            if float(score) > 0.0
        }
        for behavior in detected:
            values = per_behavior.setdefault(
                behavior,
                {
                    "support": 0,
                    "correct": 0,
                    "strict_top1_correct": 0,
                    "compatible_top1_correct": 0,
                    "exact_id_set_correct": 0,
                    "predicted": 0,
                    "false_positive": 0,
                },
            )
            values["predicted"] += 1
            values["false_positive"] += int(behavior != target)
        if result["target_hit"]:
            correct += 1
            per_behavior[target]["correct"] += 1
        predicted = result["predicted_behavior"]
        covered += int(predicted is not None)
        available += int(result["target_ids_available"])
        confusion[(target, str(predicted) if predicted is not None else "<none>")] += 1
        if result["correct"]:
            strict_correct += 1
            per_behavior[target]["strict_top1_correct"] += 1
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction: compatible now mirrors strict.
        if result.get("compatible_correct", result["correct"]):
            per_behavior[target]["compatible_top1_correct"] += 1
        if bool(result.get("exact_ids_correct", False)):
            per_behavior[target]["exact_id_set_correct"] += 1
    total = len(rows)
    macro_accuracy = float(
        np.mean(
            [
                values["correct"] / values["support"]
                for values in per_behavior.values()
                if values["support"]
            ]
        )
        if any(values["support"] for values in per_behavior.values())
        else 0.0
    )
    macro_strict_accuracy = float(
        np.mean(
            [
                values["strict_top1_correct"] / values["support"]
                for values in per_behavior.values()
                if values["support"]
            ]
        )
        if any(values["support"] for values in per_behavior.values())
        else 0.0
    )
    per_behavior_metrics: dict[str, dict[str, Any]] = {}
    for key, value in sorted(per_behavior.items()):
        if not value["support"]:
            continue
        precision = value["correct"] / value["predicted"] if value["predicted"] else 0.0
        recall = value["correct"] / value["support"]
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_behavior_metrics[key] = {
            **value,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "strict_top1_accuracy": value["strict_top1_correct"] / value["support"],
            "compatible_top1_accuracy": value["compatible_top1_correct"] / value["support"],
            "exact_id_set_accuracy": value["exact_id_set_correct"] / value["support"],
        }
    macro_f1 = float(
        np.mean([values["f1"] for values in per_behavior_metrics.values()])
        if per_behavior_metrics
        else 0.0
    )
    return {
        "n_samples": total,
        "target_id_accuracy": correct / total if total else 0.0,
        "macro_target_id_accuracy": macro_accuracy,
        "strict_target_id_accuracy": strict_correct / total if total else 0.0,
        "compatible_top1_accuracy": (
            sum(value["compatible_top1_correct"] for value in per_behavior.values()) / total
            if total
            else 0.0
        ),
        "exact_target_id_set_accuracy": (
            sum(values["exact_id_set_correct"] for values in per_behavior.values()) / total
            if total
            else 0.0
        ),
        "macro_strict_target_id_accuracy": macro_strict_accuracy,
        "macro_f1": macro_f1,
        "prediction_coverage": covered / total if total else 0.0,
        "target_id_availability_rate": available / total if total else 0.0,
        "per_behavior": per_behavior_metrics,
        "confusion": {
            f"{truth}->{prediction}": count
            for (truth, prediction), count in sorted(confusion.items())
        },
    }


def calibration_objective(metrics: Mapping[str, Any]) -> float:
    """Optimize target-ID recognition while penalizing always-on rules via macro F1."""

    return float(
        0.70 * float(metrics.get("macro_f1", 0.0))
        + 0.30 * float(metrics.get("macro_target_id_accuracy", 0.0))
    )


def group_rule_calibration_rank(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Rank group-rule candidates by macro F1, then strict target Top-1."""

    rows = dict(metrics.get("per_behavior", {}))
    focus = [
        rows[name]
        for name in ("huddle", "isolation", "social_clustering")
        if int(rows.get(name, {}).get("support", 0)) > 0
    ]
    macro_f1 = sum(float(row.get("f1", 0.0)) for row in focus) / len(focus) if focus else 0.0
    macro_strict = (
        sum(float(row.get("strict_top1_accuracy", 0.0)) for row in focus) / len(focus)
        if focus
        else 0.0
    )
    return macro_f1, macro_strict


def strict_top1_regressions(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    excluded_behaviors: set[str] | None = None,
) -> list[str]:
    """List behavior classes whose strict Top-1 count fell from a baseline."""

    excluded = excluded_behaviors or set()
    current_rows = dict(metrics.get("per_behavior", {}))
    baseline_rows = dict(baseline.get("per_behavior", {}))
    return sorted(
        behavior
        for behavior, baseline_values in baseline_rows.items()
        if behavior not in excluded
        and int(baseline_values.get("support", 0)) > 0
        and int(current_rows.get(behavior, {}).get("strict_top1_correct", 0))
        < int(baseline_values.get("strict_top1_correct", 0))
    )


def group_rule_parameter_tie_break(
    parameter_name: str, value: float, candidate_index: int
) -> tuple[float, float]:
    """Prefer conservative, empirically supported settings when scores tie."""

    if parameter_name == "clustering_initial_max_distance_cm":
        return -float(value), -float(candidate_index)
    if parameter_name == "huddle_min_member_support_fraction":
        return -abs(float(value) - 0.25), -float(candidate_index)
    return -float(candidate_index), -float(value)


def minimum_behavior_accuracy(metrics: Mapping[str, Any]) -> float:
    """Return the lowest target-ID accuracy among behaviors with labeled support."""

    values = [
        float(details.get("accuracy", 0.0))
        for details in dict(metrics.get("per_behavior", {})).values()
        if int(details.get("support", 0)) > 0
    ]
    return min(values, default=0.0)


def training_target_reached(metrics: Mapping[str, Any], target_accuracy: float = 0.95) -> bool:
    """Require both aggregate and every-class target-ID accuracy to reach target."""

    target = float(target_accuracy)
    return (
        float(metrics.get("target_id_accuracy", 0.0)) + 1e-12 >= target
        and minimum_behavior_accuracy(metrics) + 1e-12 >= target
    )


def calibration_rank(
    metrics: Mapping[str, Any], target_accuracy: float = 0.95
) -> tuple[float, ...]:
    """Rank training candidates under an accuracy constraint.

    Before the target is reached, the worst behavior is improved first so a
    frequent class cannot hide a failed minority class.  Once every behavior
    reaches the requested accuracy, macro F1 and strict top-1 accuracy become
    the tie breakers.  This prevents the search from preferring an always-on
    detector when a more selective feasible candidate exists.
    """

    feasible = training_target_reached(metrics, target_accuracy)
    worst = minimum_behavior_accuracy(metrics)
    macro_accuracy = float(metrics.get("macro_target_id_accuracy", 0.0))
    overall_accuracy = float(metrics.get("target_id_accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    strict_accuracy = float(metrics.get("strict_target_id_accuracy", 0.0))
    if feasible:
        return (1.0, macro_f1, strict_accuracy, macro_accuracy, overall_accuracy)
    return (0.0, worst, macro_accuracy, overall_accuracy, macro_f1)
