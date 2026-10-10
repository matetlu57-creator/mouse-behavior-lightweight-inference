"""Train-fitted, physical-unit compatibility ranges for behavior candidates.

The ranges are learned from labeled *training* target trajectories. They are
soft evidence penalties, not class priors or a license to infer missing IDs.
Group, social, and individual candidates remain available concurrently; a
higher layer receives only a calibrated bonus after its physical evidence is
compatible, rather than automatically replacing a lower-layer event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .blind_top1 import Top1Prediction
from .calibrated_rule_ranker import CandidateTable


@dataclass(frozen=True)
class RangeSpec:
    behavior: str
    contrast: str
    feature: str
    direction: str
    unit: str


RANGE_SPECS = (
    RangeSpec("isolation", "stationary", "nearest_q50", "min", "cm"),
    RangeSpec("stationary", "locomotion", "speed_q50", "max", "cm/s"),
    RangeSpec("huddle", "social_clustering", "distance_q50", "max", "cm"),
    RangeSpec("huddle", "social_clustering", "member_visibility_min", "max", "fraction"),
    RangeSpec("social_clustering", "huddle", "member_visibility_min", "min", "fraction"),
    RangeSpec("social_clustering", "huddle", "net_displacement", "min", "fraction_of_50cm"),
)
LAYER = {
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    "locomotion": 0, "stationary": 0,
    "approach": 1, "together": 1, "chase": 1,
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "avoidance": 1, "following": 1,
    "attack": 1, "nose_head_contact": 1, "nose_tail_contact": 1,
    "huddle": 2, "social_clustering": 2, "dispersal": 2,
    "group_locomotion": 2, "isolation": 2,
}


def fit_ranges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fit one-dimensional contrast rules using target-only train features."""
    result = []
    for spec in RANGE_SPECS:
        positive = np.asarray([row["features"][spec.feature] for row in rows
                               if row["behavior"] == spec.behavior], dtype=float)
        negative = np.asarray([row["features"][spec.feature] for row in rows
                               if row["behavior"] == spec.contrast], dtype=float)
        if len(positive) < 3 or len(negative) < 3 or not (np.isfinite(positive).all() and np.isfinite(negative).all()):
            raise ValueError(f"insufficient finite training examples for {spec.behavior}/{spec.contrast}")
        values = np.unique(np.r_[positive, negative])
        cutoffs = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
        sign = 1 if spec.direction == "min" else -1
        fits = []
        for cutoff in cutoffs:
            sensitivity = float(np.mean(sign * (positive - cutoff) >= 0))
            specificity = float(np.mean(sign * (negative - cutoff) < 0))
            fits.append((0.5 * (sensitivity + specificity), sensitivity, specificity, float(cutoff)))
        best = max(fits, key=lambda entry: (entry[0], min(entry[1], entry[2]), -abs(entry[3] - np.median(positive))))
        scale = max(float(np.subtract(*np.quantile(positive, [0.75, 0.25]))),
                    float(np.subtract(*np.quantile(negative, [0.75, 0.25]))), 0.025)
        # A weak contrast cannot be turned into a hard gate by optimization.
        result.append({
            "behavior": spec.behavior, "contrast": spec.contrast,
            "feature": spec.feature, "direction": spec.direction, "unit": spec.unit,
            "threshold": best[3], "scale": scale,
            "balanced_accuracy": best[0], "sensitivity": best[1], "specificity": best[2],
            "positive_n": len(positive), "negative_n": len(negative),
            "positive_q10": float(np.quantile(positive, 0.1)),
            "positive_q90": float(np.quantile(positive, 0.9)),
        })
    return result


def adjusted_scores(
    table: CandidateTable,
    ranker: Mapping[str, Any],
    ranges: Sequence[Mapping[str, Any]],
    *,
    range_strength: float,
    layer_bonus: float,
    group_size_bonus: float = 0.0,
) -> np.ndarray:
    """Score only observed-ID candidates with train-fitted physical penalties."""
    if table.feature_names != tuple(ranker["feature_names"]):
        raise ValueError("candidate feature schema differs from frozen ranker")
    if range_strength < 0 or layer_bonus < 0 or group_size_bonus < 0:
        raise ValueError("range strength and layer bonuses must be nonnegative")
    if not table.candidates:
        return np.empty(0, dtype=float)
    weights = np.asarray([ranker["coefficients"][candidate.behavior]
                          for candidate in table.candidates], dtype=float)
    scores = np.einsum("ij,ij->i", table.values, weights)
    names = {name: index for index, name in enumerate(table.feature_names)}
    for rule in ranges:
        if rule["balanced_accuracy"] < 0.55:
            continue
        feature = rule["feature"]
        if feature not in names:
            raise ValueError(f"missing physical feature: {feature}")
        direction = 1 if rule["direction"] == "min" else -1
        margin = direction * (table.values[:, names[feature]] - float(rule["threshold"]))
        violation = np.clip(-margin / max(float(rule["scale"]), 1e-6), 0, 2)
        selected = np.fromiter((c.behavior == rule["behavior"] for c in table.candidates), bool)
        scores[selected] -= range_strength * float(rule["balanced_accuracy"] - 0.5) * 2 * violation[selected]
    scores += layer_bonus * np.asarray([LAYER[c.behavior] for c in table.candidates])
    if group_size_bonus:
        member_count = table.values[:, names["member_count_log"]]
        supported_group = np.fromiter(
            (c.behavior in {"huddle", "social_clustering"} and c.evidence > 0
             for c in table.candidates), bool,
        )
        scores[supported_group] += group_size_bonus * member_count[supported_group]
    return scores


def predict_range_top1(
    table: CandidateTable,
    ranker: Mapping[str, Any],
    ranges: Sequence[Mapping[str, Any]],
    *,
    range_strength: float,
    layer_bonus: float,
    group_size_bonus: float = 0.0,
) -> Top1Prediction:
    scores = adjusted_scores(table, ranker, ranges,
                             range_strength=range_strength, layer_bonus=layer_bonus,
                             group_size_bonus=group_size_bonus)
    if not len(scores):
        return Top1Prediction(None, (), 0.0)
    # Evidence is a scored feature. A zero-evidence singleton/pair may still
    # compete, but physical range penalties cannot fabricate a track or group.
    winner = int(np.argmax(scores))
    candidate = table.candidates[winner]
    return Top1Prediction(candidate.behavior, candidate.target_ids, float(scores[winner]))


__all__ = ["RANGE_SPECS", "fit_ranges", "adjusted_scores", "predict_range_top1"]
