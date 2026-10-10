"""Apply the compositional behavior hierarchy to trajectory-only candidates.

The exported-track adapter proposes events and a train-fitted ranker scores
them. This bridge asks the hierarchical core to resolve overlapping events per
mouse, then returns the strongest surviving primary event for clip-level Top-1.
"""

from __future__ import annotations

from math import exp
from typing import Any

import numpy as np

from mouse_behavior.behavior.hierarchical_core import (
    COLLECTIVE_PRIORITY,
    SOCIAL_PRIORITY,
    BehaviorDecision,
    CollectiveLabel,
    Level,
    MotionLabel,
    SocialLabel,
    resolve_behavior_hierarchy,
)
from mouse_behavior.evaluation.blind_top1 import Candidate, Top1Prediction
from mouse_behavior.evaluation.calibrated_rule_ranker import CandidateTable


_BEHAVIOR_LAYER_AND_LABEL = {
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    "locomotion": (Level.INDIVIDUAL, MotionLabel.LOCOMOTION.value),
    "grooming": (Level.INDIVIDUAL, MotionLabel.GROOMING.value),
    "rearing": (Level.INDIVIDUAL, MotionLabel.REARING.value),
    "stationary": (Level.INDIVIDUAL, MotionLabel.STATIC.value),
    "approach": (Level.SOCIAL, SocialLabel.APPROACH.value),
    "together": (Level.SOCIAL, SocialLabel.TOGETHER.value),
    "chase": (Level.SOCIAL, SocialLabel.CHASING.value),
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "avoidance": (Level.SOCIAL, SocialLabel.AVOIDING.value),
    "attack": (Level.SOCIAL, SocialLabel.ATTACK.value),
    "nose_head_contact": (Level.SOCIAL, SocialLabel.SNOUT_HEAD.value),
    "nose_tail_contact": (Level.SOCIAL, SocialLabel.SNOUT_REAR.value),
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "following": (Level.SOCIAL, SocialLabel.FOLLOWING.value),
    "huddle": (Level.COLLECTIVE, CollectiveLabel.HUDDLING.value),
    "social_clustering": (Level.COLLECTIVE, CollectiveLabel.SOCIAL_CLUSTERING.value),
    "dispersal": (Level.COLLECTIVE, CollectiveLabel.DISPERSAL.value),
    "group_locomotion": (Level.COLLECTIVE, CollectiveLabel.GROUP_LOCOMOTION.value),
    "isolation": (Level.COLLECTIVE, CollectiveLabel.ISOLATION.value),
}
_CORE_LABEL_TO_BEHAVIOR = {value: key for key, value in _BEHAVIOR_LAYER_AND_LABEL.items()}


def _confidence(logit: float) -> float:
    """Map an unbounded ranker score to the core's validated confidence range."""

    clipped = max(-60.0, min(60.0, logit))
    return 1.0 / (1.0 + exp(-clipped))


def _decision(candidate: Candidate, score: float) -> BehaviorDecision:
    try:
        level, label = _BEHAVIOR_LAYER_AND_LABEL[candidate.behavior]
    except KeyError as exc:
        raise ValueError(f"unsupported candidate behavior: {candidate.behavior}") from exc
    if not candidate.target_ids or len(candidate.target_ids) != len(set(candidate.target_ids)):
        raise ValueError("candidate target IDs must be non-empty and unique")
    expected_size = 1 if level == Level.INDIVIDUAL else 2 if level == Level.SOCIAL else (
        1 if candidate.behavior == "isolation" else 3
    )
    if len(candidate.target_ids) < expected_size or (
        level != Level.COLLECTIVE and len(candidate.target_ids) != expected_size
    ):
        raise ValueError(f"invalid target count for {candidate.behavior}: {candidate.target_ids}")
    priority = (
        SOCIAL_PRIORITY[label] if level == Level.SOCIAL else
        COLLECTIVE_PRIORITY[label] if level == Level.COLLECTIVE else 0
    )
    return BehaviorDecision(
        level=level,
        label=label,
        members=tuple(sorted(candidate.target_ids)),
        confidence=_confidence(score),
        priority=priority,
        evidence={"ranker_score": score, "rule_evidence_seconds": candidate.evidence},
    )


def predict_hierarchical_top1(table: CandidateTable, model: dict[str, Any]) -> Top1Prediction:
    """Resolve trajectory-derived candidates by layer, then rank primaries.

    A rule candidate must have positive trajectory evidence to enter the core;
    the ranker cannot manufacture an event whose heuristic gate did not fire.
    Layer priority is resolved per mouse, so a group event suppresses lower
    layer events only for its own members rather than the entire clip.
    """

    if tuple(model["feature_names"]) != table.feature_names:
        raise ValueError("candidate feature schema differs from the frozen model")
    if not table.candidates:
        return Top1Prediction(None, (), 0.0)
    if table.values.ndim != 2 or table.values.shape[0] != len(table.candidates):
        raise ValueError("candidate feature matrix shape does not match candidates")

    coefficient_map = model.get("coefficients", {})
    scores: list[float] = []
    for index, candidate in enumerate(table.candidates):
        if candidate.behavior not in coefficient_map:
            raise ValueError(f"frozen model has no coefficients for {candidate.behavior}")
        coefficient = np.asarray(coefficient_map[candidate.behavior], dtype=float)
        if coefficient.shape != (table.values.shape[1],):
            raise ValueError(f"frozen coefficient shape differs for {candidate.behavior}")
        score = float(np.dot(table.values[index], coefficient))
        if not np.isfinite(score) or not np.isfinite(candidate.evidence):
            raise ValueError("candidate evidence and ranker scores must be finite")
        scores.append(score)

    individual_by_mouse: dict[int, BehaviorDecision] = {}
    individual_scores: dict[int, float] = {}
    social_events: list[BehaviorDecision] = []
    collective_events: list[BehaviorDecision] = []
    score_by_decision: dict[tuple[Level, str, tuple[int, ...]], float] = {}
    observed_ids = {mouse_id for candidate in table.candidates for mouse_id in candidate.target_ids}
    for mouse_id in observed_ids:
        unknown = BehaviorDecision(Level.INDIVIDUAL, MotionLabel.UNKNOWN.value, (mouse_id,))
        individual_by_mouse[mouse_id] = unknown

    for candidate, score in zip(table.candidates, scores):
        if candidate.evidence <= 0:
            continue
        decision = _decision(candidate, score)
        key = (decision.level, decision.label, decision.members)
        score_by_decision[key] = max(score_by_decision.get(key, float("-inf")), score)
        if decision.level == Level.INDIVIDUAL:
            mouse_id = decision.members[0]
            if score > individual_scores.get(mouse_id, float("-inf")):
                individual_by_mouse[mouse_id] = decision
                individual_scores[mouse_id] = score
        elif decision.level == Level.SOCIAL:
            social_events.append(decision)
        else:
            collective_events.append(decision)

    snapshot = resolve_behavior_hierarchy(
        individual_by_mouse=individual_by_mouse,
        social_events=social_events,
        collective_events=collective_events,
    )
    selected: dict[tuple[str, tuple[int, ...]], float] = {}
    for decision in snapshot.primary_by_mouse.values():
        behavior = _CORE_LABEL_TO_BEHAVIOR.get((decision.level, decision.label))
        if behavior is None:
            continue
        key = (decision.level, decision.label, tuple(sorted(decision.members)))
        score = score_by_decision.get(key)
        if score is not None:
            selected[(behavior, key[2])] = max(selected.get((behavior, key[2]), float("-inf")), score)
    if not selected:
        return Top1Prediction(None, (), 0.0)
    behavior, target_ids = min(
        selected,
        key=lambda item: (-selected[item], item[0], item[1]),
    )
    return Top1Prediction(behavior, target_ids, selected[(behavior, target_ids)])


__all__ = ["predict_hierarchical_top1"]
