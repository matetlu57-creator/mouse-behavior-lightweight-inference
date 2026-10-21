"""Explicit target membership policy for the opt-in group experiment."""
from __future__ import annotations

from .blind_top1 import evaluate_top1


def target_ids_match(behavior, expected, predicted, policy="exact"):
    if policy not in {"exact", "social_clustering_superset", "group_target_superset"}:
        raise ValueError("unknown target ID policy")
    if ((policy == "social_clustering_superset" and behavior == "social_clustering")
            or (policy == "group_target_superset" and behavior in {"social_clustering", "group_locomotion"})):
        return bool(expected) and set(expected).issubset(predicted)
    return set(expected) == set(predicted)


def evaluate_policy(rows, policy="exact"):
    metrics = evaluate_top1(rows)
    if policy == "exact":
        metrics["target_id_policy"] = policy
        return metrics
    correct = {name: 0 for name in metrics["per_behavior"]}
    for label, prediction in rows:
        correct[label.behavior] += int(prediction.behavior == label.behavior and target_ids_match(
            label.behavior, label.target_ids, prediction.target_ids, policy))
    for name, item in metrics["per_behavior"].items():
        item["joint_correct"] = correct[name]
        item["joint_top1_accuracy"] = correct[name] / item["support"]
    metrics["joint_top1_accuracy"] = sum(correct.values()) / len(rows) if rows else 0.0
    metrics["macro_joint_top1_accuracy"] = sum(
        item["joint_top1_accuracy"] for item in metrics["per_behavior"].values()
    ) / max(len(correct), 1)
    metrics["target_id_policy"] = policy
    metrics["primary_metric"] = "behavior_and_target_membership_" + policy
    return metrics
