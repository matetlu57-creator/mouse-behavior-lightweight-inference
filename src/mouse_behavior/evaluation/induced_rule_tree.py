"""Train-only induction of explicit trajectory threshold trees.

The trees are fitted from annotated training clips, exported as JSON numeric
rules, and evaluated without scikit-learn. Visual IDs are output labels only;
their numeric values and sample paths are never model features.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .blind_top1 import LabeledClip, Top1Prediction
from .exported_dataset import ExportedFeatures


SCENE_FEATURES = (
    "log_duration", "track_count", "scene_count_q10", "scene_count_q50",
    "scene_count_q90", "scene_speed_q10", "scene_speed_q50", "scene_speed_q90",
    "scene_nearest_q10", "scene_nearest_q50", "scene_nearest_q90",
    "pair_distance_q10", "pair_distance_q50", "pair_distance_q90",
    "pair_close_5_fraction", "pair_close_20_fraction",
)
MEMBER_FEATURES = SCENE_FEATURES + (
    "visibility", "first_seen", "last_seen", "speed_q10", "speed_q50",
    "speed_q90", "nearest_q10", "nearest_q50", "nearest_q90",
    "net_displacement", "median_x", "median_y", "slow_fraction",
    "moving_fraction", "near_10_fraction", "far_30_fraction",
)


def passes_session_generalization_gate(
    train_metrics: dict[str, Any], holdout_metrics: dict[str, Any], threshold: float
) -> bool:
    """Require each class to transfer to a different recording before validation."""
    classes = set(train_metrics["per_behavior"])
    return bool(classes) and classes == set(holdout_metrics["per_behavior"]) and all(
        metrics["per_behavior"][name]["joint_top1_accuracy"] >= threshold
        for metrics in (train_metrics, holdout_metrics)
        for name in classes
    )


def _quantiles(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    return np.quantile(finite, (0.1, 0.5, 0.9)) if finite.size else np.zeros(3)


def extract_rule_features(f: ExportedFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use measured trajectories and FPS, never annotations or IDs as features."""
    valid = f.valid
    speed = f.speed_cm_s
    nearest = f.nearest_distance_cm
    pair = f.pair_distance_cm
    pair_values = pair[np.isfinite(pair)]
    scene_speed = speed[valid]
    scene_nearest = nearest[valid & np.isfinite(nearest)]
    scene = np.r_[
        np.log1p(len(valid) / f.fps), len(f.track_ids) / 40,
        _quantiles(valid.sum(axis=1)), _quantiles(scene_speed) / 30,
        _quantiles(scene_nearest) / 100, _quantiles(pair_values) / 100,
        np.mean(pair_values < 5) if pair_values.size else 0,
        np.mean(pair_values < 20) if pair_values.size else 0,
    ]
    rows = []
    for slot in range(len(f.track_ids)):
        mask = valid[:, slot]
        indices = np.flatnonzero(mask)
        own_speed = speed[mask, slot]
        own_nearest = nearest[mask, slot]
        positions = f.centers_cm[mask, slot]
        position = np.median(positions, axis=0) if len(positions) else np.zeros(2)
        displacement = np.linalg.norm(positions[-1] - positions[0]) if len(positions) > 1 else 0
        finite_nearest = own_nearest[np.isfinite(own_nearest)]
        rows.append(np.r_[
            scene, mask.mean(), indices[0] / len(valid) if len(indices) else 0,
            indices[-1] / len(valid) if len(indices) else 0,
            _quantiles(own_speed) / 30, _quantiles(finite_nearest) / 100,
            displacement / 100, position / 150,
            np.mean(own_speed < 2) if own_speed.size else 0,
            np.mean(own_speed > 10) if own_speed.size else 0,
            np.mean(finite_nearest < 10) if finite_nearest.size else 0,
            np.mean(finite_nearest > 30) if finite_nearest.size else 0,
        ])
    members = np.asarray(rows, dtype=np.float32).reshape((-1, len(MEMBER_FEATURES)))
    return np.asarray(scene, dtype=np.float32), members, f.track_ids.copy()


def _export_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    return {
        "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, :].tolist(),
        "classes": estimator.classes_.tolist() if hasattr(estimator, "classes_") else None,
    }


def _tree_value(tree: dict[str, Any], values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    node = 0
    while tree["left"][node] >= 0:
        feature = tree["feature"][node]
        node = tree["left"][node] if values[feature] <= tree["threshold"][node] else tree["right"][node]
    result = np.asarray(tree["value"][node], dtype=float)
    classes = tree["classes"]
    if classes is not None:
        result /= max(float(result.sum()), 1e-12)
    return result, np.asarray(classes) if classes is not None else None


def fit_rule_trees(
    items: Sequence[tuple[LabeledClip, np.ndarray, np.ndarray, np.ndarray]],
    *,
    depth: int | None = None,
) -> dict[str, Any]:
    """Fit threshold rules; this function accepts only training labels."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if not items:
        raise ValueError("training items must not be empty")
    names = sorted({label.behavior for label, *_ in items})
    scenes = np.asarray([scene for _, scene, _, _ in items])
    y = np.asarray([names.index(label.behavior) for label, *_ in items])
    behavior = DecisionTreeClassifier(max_depth=depth, random_state=42)
    behavior.fit(scenes, y)
    member_trees: dict[str, Any] = {}
    count_trees: dict[str, Any] = {}
    for name in names:
        subset = [(label, scene, members, ids) for label, scene, members, ids in items if label.behavior == name]
        x = np.concatenate([members for _, _, members, _ in subset])
        labels = np.asarray([int(int(track_id) in label.target_ids)
                             for label, _, _, ids in subset for track_id in ids])
        if len(set(labels.tolist())) < 2:
            member_trees[name] = {"constant": float(labels[0])}
        else:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(x, labels)
            member_trees[name] = _export_tree(tree)
        if name in {"huddle", "social_clustering"}:
            sizes = np.asarray([len(label.target_ids) for label, *_ in subset])
            size_tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
            size_tree.fit(np.asarray([scene for _, scene, _, _ in subset]), sizes)
            count_trees[name] = _export_tree(size_tree)
    return {"model_type": "induced_trajectory_threshold_rules_v1", "depth": depth,
            "scene_features": list(SCENE_FEATURES), "member_features": list(MEMBER_FEATURES),
            "behaviors": names, "behavior_tree": _export_tree(behavior),
            "member_trees": member_trees, "count_trees": count_trees}


def predict_rule_trees(model: dict[str, Any], scene: np.ndarray,
                       members: np.ndarray, track_ids: np.ndarray) -> Top1Prediction:
    if len(scene) != len(model["scene_features"]) or members.shape != (len(track_ids), len(model["member_features"])):
        raise ValueError("trajectory rule feature schema mismatch")
    behavior_scores, behavior_classes = _tree_value(model["behavior_tree"], scene)
    assert behavior_classes is not None
    behavior = model["behaviors"][int(behavior_classes[np.argmax(behavior_scores)])]
    if behavior in model["count_trees"]:
        size_value, _ = _tree_value(model["count_trees"][behavior], scene)
        count = max(3, int(round(float(size_value[0]))))
    else:
        count = 1 if behavior in {"isolation", "stationary", "locomotion"} else 2
    count = min(count, len(track_ids))
    if not count:
        return Top1Prediction(None, (), 0.0)
    member_tree = model["member_trees"][behavior]
    if "constant" in member_tree:
        scores = np.full(len(track_ids), member_tree["constant"])
    else:
        scores = np.zeros(len(track_ids))
        for index, values in enumerate(members):
            probabilities, classes = _tree_value(member_tree, values)
            assert classes is not None
            positive = np.flatnonzero(classes == 1)
            scores[index] = probabilities[positive[0]] if len(positive) else 0.0
    order = np.lexsort((track_ids, -scores))
    chosen = tuple(sorted(int(track_ids[index]) for index in order[:count]))
    return Top1Prediction(behavior, chosen, float(behavior_scores.max()))
