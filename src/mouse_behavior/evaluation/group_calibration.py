"""Audited distance calibration; callers must disclose validation reuse."""
from __future__ import annotations

import numpy as np

from .blind_top1 import Candidate
from .group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG


def calibrate_isolation(labels, summaries, baseline=20.0):
    rows = []
    for label in labels:
        for track in label.target_ids:
            record = summaries[label.sample_id]["tracks"].get(str(track), {})
            rows.append({"sample_id": label.sample_id, "behavior": label.behavior,
                         "target_id": track, "weight": 1 / len(label.target_ids),
                         "positive": label.behavior == "isolation", **record})
    positives = [row for row in rows if row["positive"] and row.get("minimum_cm") is not None]
    negatives = [row for row in rows if not row["positive"] and row.get("minimum_cm") is not None]
    if not positives or not negatives:
        raise ValueError("isolation calibration needs observed positive and negative targets")
    search = []
    for distance in np.arange(5.0, 30.01, .5):
        recall = sum(row["weight"] for row in positives if row["minimum_cm"] > distance) / sum(row["weight"] for row in positives)
        specificity = sum(row["weight"] for row in negatives if row["minimum_cm"] <= distance) / sum(row["weight"] for row in negatives)
        search.append({"distance_cm": float(distance), "target_recall": recall,
                       "target_specificity": specificity, "balanced_accuracy": (recall + specificity) / 2})
    selected = max(search, key=lambda item: (item["balanced_accuracy"], -abs(item["distance_cm"] - baseline), item["distance_cm"]))
    return {"validation_labels_used": True, "method": "clip_weighted_target_balanced_accuracy_grid_5_to_30_cm_step_0.5_tie_nearest_20_cm",
            "comparison": "minimum_coobserved_target_to_any_other_mouse_distance_strictly_greater_than_threshold",
            "selected": selected, "search": search, "target_nearest_distances": rows}


def rethreshold_isolation(table, summary, threshold,
                         minimum_duration_s=DEFAULT_GROUP_HEURISTIC_CONFIG.isolation_min_duration_s):
    # Every observed visual track already has an isolation row. Replacing only
    # its rule evidence avoids re-extracting trajectories or changing IDs.
    values = table.values.copy()
    duration = summary["duration_s"]
    evidence = {}
    for index, candidate in enumerate(table.candidates):
        if candidate.behavior != "isolation":
            continue
        track = summary["tracks"][str(candidate.target_ids[0])]
        observed_s = float(track["observed_frames"]) / summary["fps"]
        evidence[index] = (observed_s if track["minimum_cm"] is not None
                           and track["minimum_cm"] > threshold and observed_s > minimum_duration_s else 0.0)
    maximum = max(max(evidence.values(), default=0), 1e-6)
    candidates = list(table.candidates)
    for index, seconds in evidence.items():
        old = candidates[index]
        candidates[index] = Candidate(old.behavior, old.target_ids, seconds)
        for column, name in enumerate(table.feature_names):
            if name == "evidence_seconds_log":
                values[index, column] = np.log1p(seconds) / 4
            elif name == "evidence_fraction" or (name.startswith("rule_") and name.endswith("_fraction")):
                values[index, column] = seconds / max(duration, 1e-6)
            elif name == "evidence_relative" or (name.startswith("rule_") and name.endswith("_relative")):
                values[index, column] = seconds / maximum
    return type(table)(tuple(candidates), values, table.feature_names)
