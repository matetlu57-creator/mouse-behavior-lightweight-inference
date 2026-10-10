#!/usr/bin/env python3
"""Summarize labeled training targets in physical units; never read validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.blind_top1 import load_training_manifest
from mouse_behavior.evaluation.calibrated_rule_ranker import _physical_features
from mouse_behavior.evaluation.exported_dataset import load_exported_features


FEATURES = (
    "clip_seconds_log", "simultaneous_fraction", "member_visibility_min",
    "speed_q50", "speed_q90", "speed_le_4", "speed_le_8",
    "nearest_q10", "nearest_q50", "nearest_q90",
    "distance_q10", "distance_q50", "distance_q90",
    "distance_early_q50", "distance_late_q50", "distance_net_drop",
    "distance_close_5_longest_fraction", "distance_close_10_longest_fraction",
    "net_displacement",
)
CM_SCALED = {"speed_q50", "speed_q90", "nearest_q10", "nearest_q50", "nearest_q90",
             "distance_q10", "distance_q50", "distance_q90", "distance_early_q50",
             "distance_late_q50", "distance_net_drop"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = load_training_manifest(args.train_manifest)
    cache = json.loads((args.candidate_cache / "training_candidates.json").read_text(encoding="utf-8"))
    if [sample["sample_id"] for sample in cache["samples"]] != [label.sample_id for label in labels]:
        raise ValueError("candidate cache row order differs from training manifest")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_examples: dict[str, list[str]] = defaultdict(list)
    target_rows: list[dict[str, object]] = []
    for index, (label, sample) in enumerate(zip(labels, cache["samples"])):
        if index % 100 == 0:
            print(f"training-only raw target audit {index}/{len(labels)}", flush=True)
        f = load_exported_features(label.tracks_path, fps=label.fps, max_tracks=64)
        present = set(map(int, f.track_ids))
        target = set(label.target_ids)
        coverage[label.behavior]["total"] += 1
        if not target.issubset(present):
            coverage[label.behavior]["missing_visual_target"] += 1
            missing_examples[label.behavior].append(label.sample_id)
            continue
        coverage[label.behavior]["all_target_ids_observed"] += 1
        candidates = sample["candidates"]
        if any(c[0] == label.behavior and target.issubset(c[1]) for c in candidates):
            coverage[label.behavior]["target_candidate"] += 1
        values = _physical_features(f, label.target_ids)
        target_rows.append({
            "sample_id": label.sample_id,
            "recording_session": label.recording_session,
            "behavior": label.behavior,
            "target_ids": list(label.target_ids),
            "features": {name: float(values[name]) for name in FEATURES},
        })
        for name in FEATURES:
            grouped[label.behavior][name].append(float(values[name]) * (30 if name in CM_SCALED else 1))
    summary = {
        behavior: {
            "coverage": dict(coverage[behavior]),
            "missing_examples": missing_examples[behavior][:10],
            "features": {
                name: {
                    "n": len(numbers),
                    "q10": float(np.quantile(numbers, 0.1)),
                    "q25": float(np.quantile(numbers, 0.25)),
                    "median": float(np.median(numbers)),
                    "q75": float(np.quantile(numbers, 0.75)),
                    "q90": float(np.quantile(numbers, 0.9)),
                }
                for name, numbers in features.items() if numbers
            },
        }
        for behavior, features in grouped.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "target_rows": target_rows},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
