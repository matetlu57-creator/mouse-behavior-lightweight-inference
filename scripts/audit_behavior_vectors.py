#!/usr/bin/env python3
"""Summarize per-behavior handmade features from training rows only.

The input reader rejects validation rows. No label reaches trajectory inference;
the training annotation selects only which candidate ID vector to summarize.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.blind_top1 import load_training_manifest
from mouse_behavior.evaluation.exported_dataset import load_exported_features
from mouse_behavior.evaluation.handcrafted_features import behavior_feature_vector


def summarize_training_vectors(train_manifest: Path, *, max_tracks: int = 64) -> dict:
    records = load_training_manifest(train_manifest)
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing: dict[str, int] = defaultdict(int)
    for record in records:
        features = load_exported_features(record.tracks_path, fps=record.fps, max_tracks=max_tracks)
        if not set(record.target_ids).issubset(set(map(int, features.track_ids))):
            missing[record.behavior] += 1
            continue
        vector = behavior_feature_vector(features, record.behavior, record.target_ids)
        for name, value in vector.items():
            if value is not None and np.isfinite(value):
                values[record.behavior][name].append(float(value))
    summary = {}
    for behavior in sorted({record.behavior for record in records}):
        summary[behavior] = {
            "support": sum(record.behavior == behavior for record in records),
            "missing_target_ids": missing[behavior],
            "features": {
                name: {
                    "n": len(numbers),
                    "q10": float(np.quantile(numbers, 0.10)),
                    "median": float(np.median(numbers)),
                    "q90": float(np.quantile(numbers, 0.90)),
                }
                for name, numbers in sorted(values[behavior].items())
                if numbers
            },
        }
    return {
        "source": "training_manifest_only",
        "scale": "estimated_cm_from_assumed_8cm_body_length",
        "n_samples": len(records),
        "per_behavior": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_tracks < 2:
        raise ValueError("--max-tracks must be at least 2")
    summary = summarize_training_vectors(args.train_manifest, max_tracks=args.max_tracks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary['n_samples']} training vectors to {args.output}")


if __name__ == "__main__":
    main()
