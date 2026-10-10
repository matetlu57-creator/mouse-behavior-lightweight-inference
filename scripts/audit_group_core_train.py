#!/usr/bin/env python3
"""Compare group-core behavior rules on the existing training manifest only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.exported_dataset import (
    HeuristicParameters,
    SampleRecord,
    classify_target_ids,
    load_exported_features,
    predict_features,
)


def _record(row: dict[str, str]) -> SampleRecord:
    return SampleRecord(
        sample_id=row["sample_id"],
        behavior=row["behavior"],
        canonical_behavior=row["canonical_behavior"],
        sample_dir=row["sample_dir"],
        annotation_path=row["annotation_path"],
        metadata_path=row["metadata_path"],
        tracks_path=row["tracks_path"],
        clip_path=row["clip_path"],
        source_video=row["source_video"],
        recording_session=row["recording_session"],
        fps=float(row["fps"]),
        frame_count=int(row["frame_count"]),
        mouse_ids=tuple(int(value) for value in row["mouse_ids"].split(",")),
        confidence=row["confidence"],
        frame_start=int(row["frame_start"]),
        frame_end=int(row["frame_end"]),
        time_start=float(row["time_start"]),
        time_end=float(row["time_end"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--baseline-parameters", type=Path, required=True)
    parser.add_argument("--candidate-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.train_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("only the existing training split is accepted")
    with args.baseline_parameters.open("r", encoding="utf-8") as handle:
        baseline = HeuristicParameters(**json.load(handle))
    with args.candidate_parameters.open("r", encoding="utf-8") as handle:
        candidate = HeuristicParameters(**json.load(handle))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    changed: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        record = _record(row)
        features = load_exported_features(record.tracks_path, fps=record.fps, max_tracks=args.max_tracks)
        previous = classify_target_ids(record, predict_features(features, baseline))
        updated = classify_target_ids(record, predict_features(features, candidate))
        bucket = counts[record.behavior]
        bucket["n"] += 1
        bucket["baseline_correct"] += int(bool(previous["correct"]))
        bucket["core_correct"] += int(bool(updated["correct"]))
        if previous["correct"] != updated["correct"]:
            changed.append(
                {
                    "sample_id": record.sample_id,
                    "behavior": record.behavior,
                    "baseline_correct": bool(previous["correct"]),
                    "core_correct": bool(updated["correct"]),
                    "baseline_prediction": previous["predicted_behavior"],
                    "core_prediction": updated["predicted_behavior"],
                }
            )
        if index % 100 == 0 or index == len(rows):
            print(f"training clips evaluated: {index}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"n": len(rows), "by_behavior": counts, "changed": changed}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
