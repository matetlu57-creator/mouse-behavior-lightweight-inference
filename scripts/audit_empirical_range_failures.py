#!/usr/bin/env python3
"""Post-receipt error attribution; never fit parameters on validation labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from calibrate_empirical_ranges import check_base_ranker, sha256, write_json
from mouse_behavior.evaluation.calibrated_rule_ranker import build_candidate_table
from mouse_behavior.evaluation.empirical_range_tuner import adjusted_scores
from mouse_behavior.evaluation.exported_dataset import HeuristicParameters, load_exported_features
from validate_hierarchical_core import _load_predictions_file, _read_labels_after_receipt


FOCUS = {"huddle", "isolation", "social_clustering", "stationary", "together"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--range-model-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.validation_dir / "prediction_receipt.json").read_text(encoding="utf-8"))
    if receipt["val_manifest_sha256"] != sha256(args.val_manifest):
        raise ValueError("validation manifest changed since prediction receipt")
    model_path = args.range_model_dir / "frozen_range_model.json"
    if receipt["range_model_sha256"] != sha256(model_path):
        raise ValueError("range model changed since prediction receipt")
    predictions = _load_predictions_file(args.validation_dir / "predictions.csv",
                                         receipt["prediction_sha256"])
    labels = _read_labels_after_receipt(args.val_manifest)
    if set(predictions) != {label.sample_id for label in labels}:
        raise ValueError("prediction/label sample set differs")
    check_base_ranker(args.candidate_cache, args.train_manifest)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ranker = json.loads((args.candidate_cache / "frozen_model.json").read_text(encoding="utf-8"))
    profiles = tuple(HeuristicParameters(**item) for item in model["rule_profiles"])
    details = []
    for index, label in enumerate(labels, start=1):
        if label.behavior not in FOCUS:
            continue
        f = load_exported_features(label.tracks_path, fps=label.fps, max_tracks=model["max_tracks"])
        table = build_candidate_table(f, profiles)
        scores = adjusted_scores(table, ranker, model["ranges"],
                                 range_strength=model["range_strength"],
                                 layer_bonus=model["layer_bonus"],
                                 group_size_bonus=model["group_size_bonus"])
        target = set(label.target_ids)
        same_class = [j for j, candidate in enumerate(table.candidates)
                      if candidate.behavior == label.behavior]
        matching = [j for j in same_class
                    if target.issubset(table.candidates[j].target_ids)]
        positive = [j for j in matching if table.candidates[j].evidence > 0]
        best_class = max(same_class, key=lambda j: scores[j]) if same_class else None
        best_match = max(matching, key=lambda j: scores[j]) if matching else None
        prediction = predictions[label.sample_id]
        joint_correct = prediction.behavior == label.behavior and target.issubset(prediction.target_ids)
        if joint_correct:
            cause = "correct"
        elif not matching:
            cause = "target_candidate_missing"
        elif label.behavior in {"huddle", "social_clustering"} and not positive:
            cause = "positive_group_evidence_missing"
        elif best_class is not None and best_class not in matching:
            cause = "within_behavior_member_rank"
        else:
            cause = "cross_behavior_rank"
        details.append({
            "sample_id": label.sample_id,
            "true_behavior": label.behavior,
            "true_target_count": len(target),
            "predicted_behavior": prediction.behavior,
            "predicted_target_count": len(prediction.target_ids),
            "target_visual_ids_observed": target.issubset(set(map(int, f.track_ids))),
            "candidate_count": len(table.candidates),
            "matching_target_candidates": len(matching),
            "positive_matching_target_candidates": len(positive),
            "best_same_class_target_included": (
                target.issubset(table.candidates[best_class].target_ids)
                if best_class is not None else False
            ),
            "best_matching_score": (float(scores[best_match]) if best_match is not None else None),
            "predicted_score": prediction.score,
            "cause": cause,
        })
        if index % 25 == 0:
            print(f"post-receipt error audit {index}/{len(labels)}", flush=True)
    by_behavior: dict[str, Counter[str]] = defaultdict(Counter)
    for row in details:
        by_behavior[row["true_behavior"]][row["cause"]] += 1
    result = {
        "scope": "post-receipt diagnostic only; validation labels were not used for fitting",
        "prediction_sha256": receipt["prediction_sha256"],
        "per_behavior": {name: dict(counts) for name, counts in sorted(by_behavior.items())},
        "details": details,
    }
    write_json(args.output, result)
    print(json.dumps(result["per_behavior"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
