#!/usr/bin/env python3
"""Fit physical behavior ranges on train sessions and evaluate frozen val inputs.

The train command has no validation path. The validate command saves a hashed
trajectory-only prediction receipt before opening the validation labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.blind_top1 import (
    Candidate, evaluate_top1, load_training_manifest,
)
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable, build_candidate_table, fit_ranker,
)
from mouse_behavior.evaluation.empirical_range_tuner import fit_ranges, predict_range_top1
from mouse_behavior.evaluation.exported_dataset import HeuristicParameters, load_exported_features
from mouse_behavior.evaluation.train_only_tuning import session_folds
from validate_hierarchical_core import (
    _read_labels_after_receipt, _read_unlabeled_inputs, _write_predictions,
    _write_scored_rows,
)

SOURCE_FILES = (
    "src/mouse_behavior/evaluation/empirical_range_tuner.py",
    "scripts/calibrate_empirical_ranges.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_train_cache(root: Path, labels) -> list[CandidateTable]:
    metadata = json.loads((root / "training_candidates.json").read_text(encoding="utf-8"))
    if [sample["sample_id"] for sample in metadata["samples"]] != [label.sample_id for label in labels]:
        raise ValueError("candidate cache does not match training label order")
    names = tuple(metadata["feature_names"])
    with np.load(root / "training_features.npz", allow_pickle=False) as arrays:
        return [CandidateTable(
            tuple(Candidate(c[0], tuple(c[1]), float(c[2])) for c in sample["candidates"]),
            arrays[f"clip_{index}"], names,
        ) for index, sample in enumerate(metadata["samples"])]


def check_base_ranker(root: Path, train_manifest: Path) -> dict[str, object]:
    """Reject a mismatched or validation-fitted upstream scoring model."""
    provenance = json.loads((root / "run_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("validation_used_for_fitting") is not False:
        raise ValueError("base ranker is not certified train-only")
    if provenance.get("train_manifest_sha256") != sha256(train_manifest):
        raise ValueError("base ranker training manifest changed")
    if provenance.get("model_sha256") != sha256(root / "frozen_model.json"):
        raise ValueError("base ranker model hash changed")
    for relative, expected in provenance["source_sha256"].items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"base ranker training source changed: {relative}")
    return provenance


def train(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    labels = load_training_manifest(args.train_manifest)
    check_base_ranker(args.candidate_cache, args.train_manifest)
    cache_provenance = json.loads((args.candidate_cache / "feature_provenance.json").read_text(encoding="utf-8"))
    if cache_provenance["train_manifest_sha256"] != sha256(args.train_manifest):
        raise ValueError("training manifest changed since candidate extraction")
    source_hashes = {relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}
    for label in labels:
        if sha256(Path(label.tracks_path)) != cache_provenance["trajectory_sha256"][label.sample_id]:
            raise ValueError(f"training trajectory changed since candidate extraction: {label.sample_id}")
    for relative, expected in cache_provenance["feature_source_sha256"].items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"training feature source changed: {relative}")
    tables = load_train_cache(args.candidate_cache, labels)
    audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    rows = audit["target_rows"]
    if [row["sample_id"] for row in rows] != [label.sample_id for label in labels]:
        raise ValueError("training target audit does not match manifest order")
    ranker = json.loads((args.candidate_cache / "frozen_model.json").read_text(encoding="utf-8"))
    if tuple(ranker["feature_names"]) != tables[0].feature_names:
        raise ValueError("frozen ranker feature schema differs from cache")
    print(f"training-only range search: {len(labels)} clips, {len(session_folds(labels))} held-out sessions", flush=True)
    heldout = []
    for session, fit_indices, check_indices in session_folds(labels):
        fold_ranker, _ = fit_ranker(
            [tables[index] for index in fit_indices], [labels[index] for index in fit_indices],
            steps=args.steps, regularization=0.001, learning_rate=0.03,
            objective="listwise", seed=42,
        )
        fold_ranges = fit_ranges([rows[index] for index in fit_indices])
        heldout.extend((labels[index], tables[index], fold_ranker, fold_ranges) for index in check_indices)
        print(f"training-only held-out session completed: {session} ({len(check_indices)} clips)", flush=True)
    search = []
    for strength in (0.0, 0.5, 1.0, 2.0, 4.0):
        for bonus in (0.0, 0.25, 0.5, 1.0):
            for group_size_bonus in (0.0, 4.0, 8.0):
                metrics = evaluate_top1([
                    (label, predict_range_top1(
                        table, fold_ranker, fold_ranges, range_strength=strength,
                        layer_bonus=bonus, group_size_bonus=group_size_bonus,
                    ))
                    for label, table, fold_ranker, fold_ranges in heldout
                ])
                search.append({"range_strength": strength, "layer_bonus": bonus,
                               "group_size_bonus": group_size_bonus,
                               "training_session_holdout": metrics})
    selected = max(search, key=lambda row: (
        row["training_session_holdout"]["macro_joint_top1_accuracy"],
        row["training_session_holdout"]["joint_top1_accuracy"],
        row["training_session_holdout"]["exact_id_set_accuracy"],
        -row["layer_bonus"], -row["range_strength"],
        -row["group_size_bonus"],
    ))
    ranges = fit_ranges(rows)
    strength, bonus, group_size_bonus = (selected["range_strength"],
                                         selected["layer_bonus"], selected["group_size_bonus"])
    training_metrics = evaluate_top1([
        (label, predict_range_top1(table, ranker, ranges,
                                    range_strength=strength, layer_bonus=bonus,
                                    group_size_bonus=group_size_bonus))
        for label, table in zip(labels, tables)
    ])
    args.output_dir.mkdir(parents=True)
    model = {
        "model_type": "empirical_physical_range_hierarchy_v1",
        "train_manifest_sha256": sha256(args.train_manifest),
        "candidate_cache_provenance_sha256": sha256(args.candidate_cache / "feature_provenance.json"),
        "base_ranker_sha256": sha256(args.candidate_cache / "frozen_model.json"),
        "base_ranker_provenance_sha256": sha256(args.candidate_cache / "run_provenance.json"),
        "training_audit_sha256": sha256(args.training_audit),
        "source_sha256": source_hashes,
        "validation_used_for_fitting": False,
        "range_strength": strength, "layer_bonus": bonus,
        "group_size_bonus": group_size_bonus,
        "ranges": ranges,
        "rule_profiles": ranker["rule_profiles"],
        "max_tracks": ranker["max_tracks"],
    }
    if any(sha256(REPO_ROOT / relative) != digest for relative, digest in source_hashes.items()):
        raise RuntimeError("range tuning source changed during fitting")
    write_json(args.output_dir / "frozen_range_model.json", model)
    write_json(args.output_dir / "train_search.json", search)
    write_json(args.output_dir / "train_in_sample_metrics.json", training_metrics)
    write_json(args.output_dir / "train_session_holdout_metrics.json", selected["training_session_holdout"])
    print(f"selected strength={strength} layer_bonus={bonus} group_size_bonus={group_size_bonus}; "
          f"train joint={training_metrics['joint_top1_accuracy']:.3f}, "
          f"session-holdout joint={selected['training_session_holdout']['joint_top1_accuracy']:.3f}", flush=True)


def validate(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    model_path = args.range_model_dir / "frozen_range_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model["validation_used_for_fitting"] is not False:
        raise ValueError("range model is not train-only")
    if model["train_manifest_sha256"] != sha256(args.train_manifest):
        raise ValueError("training manifest hash changed")
    if model["base_ranker_sha256"] != sha256(args.candidate_cache / "frozen_model.json"):
        raise ValueError("base ranker hash changed")
    if model["base_ranker_provenance_sha256"] != sha256(args.candidate_cache / "run_provenance.json"):
        raise ValueError("base ranker training provenance changed")
    check_base_ranker(args.candidate_cache, args.train_manifest)
    if model["candidate_cache_provenance_sha256"] != sha256(args.candidate_cache / "feature_provenance.json"):
        raise ValueError("candidate cache provenance changed")
    for relative, expected in model["source_sha256"].items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"range tuning source changed: {relative}")
    ranker = json.loads((args.candidate_cache / "frozen_model.json").read_text(encoding="utf-8"))
    cache_provenance = json.loads((args.candidate_cache / "feature_provenance.json").read_text(encoding="utf-8"))
    for relative, expected in cache_provenance["feature_source_sha256"].items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"feature source changed: {relative}")
    inputs = _read_unlabeled_inputs(args.val_manifest)
    train_trajectory_hashes = set(cache_provenance["trajectory_sha256"].values())
    validation_trajectory_hashes = {}
    profiles = tuple(HeuristicParameters(**item) for item in model["rule_profiles"])
    args.output_dir.mkdir(parents=True)
    predictions = []
    stationary_evidence_by_sample: dict[str, dict[int, float]] = {}
    for index, (sample_id, tracks_path, fps) in enumerate(inputs, start=1):
        trajectory_hash = sha256(Path(tracks_path))
        if trajectory_hash in train_trajectory_hashes:
            raise ValueError("validation trajectory duplicates a training trajectory")
        validation_trajectory_hashes[sample_id] = trajectory_hash
        features = load_exported_features(tracks_path, fps=fps, max_tracks=model["max_tracks"])
        table = build_candidate_table(features, profiles)
        stationary_evidence_by_sample[sample_id] = {
            int(candidate.target_ids[0]): float(candidate.evidence)
            for candidate in table.candidates
            if candidate.behavior == "stationary" and len(candidate.target_ids) == 1
        }
        prediction = predict_range_top1(table, ranker, model["ranges"],
                                        range_strength=model["range_strength"],
                                        layer_bonus=model["layer_bonus"],
                                        group_size_bonus=model["group_size_bonus"])
        predictions.append((sample_id, prediction))
        if index % 25 == 0 or index == len(inputs):
            print(f"trajectory-only validation inference {index}/{len(inputs)}", flush=True)
    prediction_path = args.output_dir / "predictions.csv"
    _write_predictions(prediction_path, predictions)
    stationary_candidates_path = args.output_dir / "stationary_candidate_evidence.csv"
    with stationary_candidates_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "track_id", "stationary_evidence_seconds")
        )
        writer.writeheader()
        for sample_id, track_scores in stationary_evidence_by_sample.items():
            for track_id, evidence in sorted(track_scores.items()):
                writer.writerow({
                    "sample_id": sample_id,
                    "track_id": track_id,
                    "stationary_evidence_seconds": evidence,
                })
    write_json(args.output_dir / "prediction_receipt.json", {
        "prediction_sha256": sha256(prediction_path),
        "range_model_sha256": sha256(model_path),
        "val_manifest_sha256": sha256(args.val_manifest),
        "validation_trajectory_sha256": validation_trajectory_hashes,
        "n_samples": len(predictions), "labels_read_before_prediction_receipt": False,
    })
    labels = _read_labels_after_receipt(args.val_manifest)
    if [label.sample_id for label in labels] != [sample_id for sample_id, _ in predictions]:
        raise ValueError("validation labels and frozen predictions are misaligned")
    scored = list(zip(labels, (prediction for _, prediction in predictions)))
    metrics = evaluate_top1(scored)
    _write_scored_rows(args.output_dir / "scored_rows.csv", scored)
    write_json(args.output_dir / "validation_metrics.json", metrics)
    static_rows = []
    prediction_by_sample = dict(predictions)
    for label in labels:
        if label.behavior != "stationary":
            continue
        prediction = prediction_by_sample[label.sample_id]
        scores = stationary_evidence_by_sample[label.sample_id]
        evidence = max((scores.get(track_id, 0.0) for track_id in label.target_ids), default=0.0)
        static_rows.append({
            "sample_id": label.sample_id,
            "target_ids": ",".join(map(str, label.target_ids)),
            "target_has_stationary_individual_evidence": evidence > 0.0,
            "stationary_evidence_seconds": evidence,
            "top1_behavior": prediction.behavior or "",
            "top1_target_ids": ",".join(map(str, prediction.target_ids)),
            "strict_top1_correct": (
                prediction.behavior == label.behavior
                and set(label.target_ids).issubset(prediction.target_ids)
            ),
        })
    with (args.output_dir / "static_target_evidence.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(static_rows[0]) if static_rows else (
            "sample_id", "target_ids", "target_has_stationary_individual_evidence",
            "stationary_evidence_seconds", "top1_behavior", "top1_target_ids",
            "strict_top1_correct",
        ))
        writer.writeheader()
        writer.writerows(static_rows)
    write_json(args.output_dir / "static_target_evidence_metrics.json", {
        "n_static_targets": len(static_rows),
        "targets_with_stationary_individual_evidence": sum(
            bool(row["target_has_stationary_individual_evidence"]) for row in static_rows
        ),
        "strict_top1_correct": sum(bool(row["strict_top1_correct"]) for row in static_rows),
        "labels_read_after_prediction_receipt": True,
    })
    print(f"validation joint={metrics['joint_top1_accuracy']:.3f}, "
          f"macro={metrics['macro_joint_top1_accuracy']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train-manifest", type=Path, required=True)
    train_parser.add_argument("--candidate-cache", type=Path, required=True)
    train_parser.add_argument("--training-audit", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=200)
    val_parser = sub.add_parser("validate")
    val_parser.add_argument("--train-manifest", type=Path, required=True)
    val_parser.add_argument("--val-manifest", type=Path, required=True)
    val_parser.add_argument("--candidate-cache", type=Path, required=True)
    val_parser.add_argument("--range-model-dir", type=Path, required=True)
    val_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
