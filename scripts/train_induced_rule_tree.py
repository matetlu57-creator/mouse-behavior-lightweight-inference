#!/usr/bin/env python3
"""Induce trajectory threshold rules on train.csv, then score frozen rules on val.csv.

The two subcommands are deliberately separate. Validation labels are not read
until all trajectory-only predictions have been written and hash-locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT

from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip, evaluate_top1, load_locked_predictions, load_training_manifest,
)
from mouse_behavior.evaluation.exported_dataset import (
    EXPORTED_DATASET_SOURCE_FILES,
    load_exported_features,
)
from mouse_behavior.evaluation.induced_rule_tree import (
    extract_rule_features, fit_rule_trees, passes_session_generalization_gate,
    predict_rule_trees,
)
from mouse_behavior.evaluation.train_only_tuning import session_folds
from tune_exported_training import _write_predictions


LOGGER = logging.getLogger(__name__)
SOURCE_FILES = (
    *EXPORTED_DATASET_SOURCE_FILES,
    "src/mouse_behavior/evaluation/blind_top1.py",
    "src/mouse_behavior/evaluation/induced_rule_tree.py",
    "scripts/train_induced_rule_tree.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def extract_train(labels: list[LabeledClip]):
    items = []
    for index, label in enumerate(labels, start=1):
        features = load_exported_features(label.tracks_path, fps=label.fps)
        items.append((label, *extract_rule_features(features)))
        if index % 100 == 0 or index == len(labels):
            LOGGER.info("training trajectory features %d/%d", index, len(labels))
    return items


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError("choose a new output directory; experiments are immutable")
    labels = load_training_manifest(args.train_manifest)
    manifest_hash = sha256(args.train_manifest)
    source_hashes = {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}
    trajectory_hashes = {label.sample_id: sha256(Path(label.tracks_path)) for label in labels}
    items = extract_train(labels)
    oof = []
    for session, fit_indices, check_indices in session_folds(labels):
        fitted = fit_rule_trees([items[index] for index in fit_indices], depth=args.depth)
        oof.extend((items[index][0], predict_rule_trees(fitted, *items[index][1:]))
                   for index in check_indices)
        LOGGER.info("training internal held-out session %s: %d clips", session, len(check_indices))
    oof_metrics = evaluate_top1(oof)
    model = fit_rule_trees(items, depth=args.depth)
    predictions = [(label, predict_rule_trees(model, scene, members, ids))
                   for label, scene, members, ids in items]
    train_metrics = evaluate_top1(predictions)
    # A high resubstitution score is not evidence that the thresholds transfer
    # to another recording. Never unlock validation on training fit alone.
    ready = passes_session_generalization_gate(train_metrics, oof_metrics, args.min_per_class)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "frozen_model.json", model)
    write_json(args.output_dir / "training_metrics.json", train_metrics)
    write_json(args.output_dir / "training_session_holdout.json", oof_metrics)
    _write_predictions(args.output_dir / "train_predictions.csv", predictions, split="train")
    if (sha256(args.train_manifest) != manifest_hash
            or any(sha256(REPO_ROOT / name) != digest for name, digest in source_hashes.items())):
        raise RuntimeError("training manifest or source changed during fitting")
    write_json(args.output_dir / "run_provenance.json", {
        "model_sha256": sha256(args.output_dir / "frozen_model.json"),
        "train_manifest_sha256": manifest_hash, "source_sha256": source_hashes,
        "training_trajectory_sha256": trajectory_hashes,
        "train_sample_ids": [label.sample_id for label in labels],
        "train_sessions": sorted({label.recording_session for label in labels}),
        "depth": args.depth, "min_per_class": args.min_per_class,
        "ready_for_existing_validation": ready,
        "readiness_requires_session_holdout": True,
        "validation_used_for_fitting": False,
    })
    LOGGER.info("training joint=%.2f%% session-holdout=%.2f%% train-and-holdout per-class gate=%s",
                100 * train_metrics["joint_top1_accuracy"],
                100 * oof_metrics["joint_top1_accuracy"], "PASS" if ready else "FAIL")


def validate(args: argparse.Namespace) -> None:
    provenance = json.loads((args.model_dir / "run_provenance.json").read_text(encoding="utf-8"))
    if not provenance["ready_for_existing_validation"]:
        raise ValueError("training per-class threshold was not reached")
    model_path = args.model_dir / "frozen_model.json"
    if sha256(model_path) != provenance["model_sha256"]:
        raise ValueError("frozen model hash changed")
    for name, digest in provenance["source_sha256"].items():
        if sha256(REPO_ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    if args.output_dir.exists():
        raise FileExistsError("choose a new validation output directory")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(args.val_manifest)
    inputs = []
    with args.val_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if row["split"] not in {"val", "validation"}:
                raise ValueError("validation manifest contains a non-validation row")
            if row["sample_id"] in provenance["train_sample_ids"] or row["recording_session"] in provenance["train_sessions"]:
                raise ValueError("training and validation recordings overlap")
            inputs.append((f"v-{index:06d}", row["tracks_path"], float(row["fps"])))
    if not inputs:
        raise ValueError("empty validation input")
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "predictions.csv"
    trajectory_hashes = {}
    train_hashes = set(provenance["training_trajectory_sha256"].values())
    with prediction_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "predicted_behavior", "predicted_target_ids", "score"))
        writer.writeheader()
        for index, (sample_id, tracks_path, fps) in enumerate(inputs, start=1):
            digest = sha256(Path(tracks_path))
            if digest in train_hashes:
                raise ValueError("validation trajectory duplicates training trajectory")
            trajectory_hashes[sample_id] = digest
            features = load_exported_features(tracks_path, fps=fps)
            prediction = predict_rule_trees(model, *extract_rule_features(features))
            writer.writerow({"sample_id": sample_id, "predicted_behavior": prediction.behavior or "",
                             "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                             "score": prediction.score})
            if index % 50 == 0 or index == len(inputs):
                LOGGER.info("validation trajectory-only predictions %d/%d", index, len(inputs))
    write_json(args.output_dir / "prediction_receipt.json", {
        "prediction_count": len(inputs), "predictions_sha256": sha256(prediction_path),
        "model_sha256": provenance["model_sha256"], "validation_manifest_sha256": manifest_hash,
        "validation_trajectory_sha256": trajectory_hashes, "contains_ground_truth": False,
    })
    predictions = load_locked_predictions(args.output_dir)
    if sha256(args.val_manifest) != manifest_hash or sha256(model_path) != provenance["model_sha256"]:
        raise ValueError("validation input or frozen model changed during prediction")
    with args.val_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [LabeledClip(f"v-{index:06d}", row["tracks_path"], float(row["fps"]),
                              row["recording_session"], row["canonical_behavior"],
                              tuple(sorted(int(value) for value in row["mouse_ids"].split(",") if value)))
                  for index, row in enumerate(csv.DictReader(handle), start=1)]
    scored = [(label, predictions[label.sample_id]) for label in labels]
    metrics = evaluate_top1(scored)
    write_json(args.output_dir / "validation_metrics.json", metrics)
    _write_predictions(args.output_dir / "validation_predictions_scored.csv", scored, split="val")
    LOGGER.info("existing validation joint=%.2f%% macro=%.2f%%",
                100 * metrics["joint_top1_accuracy"], 100 * metrics["macro_joint_top1_accuracy"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--train-manifest", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--depth", type=int, default=12)
    training.add_argument("--min-per-class", type=float, default=0.9)
    validation = sub.add_parser("validate")
    validation.add_argument("--val-manifest", type=Path, required=True)
    validation.add_argument("--model-dir", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        if args.depth < 1 or not 0 < args.min_per_class <= 1:
            raise ValueError("invalid training bounds")
        train(args)
    else:
        validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
