#!/usr/bin/env python3
"""Train on the existing train.csv, freeze, then evaluate the existing val.csv.

The train subcommand has no validation input. The validate subcommand has no
optimizer. Validation labels are joined only after predictions are saved.
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
import numpy as np

from mouse_behavior.config import load_config
from mouse_behavior.evaluation.blind_top1 import (
    Candidate, LabeledClip, evaluate_top1, load_locked_predictions, load_training_manifest,
    target_layer_candidate_diagnostics,
)
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable, average_ranker_models, build_candidate_table, fit_ranker, metric_rank,
    predict_table, rule_profiles,
)
from mouse_behavior.evaluation.exported_dataset import (
    EXPORTED_DATASET_SOURCE_FILES,
    HeuristicParameters,
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.train_only_tuning import candidate_oracle_metrics, session_folds
from tune_exported_training import _write_predictions


LOGGER = logging.getLogger(__name__)
SOURCE_FILES = (
    *EXPORTED_DATASET_SOURCE_FILES,
    "src/mouse_behavior/evaluation/blind_top1.py",
    "src/mouse_behavior/evaluation/calibrated_rule_ranker.py",
    "src/mouse_behavior/evaluation/train_only_tuning.py",
    "scripts/calibrate_existing_split.py",
    "scripts/tune_exported_training.py",
)
# Candidate tables depend on trajectory feature extraction and table-building
# code. Ranking/evaluation changes can safely reuse this cache when the
# manifest, trajectories, profiles, and these exact feature sources match.
FEATURE_SOURCE_FILES = (
    *EXPORTED_DATASET_SOURCE_FILES,
    "src/mouse_behavior/evaluation/calibrated_rule_ranker.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def training_tables(labels, profiles, output: Path, max_tracks: int):
    tables = []
    for index, label in enumerate(labels):
        f = load_exported_features(label.tracks_path, fps=label.fps, max_tracks=max_tracks)
        tables.append(build_candidate_table(f, profiles))
        if (index + 1) % 25 == 0 or index + 1 == len(labels):
            LOGGER.info("training-only candidate extraction %d/%d", index + 1, len(labels))
    # Numeric-only NPZ permits cheap train-only experiments without reloading
    # trajectories. It is a generated artifact, not a pickle or a source input.
    np.savez_compressed(output / "training_features.npz", **{f"clip_{i}": table.values for i, table in enumerate(tables)})
    write_json(output / "training_candidates.json", {
        "feature_names": list(tables[0].feature_names),
        "samples": [{"sample_id": label.sample_id,
                     "candidates": [[c.behavior, list(c.target_ids), c.evidence] for c in table.candidates]}
                    for label, table in zip(labels, tables)],
    })
    return tables


def train(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError("choose a new training output directory; existing experiments are preserved")
    labels = load_training_manifest(args.train_manifest)
    source_hashes = {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}
    manifest_hash = sha256(args.train_manifest)
    trajectory_hashes = {label.sample_id: sha256(Path(label.tracks_path)) for label in labels}
    profiles = rule_profiles(config_seed_parameters(load_config(args.config)))
    args.output_dir.mkdir(parents=True)
    if args.feature_cache:
        cache = json.loads((args.feature_cache / "feature_provenance.json").read_text(encoding="utf-8"))
        expected = {"train_manifest_sha256": manifest_hash, "trajectory_sha256": trajectory_hashes,
                    "max_tracks": args.max_tracks, "profiles": [p.as_dict() for p in profiles],
                    "feature_source_sha256": {name: source_hashes[name] for name in FEATURE_SOURCE_FILES}}
        cache_without_hashes = {key: value for key, value in cache.items() if key != "feature_source_sha256"}
        expected_without_hashes = {key: value for key, value in expected.items() if key != "feature_source_sha256"}
        cached_feature_hashes = dict(cache.get("feature_source_sha256", {}))
        if cache_without_hashes != expected_without_hashes or any(
            cached_feature_hashes.get(name) != digest
            for name, digest in expected["feature_source_sha256"].items()
        ):
            raise ValueError("training feature cache provenance does not match")
        metadata = json.loads((args.feature_cache / "training_candidates.json").read_text(encoding="utf-8"))
        with np.load(args.feature_cache / "training_features.npz", allow_pickle=False) as arrays:
            tables = [CandidateTable(tuple(Candidate(c[0], tuple(c[1]), c[2]) for c in sample["candidates"]),
                                     arrays[f"clip_{i}"], tuple(metadata["feature_names"]))
                      for i, sample in enumerate(metadata["samples"])]
        if [s["sample_id"] for s in metadata["samples"]] != [label.sample_id for label in labels]:
            raise ValueError("training cache row order changed")
    else:
        tables = training_tables(labels, profiles, args.output_dir, args.max_tracks)
        write_json(args.output_dir / "feature_provenance.json", {
            "train_manifest_sha256": manifest_hash, "trajectory_sha256": trajectory_hashes,
            "max_tracks": args.max_tracks, "profiles": [p.as_dict() for p in profiles],
            "feature_source_sha256": {name: source_hashes[name] for name in FEATURE_SOURCE_FILES},
        })
    oracle = candidate_oracle_metrics([(label, table.candidates) for label, table in zip(labels, tables)])
    write_json(args.output_dir / "training_candidate_coverage.json", oracle)
    LOGGER.info("training target-behavior candidate coverage %.2f%%", 100 * oracle["target_candidate_rate"])
    search = []
    for regularization in args.regularization:
        oof = []
        for session, fit_indices, check_indices in session_folds(labels):
            model, metrics = fit_ranker([tables[i] for i in fit_indices], [labels[i] for i in fit_indices],
                                       steps=args.steps, regularization=regularization,
                                       learning_rate=args.learning_rate, objective=args.objective,
                                       seed=args.seed)
            oof.extend((labels[i], predict_table(tables[i], model)) for i in check_indices)
            LOGGER.info("train internal fold=%s regularization=%g fitted_joint=%.2f%%",
                        session, regularization, 100 * metrics["joint_top1_accuracy"])
        oof_metrics = evaluate_top1(oof)
        search.append({"regularization": regularization, "training_internal_holdout": oof_metrics})
        LOGGER.info("train internal combined regularization=%g joint=%.2f%% macro=%.2f%%",
                    regularization, 100 * oof_metrics["joint_top1_accuracy"], 100 * oof_metrics["macro_joint_top1_accuracy"])
    selected = max(search, key=lambda item: metric_rank(item["training_internal_holdout"]))
    if args.session_ensemble:
        fold_models = []
        for session, fit_indices, _ in session_folds(labels):
            fold_model, _ = fit_ranker(
                [tables[i] for i in fit_indices], [labels[i] for i in fit_indices],
                steps=args.steps, regularization=selected["regularization"],
                learning_rate=args.learning_rate, objective=args.objective, seed=args.seed,
            )
            fold_models.append(fold_model)
        model = average_ranker_models(fold_models)
        metrics = evaluate_top1(
            [(label, predict_table(table, model)) for label, table in zip(labels, tables)]
        )
    else:
        model, metrics = fit_ranker(
            tables, labels, steps=args.steps, regularization=selected["regularization"],
            learning_rate=args.learning_rate, objective=args.objective, seed=args.seed,
        )
    model.update({"model_type": "calibrated_heuristic_ranker_v2", "rule_profiles": [p.as_dict() for p in profiles],
                  "max_tracks": args.max_tracks, "session_ensemble": bool(args.session_ensemble)})
    write_json(args.output_dir / "frozen_model.json", model)
    write_json(args.output_dir / "training_metrics.json", metrics)
    write_json(args.output_dir / "search_history.json", search)
    _write_predictions(args.output_dir / "train_predictions.csv",
                       [(label, predict_table(table, model)) for label, table in zip(labels, tables)], split="train")
    if sha256(args.train_manifest) != manifest_hash or any(sha256(REPO_ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("training manifest or code changed during fitting; model is not sealed")
    write_json(args.output_dir / "run_provenance.json", {
        "train_manifest_sha256": manifest_hash, "training_trajectory_sha256": trajectory_hashes,
        "train_sample_ids": [label.sample_id for label in labels],
        "train_sessions": sorted({label.recording_session for label in labels}),
        "source_sha256": source_hashes, "model_sha256": sha256(args.output_dir / "frozen_model.json"),
        "validation_used_for_fitting": False, "training_samples": len(labels),
        "steps_per_fit": args.steps, "seed": args.seed,
        "learning_rate": args.learning_rate, "objective": args.objective,
        "session_ensemble": bool(args.session_ensemble),
        "selected_regularization": selected["regularization"],
        "selected_training_internal_holdout": selected["training_internal_holdout"],
    })
    LOGGER.info("FROZEN training joint=%.2f%% output=%s", metrics["joint_top1_accuracy"] * 100, args.output_dir)


def validate(args) -> None:
    provenance = json.loads((args.model_dir / "run_provenance.json").read_text(encoding="utf-8"))
    model_path = args.model_dir / "frozen_model.json"
    if sha256(model_path) != provenance["model_sha256"]:
        raise ValueError("frozen model hash changed")
    for name, digest in provenance["source_sha256"].items():
        if sha256(REPO_ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    if args.output_dir.exists():
        raise FileExistsError("choose a new validation output directory")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    profiles = tuple(HeuristicParameters(**value) for value in model["rule_profiles"])
    manifest_hash = sha256(args.val_manifest)
    # Project the existing label-bearing manifest to trajectory-only inputs.
    # Opaque row numbers, not the behavior-bearing sample names, enter prediction.
    with args.val_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        inputs = []
        original_ids: set[str] = set()
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if row["split"] not in {"val", "validation"}:
                raise ValueError("validation manifest contains a non-validation row")
            original_id = row["sample_id"]
            if not original_id or original_id in original_ids:
                raise ValueError("missing or repeated validation sample ID")
            if original_id in provenance["train_sample_ids"] or row["recording_session"] in provenance["train_sessions"]:
                raise ValueError("original training and validation splits overlap")
            original_ids.add(original_id)
            inputs.append((f"v-{index:06d}", row["tracks_path"], float(row["fps"])))
    if not inputs:
        raise ValueError("empty validation inputs")
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "predictions.csv"
    candidate_path = args.output_dir / "candidate_predictions.csv"
    trajectory_hashes = {}
    candidate_count = 0
    with prediction_path.open("x", encoding="utf-8-sig", newline="") as handle, candidate_path.open(
        "x", encoding="utf-8-sig", newline=""
    ) as candidate_handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "predicted_behavior", "predicted_target_ids", "score"))
        writer.writeheader()
        candidate_writer = csv.DictWriter(
            candidate_handle,
            fieldnames=("sample_id", "candidate_behavior", "candidate_target_ids", "evidence"),
        )
        candidate_writer.writeheader()
        for index, (sample_id, tracks_path, fps) in enumerate(inputs):
            digest = sha256(Path(tracks_path))
            if digest in set(provenance["training_trajectory_sha256"].values()):
                raise ValueError("validation trajectory duplicates a training trajectory")
            trajectory_hashes[sample_id] = digest
            features = load_exported_features(tracks_path, fps=fps, max_tracks=model["max_tracks"])
            table = build_candidate_table(features, profiles)
            prediction = predict_table(table, model)
            writer.writerow({"sample_id": sample_id, "predicted_behavior": prediction.behavior or "",
                             "predicted_target_ids": ",".join(map(str, prediction.target_ids)), "score": prediction.score})
            for candidate in table.candidates:
                candidate_writer.writerow({
                    "sample_id": sample_id,
                    "candidate_behavior": candidate.behavior,
                    "candidate_target_ids": ",".join(map(str, candidate.target_ids)),
                    "evidence": candidate.evidence,
                })
                candidate_count += 1
            if (index + 1) % 25 == 0 or index + 1 == len(inputs):
                LOGGER.info("validation trajectory-only prediction %d/%d", index + 1, len(inputs))
    write_json(args.output_dir / "prediction_receipt.json", {
        "prediction_count": len(inputs), "predictions_sha256": sha256(prediction_path),
        "candidate_count": candidate_count,
        "candidate_predictions_sha256": sha256(candidate_path),
        "model_sha256": provenance["model_sha256"], "validation_manifest_sha256": manifest_hash,
        "validation_trajectory_sha256": trajectory_hashes, "contains_ground_truth": False,
    })
    predictions = load_locked_predictions(args.output_dir)
    receipt = json.loads((args.output_dir / "prediction_receipt.json").read_text(encoding="utf-8"))
    if sha256(candidate_path) != receipt["candidate_predictions_sha256"]:
        raise ValueError("candidate CSV changed after the blind prediction receipt")
    # Scoring is deliberately after the immutable prediction receipt.
    if sha256(args.val_manifest) != manifest_hash or sha256(model_path) != provenance["model_sha256"]:
        raise ValueError("validation manifest/model changed during prediction")
    with args.val_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [LabeledClip(f"v-{index:06d}", row["tracks_path"], float(row["fps"]), row["recording_session"],
                              row["canonical_behavior"], tuple(sorted(int(x) for x in row["mouse_ids"].split(",") if x)))
                  for index, row in enumerate(csv.DictReader(handle), start=1)]
    rows = [(label, predictions[label.sample_id]) for label in labels]
    candidates_by_sample: dict[str, list[Candidate]] = {label.sample_id: [] for label in labels}
    observed_candidate_count = 0
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] not in candidates_by_sample:
                raise ValueError("candidate CSV contains an unknown validation sample ID")
            candidates_by_sample[row["sample_id"]].append(
                Candidate(
                    row["candidate_behavior"],
                    tuple(sorted(int(value) for value in row["candidate_target_ids"].split(",") if value)),
                    float(row["evidence"]),
                )
            )
            observed_candidate_count += 1
    if observed_candidate_count != int(receipt["candidate_count"]):
        raise ValueError("candidate count differs from the blind prediction receipt")
    layer_diagnostics = target_layer_candidate_diagnostics(
        [(label, candidates_by_sample[label.sample_id]) for label in labels]
    )
    write_json(args.output_dir / "target_layer_candidate_diagnostics.json", layer_diagnostics)
    metrics = evaluate_top1(rows)
    write_json(args.output_dir / "validation_metrics.json", metrics)
    _write_predictions(args.output_dir / "validation_predictions_scored.csv", rows, split="val")
    training = json.loads((args.model_dir / "training_metrics.json").read_text(encoding="utf-8"))
    lines = ["# 原划分：训练调参后验证结果", "",
             "主指标：Top-1 行为正确，且预测包含标签指定的全部目标 ID；群体行为额外 ID 仅作为精确 ID 集合辅助指标。验证标签不参与候选生成、排序或参数选择；预测回执锁定后才用于评分。",
             "这是既有验证集评估，不宣称为全新未见过的独立盲测。", "",
             f"训练：{training['joint_top1_accuracy']:.2%}；验证：{metrics['joint_top1_accuracy']:.2%}；验证宏平均：{metrics['macro_joint_top1_accuracy']:.2%}。", "",
             "| 行为 | 训练正确/数量 | 训练准确率 | 验证正确/数量 | 验证准确率 | 验证仅行为正确 | 验证精确 ID 集合率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for behavior, value in metrics["per_behavior"].items():
        t = training["per_behavior"][behavior]
        lines.append(f"| {behavior} | {t['joint_correct']}/{t['support']} | {t['joint_top1_accuracy']:.2%} | {value['joint_correct']}/{value['support']} | {value['joint_top1_accuracy']:.2%} | {value['behavior_top1_accuracy']:.2%} | {value['exact_id_set_accuracy']:.2%} |")
    lines.extend(["", "仅行为正确是辅助指标，不能替代目标 ID 行为 Top-1。精确 ID 集合率作为额外诊断项。未达到 90% 的类别如实报告，不使用验证标签回调参数。"])
    lines.extend([
        "", "## 目标鼠跨层候选（辅助诊断）", "", layer_diagnostics["scope"], "",
        "| 标签行为 | 目标鼠实例 | 个体运动候选覆盖 | 社交候选覆盖 | 群体候选覆盖 |",
        "|---|---:|---:|---:|---:|",
    ])
    for behavior, value in sorted(layer_diagnostics["per_labeled_behavior"].items()):
        lines.append(
            f"| {behavior} | {value['target_mouse_instances']} | "
            f"{value['individual']['target_mouse_instances_with_candidate']}/{value['target_mouse_instances']} | "
            f"{value['social']['target_mouse_instances_with_candidate']}/{value['target_mouse_instances']} | "
            f"{value['group']['target_mouse_instances_with_candidate']}/{value['target_mouse_instances']} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("VALIDATION joint=%.2f%% macro=%.2f%%", metrics["joint_top1_accuracy"] * 100, metrics["macro_joint_top1_accuracy"] * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--train-manifest", type=Path, required=True)
    training.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--feature-cache", type=Path)
    training.add_argument("--max-tracks", type=int, default=64)
    training.add_argument("--steps", type=int, default=500)
    training.add_argument("--learning-rate", type=float, default=0.03)
    training.add_argument("--objective", choices=("hard_margin", "listwise"), default="hard_margin")
    training.add_argument("--session-ensemble", action="store_true")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--regularization", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    validation = sub.add_parser("validate")
    validation.add_argument("--val-manifest", type=Path, required=True)
    validation.add_argument("--model-dir", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        if (args.steps < 1 or args.max_tracks < 2 or args.learning_rate <= 0
                or any(value < 0 for value in args.regularization)):
            raise ValueError("invalid search bounds")
        train(args)
    else:
        validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
