#!/usr/bin/env python3
"""Tune joint behavior/visual-ID Top-1 using an explicit train.csv only.

No dataset-root discovery, validation manifest, or test path is accepted.  The
previously inspected validation set is not a blind holdout and is never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from _bootstrap import REPO_ROOT
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    Top1Prediction,
    evaluate_top1,
    load_training_manifest,
)
from mouse_behavior.evaluation.exported_dataset import (
    EXPORTED_DATASET_SOURCE_FILES,
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.train_only_tuning import (
    LoadedClip,
    candidate_oracle_metrics,
    evidence_rows,
    fit_rules,
    predict_rows,
    session_folds,
)


LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions(
    path: Path,
    rows: Sequence[tuple[LabeledClip, Top1Prediction]],
    *,
    split: str,
) -> None:
    fields = (
        "split", "sample_id", "recording_session", "true_behavior", "true_target_ids",
        "predicted_behavior", "predicted_target_ids", "score", "joint_top1_correct",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, prediction in rows:
            writer.writerow(
                {
                    "split": split,
                    "sample_id": record.sample_id,
                    "recording_session": record.recording_session,
                    "true_behavior": record.behavior,
                    "true_target_ids": ",".join(map(str, record.target_ids)),
                    "predicted_behavior": prediction.behavior or "",
                    "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                    "score": prediction.score,
                    "joint_top1_correct": (
                        prediction.behavior == record.behavior
                        and tuple(sorted(prediction.target_ids)) == record.target_ids
                    ),
                }
            )


def _report(
    path: Path,
    *,
    train_metrics: dict[str, Any],
    oof_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
    track_id_coverage: dict[str, Any],
    folds: Sequence[dict[str, Any]],
    target: float,
    ready: bool,
) -> None:
    lines = [
        "# Train-only joint Top-1 calibration",
        "",
        "No test or previously inspected validation data was opened by this run. "
        "The search input was the explicit train.csv manifest only.",
        "",
        "A sample is correct only when its predicted behavior **and exact visual-ID set** "
        "match the annotation. RFID animal identities are not inferred from visual track IDs.",
        "",
        f"- In-sample training joint Top-1: {train_metrics['joint_top1_accuracy']:.2%}",
        f"- Session-held-out joint Top-1: {oof_metrics['joint_top1_accuracy']:.2%}",
        f"- Session-held-out macro joint Top-1: {oof_metrics['macro_joint_top1_accuracy']:.2%}",
        f"- Fixed-rule labeled-target candidate coverage on training: {oracle_metrics['target_candidate_rate']:.2%}",
        f"- Annotated IDs present among selected visual tracks: {track_id_coverage['rate']:.2%}",
        f"- Readiness gate (both training and session-held-out >= {target:.0%}): {'PASS' if ready else 'FAIL'}",
        "- The old validation set is not a pristine blind test. No final test accuracy is reported here.",
        "- Only three training sessions are available; behavior coverage and generalization estimates are limited.",
        "",
        "## Recording-session holdouts",
        "",
        "| Held-out recording | Samples | Joint Top-1 | Macro joint Top-1 |",
        "|---|---:|---:|---:|",
    ]
    for fold in folds:
        metrics = fold["metrics"]
        lines.append(
            f"| {fold['session']} | {metrics['n_samples']} | "
            f"{metrics['joint_top1_accuracy']:.2%} | {metrics['macro_joint_top1_accuracy']:.2%} |"
        )
    for name, metrics in (("In-sample training", train_metrics), ("Session-held-out", oof_metrics)):
        lines.extend(
            [
                "",
                f"## {name} per behavior",
                "",
                "| Behavior | Samples | Joint Top-1 | Behavior only | IDs only |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for behavior, item in metrics["per_behavior"].items():
            lines.append(
                f"| {behavior} | {item['support']} | {item['joint_top1_accuracy']:.2%} | "
                f"{item['behavior_top1_accuracy']:.2%} | {item['id_top1_accuracy']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Fixed-rule candidate ceiling on the labeled training set",
            "",
            "| Behavior | Samples | Exact behavior+ID candidate rate |",
            "|---|---:|---:|",
        ]
    )
    for behavior, item in oracle_metrics["per_behavior"].items():
        lines.append(
            f"| {behavior} | {item['support']} | {item['target_candidate_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The session-held-out score is an internal development estimate. "
            "With only three recordings, repeated method changes can still overfit it; "
            "a new independent blind set is needed after the algorithm is frozen.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "train_only_top1")
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--max-rule-trials", type=int, default=24)
    parser.add_argument("--target-train-accuracy", type=float, default=0.95)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.max_tracks < 2 or args.max_rule_trials < 0:
        raise ValueError("max-tracks must be >= 2 and max-rule-trials must be >= 0")
    if not 0 < args.target_train_accuracy <= 1:
        raise ValueError("target-train-accuracy must be in (0, 1]")
    source_files = [
        *(REPO_ROOT / path for path in EXPORTED_DATASET_SOURCE_FILES),
        REPO_ROOT / "src/mouse_behavior/evaluation/blind_top1.py",
        REPO_ROOT / "src/mouse_behavior/evaluation/train_only_tuning.py",
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/predict_exported_blind.py",
        REPO_ROOT / "scripts/score_exported_blind.py",
    ]
    manifest_hash = _sha256(args.train_manifest)
    config_hash = _sha256(args.config)
    source_hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files}
    labels = load_training_manifest(args.train_manifest)
    folds = session_folds(labels)
    LOGGER.info("train-only rows=%d recording_sessions=%d", len(labels), len(folds))
    loaded: list[LoadedClip] = []
    for index, label in enumerate(labels, start=1):
        loaded.append(
            LoadedClip(
                label,
                load_exported_features(label.tracks_path, fps=label.fps, max_tracks=args.max_tracks),
            )
        )
        if index % 100 == 0 or index == len(labels):
            LOGGER.info("loaded train-only trajectories %d/%d", index, len(labels))
    seed = config_seed_parameters(load_config(args.config))
    oof_rows: list[tuple[LabeledClip, Top1Prediction]] = []
    fold_results = []
    for session, train_indices, held_out_indices in folds:
        LOGGER.info("fitting fold held_out=%s train=%d held_out=%d", session, len(train_indices), len(held_out_indices))
        fitted = fit_rules(
            [loaded[index] for index in train_indices],
            seed=seed,
            max_rule_trials=args.max_rule_trials,
        )
        held_out_rows = predict_rows(
            evidence_rows([loaded[index] for index in held_out_indices], fitted.parameters),
            fitted.weights,
        )
        metrics = evaluate_top1(held_out_rows)
        fold_results.append(
            {"session": session, "train_count": len(train_indices), "metrics": metrics,
             "parameters": fitted.parameters.as_dict(), "weights": fitted.weights}
        )
        oof_rows.extend(held_out_rows)
        LOGGER.info("held_out=%s joint=%.4f macro=%.4f", session,
                    metrics["joint_top1_accuracy"], metrics["macro_joint_top1_accuracy"])
    oof_metrics = evaluate_top1(oof_rows)
    LOGGER.info("fitting all %d training samples", len(loaded))
    final = fit_rules(loaded, seed=seed, max_rule_trials=args.max_rule_trials)
    final_evidence = evidence_rows(loaded, final.parameters)
    train_rows = predict_rows(final_evidence, final.weights)
    train_metrics = evaluate_top1(train_rows)
    oracle_metrics = candidate_oracle_metrics(final_evidence)
    track_id_present = sum(
        set(item.label.target_ids).issubset(set(item.features.track_ids.tolist()))
        for item in loaded
    )
    track_id_coverage = {"count": track_id_present, "total": len(loaded),
                         "rate": track_id_present / len(loaded)}
    target = args.target_train_accuracy
    ready = (train_metrics["joint_top1_accuracy"] >= target and
             oof_metrics["joint_top1_accuracy"] >= target)
    if (_sha256(args.train_manifest) != manifest_hash or _sha256(args.config) != config_hash
            or any(_sha256(path) != source_hashes[str(path.relative_to(REPO_ROOT))]
                   for path in source_files)):
        raise RuntimeError("training input or source changed during calibration; discard this run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "training_metrics.json", train_metrics)
    _write_json(args.output_dir / "session_holdout_metrics.json", oof_metrics)
    _write_json(args.output_dir / "candidate_oracle_metrics.json", oracle_metrics)
    _write_json(args.output_dir / "target_track_id_coverage.json", track_id_coverage)
    _write_json(args.output_dir / "folds.json", fold_results)
    _write_json(args.output_dir / "search_history.json", final.search_history)
    _write_predictions(args.output_dir / "train_predictions.csv", train_rows, split="train")
    _write_predictions(args.output_dir / "session_holdout_predictions.csv", oof_rows, split="train-held-out")
    parameters_name = "frozen_model.json" if ready else "provisional_model.json"
    _write_json(
        args.output_dir / parameters_name,
        {"heuristic_parameters": final.parameters.as_dict(), "ranking_weights": final.weights},
    )
    _report(
        args.output_dir / "report.md", train_metrics=train_metrics, oof_metrics=oof_metrics,
        oracle_metrics=oracle_metrics, track_id_coverage=track_id_coverage,
        folds=fold_results, target=target, ready=ready,
    )
    _write_json(
        args.output_dir / "run_provenance.json",
        {
            "train_manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "model_file": parameters_name,
            "model_sha256": _sha256(args.output_dir / parameters_name),
            "source_sha256": source_hashes,
            "test_data_accessed": False,
            "ready_for_blind": ready,
            "target_train_accuracy": target,
            "max_rule_trials_per_fit": args.max_rule_trials,
            "max_tracks": args.max_tracks,
        },
    )
    LOGGER.info("train joint=%.4f held-out joint=%.4f ready=%s output=%s",
                train_metrics["joint_top1_accuracy"], oof_metrics["joint_top1_accuracy"],
                ready, args.output_dir.resolve())
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
