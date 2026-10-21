#!/usr/bin/env python3
"""Train and validate only the approved group-behavior heuristics.

The frozen recording-session split is filtered to the five group labels. Group
ranker parameters are fitted on approved training clips only. Optional explicit
validation-derived isolation calibration is disclosed as diagnostic reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    Top1Prediction,
    evaluate_top1,
    load_locked_predictions,
)
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable,
    build_candidate_table,
    fit_ranker,
    metric_rank,
    predict_table,
)
from mouse_behavior.evaluation.confirmed_dataset import file_sha256
from mouse_behavior.evaluation.current_validation import resolved_input_path
from mouse_behavior.evaluation.group_calibration import calibrate_isolation, rethreshold_isolation
from mouse_behavior.evaluation.group_event_rules import isolation_summaries
from mouse_behavior.evaluation.group_scoring import evaluate_policy, target_ids_match
from mouse_behavior.evaluation.confirmed_rules import confirmed_profiles, policy_receipt
from mouse_behavior.evaluation.exported_dataset import (
    EXPORTED_DATASET_SOURCE_FILES,
    GROUP_BEHAVIORS,
    LABEL_TO_CANONICAL,
    HeuristicParameters,
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.train_only_tuning import candidate_oracle_metrics, session_folds
from mouse_behavior.evaluation.scale_migration import rescale_frozen_source_scale


REQUIRED_SAMPLE_FILES = ("annotation.json", "metadata.json", "tracks.json", "clip.mp4", "review_receipt.json")
CODE_FILES = tuple(dict.fromkeys((
    "scripts/validate_group_heuristics.py",
    *EXPORTED_DATASET_SOURCE_FILES,
    "src/mouse_behavior/evaluation/blind_top1.py",
    "src/mouse_behavior/evaluation/calibrated_rule_ranker.py",
    "src/mouse_behavior/evaluation/confirmed_dataset.py",
    "src/mouse_behavior/evaluation/current_validation.py",
    "src/mouse_behavior/evaluation/confirmed_rules.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v17.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v18.py",
    "src/mouse_behavior/evaluation/event_frames.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v19.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v20.py",
    "src/mouse_behavior/evaluation/group_event_rules.py",
    "src/mouse_behavior/evaluation/group_heuristic_config.py",
    "src/mouse_behavior/evaluation/group_scoring.py",
    "src/mouse_behavior/evaluation/scale_migration.py",
    "src/mouse_behavior/evaluation/group_calibration.py",
    "src/mouse_behavior/evaluation/train_only_tuning.py",
    "src/mouse_behavior/evaluation/temporal_screen.py",
    "src/mouse_behavior/evaluation/screening_signals.py",
    "configs/default.yaml",
    "configs/scale/rig_a.json",
)))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _read_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty {split} manifest: {path}")
    expected = {"train"} if split == "train" else {"val", "validation"}
    if any(row.get("split") not in expected for row in rows):
        raise ValueError(f"unexpected split value in {path}")
    return rows


def _group_rows(rows: list[dict[str, str]], decisions: dict[str, Any]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if row.get("canonical_behavior") not in GROUP_BEHAVIORS:
            continue
        decision = decisions.get(row["sample_id"], {})
        if decision.get("decision") not in {"approved", "rejected"}:
            raise ValueError(f"group sample has no terminal manual decision: {row['sample_id']}")
        if decision["decision"] == "approved":
            output.append(row)
    return output


def _verify_latest_approvals(
    manifest_path: Path,
    decisions: dict[str, Any],
    split_rows: list[dict[str, str]],
) -> dict[str, Any]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("approved dataset manifest must be a list")
    latest = [
        row for row in entries
        if row.get("decision") == "approved"
        and LABEL_TO_CANONICAL.get(row.get("behavior")) in GROUP_BEHAVIORS
    ]
    by_id: dict[str, dict[str, Any]] = {}
    checked_files = 0
    for row in latest:
        sample_id = str(row.get("review_id", ""))
        if not sample_id or sample_id in by_id:
            raise ValueError(f"missing or duplicate approved group review ID: {sample_id!r}")
        by_id[sample_id] = row
        decision = decisions.get(sample_id, {})
        expected_ids = sorted(int(value) for value in row.get("mouse_ids", []))
        current_ids = sorted(int(value) for value in decision.get("mouse_ids", []))
        if (decision.get("decision") != "approved"
                or decision.get("behavior") != row.get("behavior")
                or current_ids != expected_ids):
            raise ValueError(f"latest manifest and decision disagree: {sample_id}")
        hashes = row.get("dataset_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"approved group sample has no data hashes: {sample_id}")
        sample_dir = Path(row["dataset_path"]).resolve()
        for name in REQUIRED_SAMPLE_FILES:
            path = sample_dir / name
            if not path.is_file() or file_sha256(path) != hashes.get(name):
                raise ValueError(f"latest approved sample file hash differs: {path}")
            checked_files += 1
        annotation = json.loads((sample_dir / "annotation.json").read_text(encoding="utf-8"))
        if (annotation.get("behavior") != row["behavior"]
                or sorted(int(value) for value in annotation.get("mouse_ids", [])) != expected_ids
                or annotation.get("manual_review_id") != sample_id
                or annotation.get("requires_review") is not False):
            raise ValueError(f"approved group annotation differs from manifest: {sample_id}")

    split_by_id = {str(row["sample_id"]): row for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("group split contains duplicate sample IDs")
    if set(split_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(split_by_id))[:8]
        extra = sorted(set(split_by_id) - set(by_id))[:8]
        raise ValueError(f"frozen split differs from latest approved groups: missing={missing}, extra={extra}")
    for sample_id, row in split_by_id.items():
        approved = by_id[sample_id]
        ids = sorted(int(value) for value in row["mouse_ids"].split(",") if value.strip())
        if (row["canonical_behavior"] != LABEL_TO_CANONICAL[approved["behavior"]]
                or ids != sorted(int(value) for value in approved["mouse_ids"])):
            raise ValueError(f"group split label/visual IDs differ from approval: {sample_id}")
    return {
        "approved_manifest_sha256": file_sha256(manifest_path),
        "approved_group_samples": len(by_id),
        "group_split_samples": len(split_by_id),
        "split_matches_latest_approved_group_ids_and_labels": True,
        "approval_decision_mismatches": 0,
        "approved_sample_file_hash_checks": checked_files,
        "approved_sample_file_hash_mismatches": 0,
        "support": dict(sorted(Counter(LABEL_TO_CANONICAL[row["behavior"]] for row in latest).items())),
    }


def _verified_fixed_scale(
    row: dict[str, str], data_root: Path = REPO_ROOT,
    active_rig_path: Path | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    path = resolved_input_path(row["scale_receipt_path"], data_root)
    receipt_hash = file_sha256(path)
    if receipt_hash != row["scale_receipt_sha256"]:
        raise ValueError(f"scale receipt hash differs: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tracks = resolved_input_path(row["tracks_path"], data_root)
    tracks_hash = file_sha256(tracks)
    if tracks_hash not in receipt.get("crop_tracks_sha256", []):
        raise ValueError(f"scale receipt belongs to another clip: {tracks}")
    source = resolved_input_path(receipt["source_tracks_path"], data_root)
    source_hash = file_sha256(source)
    if source_hash != receipt["source_tracks_sha256"]:
        raise ValueError(f"source tracks changed after scale freeze: {source}")
    scale = dict(receipt["scale_config"])
    if (scale.get("mode") != "fixed" or not math.isfinite(float(scale.get("cm_per_pixel", 0)))
            or float(scale["cm_per_pixel"]) <= 0):
        raise ValueError(f"no fixed positive scale in receipt: {path}")
    config_drift = []
    active_rig = (Path(active_rig_path).resolve() if active_rig_path is not None else
                  (REPO_ROOT / "configs" / "scale" / "rig_a.json").resolve())
    hashes = {str(path): receipt_hash, str(tracks): tracks_hash, str(source): source_hash}
    for asset_text, expected_hash in receipt["scale_asset_sha256"].items():
        asset = resolved_input_path(asset_text, data_root)
        actual_hash = file_sha256(asset)
        hashes[str(asset)] = actual_hash
        if actual_hash == expected_hash:
            continue
        if (asset.resolve() == (data_root / "configs/default.yaml").resolve()
                and scale["mode"] == "fixed"):
            config_drift.append(asset_text)
        elif (asset.resolve() == (data_root / "configs/scale/rig_a.json").resolve()
              and scale["mode"] == "fixed"):
            scale, migration = rescale_frozen_source_scale(scale, active_rig)
            config_drift.append(
                f"{asset_text} migrated by {migration['factor']:.12g} using {active_rig}"
            )
        else:
            raise ValueError(f"frozen scale asset changed: {asset}")
    return scale, config_drift, hashes


def _labels(rows: list[dict[str, str]], expected_split: str) -> list[LabeledClip]:
    labels = []
    for index, row in enumerate(rows, start=1):
        ids = tuple(sorted({int(value) for value in row["mouse_ids"].split(",") if value.strip()}))
        behavior = row["canonical_behavior"]
        if behavior not in GROUP_BEHAVIORS or row["split"] not in ({"train"} if expected_split == "train" else {"val", "validation"}):
            raise ValueError(f"unexpected group label/split at row {index + 1}")
        if behavior == "isolation":
            valid_count = len(ids) == 1
        else:
            valid_count = len(ids) >= 3
        if not valid_count:
            raise ValueError(f"invalid exact visual-ID set for {behavior}: {row['sample_id']}")
        labels.append(LabeledClip(
            sample_id=row["sample_id"], tracks_path=row["tracks_path"], fps=float(row["fps"]),
            recording_session=row["recording_session"], behavior=behavior, target_ids=ids,
            scale_receipt_path=row["scale_receipt_path"],
            scale_receipt_sha256=row["scale_receipt_sha256"],
        ))
    return labels


def _feature_table_one(label: LabeledClip, profiles: tuple[HeuristicParameters, ...],
                       max_tracks: int, data_root: Path) -> tuple[CandidateTable, str, list[str], dict, dict[str, str]]:
    row = {
        "scale_receipt_path": label.scale_receipt_path,
        "scale_receipt_sha256": label.scale_receipt_sha256,
        "tracks_path": label.tracks_path,
    }
    scale, drift, scale_hashes = _verified_fixed_scale(
        row, data_root, REPO_ROOT / "configs" / "scale" / "rig_a.json",
    )
    path = resolved_input_path(label.tracks_path, data_root)
    trajectory_hash = file_sha256(path)
    features = load_exported_features(path, fps=label.fps, max_tracks=max_tracks, scale_config=scale)
    table = build_candidate_table(features, profiles, behavior_subset=GROUP_BEHAVIORS)
    return table, trajectory_hash, drift, {"fps": features.fps, "duration_s": len(features.valid) / features.fps,
                                          "tracks": isolation_summaries(features)}, scale_hashes


def _feature_tables(labels: list[LabeledClip], profiles: tuple[HeuristicParameters, ...],
                    max_tracks: int, workers: int, data_root: Path):
    tables: list[CandidateTable | None] = [None] * len(labels)
    trajectory_hashes = {}
    summaries = {}
    scale_drift: set[str] = set()
    scale_hashes: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_feature_table_one, label, profiles, max_tracks, data_root): index
            for index, label in enumerate(labels)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            table, digest, drift, summary, hashed_inputs = future.result()
            summaries[labels[index].sample_id] = summary
            label = labels[index]
            tables[index] = table
            trajectory_hashes[label.sample_id] = digest
            scale_drift.update(drift)
            scale_hashes.update(hashed_inputs)
            if completed % 5 == 0 or completed == len(labels):
                print(f"group feature extraction {completed}/{len(labels)}", flush=True)
    if any(table is None for table in tables):
        raise RuntimeError("group feature extraction ended without all clip tables")
    return ([table for table in tables if table is not None], trajectory_hashes,
            sorted(scale_drift), summaries, scale_hashes)


def _precision_recall(rows: list[tuple[LabeledClip, Top1Prediction]], target_id_policy: str = "exact") -> dict[str, Any]:
    matrix: Counter[tuple[str, str]] = Counter()
    result = {}
    for behavior in GROUP_BEHAVIORS:
        support = sum(label.behavior == behavior for label, _ in rows)
        predicted = sum(prediction.behavior == behavior for _, prediction in rows)
        behavior_tp = sum(
            label.behavior == behavior and prediction.behavior == behavior
            for label, prediction in rows
        )
        joint_tp = sum(
            label.behavior == behavior and prediction.behavior == behavior
            and tuple(sorted(label.target_ids)) == tuple(sorted(prediction.target_ids))
            for label, prediction in rows
        )
        requested_tp = sum(label.behavior == behavior and prediction.behavior == behavior
                           and target_ids_match(behavior, label.target_ids, prediction.target_ids, target_id_policy)
                           for label, prediction in rows)
        result[behavior] = {
            "requested_joint_tp": requested_tp,
            "requested_joint_precision": requested_tp / predicted if predicted else None,
            "requested_joint_recall": requested_tp / support if support else None,
            "support": support,
            "predicted": predicted,
            "behavior_tp": behavior_tp,
            "behavior_precision": behavior_tp / predicted if predicted else None,
            "behavior_recall": behavior_tp / support if support else None,
            "exact_behavior_and_visual_id_tp": joint_tp,
            "exact_joint_precision": joint_tp / predicted if predicted else None,
            "exact_joint_recall": joint_tp / support if support else None,
        }
    for label, prediction in rows:
        matrix[(label.behavior, prediction.behavior or "unrecognized")] += 1
    supports = sum(item["support"] for item in result.values())
    predicted_count = sum(item["predicted"] for item in result.values())
    behavior_tp = sum(item["behavior_tp"] for item in result.values())
    joint_tp = sum(item["exact_behavior_and_visual_id_tp"] for item in result.values())
    return {
        "target_id_policy": target_id_policy,
        "requested_joint_micro_precision": sum(item["requested_joint_tp"] for item in result.values()) / predicted_count if predicted_count else None,
        "requested_joint_micro_recall": sum(item["requested_joint_tp"] for item in result.values()) / supports if supports else None,
        "n_validation_clips": len(rows),
        "behavior_micro_precision": behavior_tp / predicted_count if predicted_count else None,
        "behavior_micro_recall": behavior_tp / supports if supports else None,
        "exact_joint_micro_precision": joint_tp / predicted_count if predicted_count else None,
        "exact_joint_micro_recall": joint_tp / supports if supports else None,
        "per_behavior": result,
        "confusion": {
            truth: {
                predicted: matrix[(truth, predicted)]
                for predicted in (*GROUP_BEHAVIORS, "unrecognized")
            }
            for truth in GROUP_BEHAVIORS
        },
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError("choose a new group-validation output directory; prior runs are immutable")
    train_all = _read_rows(args.train_manifest, "train")
    val_all = _read_rows(args.val_manifest, "val")
    decisions_data = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = decisions_data["decisions"]
    train_rows = _group_rows(train_all, decisions)
    val_rows = _group_rows(val_all, decisions)
    group_rows = [*train_rows, *val_rows]
    latest_audit = _verify_latest_approvals(args.approved_manifest, decisions, group_rows)
    if not train_rows or not val_rows:
        raise ValueError("approved group labels are required on both train and validation sides")
    train_sessions = {row["recording_session"] for row in train_rows}
    val_sessions = {row["recording_session"] for row in val_rows}
    if train_sessions & val_sessions:
        raise ValueError("group train and validation recording sessions overlap")
    train_labels, val_labels = _labels(train_rows, "train"), _labels(val_rows, "val")
    for behavior in GROUP_BEHAVIORS:
        if not any(label.behavior == behavior for label in train_labels):
            raise ValueError(f"group training side has no approved {behavior} examples")

    args.output_dir.mkdir(parents=True)
    out_train = args.output_dir / "group_train.csv"
    out_val = args.output_dir / "group_validation.csv"
    _write_csv(out_train, tuple(train_rows[0]), train_rows)
    _write_csv(out_val, tuple(val_rows[0]), val_rows)
    source_hashes = {name: file_sha256(REPO_ROOT / name) for name in CODE_FILES}
    input_hashes = {
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.val_manifest),
        "group_train_manifest_sha256": file_sha256(out_train),
        "group_validation_manifest_sha256": file_sha256(out_val),
        "decisions_sha256": file_sha256(args.decisions),
        "approved_dataset_manifest_sha256": latest_audit["approved_manifest_sha256"],
        "config_sha256": file_sha256(args.config),
        "data_root": str(args.data_root),
        "frozen_scale_receipts_modified": False,
        "physical_scale_migration": {
            "factor": 7.918 / 10.7,
            "arena_cm": [74.0, 74.0],
            "body_length_cm": 7.918,
            "body_width_cm": 3.068558,
            "source_receipts_modified": False,
        },
    }
    config = load_config(args.config)
    profiles = (confirmed_profiles(config_seed_parameters(config))[0],)
    train_tables, train_track_hashes, train_scale_drift, train_summaries, train_scale_hashes = _feature_tables(
        train_labels, profiles, args.max_tracks, args.workers, args.data_root
    )
    val_tables, val_track_hashes, val_scale_drift, val_summaries, val_scale_hashes = _feature_tables(
        val_labels, profiles, args.max_tracks, args.workers, args.data_root
    )
    frozen_scale_hashes = {**train_scale_hashes, **val_scale_hashes}
    input_hashes["frozen_scale_input_sha256"] = frozen_scale_hashes

    isolation_audit = {"validation_labels_used": False, "selected": {"distance_cm": profiles[0].isolation_distance_cm}}
    if args.isolation_calibration == "validation":
        isolation_audit = calibrate_isolation(val_labels, val_summaries)
        threshold = isolation_audit["selected"]["distance_cm"]
        profiles = tuple(replace(profile, isolation_distance_cm=threshold) for profile in profiles)
        train_tables = [rethreshold_isolation(table, train_summaries[label.sample_id], threshold)
                        for table, label in zip(train_tables, train_labels)]
        val_tables = [rethreshold_isolation(table, val_summaries[label.sample_id], threshold)
                      for table, label in zip(val_tables, val_labels)]
    _write_json(args.output_dir / "isolation_distance_calibration.json", isolation_audit)
    _write_json(args.output_dir / "trajectory_group_summaries.json", {**train_summaries, **val_summaries})

    search = []
    for regularization in args.regularization:
        oof = []
        for session, fit_indices, check_indices in session_folds(train_labels):
            model, _ = fit_ranker(
                [train_tables[index] for index in fit_indices],
                [train_labels[index] for index in fit_indices],
                steps=args.steps, regularization=regularization,
                learning_rate=args.learning_rate, seed=args.seed,
                objective="hard_margin", require_positive_evidence=True, target_id_policy=args.target_id_policy,
            )
            oof.extend(
                (train_labels[index], predict_table(train_tables[index], model))
                for index in check_indices
            )
        search.append({
            "regularization": regularization,
            "held_out_training_sessions": len(session_folds(train_labels)),
            "out_of_fold_metrics": evaluate_policy(oof, args.target_id_policy),
        })
    selected = max(search, key=lambda row: metric_rank(row["out_of_fold_metrics"]))
    model, training_metrics = fit_ranker(
        train_tables, train_labels, steps=args.steps,
        regularization=selected["regularization"], learning_rate=args.learning_rate,
        seed=args.seed, objective="hard_margin", require_positive_evidence=True, target_id_policy=args.target_id_policy,
    )
    model.update({
        "model_type": "confirmed_group_only_ranker_v1",
        "rule_profiles": [profile.as_dict() for profile in profiles],
        "max_tracks": args.max_tracks,
        "behavior_subset": list(GROUP_BEHAVIORS),
        "selected_regularization": selected["regularization"],
        "rule_policy": {**policy_receipt(), "group_rules": {
            **policy_receipt()["group_rules"], "isolation_distance_cm": profiles[0].isolation_distance_cm}},
        "validation_used_for_isolation_calibration": isolation_audit["validation_labels_used"],
    })
    model_path = args.output_dir / "frozen_group_model.json"
    _write_json(model_path, model)
    _write_json(args.output_dir / "training_metrics.json", training_metrics)
    _write_json(args.output_dir / "training_session_search.json", search)
    _write_json(args.output_dir / "training_candidate_coverage.json", candidate_oracle_metrics(
        list(zip(train_labels, [table.candidates for table in train_tables])),
        require_positive_evidence=True,
    ))

    prediction_path = args.output_dir / "predictions.csv"
    predictions: list[tuple[LabeledClip, Top1Prediction]] = []
    for index, (label, table) in enumerate(zip(val_labels, val_tables), start=1):
        predictions.append((label, predict_table(table, model)))
        if index % 25 == 0 or index == len(val_labels):
            print(f"group validation prediction {index}/{len(val_labels)}", flush=True)
    _write_csv(prediction_path, ("sample_id", "predicted_behavior", "predicted_target_ids", "score"), [
        {"sample_id": label.sample_id, "predicted_behavior": prediction.behavior or "",
         "predicted_target_ids": ",".join(map(str, prediction.target_ids)), "score": prediction.score}
        for label, prediction in predictions
    ])
    prediction_receipt = {
        **input_hashes,
        "prediction_count": len(predictions),
        "predictions_sha256": file_sha256(prediction_path),
        "model_sha256": file_sha256(model_path),
        "validation_trajectory_sha256": val_track_hashes,
        "contains_ground_truth": False,
    }
    _write_json(args.output_dir / "prediction_receipt.json", prediction_receipt)
    locked = load_locked_predictions(args.output_dir)
    scored_rows = [(label, locked[label.sample_id]) for label in val_labels]
    metrics = _precision_recall(scored_rows, args.target_id_policy)
    metrics["requested_top1"] = evaluate_policy(scored_rows, args.target_id_policy)
    metrics["strict_top1"] = evaluate_top1(scored_rows)
    metrics["validation_candidate_coverage"] = candidate_oracle_metrics(
        list(zip(val_labels, [table.candidates for table in val_tables])),
        require_positive_evidence=True,
    )
    metrics["training_candidate_coverage"] = json.loads(
        (args.output_dir / "training_candidate_coverage.json").read_text(encoding="utf-8")
    )
    metrics["requested_candidate_coverage"] = {
        behavior: sum(label.behavior == behavior and any(c.evidence > 0 and c.behavior == behavior
                      and target_ids_match(behavior, label.target_ids, c.target_ids, args.target_id_policy)
                      for c in table.candidates) for label, table in zip(val_labels, val_tables))
        for behavior in GROUP_BEHAVIORS}
    _write_json(args.output_dir / "group_validation_metrics.json", metrics)
    _write_csv(args.output_dir / "scored_predictions.csv", (
        "sample_id", "true_behavior", "true_target_ids", "predicted_behavior",
        "predicted_target_ids", "behavior_correct", "exact_target_ids_correct", "strict_joint_correct", "requested_joint_correct",
    ), [
        {"sample_id": label.sample_id, "true_behavior": label.behavior,
         "true_target_ids": ",".join(map(str, label.target_ids)),
         "predicted_behavior": prediction.behavior or "",
         "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
         "behavior_correct": prediction.behavior == label.behavior,
         "exact_target_ids_correct": tuple(sorted(prediction.target_ids)) == tuple(sorted(label.target_ids)),
         "requested_joint_correct": prediction.behavior == label.behavior and target_ids_match(label.behavior, label.target_ids, prediction.target_ids, args.target_id_policy),
         "strict_joint_correct": (prediction.behavior == label.behavior
                                  and tuple(sorted(prediction.target_ids)) == tuple(sorted(label.target_ids)))}
        for label, prediction in scored_rows
    ])
    group_support_train = dict(sorted(Counter(label.behavior for label in train_labels).items()))
    group_support_val = dict(sorted(Counter(label.behavior for label in val_labels).items()))
    report = [
        "# 群体行为启发式验证",
        "",
        "本次只训练和评估 Huddling、Isolation、Social clustering、Group locomotion、Dispersal。候选类别中没有个体或双鼠社交行为。",
        "",
        f"群体训练支持数：`{json.dumps(group_support_train, ensure_ascii=False, sort_keys=True)}`。",
        f"群体验证支持数：`{json.dumps(group_support_val, ensure_ascii=False, sort_keys=True)}`。",
        f"训练会话留一 OOF 选择正则化：`{selected['regularization']}`；排序器正则化只用训练会话选择；孤立距离校准使用验证标签 `{isolation_audit['validation_labels_used']}`，阈值 `{profiles[0].isolation_distance_cm}` cm。群体严格 Top-1 准确率 `{metrics['strict_top1']['joint_top1_accuracy']:.4f}`。",
        "",
        "| 行为 | 验证支持 | 预测数 | 请求口径 Precision | 请求口径 Recall | 行为+完整视觉 ID Precision | 行为+完整视觉 ID Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for behavior in GROUP_BEHAVIORS:
        row = metrics["per_behavior"][behavior]
        def fmt_metric(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4f}"
        report.append(
            f"| {behavior} | {row['support']} | {row['predicted']} | {fmt_metric(row['requested_joint_precision'])} | "
            f"{fmt_metric(row['requested_joint_recall'])} | {fmt_metric(row['exact_joint_precision'])} | "
            f"{fmt_metric(row['exact_joint_recall'])} |"
        )
    report.extend([
        "",
        f"请求计分策略：`{args.target_id_policy}`。group_target_superset 只对社会聚集和群体同步运动允许额外 ID，但必须包含全部标注目标 ID；其他三类仍要求完整 ID 集合一致。严格联合 P/R 始终要求完整集合一致。",
        "该结果是在已使用过的固定录制场次验证划分上做的重新评估，不是新独立盲测；各类支持不均衡，尤其 Group locomotion 验证只有 1 段。",
        "尺度取每个已审核样本冻结的 cm/pixel 回执；完整视频未在此脚本中解码，ID 均为视觉轨迹 ID。",
    ])
    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    for name, digest in source_hashes.items():
        if file_sha256(REPO_ROOT / name) != digest:
            raise RuntimeError(f"source changed during group validation: {name}")
    if file_sha256(args.decisions) != input_hashes["decisions_sha256"]:
        raise RuntimeError("manual decisions changed during validation")
    if file_sha256(args.approved_manifest) != input_hashes["approved_dataset_manifest_sha256"]:
        raise RuntimeError("approved dataset manifest changed during validation")
    if file_sha256(args.train_manifest) != input_hashes["train_manifest_sha256"]:
        raise RuntimeError("frozen training manifest changed during validation")
    if file_sha256(args.val_manifest) != input_hashes["validation_manifest_sha256"]:
        raise RuntimeError("frozen validation manifest changed during validation")
    for label in [*train_labels, *val_labels]:
        if file_sha256(resolved_input_path(label.tracks_path, args.data_root)) != (
            train_track_hashes.get(label.sample_id) or val_track_hashes.get(label.sample_id)
        ):
            raise RuntimeError(f"group trajectory changed during validation: {label.sample_id}")
    for path, digest in frozen_scale_hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"group frozen scale input changed during validation: {path}")
    receipt = {
        **prediction_receipt,
        "source_sha256": source_hashes,
        "train_trajectory_sha256": train_track_hashes,
        "model_sha256": file_sha256(model_path),
        "training_metrics_sha256": file_sha256(args.output_dir / "training_metrics.json"),
        "validation_metrics_sha256": file_sha256(args.output_dir / "group_validation_metrics.json"),
        "report_sha256": file_sha256(report_path),
        "latest_approved_group_audit": latest_audit,
        "group_training_support": group_support_train,
        "group_validation_support": group_support_val,
        "training_scale_config_drift": train_scale_drift,
        "validation_scale_config_drift": val_scale_drift,
        "validation_used_for_fitting": isolation_audit["validation_labels_used"],
        "validation_used_for_rule_calibration": isolation_audit["validation_labels_used"],
        "validation_used_for_ranker_fitting": False,
        "target_id_policy": args.target_id_policy,
        "isolation_distance_calibration_sha256": file_sha256(args.output_dir / "isolation_distance_calibration.json"),
        "candidate_behaviors": list(GROUP_BEHAVIORS),
        "validation_independent_blind_test": False,
        "media_decode_checked": False,
    }
    _write_json(args.output_dir / "run_receipt.json", receipt)


def parse_args() -> argparse.Namespace:
    data_root = REPO_ROOT.parent.parent
    default_root = data_root / "outputs" / "confirmed_calibration_v1_20261010"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=default_root / "dataset_v2" / "train.csv")
    parser.add_argument("--val-manifest", type=Path, default=default_root / "dataset_v2" / "val.csv")
    parser.add_argument("--decisions", type=Path, default=data_root / "outputs" / "behavior_confirmed_dataset" / "decisions.json")
    parser.add_argument("--approved-manifest", type=Path,
                        default=data_root / "outputs" / "behavior_confirmed_dataset" / "dataset_manifest.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=data_root,
                        help="Read-only root for inherited data and frozen scale receipt paths.")
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--isolation-calibration", choices=["fixed", "validation"], default="fixed")
    parser.add_argument("--target-id-policy", choices=["exact", "social_clustering_superset", "group_target_superset"], default="exact")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("data_root", "train_manifest", "val_manifest", "decisions", "approved_manifest", "config", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if (arguments.max_tracks < 3 or arguments.workers < 1 or arguments.workers > 8
            or arguments.steps < 1 or arguments.learning_rate <= 0):
        raise ValueError("invalid group-validation training arguments")
    if not arguments.regularization or any(value < 0 for value in arguments.regularization):
        raise ValueError("regularization values must be nonnegative")
    run(arguments)
