#!/usr/bin/env python3
"""Train and validate only the six pair-social heuristics on the inherited split.

Training uses approved social clips only. Validation uses only approved clips
from the same six social classes, so reported precision is closed-set social
precision and does not estimate non-social deployment false positives.
Validation labels never enter fitting or threshold selection. All artifacts
are written to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT

from mouse_behavior.config import load_config
from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    Top1Prediction,
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
from mouse_behavior.evaluation.scale_migration import rescale_frozen_source_scale
from mouse_behavior.evaluation.confirmed_rules import (
    policy_receipt,
    social_profiles,
)
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    LABEL_TO_CANONICAL,
    PAIR_BEHAVIORS,
    HeuristicParameters,
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.train_only_tuning import session_folds


SOCIAL_BEHAVIORS = tuple(PAIR_BEHAVIORS)
CODE_FILES = (
    "scripts/validate_social_heuristics.py",
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
    "src/mouse_behavior/evaluation/requested_thresholds_v27.py",
    "src/mouse_behavior/evaluation/requested_thresholds_v28.py",
    "src/mouse_behavior/evaluation/exported_dataset.py",
    "src/mouse_behavior/evaluation/exported_dataset_config.py",
    "src/mouse_behavior/evaluation/exported_dataset_features.py",
    "src/mouse_behavior/evaluation/exported_dataset_models.py",
    "src/mouse_behavior/evaluation/exported_dataset_prediction.py",
    "src/mouse_behavior/evaluation/exported_dataset_temporal.py",
    "src/mouse_behavior/evaluation/screening_signals.py",
    "src/mouse_behavior/evaluation/social_heuristic_config.py",
    "src/mouse_behavior/evaluation/social_event_geometry.py",
    "src/mouse_behavior/evaluation/scale_migration.py",
    "src/mouse_behavior/behavior/individual_rules.py",
    "src/mouse_behavior/behavior/static_rule.py",
    "src/mouse_behavior/evaluation/train_only_tuning.py",
    "configs/default.yaml",
    "configs/scale/rig_a.json",
    "docs/confirmed-calibration-v1.md",
)


def _write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _read_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty {expected_split} manifest: {path}")
    for row in rows:
        expected = {"train"} if expected_split == "train" else {"val", "validation"}
        if row.get("split") not in expected:
            raise ValueError(f"unexpected split value in {path}: {row.get('split')!r}")
    return rows


def _current_approval_rows(rows: list[dict[str, str]], decisions: dict[str, Any]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        state = decisions.get(row["sample_id"], {}).get("decision")
        if state not in {"approved", "rejected"}:
            raise ValueError(f"sample has no terminal manual decision: {row['sample_id']}")
        if state == "approved":
            output.append(row)
    return output


def _verify_latest_approved_social_manifest(
    manifest_path: Path,
    decisions: dict[str, Any],
    split_social_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Bind the inherited split to the exact current approved dataset and files."""
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("latest approved dataset manifest must be a list")
    latest = [
        row for row in entries
        if row.get("decision") == "approved"
        and LABEL_TO_CANONICAL.get(row.get("behavior")) in SOCIAL_BEHAVIORS
    ]
    by_id: dict[str, dict[str, Any]] = {}
    file_hash_checks = 0
    for row in latest:
        sample_id = str(row.get("review_id", ""))
        if not sample_id or sample_id in by_id:
            raise ValueError(f"missing or duplicate current approved social ID: {sample_id!r}")
        by_id[sample_id] = row
        decision = decisions.get(sample_id, {})
        expected_ids = sorted(int(value) for value in row.get("mouse_ids", []))
        current_ids = sorted(int(value) for value in decision.get("mouse_ids", []))
        if (decision.get("decision") != "approved"
                or decision.get("behavior") != row.get("behavior")
                or current_ids != expected_ids):
            raise ValueError(f"latest manifest and current decision disagree: {sample_id}")
        expected_hashes = row.get("dataset_sha256")
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError(f"approved social sample has no data-file hashes: {sample_id}")
        sample_dir = Path(row["dataset_path"]).resolve()
        for name, expected_hash in expected_hashes.items():
            path = sample_dir / name
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError(f"latest approved sample file hash differs: {path}")
            file_hash_checks += 1

    split_by_id = {str(row["sample_id"]): row for row in split_social_rows}
    if len(split_by_id) != len(split_social_rows):
        raise ValueError("social training/validation split contains duplicate IDs")
    if set(split_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(split_by_id))[:8]
        extra = sorted(set(split_by_id) - set(by_id))[:8]
        raise ValueError(f"inherited social split does not match latest approvals: missing={missing}, extra={extra}")
    for sample_id, row in split_by_id.items():
        approved = by_id[sample_id]
        ids = sorted(int(value) for value in row["mouse_ids"].split(",") if value.strip())
        if (row["canonical_behavior"] != LABEL_TO_CANONICAL[approved["behavior"]]
                or ids != sorted(int(value) for value in approved["mouse_ids"])):
            raise ValueError(f"social split behavior/target IDs differ from latest approval: {sample_id}")
    return {
        "approved_manifest_sha256": file_sha256(manifest_path),
        "approved_social_samples": len(by_id),
        "social_split_samples": len(split_by_id),
        "social_split_exact_id_coverage": True,
        "decision_behavior_id_mismatches": 0,
        "dataset_file_hash_checks": file_hash_checks,
        "dataset_file_hash_mismatches": 0,
        "behavior_support": {
            behavior: sum(LABEL_TO_CANONICAL[row["behavior"]] == behavior for row in latest)
            for behavior in SOCIAL_BEHAVIORS
        },
    }


def _verified_frozen_scale(
    row: dict[str, str], data_root: Path = REPO_ROOT,
    active_rig_path: Path | None = None,
    hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Use the per-sample fixed scale in its receipt after source/hash checks.

    The historical receipt also fingerprints default.yaml. That file has since
    changed in this intentionally dirty worktree. A fixed cm/pixel value is
    safe to replay when the crop, full source tracks, receipt, and rig asset all
    still match; any other asset drift is rejected.
    """
    receipt_path = resolved_input_path(row["scale_receipt_path"], data_root)
    receipt_hash = file_sha256(receipt_path)
    if receipt_hash != row["scale_receipt_sha256"]:
        raise ValueError(f"scale receipt hash differs: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tracks_path = resolved_input_path(row["tracks_path"], data_root)
    tracks_hash = file_sha256(tracks_path)
    if tracks_hash not in receipt.get("crop_tracks_sha256", []):
        raise ValueError(f"scale receipt belongs to another crop: {tracks_path}")
    source_path = resolved_input_path(receipt["source_tracks_path"], data_root)
    source_hash = file_sha256(source_path)
    if source_hash != receipt["source_tracks_sha256"]:
        raise ValueError(f"source tracks changed after scale freeze: {source_path}")
    scale = dict(receipt["scale_config"])
    if (scale.get("mode") != "fixed" or not math.isfinite(float(scale.get("cm_per_pixel", 0)))
            or float(scale["cm_per_pixel"]) <= 0):
        raise ValueError(f"scale receipt has no fixed positive cm/pixel: {receipt_path}")
    allowed_config_drift = []
    active_rig = (Path(active_rig_path).resolve() if active_rig_path is not None else
                  (REPO_ROOT / "configs" / "scale" / "rig_a.json").resolve())
    for asset_text, expected_hash in receipt["scale_asset_sha256"].items():
        asset = resolved_input_path(asset_text, data_root)
        actual_hash = file_sha256(asset)
        if actual_hash == expected_hash:
            if hashes is not None:
                hashes[str(asset)] = actual_hash
            continue
        expected_drifting_config = data_root / "configs" / "default.yaml"
        expected_drifting_rig = data_root / "configs" / "scale" / "rig_a.json"
        if asset.resolve() == expected_drifting_config.resolve() and scale["mode"] == "fixed":
            allowed_config_drift.append(asset_text)
        elif asset.resolve() == expected_drifting_rig.resolve() and scale["mode"] == "fixed":
            scale, migration = rescale_frozen_source_scale(scale, active_rig)
            allowed_config_drift.append(
                f"{asset_text} migrated by {migration['factor']:.12g} using {active_rig}"
            )
        else:
            raise ValueError(f"frozen scale asset changed: {asset}")
        if hashes is not None:
            hashes[str(asset)] = actual_hash
    if hashes is not None:
        hashes[str(receipt_path)] = receipt_hash
        hashes[str(tracks_path)] = tracks_hash
        hashes[str(source_path)] = source_hash
    return scale, allowed_config_drift


def _label(row: dict[str, str]) -> LabeledClip:
    ids = tuple(sorted({int(value) for value in row["mouse_ids"].split(",") if value.strip()}))
    behavior = row["canonical_behavior"]
    if behavior not in CANONICAL_BEHAVIORS:
        raise ValueError(f"unknown canonical label: {behavior}")
    if len(ids) != 2:
        raise ValueError(f"social validation requires exactly two target IDs: {row['sample_id']}")
    return LabeledClip(
        sample_id=row["sample_id"],
        tracks_path=row["tracks_path"],
        fps=float(row["fps"]),
        recording_session=row["recording_session"],
        behavior=behavior,
        target_ids=ids,
        scale_receipt_path=row["scale_receipt_path"],
        scale_receipt_sha256=row["scale_receipt_sha256"],
    )


def _social_table(
    row: dict[str, str],
    profiles: tuple[HeuristicParameters, ...],
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, data_root: Path = REPO_ROOT, hashes: dict[str, str] | None = None,
) -> CandidateTable:
    scale, _ = _verified_frozen_scale(
        row, data_root, REPO_ROOT / "configs" / "scale" / "rig_a.json", hashes,
    )
    tracks_path = resolved_input_path(row["tracks_path"], data_root)
    features = load_exported_features(
        tracks_path, fps=float(row["fps"]), max_tracks=64, scale_config=scale
    )
    return build_candidate_table(
        features, profiles, behavior_subset=SOCIAL_BEHAVIORS,
        social_config=social_config,
    )


def _social_metrics(rows, predictions: dict[str, Top1Prediction]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_behavior: dict[str, dict[str, Any]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    false_positive_sources: dict[str, Counter[str]] = defaultdict(Counter)
    error_types: Counter[str] = Counter()
    exact_by_id = {}
    for row in rows:
        prediction = predictions[row["sample_id"]]
        truth = row["canonical_behavior"]
        truth_ids = tuple(sorted(
            int(value) for value in row["mouse_ids"].split(",") if value.strip()
        ))
        predicted_ids = tuple(sorted(prediction.target_ids))
        predicted_behavior = prediction.behavior or "unrecognized"
        confusion[(truth, predicted_behavior)] += 1
        exact = prediction.behavior == truth and predicted_ids == truth_ids
        exact_by_id[row["sample_id"]] = exact
        if exact:
            error_types["correct_behavior_and_target_ids"] += 1
        elif prediction.behavior is None:
            error_types["unrecognized"] += 1
        elif prediction.behavior == truth:
            error_types["correct_behavior_wrong_target_ids"] += 1
        elif predicted_ids == truth_ids:
            error_types["wrong_behavior_correct_target_ids"] += 1
        else:
            error_types["wrong_behavior_and_wrong_target_ids"] += 1
    for behavior in SOCIAL_BEHAVIORS:
        support = predicted = exact_tp = behavior_wrong_id = 0
        for row in rows:
            truth = row["canonical_behavior"]
            prediction = predictions[row["sample_id"]]
            if truth == behavior:
                support += 1
                if prediction.behavior == behavior:
                    if exact_by_id[row["sample_id"]]:
                        exact_tp += 1
                    else:
                        behavior_wrong_id += 1
            if prediction.behavior == behavior:
                predicted += 1
                if truth != behavior:
                    false_positive_sources[behavior][truth] += 1
        by_behavior[behavior] = {
            "support": support,
            "predicted": predicted,
            "exact_behavior_target_id_tp": exact_tp,
            "correct_behavior_wrong_target_ids": behavior_wrong_id,
            "precision": exact_tp / predicted if predicted else None,
            "recall": exact_tp / support if support else None,
            "false_positive_truths": dict(false_positive_sources[behavior].most_common()),
        }
    matrix = {
        "true_behavior": sorted({row["canonical_behavior"] for row in rows}),
        "predicted_behavior": [*SOCIAL_BEHAVIORS, "unrecognized"],
        "counts": {
            truth: {
                guessed: confusion[(truth, guessed)]
                for guessed in [*SOCIAL_BEHAVIORS, "unrecognized"]
            }
            for truth in sorted({row["canonical_behavior"] for row in rows})
        },
    }
    total_predicted = sum(value["predicted"] for value in by_behavior.values())
    total_support = sum(value["support"] for value in by_behavior.values())
    total_exact_tp = sum(value["exact_behavior_target_id_tp"] for value in by_behavior.values())
    summary = {
        "n_approved_validation_samples": len(rows),
        "n_social_validation_samples": total_support,
        "n_social_predictions": total_predicted,
        "exact_behavior_target_id_tp": total_exact_tp,
        "joint_micro_precision": total_exact_tp / total_predicted if total_predicted else None,
        "joint_micro_recall": total_exact_tp / total_support if total_support else None,
        "per_behavior": by_behavior,
        "error_types": dict(error_types),
        "confusion": matrix,
    }
    return summary, matrix


def _historical_social_baseline(
    all_val_rows: list[dict[str, str]], approved_rows: list[dict[str, str]], baseline_path: Path
) -> dict[str, Any]:
    """Read prior frozen predictions for context only; never use them in fitting."""
    with baseline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != len(all_val_rows):
        raise ValueError(f"historical baseline row count differs: {baseline_path}")
    by_id = {row["sample_id"]: row for row in all_val_rows}
    prediction_ids = {prediction["sample_id"] for prediction in predictions}
    if set(by_id) == prediction_ids:
        joined = {prediction["sample_id"]: prediction for prediction in predictions}
    else:
        # Older receipts pseudonymized IDs as v-000001, etc. Their rows can
        # only be paired by the inherited split order after all label, exact
        # target-ID and session fields match row-for-row.
        for row, prediction in zip(all_val_rows, predictions):
            manifest_ids = tuple(sorted(int(value) for value in row["mouse_ids"].split(",") if value.strip()))
            prediction_ids = tuple(sorted(
                int(value) for value in prediction["true_target_ids"].split(",") if value.strip()
            ))
            if (row["canonical_behavior"] != prediction["true_behavior"]
                    or manifest_ids != prediction_ids
                    or row["recording_session"] != prediction["recording_session"]):
                raise ValueError(f"historical baseline row order does not match labels/IDs/session: {baseline_path}")
        joined = {
            row["sample_id"]: prediction
            for row, prediction in zip(all_val_rows, predictions)
        }
    result = {}
    for behavior in SOCIAL_BEHAVIORS:
        support = sum(row["canonical_behavior"] == behavior for row in approved_rows)
        tp = sum(
            row["canonical_behavior"] == behavior
            and joined[row["sample_id"]]["predicted_behavior"] == behavior
            for row in approved_rows
        )
        predicted = sum(
            joined[row["sample_id"]]["predicted_behavior"] == behavior
            for row in approved_rows
        )
        result[behavior] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
        }
    return result


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError("choose a new social-validation output directory")
    train_manifest = _read_manifest(args.train_manifest, "train")
    val_manifest = _read_manifest(args.val_manifest, "val")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"]
    approved_train = _current_approval_rows(train_manifest, decisions)
    approved_val = _current_approval_rows(val_manifest, decisions)
    train_rows = [row for row in approved_train if row["canonical_behavior"] in SOCIAL_BEHAVIORS]
    val_rows = [row for row in approved_val if row["canonical_behavior"] in SOCIAL_BEHAVIORS]
    latest_approval_audit = _verify_latest_approved_social_manifest(
        args.approved_manifest, decisions, [*train_rows, *val_rows]
    )
    train_labels = [_label(row) for row in train_rows]
    social_support = Counter(label.behavior for label in train_labels)
    if not train_labels:
        raise ValueError("no approved social training samples")

    scale_drift_by_sample = {}
    frozen_input_hashes: dict[str, str] = {}
    for row in [*train_rows, *val_rows]:
        _, drift = _verified_frozen_scale(
            row, args.data_root, REPO_ROOT / "configs" / "scale" / "rig_a.json",
            frozen_input_hashes,
        )
        if drift:
            scale_drift_by_sample[row["sample_id"]] = drift
    args.output_dir.mkdir(parents=True)
    config = load_config(args.config)
    seed = config_seed_parameters(config)
    social_config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    profiles = social_profiles(seed, social_config)
    source_hashes = {name: file_sha256(REPO_ROOT / name) for name in CODE_FILES}
    input_hashes = {
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.val_manifest),
        "decisions_sha256": file_sha256(args.decisions),
        "approved_dataset_manifest_sha256": latest_approval_audit["approved_manifest_sha256"],
        "config_sha256": file_sha256(args.config),
        "data_root": str(args.data_root),
        "frozen_source_input_sha256": frozen_input_hashes,
        "frozen_scale_receipts_modified": False,
        "frozen_scale_rig_migration": {
            "factor": 7.918 / 10.7,
            "arena_cm": [74.0, 74.0],
            "body_length_cm": 7.918,
            "body_width_cm": 3.068558,
            "samples_with_allowed_scale_drift": len(scale_drift_by_sample),
            "sample_details": scale_drift_by_sample,
        },
    }

    train_tables = []
    for index, row in enumerate(train_rows, start=1):
        train_tables.append(_social_table(
            row, profiles, social_config, data_root=args.data_root, hashes=frozen_input_hashes,
        ))
        if index % 50 == 0 or index == len(train_rows):
            print(f"social training feature extraction {index}/{len(train_rows)}", flush=True)

    regularization_values = (0.001, 0.01, 0.1)
    search = []
    for regularization in regularization_values:
        oof = []
        for _session, fit_indices, check_indices in session_folds(train_labels):
            fold_model, _ = fit_ranker(
                [train_tables[i] for i in fit_indices],
                [train_labels[i] for i in fit_indices],
                steps=args.steps,
                regularization=regularization,
                learning_rate=args.learning_rate,
                seed=args.seed,
                objective="hard_margin",
                require_positive_evidence=True,
            )
            oof.extend(
                (train_labels[i], predict_table(train_tables[i], fold_model))
                for i in check_indices
            )
        from mouse_behavior.evaluation.blind_top1 import evaluate_top1
        metrics = evaluate_top1(oof)
        search.append({"regularization": regularization, "train_session_oof": metrics})
    selected = max(search, key=lambda item: metric_rank(item["train_session_oof"]))
    model, train_metrics = fit_ranker(
        train_tables,
        train_labels,
        steps=args.steps,
        regularization=selected["regularization"],
        learning_rate=args.learning_rate,
        seed=args.seed,
        objective="hard_margin",
        require_positive_evidence=True,
    )
    model.update({
        "model_type": "social_only_confirmed_heuristic_ranker_v11",
        "social_behaviors": list(SOCIAL_BEHAVIORS),
        "rule_profiles": [profile.as_dict() for profile in profiles],
        "social_heuristic_config": social_config.as_dict(),
        "rule_policy": policy_receipt(social_config),
        "training_behavior_support": dict(sorted(social_support.items())),
        "validation_used_for_fitting": False,
        "approved_social_manifest": latest_approval_audit,
    })
    model_path = args.output_dir / "frozen_social_model.json"
    _write_json(model_path, model)
    model_hash = file_sha256(model_path)
    _write_json(args.output_dir / "training_session_oof.json", {"search": search})
    _write_json(args.output_dir / "training_metrics.json", train_metrics)
    _write_json(args.output_dir / "training_receipt.json", {
        **input_hashes,
        "code_sha256": source_hashes,
        "model_sha256": model_hash,
        "training_samples": len(train_labels),
        "training_behavior_support": dict(sorted(social_support.items())),
        "selected_regularization": selected["regularization"],
        "selected_train_session_oof": selected["train_session_oof"],
        "validation_used_for_fitting": False,
        "latest_approved_dataset_audit": latest_approval_audit,
    })
    if any(file_sha256(REPO_ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("social heuristic source changed while fitting; model is unsealed")

    predictions: dict[str, Top1Prediction] = {}
    prediction_rows = []
    rule_gate_support = Counter()
    rule_gate_hits = Counter()
    for index, row in enumerate(val_rows, start=1):
        table = _social_table(
            row, profiles, social_config, data_root=args.data_root, hashes=frozen_input_hashes,
        )
        prediction = predict_table(table, model)
        predictions[row["sample_id"]] = prediction
        label = _label(row)
        active_candidates = [candidate for candidate in table.candidates if candidate.evidence > 0]
        active_exact_pair_behaviors = sorted({
            candidate.behavior for candidate in active_candidates
            if tuple(sorted(candidate.target_ids)) == label.target_ids
        })
        rule_gate_support[label.behavior] += 1
        rule_gate_hits[label.behavior] += int(label.behavior in active_exact_pair_behaviors)
        predicted_ids = tuple(sorted(prediction.target_ids))
        exact_match = (
            prediction.behavior == row["canonical_behavior"]
            and predicted_ids == label.target_ids
        )
        if exact_match:
            error_type = "correct_behavior_and_target_ids"
        elif prediction.behavior is None:
            error_type = "unrecognized"
        elif prediction.behavior == row["canonical_behavior"]:
            error_type = "correct_behavior_wrong_target_ids"
        elif predicted_ids == label.target_ids:
            error_type = "wrong_behavior_correct_target_ids"
        else:
            error_type = "wrong_behavior_and_wrong_target_ids"
        prediction_rows.append({
            "sample_id": row["sample_id"],
            "recording_session": row["recording_session"],
            "true_behavior": row["canonical_behavior"],
            "true_target_ids": row["mouse_ids"],
            "predicted_behavior": prediction.behavior or "unrecognized",
            "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
            "score": prediction.score,
            "social_candidate_count": len(table.candidates),
            "active_social_candidate_count": len(active_candidates),
            "active_exact_pair_behaviors": json.dumps(active_exact_pair_behaviors, ensure_ascii=False),
            "true_pair_rule_candidate_present": label.behavior in active_exact_pair_behaviors,
            "strict_behavior_and_id_correct": exact_match,
            "error_type": error_type,
        })
        if index % 50 == 0 or index == len(val_rows):
            print(f"social validation prediction {index}/{len(val_rows)}", flush=True)

    metrics, matrix = _social_metrics(val_rows, predictions)
    opposite_contact = {
        "nose_head_contact": "nose_tail_contact",
        "nose_tail_contact": "nose_head_contact",
    }
    cross_contact_errors = []
    for row in prediction_rows:
        true_behavior = row["true_behavior"]
        if row["predicted_behavior"] != opposite_contact.get(true_behavior):
            continue
        true_ids = tuple(sorted(int(value) for value in row["true_target_ids"].split(",") if value))
        predicted_ids = tuple(sorted(int(value) for value in row["predicted_target_ids"].split(",") if value))
        active_pair_behaviors = json.loads(row["active_exact_pair_behaviors"])
        cross_contact_errors.append({
            "sample_id": row["sample_id"],
            "true_behavior": true_behavior,
            "predicted_behavior": row["predicted_behavior"],
            "same_target_id_set": predicted_ids == true_ids,
            "true_behavior_candidate_present_for_annotated_pair": row["true_pair_rule_candidate_present"],
            "annotated_pair_rule_behaviors": active_pair_behaviors,
        })
    contact_type_diagnostics = {
        "strict_top1_opposite_contact_type_count": len(cross_contact_errors),
        "same_target_id_set_count": sum(item["same_target_id_set"] for item in cross_contact_errors),
        "different_target_id_set_count": sum(not item["same_target_id_set"] for item in cross_contact_errors),
        "correct_contact_candidate_present_count": sum(
            item["true_behavior_candidate_present_for_annotated_pair"]
            for item in cross_contact_errors
        ),
        "errors": cross_contact_errors,
        "interpretation": (
            "A head/tail label disagreement is a rule-classification error only when the "
            "predicted visual-ID set equals the annotated set. If the IDs differ, inspect "
            "the exact annotated-pair candidate list: a correct contact candidate there "
            "identifies pair ranking as the error; no such candidate identifies missed "
            "geometry/probing evidence or an annotation/track-quality question."
        ),
    }
    rule_gate_metrics = {
        behavior: {
            "support": int(rule_gate_support[behavior]),
            "true_pair_positive_rule_candidates": int(rule_gate_hits[behavior]),
            "true_pair_candidate_coverage": (
                rule_gate_hits[behavior] / rule_gate_support[behavior]
                if rule_gate_support[behavior] else None
            ),
        }
        for behavior in SOCIAL_BEHAVIORS
    }
    _write_json(args.output_dir / "social_validation_metrics.json", metrics)
    _write_json(args.output_dir / "social_confusion_matrix.json", matrix)
    _write_json(args.output_dir / "social_rule_gate_metrics.json", rule_gate_metrics)
    _write_json(args.output_dir / "social_contact_type_diagnostics.json", contact_type_diagnostics)
    _write_csv(
        args.output_dir / "social_validation_predictions.csv",
        ("sample_id", "recording_session", "true_behavior", "true_target_ids",
         "predicted_behavior", "predicted_target_ids", "score", "social_candidate_count",
         "active_social_candidate_count", "active_exact_pair_behaviors",
         "true_pair_rule_candidate_present", "strict_behavior_and_id_correct", "error_type"),
        prediction_rows,
    )
    metric_rows = []
    for behavior, values in metrics["per_behavior"].items():
        metric_rows.append({"behavior": behavior, **values})
    _write_csv(
        args.output_dir / "social_per_behavior_precision_recall.csv",
        ("behavior", "support", "predicted", "exact_behavior_target_id_tp", "precision",
         "recall", "correct_behavior_wrong_target_ids", "false_positive_truths"),
        [{**row, "false_positive_truths": json.dumps(row["false_positive_truths"], ensure_ascii=False)}
         for row in metric_rows],
    )

    def pct(value: float | None) -> str:
        return "未定义" if value is None else f"{value:.2%}"

    report_lines = [
        "# 社交行为启发式 v11：行为与目标视觉 ID 联合验证",
        "",
        "主指标只在行为标签与审核标注的完整无序视觉 ID 集合都完全一致时计为 TP。预测行为正确但 ID 错、目标 ID 正确但行为错、或多/少目标鼠都不算正确。P 的分母是该类全部预测，R 的分母是该类全部审核通过真值。",
        "",
        f"最新审核数据集共 {latest_approval_audit['approved_social_samples']} 条社交样本；训练 {len(train_rows)} 条，冻结验证 {len(val_rows)} 条。训练与验证合计 ID/行为覆盖完全匹配最新 manifest；{latest_approval_audit['dataset_file_hash_checks']} 个样本文件哈希均通过。验证集没有用于拟合或选阈值。",
        "",
        f"冻结验证联合 micro P={pct(metrics['joint_micro_precision'])}，R={pct(metrics['joint_micro_recall'])}；正确行为+目标 ID {metrics['exact_behavior_target_id_tp']}/{metrics['n_social_validation_samples']}。",
        "",
        "| 社交行为 | 真值数 | 预测数 | 行为+ID 全匹配 | 联合 P | 联合 R | 同行为但目标 ID 错 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for behavior, value in metrics["per_behavior"].items():
        report_lines.append(
            f"| {behavior} | {value['support']} | {value['predicted']} | "
            f"{value['exact_behavior_target_id_tp']} | {pct(value['precision'])} | "
            f"{pct(value['recall'])} | {value['correct_behavior_wrong_target_ids']} |"
        )
    report_lines.extend([
        "",
        "## 错误类型",
        "",
        "逐样本预测、真值行为、真值 ID、预测 ID 和错误类型见 social_validation_predictions.csv。真值→预测行为混淆见 social_confusion_matrix.json。",
        "",
        "| 结果类别 | 数量 |",
        "|---|---:|",
    ])
    for category in (
        "correct_behavior_and_target_ids",
        "correct_behavior_wrong_target_ids",
        "wrong_behavior_correct_target_ids",
        "wrong_behavior_and_wrong_target_ids",
        "unrecognized",
    ):
        report_lines.append(f"| {category} | {metrics['error_types'].get(category, 0)} |")
    report_lines.extend([
        "",
        "## 真值目标 ID 对候选覆盖",
        "",
        "此项只用于定位 FN：同一真值 ID 对没有生成真值行为候选属于规则证据漏检；候选已生成但 Top-1 选了其他行为/鼠对属于候选竞争或目标 ID 排序错误。它不是 P/R。",
        "",
        "| 行为 | 真值支持 | 正确 ID 对候选数 | 覆盖率 |",
        "|---|---:|---:|---:|",
    ])
    for behavior, values in rule_gate_metrics.items():
        coverage = values["true_pair_candidate_coverage"]
        coverage_text = pct(coverage)
        report_lines.append(
            f"| {behavior} | {values['support']} | {values['true_pair_positive_rule_candidates']} | {coverage_text} |"
        )
    report_lines.extend([
        "逐样本候选类别见 social_validation_predictions.csv；候选覆盖不等于最终 P/R。",
        "",
        "## 冻结规则与边界",
        "",
        f"同鼠对互斥顺序：{' > '.join(social_config.priority_order)}；完整定向追逐可收回接近及低优先接触占帧，但保留攻击、一起。当前参数：`{json.dumps(social_config.as_dict(), ensure_ascii=False, sort_keys=True)}`。",
        f"Attack 的各观测／遮挡通道以可执行策略回执为准：`{policy_receipt(social_config)['rules']['attack']}`。",
        f"鼻头/鼻尾接触独立起始距离≤{social_config.contact_head_onset_distance_cm:g}/{social_config.contact_tail_onset_distance_cm:g} cm，分别测鼻到头三角/尾根；基线维持距离≤{social_config.dist_nose_head_contact_max_cm:g}/{social_config.dist_nose_tail_contact_max_cm:g} cm，实测时长≥{social_config.nose_contact_min_duration_s:g}/{social_config.nose_tail_min_duration_s:g} s。近距补充时长≥{social_config.contact_head_extension_min_duration_s:g}/{social_config.contact_tail_extension_min_duration_s:g} s，使用各自更严距离。不能跨方向借起点或时间；一起和完整定向追逐占帧排除接触。",
        f"Chasing：双方各自实际相邻支持步的路程/时间≥{social_config.chasing_min_mean_speed_cm_s:g} cm/s；各路径及净位移≥{social_config.chasing_min_net_displacement_cm:g} cm；速度夹角≤{social_config.angle_chasing_dir_max_deg:g}°，前后及朝向关系合格；观测支持≥{social_config.chasing_min_duration_s:g} s；缺失不计路程、时间或正帧；无体心距离上限。",
        f"Approach：起点中心距≥{social_config.dist_approach_start_min_cm:g} cm、行动速度>{social_config.vel_approach_actor_min_cm_s:g} cm/s，行动鼠朝向目标且双方距离缩短；目标可静止、侧移或远离。终点中心距{social_config.dist_approach_end_min_cm:g}–{social_config.dist_approach_end_max_cm:g} cm，鼻到身体表面{social_config.approach_surface_endpoint_cm:g}±{social_config.approach_surface_endpoint_tolerance_cm:g} cm；≥{social_config.approach_min_duration_s:g} s。接触起点使用独立几何，不复用此终点。",
        f"Together：中心距<{social_config.dist_together_max_cm:g} cm、双方速度各自严格<{social_config.vel_together_max_cm_s:g} cm/s、持续≥{social_config.together_min_duration_s:g} s，按当前优先顺序裁决。",
        "鼻头与鼻尾几何核查：左鼠鼻尖→右鼠头部（鼻端/两耳最近点）与左鼠鼻尖→右鼠尾根使用不同特征，反方向也分别计算；合成测试断言映射不互换。此前 v10 的 6 条严格头/尾交叉错误里，预测 ID 对均与标注 ID 对不同；其中 2 条正确尾接触候选已为标注 ID 对产生。因此该统计主要说明 Top-1 选错鼠对，不能据此认定头尾关键点映射反了。v11 的新逐条核查见 social_contact_type_diagnostics.json。",
        "",
        "## 范围限制",
        "",
        f"训练集类别支持：{json.dumps(dict(sorted(social_support.items())), ensure_ascii=False)}；验证只含六类已审核社交片段，属于社交闭集，不含 Locomotion、Static 或群体行为负例，不能据此估计部署到完整视频后的误报率。验证录制场次固定且未用于本轮拟合/选择，但该划分已有过往使用，不是全新盲测。",
        "",
        f"数据回执：approved manifest SHA-256 {latest_approval_audit['approved_manifest_sha256']}；审核决定与 manifest 的行为/目标 ID 不一致数 0；样本文件哈希不一致数 0。固定尺度继续从各样本冻结的 cm/pixel 回执读取。",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if any(file_sha256(REPO_ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("social heuristic source changed during validation; report is unsealed")
    for path, digest in frozen_input_hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"frozen social input changed during validation: {path}")
    if (file_sha256(args.approved_manifest) != input_hashes["approved_dataset_manifest_sha256"]
            or file_sha256(args.decisions) != input_hashes["decisions_sha256"]):
        raise RuntimeError("latest approved data changed during social validation")
    _write_json(args.output_dir / "run_receipt.json", {
        **input_hashes,
        "code_sha256": source_hashes,
        "model_sha256": model_hash,
        "social_metrics_sha256": file_sha256(args.output_dir / "social_validation_metrics.json"),
        "rule_gate_metrics_sha256": file_sha256(args.output_dir / "social_rule_gate_metrics.json"),
        "prediction_csv_sha256": file_sha256(args.output_dir / "social_validation_predictions.csv"),
        "report_sha256": file_sha256(args.output_dir / "report.md"),
        "training_samples": len(train_labels),
        "training_behavior_support": dict(sorted(social_support.items())),
        "validation_samples": len(val_rows),
        "validation_social_support": {
            behavior: sum(row["canonical_behavior"] == behavior for row in val_rows)
            for behavior in SOCIAL_BEHAVIORS
        },
        "latest_approved_dataset_audit": latest_approval_audit,
        "metrics": metrics,
        "validation_used_for_fitting": False,
        "training_candidate_classes": list(SOCIAL_BEHAVIORS),
        "validation_candidate_classes": list(SOCIAL_BEHAVIORS),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = REPO_ROOT.parent.parent
    default_root = data_root / "outputs" / "confirmed_calibration_v1_20261010"
    parser.add_argument("--train-manifest", type=Path, default=default_root / "dataset_v2" / "train.csv")
    parser.add_argument("--val-manifest", type=Path, default=default_root / "dataset_v2" / "val.csv")
    parser.add_argument("--decisions", type=Path, default=data_root / "outputs" / "behavior_confirmed_dataset" / "decisions.json")
    parser.add_argument("--approved-manifest", type=Path,
                        default=data_root / "outputs" / "behavior_confirmed_dataset" / "dataset_manifest.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--baseline-dir", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=data_root,
                        help="Read-only root for inherited data and frozen scale receipt paths.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("data_root", "train_manifest", "val_manifest", "decisions", "approved_manifest",
                 "config", "baseline_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("invalid training parameters")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
