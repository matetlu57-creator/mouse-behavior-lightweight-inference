#!/usr/bin/env python3
"""Tune only the Huddling and Social-clustering feature gates on train.csv.

The script accepts one explicit training manifest and filters it to those two
annotated behaviors before opening any tracks. It never opens a validation or
test manifest. All non-target heuristic parameters remain loaded unchanged
from the supplied baseline parameter file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _bootstrap import REPO_ROOT
from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    evaluate_top1,
    load_training_manifest,
    Top1Prediction,
)
from mouse_behavior.evaluation.exported_dataset import (
    EXPORTED_DATASET_SOURCE_FILES,
    CANONICAL_BEHAVIORS,
    HeuristicParameters,
    _collective_rule_config,
    collective_track_quality_failures,
    collective_track_quality_metrics,
    load_exported_features,
)
from mouse_behavior.behavior.collective_state import (
    classify_collective_windows,
    collective_candidate_windows,
)


LOGGER = logging.getLogger(__name__)
FOCUS = {"huddle", "social_clustering"}
SOURCE_FILES = (
    "src/mouse_behavior/behavior/collective_state.py",
    *EXPORTED_DATASET_SOURCE_FILES,
    "src/mouse_behavior/evaluation/blind_top1.py",
    "scripts/tune_collective_features.py",
    "scripts/predict_exported_blind.py",
)
TUNING_GRID: Mapping[str, tuple[float, ...]] = {
    "social_nnd_drop_bw_min": (0.10, 0.20, 0.35, 0.50),
    "social_radius_drop_bw_min": (0.10, 0.20, 0.35, 0.50),
    "social_inward_fraction_min": (0.35, 0.45, 0.55, 0.67),
    "social_active_inward_member_fraction_min": (0.15, 0.20, 0.25, 0.33),
    "huddle_nnd_bw_max": (1.30, 1.50, 1.70, 2.00, 2.50),
    "huddle_radius_bw_max": (1.00, 1.25, 1.50, 1.75, 2.00),
    "huddle_bbox_iom_min": (0.10, 0.14, 0.18, 0.22),
    "collective_huddle_min_group_visible_fraction": (0.20, 0.30, 0.45, 0.60),
    "collective_huddle_min_member_support_fraction": (0.25, 0.40, 0.55, 0.70),
    "collective_huddle_recovery_near_group_max_bw": (2.0, 3.0, 4.0),
    "huddle_low_motion_fraction_min": (0.50, 0.60, 0.70, 0.80),
    "huddle_contact_density_min": (0.10, 0.25, 0.40, 0.55, 0.70),
    "huddle_proximity_density_min": (0.60, 0.75, 0.85, 0.95),
    "huddle_graph_stability_min": (0.30, 0.50, 0.70, 0.85),
    "collective_track_min_lifetime_s": (0.5, 1.0, 2.0, 3.0),
    "collective_track_min_presence_ratio": (0.02, 0.05, 0.10, 0.15, 0.20),
    "collective_track_min_median_detection_confidence": (0.0, 0.15, 0.25, 0.35, 0.50),
    "collective_track_min_median_keypoint_coverage": (0.0, 0.10, 0.20, 0.30, 0.40),
    "collective_track_max_spatial_duplicate_score": (0.60, 0.75, 0.85, 0.95, 1.01),
}


@dataclass(frozen=True)
class PreparedGroupWindows:
    windows: tuple[Any, ...]
    target_track_quality: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parameters(path: Path) -> HeuristicParameters:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base parameter JSON must contain an object")
    known = set(HeuristicParameters().as_dict())
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    return replace(
        HeuristicParameters(),
        **{name: float(value) for name, value in payload.items()},
    )


def _predict(
    labels: Sequence[LabeledClip],
    prepared_windows: Sequence[PreparedGroupWindows],
    parameters: HeuristicParameters,
    *,
    include_debug: bool = False,
) -> tuple[list[tuple[LabeledClip, Any]], list[dict[str, Any]]]:
    rows: list[tuple[LabeledClip, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    config = _collective_rule_config(parameters)
    for index, (label, prepared) in enumerate(zip(labels, prepared_windows), start=1):
        target_quality_passes = bool(prepared.target_track_quality) and all(
            not collective_track_quality_failures(row, parameters)
            for row in prepared.target_track_quality
        )
        decision = (
            classify_collective_windows(prepared.windows, config)
            if target_quality_passes
            else classify_collective_windows((), config)
        )
        candidates = [
            ("huddle", decision.huddling_score),
            ("social_clustering", decision.social_clustering_score),
        ]
        behavior, score = max(candidates, key=lambda item: (item[1], item[0] == "huddle"))
        if score <= 0.0:
            behavior = ""
        top1 = Top1Prediction(
            behavior or None,
            tuple(label.target_ids) if behavior else (),
            float(score),
        )
        rows.append((label, top1))
        for state, score, window, checks in (
            ("HUDDLING", decision.huddling_score, decision.huddle_window, decision.huddle_checks),
            (
                "SOCIAL_CLUSTERING",
                decision.social_clustering_score,
                decision.social_window,
                decision.social_checks,
            ),
        ):
            if window is None or not include_debug:
                continue
            debug_rows.append(
                {
                    "split": "train",
                    "sample_id": label.sample_id,
                    "label_behavior_after_prediction": label.behavior,
                    "label_target_ids_after_prediction": ",".join(map(str, label.target_ids)),
                    **window.debug_values(),
                    "group_id": ",".join(map(str, label.target_ids)),
                    "member_ids": ",".join(map(str, label.target_ids)),
                    "frame": window.end_frame,
                    "time_s": window.end_frame / max(label.fps, 1e-9),
                    "state": state,
                    "score": score,
                    "checks": json.dumps(checks, ensure_ascii=False, sort_keys=True),
                }
            )
        if index == 1 or index % 100 == 0 or index == len(labels):
            LOGGER.info("evaluated training clips %d/%d", index, len(labels))
    return rows, debug_rows


def _focus_rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    rows = metrics["per_behavior"]
    huddle = rows.get("huddle", {"joint_correct": 0, "support": 0, "joint_top1_accuracy": 0.0})
    cluster = rows.get(
        "social_clustering",
        {"joint_correct": 0, "support": 0, "joint_top1_accuracy": 0.0},
    )
    hits = int(huddle["joint_correct"]) + int(cluster["joint_correct"])
    minimum = min(
        float(huddle["joint_top1_accuracy"]),
        float(cluster["joint_top1_accuracy"]),
    )
    macro = (float(huddle["joint_top1_accuracy"]) + float(cluster["joint_top1_accuracy"])) / 2
    return float(hits), minimum, macro, float(metrics["joint_top1_accuracy"])


def _prepare_group_windows(
    labels: Sequence[LabeledClip],
    loaded: Sequence[Any],
    parameters: HeuristicParameters,
) -> list[PreparedGroupWindows]:
    """Extract target-group features once; annotations are training-only here."""

    config = _collective_rule_config(parameters)
    all_windows: list[PreparedGroupWindows] = []
    for index, (label, features) in enumerate(zip(labels, loaded), start=1):
        slots_by_id = {
            int(track_id): slot for slot, track_id in enumerate(features.track_ids.tolist())
        }
        if not set(label.target_ids).issubset(slots_by_id):
            all_windows.append(PreparedGroupWindows((), ()))
            LOGGER.warning(
                "training target IDs missing from retained tracks: %s ids=%s",
                label.sample_id,
                label.target_ids,
            )
            continue
        members = tuple(slots_by_id[track_id] for track_id in label.target_ids)
        quality_by_id = {
            int(row["track_id"]): row for row in collective_track_quality_metrics(features)
        }
        target_quality = tuple(quality_by_id[track_id] for track_id in label.target_ids)
        centers = features.collective_centers_cm
        valid = features.collective_valid
        velocity = features.collective_velocity_cm_s
        speed = features.collective_speed_cm_s
        if centers is None or valid is None or velocity is None or speed is None:
            centers, valid, velocity, speed = (
                features.centers_cm,
                features.valid,
                features.velocity_cm_s,
                features.speed_cm_s,
            )
        windows = collective_candidate_windows(
            centers_cm=centers,
            valid=valid,
            speed_cm_s=speed,
            velocity_cm_s=velocity,
            member_slots=members,
            fps=features.fps,
            body_length_ref_cm=(
                features.body_length_ref_cm
                if features.body_length_ref_cm is not None
                else np.full(len(features.track_ids), 8.0)
            ),
            body_width_ref_cm=(
                features.body_width_ref_cm
                if features.body_width_ref_cm is not None
                else np.full(len(features.track_ids), max(features.mouse_width_cm, 1.0))
            ),
            config=config,
            keypoints_cm=features.collective_keypoints_cm,
            boxes_cm=features.collective_boxes_cm,
        )
        all_windows.append(PreparedGroupWindows(windows, target_quality))
        if index == 1 or index % 100 == 0 or index == len(labels):
            LOGGER.info("prepared target group windows %d/%d", index, len(labels))
    return all_windows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--base-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--max-passes", type=int, default=1)
    parser.add_argument("--log-level", choices=("INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.max_tracks < 3 or args.max_passes < 0:
        raise ValueError("max-tracks must be >= 3 and max-passes must be >= 0")
    if args.output_dir.exists():
        raise FileExistsError("choose a new output directory; previous experiments are preserved")
    labels = [
        label
        for label in load_training_manifest(args.train_manifest)
        if label.behavior in FOCUS
    ]
    if not labels:
        raise ValueError("training manifest has no Huddling or Social-clustering samples")
    tuned = replace(
        _load_parameters(args.base_parameters), collective_feature_rules_enabled=1.0
    )

    source_hashes = {
        name: _sha256(REPO_ROOT / name)
        for name in SOURCE_FILES
        if name != "scripts/predict_exported_blind.py"
    }
    manifest_hash = _sha256(args.train_manifest)
    LOGGER.info("loading %d training trajectories", len(labels))
    loaded = [
        load_exported_features(label.tracks_path, fps=label.fps, max_tracks=args.max_tracks)
        for label in labels
    ]

    LOGGER.info("precomputing the labeled target-group windows only")
    prepared_windows = _prepare_group_windows(labels, loaded, tuned)
    current_rows, _ = _predict(labels, prepared_windows, tuned)
    current_metrics = evaluate_top1(current_rows)
    history: list[dict[str, Any]] = []
    current_rank = _focus_rank(current_metrics)

    for pass_index in range(1, args.max_passes + 1):
        changed = False
        for name, values in TUNING_GRID.items():
            candidate_results = []
            for value in (getattr(tuned, name), *values):
                candidate = replace(tuned, **{name: float(value)})
                rows, _ = _predict(labels, prepared_windows, candidate)
                metrics = evaluate_top1(rows)
                rank = _focus_rank(metrics)
                candidate_results.append((candidate, rows, metrics, rank))
                history.append(
                    {
                        "pass": pass_index,
                        "parameter": name,
                        "value": getattr(candidate, name),
                        "huddle_strict_correct": metrics["per_behavior"].get("huddle", {}).get("joint_correct", 0),
                        "huddle_support": metrics["per_behavior"].get("huddle", {}).get("support", 0),
                        "social_clustering_strict_correct": metrics["per_behavior"].get("social_clustering", {}).get("joint_correct", 0),
                        "social_clustering_support": metrics["per_behavior"].get("social_clustering", {}).get("support", 0),
                        "overall_strict_accuracy": metrics["joint_top1_accuracy"],
                        "focus_rank": json.dumps(rank),
                    }
                )
            best = max(candidate_results, key=lambda item: (item[3], item[0] == tuned))
            if best[3] > current_rank:
                tuned, current_rows, current_metrics, current_rank = best
                changed = True
                LOGGER.info("selected %s=%s train_focus=%s", name, getattr(tuned, name), current_rank)
        if not changed:
            break

    final_rows, debug_rows = _predict(
        labels, prepared_windows, tuned, include_debug=True
    )
    metrics = evaluate_top1(final_rows)
    metrics["training_scope"] = (
        "target-ID Huddling/Social-clustering group rules only; "
        "non-collective competitors are not part of this conditional score"
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "tuned_parameters.json").open("x", encoding="utf-8") as handle:
        json.dump(tuned.as_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output_dir / "training_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output_dir / "tuning_history.json").open("x", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _write_csv(
        args.output_dir / "training_predictions.csv",
        [
            {
                "sample_id": record.sample_id,
                "true_behavior": record.behavior,
                "true_target_ids": ",".join(map(str, record.target_ids)),
                "predicted_behavior": prediction.behavior or "",
                "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                "strict_top1_correct": (
                    prediction.behavior == record.behavior
                    and tuple(prediction.target_ids) == record.target_ids
                ),
            }
            for record, prediction in final_rows
        ],
    )
    _write_csv(args.output_dir / "feature_debug.csv", debug_rows)

    frozen_model = {
        "heuristic_parameters": tuned.as_dict(),
        "ranking_weights": {name: 1.0 for name in CANONICAL_BEHAVIORS},
        "max_tracks": args.max_tracks,
    }
    model_path = args.output_dir / "frozen_model.json"
    with model_path.open("x", encoding="utf-8") as handle:
        json.dump(frozen_model, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    predictor_path = REPO_ROOT / "scripts/predict_exported_blind.py"
    source_hashes["scripts/predict_exported_blind.py"] = _sha256(predictor_path)
    provenance = {
        "ready_for_blind": True,
        "model_file": "frozen_model.json",
        "model_sha256": _sha256(model_path),
        "source_sha256": source_hashes,
        "max_tracks": args.max_tracks,
        "training_manifest_sha256": manifest_hash,
        "training_sample_count": len(labels),
        "training_label_scope": sorted(FOCUS),
        "training_trajectory_sha256": {
            label.sample_id: _sha256(Path(label.tracks_path)) for label in labels
        },
        "validation_used_for_fitting": False,
        "non_target_behavior_parameters_modified": False,
        "collective_features_enabled": True,
        "training_scope": "Huddling/Social-clustering labeled target IDs only",
    }
    with (args.output_dir / "run_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOGGER.info(
        "frozen train strict=%.2f%% huddle=%s/%s social_clustering=%s/%s",
        metrics["joint_top1_accuracy"] * 100,
        metrics["per_behavior"].get("huddle", {}).get("joint_correct", 0),
        metrics["per_behavior"].get("huddle", {}).get("support", 0),
        metrics["per_behavior"].get("social_clustering", {}).get("joint_correct", 0),
        metrics["per_behavior"].get("social_clustering", {}).get("support", 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
