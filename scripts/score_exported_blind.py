#!/usr/bin/env python3
"""Score locked blind predictions after inference has completed.

This separate command is the first step allowed to read test labels.  It
verifies the prediction receipt before joining by opaque sample ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401 - ensure the src package is importable
from mouse_behavior.evaluation.blind_top1 import LabeledClip, evaluate_top1, load_locked_predictions
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
)


def load_blind_labels(path: Path) -> dict[str, LabeledClip]:
    labels: dict[str, LabeledClip] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {"sample_id", "canonical_behavior", "mouse_ids"}:
            raise ValueError("blind labels must have exactly sample_id, canonical_behavior, mouse_ids")
        for row in reader:
            sample_id = row["sample_id"]
            behavior = row["canonical_behavior"]
            if not sample_id or sample_id in labels or behavior not in CANONICAL_BEHAVIORS:
                raise ValueError(f"invalid or repeated blind label: {sample_id!r}")
            target_ids = tuple(sorted(int(value) for value in row["mouse_ids"].split(",") if value))
            minimum = 1 if behavior == "isolation" else (
                3 if behavior in GROUP_BEHAVIORS else (
                    1 if behavior in INDIVIDUAL_BEHAVIORS else 2
                )
            )
            if len(target_ids) != len(set(target_ids)) or len(target_ids) < minimum or (
                behavior not in GROUP_BEHAVIORS and len(target_ids) != minimum
            ):
                raise ValueError(f"invalid target ID set for {behavior}: {sample_id!r}")
            labels[sample_id] = LabeledClip(sample_id, "", 1.0, "", behavior, target_ids)
    if not labels:
        raise ValueError("blind label file is empty")
    return labels


def _normalized_path(value: str) -> str:
    return str(Path(value).resolve()).casefold()


def _trajectory_sha256(value: str) -> str:
    digest = hashlib.sha256()
    with Path(value).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_blind_labels_from_split(
    split_path: Path, blind_manifest_path: Path
) -> dict[str, LabeledClip]:
    """Join labels to opaque IDs only after the prediction receipt exists."""

    opaque_to_track: dict[str, str] = {}
    with blind_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            # The predictor's label-free manifest uses sample_id; older
            # scorer manifests used opaque_id. Both contain the same opaque key.
            opaque_id = row.get("opaque_id") or row.get("sample_id", "")
            track_path = row.get("tracks_path", "")
            if not opaque_id or not track_path or opaque_id in opaque_to_track:
                raise ValueError("blind manifest has a missing or duplicate ID/path")
            opaque_to_track[opaque_id] = _normalized_path(track_path)
    track_to_opaque = {track: opaque for opaque, track in opaque_to_track.items()}
    if len(track_to_opaque) != len(opaque_to_track):
        raise ValueError("blind manifest repeats a trajectory path")
    blind_hash_to_opaque: dict[str, list[str]] = {}
    for opaque_id, track_path in opaque_to_track.items():
        digest = _trajectory_sha256(track_path)
        blind_hash_to_opaque.setdefault(digest, []).append(opaque_id)

    labels: dict[str, LabeledClip] = {}
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in {"val", "validation"}:
                raise ValueError("label split contains a non-validation row")
            track_path = row.get("tracks_path", "")
            opaque_id = track_to_opaque.get(_normalized_path(track_path))
            if not opaque_id and track_path:
                matches = blind_hash_to_opaque.get(_trajectory_sha256(track_path), [])
                if len(matches) > 1:
                    raise ValueError(
                        f"ambiguous anonymous trajectory content for label: {row.get('sample_id')!r}"
                    )
                opaque_id = matches[0] if matches else None
            behavior = row.get("canonical_behavior", "")
            if not opaque_id or opaque_id in labels or behavior not in CANONICAL_BEHAVIORS:
                raise ValueError(f"invalid or unmatched validation label: {row.get('sample_id')!r}")
            target_ids = tuple(sorted(int(value) for value in row["mouse_ids"].split(",") if value))
            minimum = 1 if behavior == "isolation" else (
                3 if behavior in GROUP_BEHAVIORS else (
                    1 if behavior in INDIVIDUAL_BEHAVIORS else 2
                )
            )
            if len(target_ids) != len(set(target_ids)) or len(target_ids) < minimum or (
                behavior not in GROUP_BEHAVIORS and len(target_ids) != minimum
            ):
                raise ValueError(f"invalid target ID set for {behavior}: {row.get('sample_id')!r}")
            labels[opaque_id] = LabeledClip(
                row["sample_id"], "", 1.0, "", behavior, target_ids
            )
    if labels.keys() != opaque_to_track.keys():
        raise ValueError("validation labels and anonymous trajectory manifest differ")
    return labels


def _add_precision_recall(
    metrics: dict[str, object],
    labels: dict[str, LabeledClip],
    predictions: dict[str, object],
) -> None:
    behavior_tp: Counter[str] = Counter()
    predicted_count: Counter[str] = Counter()
    exact_joint_tp: Counter[str] = Counter()
    for sample_id, label in labels.items():
        prediction = predictions[sample_id]
        predicted = prediction.behavior
        if predicted:
            predicted_count[predicted] += 1
        if predicted == label.behavior:
            behavior_tp[label.behavior] += 1
            if tuple(prediction.target_ids) == tuple(label.target_ids):
                exact_joint_tp[label.behavior] += 1
    per_behavior = metrics["per_behavior"]
    assert isinstance(per_behavior, dict)
    for behavior, values in per_behavior.items():
        support = int(values["support"])
        values["predicted_count"] = predicted_count[behavior]
        values["strict_exact_correct"] = exact_joint_tp[behavior]
        values["behavior_precision"] = (
            behavior_tp[behavior] / predicted_count[behavior]
            if predicted_count[behavior]
            else 0.0
        )
        values["behavior_recall"] = behavior_tp[behavior] / support if support else 0.0
        values["joint_id_precision"] = (
            exact_joint_tp[behavior] / predicted_count[behavior]
            if predicted_count[behavior]
            else 0.0
        )
        values["joint_id_recall"] = exact_joint_tp[behavior] / support if support else 0.0
        values["strict_exact_top1_accuracy"] = values["joint_id_recall"]
        precision = float(values["behavior_precision"])
        recall = float(values["behavior_recall"])
        values["behavior_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    total = int(metrics["n_samples"])
    strict_exact = sum(exact_joint_tp.values())
    metrics["strict_exact_correct"] = strict_exact
    metrics["strict_exact_top1_accuracy"] = strict_exact / total if total else 0.0


_CHECK_LABELS = {
    "member_count": "群体数量",
    "duration": "持续时长",
    "group_visibility": "群体可见比例",
    "member_support": "成员轨迹支持比例",
    "nnd_contraction": "最近邻距离收缩",
    "radius_contraction": "群体半径收缩",
    "inward_motion": "向心运动",
    "active_members": "活跃成员比例",
    "graph_growth": "邻近图增长/合并",
    "compact_nnd": "紧凑最近邻距离",
    "compact_radius": "紧凑群体半径",
    "stable_nnd": "最近邻距离稳定",
    "stable_radius": "群体半径稳定",
    "low_motion": "低运动成员比例",
    "contact_density": "接触图密度",
    "proximity_density": "邻近图密度",
    "graph_stability": "接触关系稳定性",
    "not_attack": "排除攻击性高运动",
}


def _write_error_analysis(
    path: Path,
    labels: dict[str, LabeledClip],
    predictions: dict[str, object],
    debug_path: Path,
) -> None:
    if not debug_path.is_file():
        return
    debug_by_sample: dict[str, list[dict[str, str]]] = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            debug_by_sample.setdefault(row.get("opaque_id", ""), []).append(row)

    rows: list[dict[str, str]] = []
    for sample_id, label in labels.items():
        prediction = predictions[sample_id]
        if prediction.behavior == label.behavior and tuple(prediction.target_ids) == tuple(
            label.target_ids
        ):
            continue
        checks_by_state = [
            row
            for row in debug_by_sample.get(sample_id, [])
            if row.get("state", "").upper()
            == ("HUDDLING" if label.behavior == "huddle" else "SOCIAL_CLUSTERING")
            and set(label.target_ids).issubset(
                {int(value) for value in row.get("member_ids", "").split(",") if value}
            )
        ] if label.behavior in {"huddle", "social_clustering"} else []
        if checks_by_state:
            chosen = max(
                checks_by_state,
                key=lambda item: (
                    -len([key for key, value in json.loads(item["checks"]).items() if value is False]),
                    float(item.get("score", 0.0) or 0.0),
                ),
            )
            checks = json.loads(chosen["checks"])
            failed = [
                _CHECK_LABELS.get(name, name)
                for name, passed in checks.items()
                if passed is False
            ]
            reason = "；".join(failed) if failed else "目标组候选条件通过，但未成为 Top-1"
            candidate_members = chosen.get("member_ids", "")
        elif label.behavior in {"huddle", "social_clustering"}:
            reason = "未形成包含全部标注目标 ID 的群体候选窗口"
            candidate_members = ""
        else:
            reason = "该目标标签不适用 Huddling/Social clustering 条件诊断"
            candidate_members = ""
        rows.append(
            {
                "sample_id": sample_id,
                "annotated_sample_id": label.sample_id,
                "true_behavior": label.behavior,
                "true_target_ids": ",".join(map(str, label.target_ids)),
                "predicted_behavior": prediction.behavior or "unrecognized",
                "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                "matching_group_candidate_ids": candidate_members,
                "failed_group_conditions": reason,
            }
        )
    if rows:
        with path.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_scored_predictions(
    path: Path,
    labels: dict[str, LabeledClip],
    predictions: dict[str, object],
) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        fields = (
            "sample_id", "opaque_id", "split", "predicted_behavior",
            "predicted_target_ids", "score", "strict_top1_correct",
            "compatible_top1_correct", "target_id_hit", "secondary_behaviors",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for opaque_id, label in labels.items():
            prediction = predictions[opaque_id]
            ids_exact = tuple(prediction.target_ids) == tuple(label.target_ids)
            strict = prediction.behavior == label.behavior and ids_exact
            # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction:
            # column kept as a strict mirror for schema compatibility.
            dual = strict
            writer.writerow(
                {
                    "sample_id": label.sample_id,
                    "opaque_id": opaque_id,
                    "split": "validation",
                    "predicted_behavior": prediction.behavior or "",
                    "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                    "score": prediction.score,
                    "strict_top1_correct": strict,
                    "compatible_top1_correct": dual,
                    "target_id_hit": ids_exact,
                    "secondary_behaviors": ",".join(prediction.secondary_behaviors),
                }
            )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--dataset-split", type=Path)
    parser.add_argument("--blind-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_locked_predictions(args.predictions_dir)
    if args.labels is not None:
        labels = load_blind_labels(args.labels)
    elif args.dataset_split is not None and args.blind_manifest is not None:
        receipt = json.loads(
            (args.predictions_dir / "prediction_receipt.json").read_text(encoding="utf-8")
        )
        if _trajectory_sha256(str(args.blind_manifest)) != receipt.get("blind_manifest_sha256"):
            raise ValueError("blind manifest differs from the blind prediction receipt")
        labels = load_blind_labels_from_split(args.dataset_split, args.blind_manifest)
    else:
        raise ValueError("provide --labels or both --dataset-split and --blind-manifest")
    if predictions.keys() != labels.keys():
        raise ValueError(
            f"blind prediction/label sample IDs differ: "
            f"missing={len(labels.keys() - predictions.keys())} "
            f"extra={len(predictions.keys() - labels.keys())}"
        )
    metrics = evaluate_top1([(labels[key], predictions[key]) for key in sorted(labels)])
    _add_precision_recall(metrics, labels, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    scored_predictions_path = args.output.parent / "validation_predictions.csv"
    _write_scored_predictions(scored_predictions_path, labels, predictions)
    print(
        "strict exact-ID Top-1: "
        f"{metrics['strict_exact_correct']}/{metrics['n_samples']} "
        f"({metrics['strict_exact_top1_accuracy']:.2%})"
    )
    debug_path = args.predictions_dir / "feature_debug.csv"
    error_path = args.output.parent / "validation_error_analysis.csv"
    _write_error_analysis(error_path, labels, predictions, debug_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
