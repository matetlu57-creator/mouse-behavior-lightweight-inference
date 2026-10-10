#!/usr/bin/env python3
"""Apply a frozen train-only ranker plus core hierarchy to val trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT
from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    Top1Prediction,
    evaluate_top1,
    load_locked_predictions,
)
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable,
    build_candidate_table,
    predict_table,
)
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    HeuristicParameters,
    load_exported_features,
)
from mouse_behavior.evaluation.hierarchical_core_ranker import predict_hierarchical_top1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_predictions(path: Path, rows: list[tuple[str, Top1Prediction]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "predicted_behavior", "predicted_target_ids", "score"),
        )
        writer.writeheader()
        for sample_id, prediction in rows:
            writer.writerow({
                "sample_id": sample_id,
                "predicted_behavior": prediction.behavior or "",
                "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                "score": prediction.score,
            })


def _read_unlabeled_inputs(path: Path) -> list[tuple[str, str, float]]:
    """Project the val manifest to anonymous paths/FPS without accessing labels."""

    inputs: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError("validation manifest is empty")
        indexes = {name: index for index, name in enumerate(header)}
        required = {"tracks_path", "fps", "split"}
        missing = required - set(indexes)
        if missing:
            raise ValueError(f"validation manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"invalid validation row width at row {row_number + 1}")
            if row[indexes["split"]] not in {"val", "validation"}:
                raise ValueError(f"non-validation row at row {row_number + 1}")
            tracks_path = row[indexes["tracks_path"]].strip()
            try:
                fps = float(row[indexes["fps"]])
            except ValueError as exc:
                raise ValueError(f"invalid FPS at row {row_number + 1}") from exc
            if not tracks_path or fps <= 0:
                raise ValueError(f"invalid trajectory path/FPS at row {row_number + 1}")
            inputs.append((f"v-{row_number:06d}", tracks_path, fps))
    if not inputs:
        raise ValueError("validation manifest has no rows")
    return inputs


def _read_labels_after_receipt(path: Path) -> list[LabeledClip]:
    labels: list[LabeledClip] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tracks_path", "fps", "split", "recording_session",
            "canonical_behavior", "mouse_ids",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"validation labels are missing columns: {sorted(missing)}")
        for index, row in enumerate(reader, start=1):
            if row["split"] not in {"val", "validation"}:
                raise ValueError(f"non-validation label at row {index + 1}")
            behavior = row["canonical_behavior"]
            if behavior not in CANONICAL_BEHAVIORS:
                raise ValueError(f"unsupported validation label at row {index + 1}: {behavior}")
            try:
                fps = float(row["fps"])
                target_ids = tuple(sorted({
                    int(value) for value in row["mouse_ids"].split(",") if value.strip()
                }))
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"invalid validation IDs/FPS at row {index + 1}") from exc
            labels.append(LabeledClip(
                sample_id=f"v-{index:06d}",
                tracks_path=row["tracks_path"],
                fps=fps,
                recording_session=row["recording_session"],
                behavior=behavior,
                target_ids=target_ids,
            ))
    if not labels:
        raise ValueError("validation labels are empty")
    return labels


def _load_frozen_model(model_dir: Path, train_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = model_dir / "frozen_model.json"
    provenance_path = model_dir / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("validation_used_for_fitting") is not False:
        raise ValueError("model provenance does not prove train-only fitting")
    if provenance.get("train_manifest_sha256") != _sha256(train_manifest):
        raise ValueError("frozen model was not fitted on the supplied train manifest")
    model_hash = _sha256(model_path)
    if provenance.get("model_sha256") != model_hash:
        raise ValueError("frozen model hash differs from training provenance")
    for relative_path, expected_hash in provenance["source_sha256"].items():
        source = REPO_ROOT / relative_path
        if not source.is_file() or _sha256(source) != expected_hash:
            raise ValueError(f"frozen training source changed: {relative_path}")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not model.get("rule_profiles") or model.get("max_tracks", 0) < 2:
        raise ValueError("frozen model is missing rule profiles or track bound")
    return model, provenance


def _load_predictions_file(path: Path, expected_hash: str) -> dict[str, Top1Prediction]:
    if _sha256(path) != expected_hash:
        raise ValueError(f"prediction output changed after receipt: {path.name}")
    predictions: dict[str, Top1Prediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["sample_id"]
            if not sample_id or sample_id in predictions:
                raise ValueError("prediction output has missing or duplicate sample IDs")
            behavior = row["predicted_behavior"] or None
            target_ids = tuple(int(value) for value in row["predicted_target_ids"].split(",") if value)
            predictions[sample_id] = Top1Prediction(behavior, target_ids, float(row["score"]))
    return predictions


def _write_scored_rows(path: Path, rows: list[tuple[LabeledClip, Top1Prediction]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "true_behavior", "true_target_ids", "predicted_behavior",
            "predicted_target_ids", "behavior_correct", "target_ids_included", "joint_top1_correct",
        ))
        writer.writeheader()
        for label, prediction in rows:
            behavior_ok = prediction.behavior == label.behavior
            ids_ok = set(label.target_ids).issubset(set(prediction.target_ids))
            writer.writerow({
                "sample_id": label.sample_id,
                "true_behavior": label.behavior,
                "true_target_ids": ",".join(map(str, label.target_ids)),
                "predicted_behavior": prediction.behavior or "",
                "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                "behavior_correct": behavior_ok,
                "target_ids_included": ids_ok,
                "joint_top1_correct": behavior_ok and ids_ok,
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a new validation output directory; existing runs are preserved")

    model, provenance = _load_frozen_model(args.model_dir, args.train_manifest)
    model_path = args.model_dir / "frozen_model.json"
    model_hash = _sha256(model_path)
    val_manifest_hash = _sha256(args.val_manifest)
    inputs = _read_unlabeled_inputs(args.val_manifest)
    train_trajectory_hashes = set(provenance["training_trajectory_sha256"].values())

    args.output_dir.mkdir(parents=True)
    hierarchy_path = args.output_dir / "predictions.csv"
    baseline_path = args.output_dir / "baseline_predictions.csv"
    profiles = tuple(HeuristicParameters(**item) for item in model["rule_profiles"])
    hierarchy_rows: list[tuple[str, Top1Prediction]] = []
    baseline_rows: list[tuple[str, Top1Prediction]] = []
    validation_trajectory_hashes: dict[str, str] = {}
    for index, (sample_id, tracks_path, fps) in enumerate(inputs, start=1):
        trajectory = Path(tracks_path)
        trajectory_hash = _sha256(trajectory)
        if trajectory_hash in train_trajectory_hashes:
            raise ValueError("validation trajectory is present in the training set")
        validation_trajectory_hashes[sample_id] = trajectory_hash
        features = load_exported_features(
            trajectory, fps=fps, max_tracks=int(model["max_tracks"])
        )
        table: CandidateTable = build_candidate_table(features, profiles)
        baseline_rows.append((sample_id, predict_table(table, model)))
        hierarchy_rows.append((sample_id, predict_hierarchical_top1(table, model)))
        if index % 25 == 0 or index == len(inputs):
            print(f"trajectory-only validation inference {index}/{len(inputs)}")

    _write_predictions(hierarchy_path, hierarchy_rows)
    _write_predictions(baseline_path, baseline_rows)
    receipt = {
        "prediction_count": len(inputs),
        "predictions_sha256": _sha256(hierarchy_path),
        "baseline_predictions_sha256": _sha256(baseline_path),
        "validation_manifest_sha256": val_manifest_hash,
        "validation_trajectory_sha256": validation_trajectory_hashes,
        "model_sha256": model_hash,
        "hierarchy_adapter_sha256": _sha256(REPO_ROOT / "src/mouse_behavior/evaluation/hierarchical_core_ranker.py"),
        "hierarchical_core_sha256": _sha256(REPO_ROOT / "src/mouse_behavior/behavior/hierarchical_core.py"),
        "contains_ground_truth": False,
    }
    _write_json(args.output_dir / "prediction_receipt.json", receipt)

    # The label-bearing columns are opened only after both predictions have
    # been written and their hashes sealed in the receipt above.
    predictions = load_locked_predictions(args.output_dir)
    baseline_predictions = _load_predictions_file(
        baseline_path, receipt["baseline_predictions_sha256"]
    )
    if set(predictions) != {sample_id for sample_id, _, _ in inputs}:
        raise ValueError("hierarchical prediction IDs differ from blind validation inputs")
    if set(baseline_predictions) != set(predictions):
        raise ValueError("baseline prediction IDs differ from hierarchical prediction IDs")
    if _sha256(args.val_manifest) != val_manifest_hash or _sha256(model_path) != model_hash:
        raise ValueError("validation manifest or frozen model changed during inference")

    labels = _read_labels_after_receipt(args.val_manifest)
    if [item.sample_id for item in labels] != [sample_id for sample_id, _, _ in inputs]:
        raise ValueError("validation label order/count differs from anonymous prediction order")
    session_overlap = set(provenance.get("train_sessions", ())) & {
        item.recording_session for item in labels
    }
    if session_overlap:
        raise ValueError(f"training and validation recording sessions overlap: {sorted(session_overlap)}")
    hierarchy_scored = [(label, predictions[label.sample_id]) for label in labels]
    baseline_scored = [(label, baseline_predictions[label.sample_id]) for label in labels]
    hierarchy_metrics = evaluate_top1(hierarchy_scored)
    baseline_metrics = evaluate_top1(baseline_scored)
    _write_json(args.output_dir / "validation_metrics.json", hierarchy_metrics)
    _write_json(args.output_dir / "baseline_metrics.json", baseline_metrics)
    _write_scored_rows(args.output_dir / "validation_predictions_scored.csv", hierarchy_scored)
    _write_scored_rows(args.output_dir / "baseline_predictions_scored.csv", baseline_scored)

    training_metrics = json.loads((args.model_dir / "training_metrics.json").read_text(encoding="utf-8"))
    lines = [
        "# 新层级输出适配器：原验证集评估", "",
        "评估对象：训练集冻结的阈值候选与线性评分器，加上 hierarchical_core 的个体→社交→群体逐鼠主行为解析。只有正轨迹规则证据的候选进入层级核心；较高层只覆盖参与同一鼠的低层行为。",
        "推理阶段仅使用轨迹文件、FPS 与片段内视觉 track ID。匿名预测 CSV 已写入并经 SHA-256 回执锁定后，才读取验证标签进行评分。此划分此前使用过，因此这是原验证集评估，不是全新盲测。", "",
        f"验证样本：{len(labels)}。联合 Top-1（行为正确且预测目标 ID 包含全部标签目标 ID）：**{hierarchy_metrics['joint_top1_accuracy']:.2%}**；宏平均：**{hierarchy_metrics['macro_joint_top1_accuracy']:.2%}**。",
        f"对照：同一冻结模型、不应用新层级解析的 Top-1 为 {baseline_metrics['joint_top1_accuracy']:.2%}（宏平均 {baseline_metrics['macro_joint_top1_accuracy']:.2%}）。训练模型原始全训练集 Top-1（未应用层级后处理）为 {training_metrics['joint_top1_accuracy']:.2%}。", "",
        "| 行为 | 正确/数量 | 联合 Top-1 | 仅行为正确 | ID 包含率 | 对照 Top-1 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for behavior, value in hierarchy_metrics["per_behavior"].items():
        base = baseline_metrics["per_behavior"].get(behavior, {})
        lines.append(
            f"| {behavior} | {value['joint_correct']}/{value['support']} | "
            f"{value['joint_top1_accuracy']:.2%} | {value['behavior_top1_accuracy']:.2%} | "
            f"{value['id_top1_accuracy']:.2%} | {base.get('joint_top1_accuracy', 0.0):.2%} |"
        )
    lines.extend([
        "", "## 限制", "",
        "层级核心的冲突解析已接入并按验证轨迹运行；底层候选仍由现有导出轨迹规则适配器产生，尚未将新的逐窗 PoseObservation/PairWindowStats/GroupWindowStats 原生检测器替换进连续生产推理和渲染链路。因此该结果验证的是‘训练集冻结候选 + 新层级主行为输出’，不是端到端生产流水线，也不能据此宣称所有重构规则都已验证。",
        "训练 Top-1 采用既有 ranker 的训练指标（无层级后处理）；本实验没有用验证结果调整任何参数。若层级结果不佳，后续只能在训练集上重训/改规则后再使用未查看过的数据作独立盲测。",
    ])
    with (args.output_dir / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(
        f"hierarchical validation joint Top-1={hierarchy_metrics['joint_top1_accuracy']:.2%}; "
        f"baseline={baseline_metrics['joint_top1_accuracy']:.2%}; output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
