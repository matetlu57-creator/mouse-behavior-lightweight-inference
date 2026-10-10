#!/usr/bin/env python3
"""Re-score frozen validation predictions under the compatible-ID contract.

This post-hoc builder never reruns behavior inference. It derives strict and
compatible labels from the frozen prediction CSV, then asks the existing
renderer to annotate a new output directory.

[SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction.
Compatible scoring keeps only the same-class target-ID containment path; the
frozen trajectory evidence is still checked for ID integrity but no longer
contributes contact-duration credit.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "isolation_10s_cumulative_20260927"
DEFAULT_RENDER = DEFAULT_RUN / "validation" / "rendered_compatible_20260930_v1"
DEFAULT_SITE_DATA = ROOT / "tools" / "video_review_viewer" / "public" / "review-data-v11.json"

BEHAVIOR_TO_PREDICTION = {
    "Approach": "approach",
    "Attack": "attack",
    "Huddling": "huddle",
    "Isolation": "isolation",
    "Snout-head_contact": "nose_head_contact",
    "Snout-rear_contact": "nose_tail_contact",
    "Social clustering": "social_clustering",
    "Static": "stationary",
    "Together": "together",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion: the merged display label plus legacy render aliases.
    "Locomotion": "locomotion",
    "Walking": "locomotion",
    "Running": "locomotion",
}
DISPLAY_NAMES = {
    "approach": "接近",
    "attack": "攻击/被攻击",
    "huddle": "扎堆",
    "isolation": "孤立",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "social_clustering": "社会聚集",
    "stationary": "静止",
    "together": "一起",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    "locomotion": "移动",
    "chase": "追逐/被追逐",
    "following": "跟随",
    "avoidance": "回避/被回避",
    "dispersal": "群体分散",
    "group_locomotion": "群体同步运动",
}
# [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
# ALLOWED_DUALS 同层双标签白名单（Together/Approach ↔ 鼻头/鼻尾接触）已整体删除。


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("opaque_id", ""))
            if not key or key in result:
                raise ValueError(f"trajectory evidence has missing/duplicate opaque_id: {key!r}")
            result[key] = item
    return result


def _ids(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


# [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
# _contact_seconds 累计接触时长助手随双标签信用一并删除；>0.5s 阈值不再参与计分。


def _score_row(
    prediction: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    truth_label = prediction["label_behavior"]
    truth = BEHAVIOR_TO_PREDICTION[truth_label]
    predicted = prediction.get("predicted_behavior", "")
    target_ids = _ids(prediction.get("label_target_ids", ""))
    predicted_ids = _ids(prediction.get("predicted_target_ids", ""))
    exact_ids = target_ids == predicted_ids
    includes_targets = bool(target_ids) and set(target_ids).issubset(predicted_ids)
    strict = predicted == truth and exact_ids
    same_class_containment = predicted == truth and includes_targets
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # Together/Approach 与鼻头/鼻尾接触的同层双标签信用（含 >0.5s 累计接触时长阈值）
    # 已整体删除；compatible 判定只保留同行为目标 ID 包含路径（same_class_containment）。
    compatible = same_class_containment

    if strict:
        reason = "严格正确：行为和目标 ID 集合均完全匹配。"
        id_relation = "exact"
    elif same_class_containment:
        extras = sorted(set(predicted_ids) - set(target_ids))
        reason = "兼容命中：预测 ID 包含全部标签目标"
        if extras:
            reason += f"；额外并入 ID {','.join(map(str, extras))}。"
        else:
            reason += "。"
        id_relation = "contains_label_ids" if extras else "exact"
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # 原 “elif dual_label: 双标签兼容命中” 说明分支已整体删除；跨行为预测一律走错误归因。
    else:
        missing = sorted(set(target_ids) - set(predicted_ids))
        extras = sorted(set(predicted_ids) - set(target_ids))
        if not predicted:
            reason = "未识别到行为，未输出目标 ID。"
        elif predicted == truth:
            reason = "行为类别正确，但预测 ID 未包含全部标签目标。"
        else:
            reason = (
                f"标签为{DISPLAY_NAMES.get(truth, truth)}，识别为"
                f"{DISPLAY_NAMES.get(predicted, predicted)}。"
            )
        if missing:
            reason += f" 缺少标签 ID {','.join(map(str, missing))}。"
        if extras:
            reason += f" 多出 ID {','.join(map(str, extras))}。"
        id_relation = (
            "missing_and_extra" if missing and extras
            else "missing" if missing
            else "extra" if extras
            else "none" if not predicted_ids
            else "exact"
        )

    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # compatible_labels 不再输出双标签组合；contact_seconds 与 compatible_dual_label 输出列
    # 已移除。旧输入 CSV 若仍带这些列，会随行透传但不再参与计分。
    compatible_labels = [truth] if same_class_containment and not exact_ids else []
    return {
        **prediction,
        "strict_top1_correct": str(strict).lower(),
        "compatible_top1_correct": str(compatible).lower(),
        "target_id_hit": str(includes_targets).lower(),
        "exact_target_id_hit": str(exact_ids).lower(),
        "compatible_labels": ",".join(compatible_labels),
        "compatibility_reason": reason,
        "id_relation": id_relation,
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = Counter(row["label_behavior"] for row in rows)
    predicted_count = Counter(row["predicted_behavior"] for row in rows if row["predicted_behavior"])
    strict_pred = Counter()
    compatible_pred = Counter()
    strict_truth = Counter()
    compatible_truth = Counter()
    containment_truth = Counter()
    for row in rows:
        truth = BEHAVIOR_TO_PREDICTION[row["label_behavior"]]
        predicted = row["predicted_behavior"]
        strict = row["strict_top1_correct"] == "true"
        compatible = row["compatible_top1_correct"] == "true"
        if strict:
            strict_pred[predicted] += 1
            strict_truth[truth] += 1
        if compatible:
            compatible_pred[predicted] += 1
            compatible_truth[truth] += 1
        if predicted == truth and set(_ids(row["label_target_ids"])).issubset(
            _ids(row["predicted_target_ids"])
        ):
            containment_truth[truth] += 1

    per_behavior = []
    for label, key in BEHAVIOR_TO_PREDICTION.items():
        n = support[label]
        p_n = predicted_count[key]
        per_behavior.append(
            {
                "key": key,
                "name": DISPLAY_NAMES[key],
                "support": n,
                "predictedCount": p_n,
                "strictPrecisionCorrect": strict_pred[key],
                "compatiblePrecisionCorrect": compatible_pred[key],
                "strictRecallCorrect": strict_truth[key],
                "compatibleRecallCorrect": compatible_truth[key],
                "precisionStrict": round(100 * strict_pred[key] / p_n, 2) if p_n else 0.0,
                "precisionCompatible": round(100 * compatible_pred[key] / p_n, 2) if p_n else 0.0,
                "recallStrict": round(100 * strict_truth[key] / n, 2) if n else 0.0,
                "recallCompatible": round(100 * compatible_truth[key] / n, 2) if n else 0.0,
                "strictTop1": round(100 * strict_truth[key] / n, 2) if n else 0.0,
                "sameClassContainmentTop1": round(100 * containment_truth[key] / n, 2) if n else 0.0,
                # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
                # dualTop1 仅为消费端 schema 兼容保留；其值为同行为 ID 包含兼容命中率，不再含双标签信用。
                "dualTop1": round(100 * compatible_truth[key] / n, 2) if n else 0.0,
            }
        )
    strict_total = sum(strict_truth.values())
    compatible_total = sum(compatible_truth.values())
    prediction_total = sum(predicted_count.values())
    return {
        "total": len(rows),
        "strictCorrect": strict_total,
        "compatibleCorrect": compatible_total,
        "strictTop1Rate": round(100 * strict_total / len(rows), 2) if rows else 0.0,
        "compatibleTop1Rate": round(100 * compatible_total / len(rows), 2) if rows else 0.0,
        "strictOverallPrecision": round(100 * strict_total / prediction_total, 2) if prediction_total else 0.0,
        "compatibleOverallPrecision": round(100 * compatible_total / prediction_total, 2) if prediction_total else 0.0,
        "strictOverallRecall": round(100 * strict_total / len(rows), 2) if rows else 0.0,
        "compatibleOverallRecall": round(100 * compatible_total / len(rows), 2) if rows else 0.0,
        "predictionCount": prediction_total,
        "perBehavior": per_behavior,
    }


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _browser_compatible_video_path(rendered_video: Path) -> Path:
    """Prefer the H.264 browser copy while keeping the rendered source intact."""
    rendered_video = rendered_video.resolve()
    copy_path = (
        rendered_video.parent.parent
        / "video_viewer_media"
        / rendered_video.parent.name
        / rendered_video.name
    )
    return copy_path if copy_path.is_file() else rendered_video


def _build_site_data(
    manifest_path: Path,
    scored_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_video_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_sample = {row["sample_id"]: row for row in scored_rows}
    entries = []
    for index, item in enumerate(manifest):
        source = rows_by_sample[item["sample_id"]]
        strict = source["strict_top1_correct"] == "true"
        compatible = source["compatible_top1_correct"] == "true"
        relative_video = item["output_video"]
        try:
            rendered_video = Path(relative_video).resolve()
            video_relative = _browser_compatible_video_path(rendered_video).relative_to(
                ROOT / "outputs"
            ).as_posix()
        except ValueError as exc:
            raise ValueError(f"rendered video must be inside outputs: {relative_video}") from exc
        entries.append(
            {
                "id": f"compatible-{index + 1:03d}",
                "sampleId": item["sample_id"],
                "truth": item["behavior"],
                "truthName": DISPLAY_NAMES.get(item["behavior"], item["behavior"]),
                "predicted": item["predicted_behavior"],
                "predictedName": DISPLAY_NAMES.get(item["predicted_behavior"], "未识别"),
                "status": "success" if compatible else "error",
                "strictTop1": strict,
                "compatibleTop1": compatible,
                "targetHit": source["target_id_hit"] == "true",
                "exactTargetHit": source["exact_target_id_hit"] == "true",
                "idRelation": source["id_relation"],
                "compatibilityReason": source["compatibility_reason"],
                # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
                # compatibleDualLabel/contactSeconds 站点字段已随双标签信用删除。
                "errorReason": source["compatibility_reason"],
                "targetIds": [str(x) for x in _ids(source["label_target_ids"])],
                "predictedTargetIds": [str(x) for x in _ids(source["predicted_target_ids"])],
                "videoUrl": f"/video-files/{video_relative}",
                "frames": int(item["frames"]),
                "fps": round(float(item["fps"]), 3),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "recordingSession": item.get("recording_session", ""),
                "selectedTrackCount": int(item.get("selected_track_count", 0)),
                "auxiliaryEvents": [
                    {
                        "behavior": event["behavior"],
                        "behaviorName": DISPLAY_NAMES.get(event["behavior"], event["behavior"]),
                        "ids": [str(x) for x in event.get("ids", [])],
                        "score": float(event.get("score", 0.0)),
                        "eventScope": event.get("event_scope", ""),
                    }
                    for event in item.get("auxiliary_events", [])
                ],
                "auxiliaryBehaviors": item.get("auxiliary_behaviors", []),
            }
        )
    success = sum(item["compatibleTop1"] for item in entries)
    behaviors = [
        {"key": row["key"], "name": row["name"]}
        for row in metrics["perBehavior"]
        if row["support"]
    ]
    return {
        "reviewVersion": "v11-compatible",
        "generatedFrom": [
            "frozen validation_predictions.csv",
            "frozen trajectory_evidence.jsonl",
            "existing rendered validation videos and tracks",
            "browser-compatible H.264 copies when present",
        ],
        "notes": [
            "本页只对已冻结的验证预测重新计分，没有重跑推理或调参。",
            "严格指标要求行为相同且 ID 集合完全相同；兼容指标按包含标签目标 ID 计命中。",
            # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
            "同层双标签兼容信用已删除：Together 新定义允许接触（无互动）；行为验证仅保留严格 Top-1 与同行为目标 ID 包含的兼容口径。",
            "视频继续显示算法实际输出，不会把兼容评分反写成模型预测行为。",
            "该验证集此前已被查看，本页是诊断性复核，不是新的盲测。",
        ],
        "metrics": {
            "total": len(entries),
            "success": success,
            "errors": len(entries) - success,
            "strictTop1Rate": metrics["strictTop1Rate"],
            "compatibleTop1Rate": metrics["compatibleTop1Rate"],
            "strictCorrect": metrics["strictCorrect"],
            "compatibleCorrect": metrics["compatibleCorrect"],
            "strictOverallPrecision": metrics["strictOverallPrecision"],
            "compatibleOverallPrecision": metrics["compatibleOverallPrecision"],
            "strictOverallRecall": metrics["strictOverallRecall"],
            "compatibleOverallRecall": metrics["compatibleOverallRecall"],
            "predictionCount": metrics["predictionCount"],
            "perBehavior": metrics["perBehavior"],
        },
        "behaviors": behaviors,
        "items": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER)
    parser.add_argument("--site-data", type=Path, default=DEFAULT_SITE_DATA)
    parser.add_argument("--skip-render", action="store_true", help="use a completed manifest already in --render-dir")
    args = parser.parse_args()

    validation_dir = args.run_root / "validation"
    split_path = args.run_root / "dataset_split" / "val.csv"
    prediction_path = validation_dir / "validation_predictions.csv"
    evidence_path = validation_dir / "trajectory_evidence.jsonl"
    predictions = _read_csv(prediction_path)
    evidence = _read_jsonl(evidence_path)
    if len(predictions) != 265 or len(predictions) != len(evidence):
        raise ValueError("frozen validation prediction/evidence row count is incomplete or mismatched")
    if {row["opaque_id"] for row in predictions} != set(evidence):
        raise ValueError("frozen prediction and trajectory-evidence opaque IDs differ")
    scored = [_score_row(row, evidence[row["opaque_id"]]) for row in predictions]
    metrics = _metrics(scored)
    args.render_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.render_dir / "compatible_validation_predictions.csv"
    metrics_path = args.render_dir / "compatible_validation_metrics.json"
    _write_predictions(scored_path, scored)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    render_manifest = args.render_dir / "videos" / "render_manifest.json"
    if not args.skip_render:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "render_validation_result.py"),
            "--dataset-split", str(split_path),
            "--predictions", str(scored_path),
            "--output-dir", str(render_manifest.parent),
            "--trajectory-evidence", str(evidence_path),
            "--max-per-behavior", "10000",
            "--include-auxiliary",
            "--resume",
            "--workers", "4",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    if not render_manifest.is_file():
        raise FileNotFoundError(f"completed renderer manifest not found: {render_manifest}")
    payload = _build_site_data(render_manifest, scored, metrics, render_manifest.parent)
    args.site_data.parent.mkdir(parents=True, exist_ok=True)
    args.site_data.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "render_dir": str(args.render_dir),
        "rendered_videos": len(json.loads(render_manifest.read_text(encoding="utf-8"))),
        "site_data": str(args.site_data),
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
