"""Build the review data consumed by the local validation-video viewer.

The viewer intentionally reads only the frozen validation render manifests and
the already exported prediction table.  It does not run inference or inspect
the validation labels at runtime.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exported_dataset_calibration"
VIEWER_ROOT = REPO_ROOT / "tools" / "video_review_viewer"
DATA_PATH = VIEWER_ROOT / "public" / "review-data.json"
DEFAULT_CORRECT_MANIFEST = "rendered_validation_with_behavior_v5"
DEFAULT_ERROR_MANIFEST = "rendered_validation_errors_with_behavior_v5"
DEFAULT_PREDICTIONS = REPO_ROOT / "outputs" / "exported_dataset_calibration_v5" / "validation_predictions.csv"

DISPLAY_NAMES = {
    "approach": "接近",
    "attack": "攻击/被攻击",
    "huddle": "扎堆",
    "isolation": "孤立",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "social_clustering": "社会聚集",
    "group_locomotion": "群体同步运动",
    "dispersal": "群体分散",
    "stationary": "静止",
    "together": "一起",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    "locomotion": "移动",
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "following": "跟随",
    "chase": "追逐/被追逐",
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "avoidance": "回避/被回避",
}

BEHAVIOR_RULES = {
    "social_clustering": {
        "title": "形成过程 + 身份连续",
        "summary": "目标 ID 映射连续，且小鼠从较远位置向稳定聚集区域靠近。",
        "cue": "短时检测框缺失不自动等同于身份丢失；以 ID lineage 是否连续为准。",
    },
    "huddle": {
        "title": "稳定 ID + 低运动休息",
        "summary": "轨迹 ID 保持稳定，目标小鼠长期低运动并维持近距离休息关系。",
        "cue": "允许短时漏框；漏框本身不能把稳定身份改判为社会聚集。",
    },
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("sample_id", "")
            if not sample_id:
                continue
            target_ids = [part.strip() for part in row.get("target_ids", "").split(",") if part.strip()]
            try:
                candidate_scores = json.loads(row.get("candidate_scores", "{}"))
            except json.JSONDecodeError:
                candidate_scores = {}
            result[sample_id] = {
                "recordingSession": row.get("recording_session", ""),
                "targetIds": target_ids,
                "targetIdsAvailable": row.get("target_ids_available", "").lower() == "true",
                "targetIdHit": row.get("target_id_hit", "").lower() == "true",
                "candidateScores": candidate_scores,
                "allDetectedBehaviors": [
                    item.strip() for item in row.get("all_detected_behaviors", "").split(",") if item.strip()
                ],
                "selectedTrackCount": int(row.get("selected_track_count") or 0),
            }
    return result


def _relative_video_url(video_path: str, output_root: Path) -> str:
    """Return a safe URL consumed by the Vite video-files middleware."""

    root = output_root.resolve()
    raw = Path(video_path)
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (REPO_ROOT / raw).resolve()
    relative_path = resolved.relative_to(root)
    media_candidate = root / "video_viewer_media" / relative_path
    if media_candidate.exists():
        relative_path = media_candidate.relative_to(root)
    try:
        output_prefix = root.relative_to((REPO_ROOT / "outputs").resolve())
    except ValueError:
        output_prefix = Path(root.name)
    relative = (output_prefix / relative_path).as_posix()
    return f"/video-files/{relative}"


def _load_manifest(
    output_root: Path,
    directory_name: str,
    source_kind: str,
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_path = output_root / directory_name / "render_manifest.json"
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(_read_json(manifest_path)):
        sample_id = str(item.get("sample_id", ""))
        truth = str(item.get("behavior", ""))
        predicted = str(item.get("predicted_behavior", ""))
        prediction = predictions.get(sample_id, {})
        rule = BEHAVIOR_RULES.get(truth) or BEHAVIOR_RULES.get(predicted) or {}
        target_id_set = {str(value) for value in prediction.get("targetIds", [])}
        auxiliary_events = []
        for raw_event in item.get("auxiliary_events", []) or []:
            if not isinstance(raw_event, dict):
                continue
            ids = [str(value) for value in raw_event.get("ids", [])]
            # The HTML section is specifically for non-target animals.  The
            # rendered video still keeps target-participating pair/group
            # events, but the side panel avoids repeating the target result.
            if ids and not any(value not in target_id_set for value in ids):
                continue
            behavior_key = str(raw_event.get("behavior", ""))
            auxiliary_events.append(
                {
                    "behavior": behavior_key,
                    "behaviorName": DISPLAY_NAMES.get(behavior_key, behavior_key),
                    "ids": ids,
                    "score": float(raw_event.get("score", 0.0) or 0.0),
                    "eventScope": str(raw_event.get("event_scope", "")),
                }
            )
        auxiliary_events.sort(key=lambda event: (-event["score"], event["behavior"], event["ids"]))
        strict_top1 = bool(item.get("strict_top1", False))
        target_hit = bool(item.get("target_hit", False))
        if strict_top1:
            review_reason = "行为与目标 ID 均匹配"
        elif predicted != truth:
            review_reason = f"预测为{DISPLAY_NAMES.get(predicted, predicted)}，标签为{DISPLAY_NAMES.get(truth, truth)}"
        elif not target_hit:
            review_reason = "行为一致，但目标 ID 未命中"
        else:
            review_reason = str(item.get("error_reason", "识别与标签不一致"))
        entries.append(
            {
                "id": f"{source_kind}-{index:03d}",
                "sampleId": sample_id,
                "truth": truth,
                "truthName": DISPLAY_NAMES.get(truth, truth),
                "predicted": predicted,
                "predictedName": DISPLAY_NAMES.get(predicted, predicted),
                "targetHit": target_hit,
                "strictTop1": strict_top1,
                "status": "success" if strict_top1 else "error",
                "sourceKind": source_kind,
                "errorReason": review_reason,
                "sourceClip": item.get("source_clip", ""),
                "sourceTracks": item.get("source_tracks", ""),
                "videoUrl": _relative_video_url(str(item.get("output_video", "")), output_root),
                "frames": int(item.get("frames") or 0),
                "fps": round(float(item.get("fps") or 0), 3),
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "recordingSession": prediction.get("recordingSession", ""),
                "targetIds": prediction.get("targetIds", []),
                "targetIdsAvailable": prediction.get("targetIdsAvailable", False),
                "candidateScores": prediction.get("candidateScores", {}),
                "allDetectedBehaviors": prediction.get("allDetectedBehaviors", []),
                "selectedTrackCount": prediction.get("selectedTrackCount", 0),
                "auxiliaryEvents": auxiliary_events,
                "auxiliaryBehaviors": sorted({event["behavior"] for event in auxiliary_events}),
                "ruleTitle": rule.get("title", ""),
                "ruleSummary": rule.get("summary", ""),
                "ruleCue": rule.get("cue", ""),
            }
        )
    return entries


def build(
    output_root: Path = OUTPUT_ROOT,
    data_path: Path = DATA_PATH,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    correct_manifest: str = DEFAULT_CORRECT_MANIFEST,
    error_manifest: str = DEFAULT_ERROR_MANIFEST,
) -> dict[str, Any]:
    predictions = _read_predictions(prediction_path)
    behavior_examples = _load_manifest(output_root, correct_manifest, "behavior-example", predictions)
    errors = _load_manifest(output_root, error_manifest, "error", predictions)

    # A sample can occur in the one-per-behavior examples and in the complete
    # error export.  Prefer the complete error export so the same clip is not
    # shown twice in the review list.
    by_sample: dict[str, dict[str, Any]] = {}
    for item in behavior_examples + errors:
        current = by_sample.get(item["sampleId"])
        if current is None or item["sourceKind"] == "error":
            by_sample[item["sampleId"]] = item
    records = list(by_sample.values())

    records.sort(key=lambda item: (item["status"] != "success", item["truthName"], item["sampleId"]))
    success_count = sum(item["status"] == "success" for item in records)
    error_count = len(records) - success_count
    behaviors = sorted({item["truth"] for item in records}, key=lambda key: DISPLAY_NAMES.get(key, key))
    payload = {
        "generatedFrom": [
            str((output_root / correct_manifest / "render_manifest.json").resolve().relative_to(REPO_ROOT.resolve())),
            str((output_root / error_manifest / "render_manifest.json").resolve().relative_to(REPO_ROOT.resolve())),
            str(prediction_path.resolve().relative_to(REPO_ROOT.resolve()) if prediction_path.resolve().is_relative_to(REPO_ROOT.resolve()) else prediction_path),
        ],
        "metrics": {
            "total": len(records),
            "success": success_count,
            "errors": error_count,
            "strictTop1Rate": round(success_count / len(records) * 100, 2) if records else 0,
        },
        "behaviors": [{"key": key, "name": DISPLAY_NAMES.get(key, key)} for key in behaviors],
        "items": records,
    }
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prediction-file", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--correct-manifest", default=DEFAULT_CORRECT_MANIFEST)
    parser.add_argument("--error-manifest", default=DEFAULT_ERROR_MANIFEST)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    payload = build(
        output_root=args.output_root,
        data_path=args.output,
        prediction_path=args.prediction_file,
        correct_manifest=args.correct_manifest,
        error_manifest=args.error_manifest,
    )
    print(json.dumps({"path": str(args.output), "items": len(payload["items"]), "metrics": payload["metrics"]}, ensure_ascii=False))
