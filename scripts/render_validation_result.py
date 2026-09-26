"""Render exported validation clips with frozen prediction summaries.

Exported-dataset validation samples contain a clip-local ``tracks.json`` and
``clip.mp4`` rather than a project trajectory cache, so the production cache
loader cannot be used directly.  The frame renderer nevertheless reuses the
project's shared ``visualization.overlay`` behavior labels, role colors, box
labels, and ID sidebar.  The behavior result is read from the already-frozen
validation prediction CSV; no recalibration is run and no label is passed back
into the behavior rule evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mouse_behavior.visualization.overlay import (  # noqa: E402
    BoxLabel,
    DISPLAY_NAMES_ZH,
    MouseOverlay,
    build_mouse_overlays,
    canonical_behavior,
    draw_behavior_sidebar,
    draw_text_overlay,
    format_mouse_id,
    focus_behavior_name_zh,
    font_size_for_frame,
    load_font,
    select_display_events,
    sidebar_width_for_frame,
)
from mouse_behavior.evaluation.exported_dataset import (  # noqa: E402
    GROUP_BEHAVIORS,
    PAIR_BEHAVIORS,
    HeuristicParameters,
    load_exported_features,
    predict_features,
)


GROUP_RENDER_BEHAVIORS = frozenset(
    {
        "huddle",
        "social_clustering",
        "group_locomotion",
        "dispersal",
        "isolation",
    }
)
UNRECOGNIZED_BEHAVIOR_COLOR_BGR = (60, 60, 220)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_tracks(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"tracks.json 必须是帧列表: {path}")
    return {
        int(frame.get("frame", index)): list(frame.get("detections", []))
        for index, frame in enumerate(payload)
        if isinstance(frame, dict)
    }


def _safe_name(value: str) -> str:
    chars = [char if char.isalnum() or char in "-_" else "_" for char in value]
    return "".join(chars).strip("_") or "validation"


def _target_ids(row: dict[str, str]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {int(value.strip()) for value in row.get("mouse_ids", "").split(",") if value.strip()}
        )
    )


def _is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _prediction_event(
    prediction: dict[str, str], target_ids: tuple[int, ...], total_frames: int
) -> dict[str, Any] | None:
    predicted = canonical_behavior(prediction.get("predicted_behavior"))
    if not predicted or predicted == "none" or not target_ids or total_frames <= 0:
        return None
    event: dict[str, Any] = {
        "behavior": predicted,
        "behavior_name_zh": focus_behavior_name_zh(predicted),
        "candidate_level": "strong",
        "event_scope": "individual" if len(target_ids) == 1 else "pair",
        "start_frame": 0,
        "end_frame": total_frames - 1,
        "peak_frame": max(total_frames // 2, 0),
        "peak_score": 1.0,
        # The frozen Top-1 result is the target review label.  Auxiliary
        # events may be rendered for other IDs, but must not replace it for
        # the labelled target mouse.
        "_render_priority": 10000,
        "event_source": "target_top1",
    }
    if predicted in GROUP_RENDER_BEHAVIORS:
        # Group overlays resolve participants from member_ids.  Isolation is
        # a one-member group relation, so storing its target as actor_id
        # would make the event visible in the sidebar but leave the mouse
        # box in the generic “仅追踪” state.
        event["event_scope"] = "group"
        event["member_ids"] = ",".join(str(value) for value in target_ids)
    elif len(target_ids) == 1:
        event["actor_id"] = target_ids[0]
        event["event_scope"] = "individual"
    elif len(target_ids) == 2:
        event["pair_key"] = f"{target_ids[0]},{target_ids[1]}"
    else:
        event["event_scope"] = "group"
        event["member_ids"] = ",".join(str(value) for value in target_ids)
    return event


def _best_positive_score(
    values: dict[str, Any], behaviors: tuple[str, ...]
) -> tuple[str, float] | None:
    candidates: list[tuple[str, float]] = []
    for behavior in behaviors:
        try:
            score = float(values.get(behavior, 0.0))
        except (TypeError, ValueError):
            continue
        if score > 0.0:
            candidates.append((behavior, score))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], item[0]))


def _auxiliary_events_from_evidence(
    evidence: dict[str, Any],
    target_ids: tuple[int, ...],
    total_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert blind trajectory evidence into display-only events.

    The target Top-1 event remains sourced from the frozen prediction CSV.
    These events are recomputed from the clip-local tracks and frozen
    heuristic parameters, so they never consume validation labels. Keep one
    locomotion event per visual ID and a separate isolation group event;
    distinct behavior layers can be simultaneously true for the same mouse.
    """

    if total_frames <= 0:
        return [], []
    target_set = set(target_ids)
    events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    end_frame = total_frames - 1

    def add_event(
        behavior: str,
        ids: tuple[int, ...],
        score: float,
        event_scope: str,
        *,
        actor_id: int | None = None,
        pair_key: str = "",
    ) -> None:
        if not ids or not behavior or score <= 0.0:
            return
        if event_scope == "individual" and actor_id is None:
            return
        event: dict[str, Any] = {
            "behavior": behavior,
            "behavior_name_zh": focus_behavior_name_zh(behavior),
            "candidate_level": "auxiliary",
            "event_source": "trajectory_heuristic",
            "event_scope": event_scope,
            "start_frame": 0,
            "end_frame": end_frame,
            "peak_frame": max(total_frames // 2, 0),
            "peak_score": float(score),
            "_render_priority": 0,
        }
        if event_scope == "individual":
            event["actor_id"] = int(actor_id)
        elif event_scope == "pair":
            event["pair_key"] = pair_key or ",".join(map(str, ids))
        else:
            member_text = ",".join(map(str, ids))
            event["member_ids"] = member_text
            # A stable key keeps separate groups from being deduplicated.
            event["pair_key"] = member_text
        events.append(event)
        summaries.append(
            {
                "behavior": behavior,
                "ids": list(ids),
                "score": round(float(score), 4),
                "event_scope": event_scope,
                "includes_target": bool(target_set.intersection(ids)),
            }
        )

    identity_scores = dict(evidence.get("identity_scores", {}))
    for text_id, raw_scores in identity_scores.items():
        try:
            track_id = int(text_id)
        except (TypeError, ValueError):
            continue
        scores = dict(raw_scores)
        best = _best_positive_score(scores, ("stationary", "walking"))
        if best is not None:
            add_event(best[0], (track_id,), best[1], "individual", actor_id=track_id)
        isolation_score = _best_positive_score(scores, ("isolation",))
        if isolation_score is not None:
            add_event(isolation_score[0], (track_id,), isolation_score[1], "group")

    for text_pair, raw_scores in dict(evidence.get("pair_scores", {})).items():
        try:
            ids = tuple(sorted(int(value) for value in str(text_pair).split(",")))
        except (TypeError, ValueError):
            continue
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        best = _best_positive_score(dict(raw_scores), PAIR_BEHAVIORS)
        if best is not None:
            add_event(best[0], ids, best[1], "pair", pair_key=",".join(map(str, ids)))

    for text_group, raw_scores in dict(evidence.get("group_scores", {})).items():
        try:
            ids = tuple(sorted(int(value) for value in str(text_group).split(",")))
        except (TypeError, ValueError):
            continue
        if len(ids) < 3 or len(ids) != len(set(ids)):
            continue
        best = _best_positive_score(dict(raw_scores), GROUP_BEHAVIORS)
        if best is not None:
            add_event(best[0], ids, best[1], "group")

    summaries.sort(key=lambda item: (-float(item["score"]), item["behavior"], item["ids"]))
    return events, summaries


def _error_reason(prediction: dict[str, str], truth_name: str, predicted_name: str) -> str:
    target_hit = _is_true(prediction.get("target_id_hit", "False"))
    if not predicted_name or predicted_name == "未识别":
        return f"未识别到{truth_name}"
    if predicted_name == truth_name:
        if not target_hit:
            return f"行为识别正确，但目标 ID 未命中（标签：{truth_name}）"
        return f"行为与目标 ID 未共同达到严格 Top-1（识别：{predicted_name}）"
    if target_hit:
        return f"目标 ID 已命中，但行为识别为{predicted_name}（标签为{truth_name}）"
    return f"行为识别为{predicted_name}（标签为{truth_name}），且目标 ID 未命中"


def _mark_unrecognized_targets(
    mouse_overlays: dict[int, MouseOverlay],
    target_ids: tuple[int, ...],
    predicted: str,
) -> None:
    """Mark labelled targets when the blind predictor emitted no behavior."""

    if predicted not in {"", "none"}:
        return
    for target_id in target_ids:
        if target_id not in mouse_overlays:
            continue
        mouse_overlays[target_id] = MouseOverlay(
            text=f"{format_mouse_id(target_id)}｜未识别",
            color_bgr=UNRECOGNIZED_BEHAVIOR_COLOR_BGR,
            priority=(1, 0.0),
        )


def render_row(
    row: dict[str, str],
    prediction: dict[str, str],
    output_path: Path,
    parameters: HeuristicParameters | None = None,
    include_auxiliary: bool = False,
) -> dict[str, Any]:
    clip_path = Path(row["clip_path"])
    tracks_path = Path(row["tracks_path"])
    if not clip_path.exists():
        raise FileNotFoundError(f"验证片段不存在: {clip_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"验证轨迹不存在: {tracks_path}")

    track_frames = _load_tracks(tracks_path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开验证片段: {clip_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"验证片段尺寸无效: {clip_path}")
    if fps <= 0:
        fps = float(row.get("fps") or 30.0)

    target_ids = _target_ids(row)
    predicted = canonical_behavior(prediction.get("predicted_behavior"))
    truth = canonical_behavior(row.get("canonical_behavior"))
    predicted_name = DISPLAY_NAMES_ZH.get(predicted, predicted) if predicted else "未识别"
    truth_name = DISPLAY_NAMES_ZH.get(truth, truth)
    strict = prediction.get("strict_top1_correct", "False")
    target_hit = prediction.get("target_id_hit", "False")
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(track_frames))
    predicted_event = _prediction_event(prediction, target_ids, total_frames)
    # The v6 review renderer can add label-free trajectory candidates for all
    # visible IDs.  The target Top-1 event carries a dedicated render priority,
    # while the overlay module applies social > group > individual hierarchy
    # to every other candidate.
    auxiliary_summaries: list[dict[str, Any]] = []
    auxiliary_events: list[dict[str, Any]] = []
    if include_auxiliary and parameters is not None:
        features = load_exported_features(tracks_path, fps=fps, max_tracks=64)
        evidence = predict_features(features, parameters)
        auxiliary_events, auxiliary_summaries = _auxiliary_events_from_evidence(
            evidence, target_ids, total_frames
        )
    event_rows: list[dict[str, Any]] = auxiliary_events
    if predicted_event is not None:
        event_rows.append(predicted_event)
    reason = (
        "正确识别" if _is_true(strict) else _error_reason(prediction, truth_name, predicted_name)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidebar_width = sidebar_width_for_frame(width, height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + sidebar_width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建渲染视频: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            active_events, display_layer = select_display_events(event_rows)
            detections = track_frames.get(frame_index, [])
            valid_ids = []
            for detection in detections:
                try:
                    valid_ids.append(int(detection["track_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
            mouse_overlays = build_mouse_overlays(
                active_events,
                display_layer,
                valid_ids,
            )
            # If the target Top-1 predictor emits no class, mark only the
            # labelled target IDs explicitly. Auxiliary events remain
            # independent of that target-only fallback.
            _mark_unrecognized_targets(mouse_overlays, target_ids, predicted)
            box_labels: list[BoxLabel] = []
            for detection in detections:
                try:
                    track_id = int(detection["track_id"])
                    x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
                except (KeyError, TypeError, ValueError):
                    continue
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                overlay = mouse_overlays.get(track_id)
                if overlay is None:
                    overlay = MouseOverlay(
                        text=f"ID {track_id:02d}｜仅追踪",
                        color_bgr=(170, 170, 170),
                        priority=(0, 0.0),
                    )
                color = overlay.color_bgr
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                box_labels.append(BoxLabel((x1, y1, x2, y2), overlay.text, color))
            panel_lines = [
                f"验证审查｜标签行为：{truth_name}",
                f"识别行为：{predicted_name}",
                f"结果：{reason}",
                f"target_hit={target_hit}  strict_top1={strict}  辅助事件={len(auxiliary_events)}",
            ]
            frame = draw_text_overlay(
                frame,
                box_labels,
                panel_lines,
                load_font(None, font_size_for_frame(width, height)),
            )
            frame = draw_behavior_sidebar(
                frame,
                frame_index=frame_index,
                total_frames=total_frames,
                fps=fps,
                active_event_count=len(active_events),
                display_events=active_events,
                display_layer=display_layer,
                mouse_overlays=mouse_overlays,
                valid_ids=valid_ids,
                panel_width=sidebar_width,
                empty_event_text=(
                    "当前帧无行为预测，目标框显示“未识别”" if predicted in {"", "none"} else None
                ),
            )
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return {
        "sample_id": row["sample_id"],
        "behavior": row["canonical_behavior"],
        "predicted_behavior": predicted,
        "target_hit": _is_true(target_hit),
        "strict_top1": _is_true(strict),
        "behavior_correct": bool(predicted and predicted != "none" and predicted == truth),
        "prediction_available": bool(predicted and predicted != "none"),
        "error_reason": reason,
        "rendering_backend": "mouse_behavior.visualization.overlay",
        "source_clip": str(clip_path),
        "source_tracks": str(tracks_path),
        "output_video": str(output_path),
        "frames": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
        "auxiliary_events": auxiliary_summaries,
        "auxiliary_behaviors": sorted({item["behavior"] for item in auxiliary_summaries}),
    }


def _manifest_from_existing(
    row: dict[str, str], prediction: dict[str, str], output_path: Path
) -> dict[str, Any]:
    """Rebuild a manifest entry without re-encoding an existing render."""

    if not output_path.exists():
        raise FileNotFoundError(f"已有渲染视频不存在: {output_path}")
    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"已有渲染视频无法打开: {output_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(row.get("fps") or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    predicted = canonical_behavior(prediction.get("predicted_behavior"))
    truth = canonical_behavior(row.get("canonical_behavior"))
    predicted_name = DISPLAY_NAMES_ZH.get(predicted, predicted) if predicted else "未识别"
    truth_name = DISPLAY_NAMES_ZH.get(truth, truth)
    strict = prediction.get("strict_top1_correct", "False")
    target_hit = prediction.get("target_id_hit", "False")
    reason = (
        "正确识别" if _is_true(strict) else _error_reason(prediction, truth_name, predicted_name)
    )
    return {
        "sample_id": row["sample_id"],
        "behavior": row["canonical_behavior"],
        "predicted_behavior": predicted,
        "target_hit": _is_true(target_hit),
        "strict_top1": _is_true(strict),
        "behavior_correct": bool(predicted and predicted != "none" and predicted == truth),
        "prediction_available": bool(predicted and predicted != "none"),
        "error_reason": reason,
        "rendering_backend": "mouse_behavior.visualization.overlay",
        "source_clip": row["clip_path"],
        "source_tracks": row["tracks_path"],
        "output_video": str(output_path),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "auxiliary_events": [],
        "auxiliary_behaviors": [],
    }


def _load_heuristic_parameters(path: Path | None) -> HeuristicParameters | None:
    """Load frozen train-only parameters when auxiliary overlays are requested."""

    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"启发式参数文件必须是 JSON 对象: {path}")
    return HeuristicParameters(**{key: float(value) for key, value in payload.items()})


def _select_rows(
    rows: list[dict[str, str]], max_per_behavior: int, sample_id: str | None
) -> list[dict[str, str]]:
    if sample_id:
        selected = [row for row in rows if row.get("sample_id") == sample_id]
        if not selected:
            raise ValueError(f"验证集没有 sample_id: {sample_id}")
        return selected
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        behavior = row.get("canonical_behavior", "unknown")
        if counts.get(behavior, 0) >= max_per_behavior:
            continue
        selected.append(row)
        counts[behavior] = counts.get(behavior, 0) + 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--heuristic-parameters",
        type=Path,
        help=("训练集冻结的启发式参数 JSON；提供后会为所有可见 ID 生成辅助行为覆盖。"),
    )
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="按层级渲染轨迹推断的其他 ID 行为；不读取验证标签。",
    )
    parser.add_argument("--max-per-behavior", type=int, default=1)
    parser.add_argument("--sample-id")
    status_group = parser.add_mutually_exclusive_group()
    status_group.add_argument(
        "--correct-only",
        action="store_true",
        help="仅渲染 strict_top1_correct=True 的样本（行为和目标 ID 均正确）。",
    )
    status_group.add_argument(
        "--errors-only",
        action="store_true",
        help="仅渲染 strict_top1_correct=False 的样本。",
    )
    status_group.add_argument(
        "--unrecognized-only",
        action="store_true",
        help="仅渲染 predicted_behavior 为空或为 none 的样本。",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="只从已有 MP4 重建清单，不重新编码视频。",
    )
    args = parser.parse_args()

    parameter_path = args.heuristic_parameters
    if parameter_path is None:
        inferred = args.predictions.resolve().parent / "tuned_parameters.json"
        parameter_path = inferred if inferred.exists() else None
    parameters = _load_heuristic_parameters(parameter_path)

    rows = _read_rows(args.dataset_split)
    predictions = {
        row["sample_id"]: row
        for row in _read_rows(args.predictions)
        if row.get("split") == "validation"
    }
    if args.sample_id:
        # An explicit sample request is a debugging override and should not
        # disappear merely because it is an error case.
        rows = rows
    elif args.correct_only:
        rows = [
            row
            for row in rows
            if row["sample_id"] in predictions
            and _is_true(predictions[row["sample_id"]].get("strict_top1_correct", ""))
        ]
    elif args.errors_only:
        rows = [
            row
            for row in rows
            if row["sample_id"] in predictions
            and not _is_true(predictions[row["sample_id"]].get("strict_top1_correct", ""))
        ]
    elif args.unrecognized_only:
        rows = [
            row
            for row in rows
            if row["sample_id"] in predictions
            and canonical_behavior(predictions[row["sample_id"]].get("predicted_behavior"))
            in {"", "none"}
        ]
    selected = _select_rows(rows, max(int(args.max_per_behavior), 1), args.sample_id)
    manifest: list[dict[str, Any]] = []
    for row in selected:
        prediction = predictions.get(row["sample_id"])
        if prediction is None:
            raise ValueError(f"预测文件缺少验证样本: {row['sample_id']}")
        filename = f"{_safe_name(row['canonical_behavior'])}_{_safe_name(row['sample_id'].split('/')[-1])}.mp4"
        output_path = args.output_dir / filename
        if args.manifest_only:
            manifest.append(_manifest_from_existing(row, prediction, output_path))
        else:
            manifest.append(
                render_row(
                    row,
                    prediction,
                    output_path,
                    parameters,
                    include_auxiliary=args.include_auxiliary,
                )
            )

    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(manifest), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
