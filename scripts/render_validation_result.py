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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    INDIVIDUAL_BEHAVIORS,
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


def _prediction_target_ids(prediction: dict[str, str]) -> tuple[int, ...]:
    """Return the IDs selected by the frozen predictor, never the label IDs."""

    values: set[int] = set()
    for raw_value in prediction.get("predicted_target_ids", "").split(","):
        value = raw_value.strip()
        if not value:
            continue
        try:
            values.add(int(value))
        except ValueError:
            continue
    return tuple(sorted(values))


def _event_member_ids(event: dict[str, Any]) -> tuple[int, ...]:
    raw_values = event.get("member_ids", ())
    if isinstance(raw_values, str):
        raw_values = raw_values.split(",")
    try:
        return tuple(sorted({int(value) for value in raw_values}))
    except (TypeError, ValueError):
        return ()


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
        # The frozen Top-1 result belongs on the IDs actually predicted.
        # Ground-truth target IDs are separately marked for post-hoc review.
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


def _prediction_event_for_row(
    row: dict[str, str], prediction: dict[str, str], total_frames: int
) -> dict[str, Any] | None:
    """Build the primary event from model output, not the annotation row."""

    return _prediction_event(prediction, _prediction_target_ids(prediction), total_frames)


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
    These events use the frozen, label-free trajectory evidence export when
    available. The compatibility fallback recomputes them from clip-local
    tracks and frozen parameters, never from validation labels. Keep one
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
        start_frame: int = 0,
        stop_frame: int | None = None,
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
            "start_frame": max(0, min(start_frame, end_frame)),
            "end_frame": max(0, min(stop_frame if stop_frame is not None else end_frame, end_frame)),
            "peak_frame": max(0, min((start_frame + (stop_frame if stop_frame is not None else end_frame)) // 2, end_frame)),
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
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        best = _best_positive_score(scores, INDIVIDUAL_BEHAVIORS)
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

    timed_group_behaviors: set[tuple[str, tuple[int, ...]]] = set()
    for raw_event in evidence.get("group_events", ()):
        try:
            behavior = str(raw_event["behavior"])
            ids = tuple(sorted(int(value) for value in raw_event["member_ids"]))
            start = int(raw_event["start_frame"])
            stop = int(raw_event["end_frame"])
            score = float(raw_event["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if behavior not in GROUP_BEHAVIORS or len(ids) < 3 or not 0 <= start <= stop < total_frames:
            continue
        timed_group_behaviors.add((behavior, ids))
        add_event(behavior, ids, score, "group", start_frame=start, stop_frame=stop)

    for text_group, raw_scores in dict(evidence.get("group_scores", {})).items():
        try:
            ids = tuple(sorted(int(value) for value in str(text_group).split(",")))
        except (TypeError, ValueError):
            continue
        if len(ids) < 3 or len(ids) != len(set(ids)):
            continue
        best = _best_positive_score(
            {
                behavior: score
                for behavior, score in dict(raw_scores).items()
                if (behavior, ids) not in timed_group_behaviors
            },
            GROUP_BEHAVIORS,
        )
        if best is not None:
            add_event(best[0], ids, best[1], "group")

    summaries.sort(key=lambda item: (-float(item["score"]), item["behavior"], item["ids"]))
    return events, summaries


def _auxiliary_events_for_render(
    prediction: dict[str, Any],
    tracks_path: Path,
    fps: float,
    target_ids: tuple[int, ...],
    total_frames: int,
    parameters: HeuristicParameters | None,
    enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use frozen evidence when available; parameters are only a fallback."""

    if not enabled:
        return [], []
    evidence = prediction.get("_trajectory_evidence")
    if evidence is None and parameters is not None:
        features = load_exported_features(tracks_path, fps=fps, max_tracks=64)
        evidence = predict_features(features, parameters)
    if evidence is None:
        return [], []
    return _auxiliary_events_from_evidence(evidence, target_ids, total_frames)


def _events_at_frame(events: list[dict[str, Any]], frame_index: int, total_frames: int) -> list[dict[str, Any]]:
    """Keep phase annotations local to their inferred trajectory window."""

    return [
        event for event in events
        if int(event.get("start_frame", 0)) <= frame_index
        <= int(event.get("end_frame", total_frames - 1))
    ]


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


def _prediction_for_render(
    row: dict[str, str], prediction: dict[str, str]
) -> dict[str, str]:
    """Normalize current and legacy frozen prediction exports for rendering."""

    normalized = dict(prediction)
    if canonical_behavior(normalized.get("predicted_behavior")) in {
        "none",
        "unrecognized",
        "未识别",
    }:
        normalized["predicted_behavior"] = ""
    truth_ids = _target_ids(row)
    predicted_ids = tuple(
        sorted(
            {
                int(value.strip())
                for value in prediction.get("predicted_target_ids", "").split(",")
                if value.strip()
            }
        )
    )
    # New blind exports carry the Top-1 prediction and its candidate IDs, not
    # the old scorer's derived target_id_hit column. Derive this display-only
    # diagnostic after prediction has been frozen and labels are joined.
    if "target_id_hit" not in normalized:
        normalized["target_id_hit"] = str(bool(truth_ids and predicted_ids == truth_ids))
    if "strict_top1_correct" not in normalized:
        normalized["strict_top1_correct"] = str(
            canonical_behavior(prediction.get("predicted_behavior"))
            == canonical_behavior(row.get("canonical_behavior"))
            and predicted_ids == truth_ids
        )
    if "compatible_top1_correct" not in normalized:
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # 旧输入缺少 compatible 列时只退化为严格 Top-1；不再推断 Together↔鼻头/鼻尾接触同层双标签兼容。
        normalized["compatible_top1_correct"] = str(
            _is_true(normalized["strict_top1_correct"])
        )
    return normalized


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


def _mark_unselected_label_targets(
    mouse_overlays: dict[int, MouseOverlay],
    label_ids: tuple[int, ...],
    predicted_ids: tuple[int, ...],
) -> None:
    """Distinguish annotated targets omitted by the predicted ID set."""

    predicted_set = set(predicted_ids)
    for target_id in label_ids:
        if target_id in predicted_set:
            continue
        previous = mouse_overlays.get(target_id)
        text = f"{format_mouse_id(target_id)}｜标签目标（算法未选中）"
        if previous is not None:
            text = f"{previous.text}｜标签目标（算法未选中）"
        mouse_overlays[target_id] = MouseOverlay(
            text=text,
            color_bgr=(0, 165, 255) if previous is None else previous.color_bgr,
            priority=(2, 0.0) if previous is None else previous.priority,
        )


def _add_compatible_pair_overlay(
    mouse_overlays: dict[int, MouseOverlay],
    target_ids: tuple[int, ...],
    prediction: dict[str, str],
) -> None:
    """Show the trajectory-supported secondary label without changing Top-1."""

    if len(target_ids) != 2:
        return
    labels = [
        canonical_behavior(value)
        for value in prediction.get("compatible_labels", "").split(",")
        if value.strip()
    ]
    predicted = canonical_behavior(prediction.get("predicted_behavior"))
    compatible_pairs = {
        frozenset(("together", "nose_head_contact")),
        frozenset(("together", "nose_tail_contact")),
        frozenset(("approach", "nose_head_contact")),
        frozenset(("approach", "nose_tail_contact")),
    }
    if len(labels) != 2 or predicted not in labels or frozenset(labels) not in compatible_pairs:
        return
    secondary = next(value for value in labels if value != predicted)
    if secondary not in {"approach", "together", "nose_head_contact", "nose_tail_contact"}:
        return
    secondary_name = DISPLAY_NAMES_ZH.get(secondary, secondary)
    for target_id in target_ids:
        overlay = mouse_overlays.get(target_id)
        if overlay is not None:
            mouse_overlays[target_id] = MouseOverlay(
                text=f"{overlay.text}｜同层：{secondary_name}",
                color_bgr=overlay.color_bgr,
                priority=overlay.priority,
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
    predicted_ids = _prediction_target_ids(prediction)
    review_target_ids = tuple(sorted(set(target_ids).union(predicted_ids)))
    predicted_event = _prediction_event_for_row(row, prediction, total_frames)
    # The v6 review renderer can add label-free trajectory candidates for all
    # visible IDs.  The target Top-1 event carries a dedicated render priority,
    # while the overlay module applies social > group > individual hierarchy
    # to every other candidate.
    auxiliary_events, auxiliary_summaries = _auxiliary_events_for_render(
        prediction,
        tracks_path,
        fps,
        review_target_ids,
        total_frames,
        parameters,
        include_auxiliary,
    )
    evidence = prediction.get("_trajectory_evidence")
    if (
        evidence is not None
        and predicted_event is not None
        and predicted_event.get("event_scope") == "group"
    ):
        matching_phase = [
            event for event in evidence.get("group_events", ())
            if event.get("behavior") == predicted_event["behavior"]
            and _event_member_ids(event) == predicted_ids
        ]
        if matching_phase:
            phase = max(matching_phase, key=lambda event: float(event.get("score", 0.0)))
            predicted_event["start_frame"] = int(phase["start_frame"])
            predicted_event["end_frame"] = int(phase["end_frame"])
            predicted_event["peak_frame"] = (
                predicted_event["start_frame"] + predicted_event["end_frame"]
            ) // 2
    event_rows: list[dict[str, Any]] = auxiliary_events
    if predicted_event is not None:
        event_rows.append(predicted_event)
    reason = prediction.get("compatibility_reason") or (
        "严格正确：行为和目标 ID 均匹配。"
        if _is_true(strict)
        else _error_reason(prediction, truth_name, predicted_name)
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
            current_events = _events_at_frame(event_rows, frame_index, total_frames)
            active_events, display_layer = select_display_events(current_events)
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
            _mark_unselected_label_targets(mouse_overlays, target_ids, predicted_ids)
            _add_compatible_pair_overlay(mouse_overlays, predicted_ids, prediction)
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
                f"compatible_top1={prediction.get('compatible_top1_correct', 'False')}",
                # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
                # 原“兼容双标签/接触累计”面板行已删除；计分仅保留严格 Top-1 口径。
                f"标签目标 ID：{','.join(map(str, target_ids)) or '无'}",
                f"预测目标 ID：{','.join(map(str, predicted_ids)) or '无'}",
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
        "compatible_top1": _is_true(prediction.get("compatible_top1_correct", "False")),
        "compatible_labels": prediction.get("compatible_labels", ""),
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # compatible_dual_label/contact_seconds manifest 字段已移除；旧 CSV 仍带这些列时读取会被直接忽略。
        "compatibility_reason": prediction.get("compatibility_reason", reason),
        "id_relation": prediction.get("id_relation", ""),
        "target_ids": list(target_ids),
        "predicted_target_ids": list(predicted_ids),
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
    row: dict[str, str],
    prediction: dict[str, str],
    output_path: Path,
    parameters: HeuristicParameters | None = None,
    include_auxiliary: bool = False,
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
    target_ids = _target_ids(row)
    predicted_ids = _prediction_target_ids(prediction)
    predicted_name = DISPLAY_NAMES_ZH.get(predicted, predicted) if predicted else "未识别"
    truth_name = DISPLAY_NAMES_ZH.get(truth, truth)
    strict = prediction.get("strict_top1_correct", "False")
    target_hit = prediction.get("target_id_hit", "False")
    _, auxiliary_summaries = _auxiliary_events_for_render(
        prediction,
        Path(row["tracks_path"]),
        fps,
        tuple(sorted(set(target_ids).union(predicted_ids))),
        frames,
        parameters,
        include_auxiliary,
    )
    reason = prediction.get("compatibility_reason") or (
        "严格正确：行为和目标 ID 均匹配。"
        if _is_true(strict)
        else _error_reason(prediction, truth_name, predicted_name)
    )
    return {
        "sample_id": row["sample_id"],
        "behavior": row["canonical_behavior"],
        "predicted_behavior": predicted,
        "target_hit": _is_true(target_hit),
        "strict_top1": _is_true(strict),
        "compatible_top1": _is_true(prediction.get("compatible_top1_correct", "False")),
        "compatible_labels": prediction.get("compatible_labels", ""),
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # compatible_dual_label/contact_seconds manifest 字段已移除；旧 CSV 仍带这些列时读取会被直接忽略。
        "compatibility_reason": prediction.get("compatibility_reason", reason),
        "id_relation": prediction.get("id_relation", ""),
        "target_ids": list(target_ids),
        "predicted_target_ids": list(predicted_ids),
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
        "auxiliary_events": auxiliary_summaries,
        "auxiliary_behaviors": sorted({item["behavior"] for item in auxiliary_summaries}),
    }


def _render_matches_source(row: dict[str, str], output_path: Path) -> bool:
    """Only reuse a rendered clip when it has every source frame."""

    if not output_path.exists():
        return False
    source = cv2.VideoCapture(str(Path(row["clip_path"])))
    rendered = cv2.VideoCapture(str(output_path))
    try:
        if not source.isOpened() or not rendered.isOpened():
            return False
        source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        rendered_frames = int(rendered.get(cv2.CAP_PROP_FRAME_COUNT))
        return source_frames > 0 and rendered_frames == source_frames
    finally:
        source.release()
        rendered.release()


def _render_one_sample(task: tuple[Any, ...]) -> tuple[int, dict[str, Any]]:
    """Render or resume one independent clip in a worker process."""

    (
        index,
        row,
        prediction,
        output_path,
        parameters,
        include_auxiliary,
        resume,
        manifest_only,
        trajectory_evidence,
        limit_cv_threads,
    ) = task
    if limit_cv_threads:
        # Each worker owns a separate clip; avoid a nested OpenCV thread pool.
        cv2.setNumThreads(1)
    normalized_prediction = _prediction_for_render(row, prediction)
    if trajectory_evidence is not None:
        normalized_prediction["_trajectory_evidence"] = trajectory_evidence
    if manifest_only or (resume and _render_matches_source(row, output_path)):
        result = _manifest_from_existing(
            row,
            normalized_prediction,
            output_path,
            parameters,
            include_auxiliary=include_auxiliary,
        )
    else:
        result = render_row(
            row,
            normalized_prediction,
            output_path,
            parameters,
            include_auxiliary=include_auxiliary,
        )
    return index, result


def _read_trajectory_evidence(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            opaque_id = str(item.get("opaque_id", ""))
            if not opaque_id or opaque_id in evidence:
                raise ValueError(f"trajectory evidence has missing/duplicate opaque_id: {opaque_id!r}")
            evidence[opaque_id] = item
    return evidence


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
    parser.add_argument(
        "--trajectory-evidence",
        type=Path,
        help="可选：使用已冻结的无标签轨迹证据渲染其他小鼠行为，不重复推理。",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用帧数与源视频一致的已有渲染，其余样本重新渲染。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行渲染进程数；每个进程处理独立验证片段，默认 1。",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须至少为 1")

    parameter_path = args.heuristic_parameters
    if parameter_path is None:
        inferred = args.predictions.resolve().parent / "tuned_parameters.json"
        parameter_path = inferred if inferred.exists() else None
    parameters = _load_heuristic_parameters(parameter_path)

    rows = _read_rows(args.dataset_split)
    predictions = {
        row["sample_id"]: row
        for row in _read_rows(args.predictions)
        if row.get("split", "validation") == "validation"
    }
    trajectory_evidence = (
        _read_trajectory_evidence(args.trajectory_evidence)
        if args.trajectory_evidence is not None
        else {}
    )
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
    tasks: list[tuple[Any, ...]] = []
    for index, row in enumerate(selected):
        prediction = predictions.get(row["sample_id"])
        if prediction is None:
            raise ValueError(f"预测文件缺少验证样本: {row['sample_id']}")
        opaque_id = prediction.get("opaque_id", "")
        sample_evidence = None
        if trajectory_evidence:
            if opaque_id not in trajectory_evidence:
                raise ValueError(f"轨迹证据文件缺少 opaque_id: {opaque_id!r}")
            sample_evidence = trajectory_evidence[opaque_id]
        filename = f"{_safe_name(row['canonical_behavior'])}_{_safe_name(row['sample_id'].split('/')[-1])}.mp4"
        output_path = args.output_dir / filename
        tasks.append(
            (
                index,
                row,
                prediction,
                output_path,
                parameters,
                args.include_auxiliary,
                args.resume,
                args.manifest_only,
                sample_evidence,
                args.workers > 1,
            )
        )

    manifest: list[dict[str, Any] | None] = [None] * len(tasks)
    workers = min(args.workers, max(len(tasks), 1))
    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            index, result = _render_one_sample(task)
            manifest[index] = result
            print(
                json.dumps(
                    {"rendered": done, "total": len(selected), "sample_id": result["sample_id"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_one_sample, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                index, result = future.result()
                manifest[index] = result
                if done % 10 == 0 or done == len(tasks):
                    print(
                        json.dumps(
                            {"rendered": done, "total": len(selected), "sample_id": result["sample_id"]},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    completed_manifest = [item for item in manifest if item is not None]
    if len(completed_manifest) != len(selected):
        raise RuntimeError("渲染完成数与验证片段数不匹配")

    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(completed_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"videos": len(completed_manifest), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
