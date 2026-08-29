#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Beiyi events exactly as the unbiased renderer would display them.

Folder labels are read only after inference and are used as video-level review
truth.  They never influence event ordering, FSM transitions, or rendering.
This is deliberately stricter than the legacy ``HIT`` report: a behavior that
appears for one frame no longer makes an entire example look successful.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from mouse_behavior.visualization.overlay import (
    canonical_behavior,
    event_category,
    normalize_contact_events,
    normalize_focus_behavior,
    resolve_event_for_frame,
    select_display_events,
)


NONE_LABEL = "none"


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _safe_frame(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _truthy(value: Any) -> bool:
    if value is None or bool(pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _frame_events(
    behavior_rows: Iterable[Mapping[str, Any]],
    contact_rows: Iterable[Mapping[str, Any]],
    frame_count: int,
) -> list[list[dict[str, Any]]]:
    frame_map: list[list[dict[str, Any]]] = [[] for _ in range(frame_count)]
    events = [dict(row) for row in behavior_rows]
    events.extend(normalize_contact_events(contact_rows))
    for event in events:
        start = max(_safe_frame(event.get("start_frame"), 0), 0)
        end = min(_safe_frame(event.get("end_frame"), -1), frame_count - 1)
        if end < start:
            continue
        for frame_index in range(start, end + 1):
            resolved = resolve_event_for_frame(event, frame_index)
            if resolved is not None:
                frame_map[frame_index].append(resolved)
    return frame_map


def _dominant(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return NONE_LABEL, 0
    return counter.most_common(1)[0]


def _expected_category(behavior: str) -> str:
    category = event_category({"behavior": behavior})
    if category is None:
        raise ValueError(f"Unsupported expected behavior: {behavior}")
    return category


def evaluate_case(row: Mapping[str, Any]) -> dict[str, Any]:
    frame_count = max(_safe_frame(row.get("source_frames"), 0), 0)
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count for {row.get('relative_video')}")
    expected_raw = str(row.get("expected_behavior", ""))
    expected = normalize_focus_behavior(expected_raw)
    if expected is None:
        raise ValueError(f"Unsupported expected behavior: {expected_raw}")
    expected_category = _expected_category(expected)

    analysis_dir = Path(str(row["case_output"])) / "analysis"
    behavior_rows = _records(analysis_dir / "lightweight_behavior_events.csv")
    contact_rows = _records(analysis_dir / "lightweight_contact_events.csv")
    frame_map = _frame_events(behavior_rows, contact_rows, frame_count)

    top_behavior_counts: Counter[str] = Counter()
    top_category_counts: Counter[str] = Counter()
    selected_behavior_frames: Counter[str] = Counter()
    target_selected_frames = 0
    target_top_frames = 0
    target_category_top_frames = 0
    false_group_frames = 0

    for active_events in frame_map:
        selected, _ = select_display_events(active_events, focus_behavior=None)
        selected_behaviors = {canonical_behavior(event.get("behavior")) for event in selected}
        for behavior in selected_behaviors:
            selected_behavior_frames[behavior] += 1
        if expected in selected_behaviors:
            target_selected_frames += 1

        if selected:
            top_event = selected[0]
            top_behavior = canonical_behavior(top_event.get("behavior"))
            top_category = event_category(top_event) or NONE_LABEL
        else:
            top_behavior = NONE_LABEL
            top_category = NONE_LABEL
        top_behavior_counts[top_behavior] += 1
        top_category_counts[top_category] += 1
        target_top_frames += int(top_behavior == expected)
        target_category_top_frames += int(top_category == expected_category)
        false_group_frames += int(expected_category != "group" and top_category == "group")

    dominant_behavior, dominant_behavior_frames = _dominant(top_behavior_counts)
    dominant_category, dominant_category_frames = _dominant(top_category_counts)
    non_target_top = {
        behavior: count
        for behavior, count in top_behavior_counts.most_common()
        if behavior not in {expected, NONE_LABEL}
    }
    bridge_count = sum(_truthy(event.get("identity_bridge")) for event in behavior_rows)

    return {
        "relative_video": str(row.get("relative_video", "")),
        "category": str(row.get("category", "")),
        "label_folder": str(row.get("label_folder", "")),
        "expected_behavior": expected_raw,
        "expected_display_behavior": expected,
        "expected_category": expected_category,
        "source_frames": frame_count,
        "target_selected_frames": target_selected_frames,
        "target_selected_fraction": target_selected_frames / frame_count,
        "target_top_frames": target_top_frames,
        "target_top_fraction": target_top_frames / frame_count,
        "target_category_top_frames": target_category_top_frames,
        "target_category_top_fraction": target_category_top_frames / frame_count,
        "dominant_behavior": dominant_behavior,
        "dominant_behavior_frames": dominant_behavior_frames,
        "dominant_behavior_fraction": dominant_behavior_frames / frame_count,
        "dominant_behavior_matches": dominant_behavior == expected,
        "dominant_category": dominant_category,
        "dominant_category_frames": dominant_category_frames,
        "dominant_category_fraction": dominant_category_frames / frame_count,
        "dominant_category_matches": dominant_category == expected_category,
        "false_group_frames": false_group_frames,
        "false_group_fraction": false_group_frames / frame_count,
        "identity_bridged_event_count": bridge_count,
        "non_target_top_behavior_counts_json": json.dumps(
            non_target_top,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "selected_behavior_frame_counts_json": json.dumps(
            dict(selected_behavior_frames.most_common()),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    behavior_rows: list[dict[str, Any]] = []
    for behavior, group in cases.groupby("expected_behavior", sort=False):
        behavior_rows.append(
            {
                "expected_behavior": behavior,
                "video_count": int(len(group)),
                "dominant_behavior_match_count": int(group["dominant_behavior_matches"].sum()),
                "dominant_category_match_count": int(group["dominant_category_matches"].sum()),
                "mean_target_selected_fraction": float(group["target_selected_fraction"].mean()),
                "mean_target_top_fraction": float(group["target_top_fraction"].mean()),
                "mean_false_group_fraction": float(group["false_group_fraction"].mean()),
            }
        )
    by_behavior = pd.DataFrame(behavior_rows)
    summary = {
        "video_count": int(len(cases)),
        "dominant_behavior_match_count": int(cases["dominant_behavior_matches"].sum()),
        "dominant_behavior_match_fraction": float(cases["dominant_behavior_matches"].mean()),
        "dominant_category_match_count": int(cases["dominant_category_matches"].sum()),
        "dominant_category_match_fraction": float(cases["dominant_category_matches"].mean()),
        "mean_target_selected_fraction": float(cases["target_selected_fraction"].mean()),
        "mean_target_top_fraction": float(cases["target_top_fraction"].mean()),
        "non_group_false_group_video_count": int(
            ((cases["expected_category"] != "group") & (cases["false_group_frames"] > 0)).sum()
        ),
        "identity_bridged_event_count": int(cases["identity_bridged_event_count"].sum()),
        "method": (
            "Unbiased frame replay using the production renderer's event "
            "normalization, identity-role trace, hierarchy, and priority."
        ),
        "limitation": (
            "Folder names are video-level labels, not dense per-frame or actor/target ground truth."
        ),
    }
    return by_behavior, summary


def evaluate(validation_root: Path) -> dict[str, Any]:
    manifest_path = validation_root / "beiyi_video_validation.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    cases = pd.DataFrame(evaluate_case(row) for row in manifest.to_dict("records"))
    by_behavior, summary = summarize(cases)

    case_path = validation_root / "rendered_behavior_accuracy_by_video.csv"
    behavior_path = validation_root / "rendered_behavior_accuracy_by_behavior.csv"
    summary_path = validation_root / "rendered_behavior_accuracy_summary.json"
    cases.to_csv(case_path, index=False, encoding="utf-8-sig")
    by_behavior.to_csv(behavior_path, index=False, encoding="utf-8-sig")
    summary.update(
        {
            "case_csv": str(case_path),
            "behavior_csv": str(behavior_path),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        type=Path,
        required=True,
        help="Directory containing beiyi_video_validation.csv and case outputs.",
    )
    args = parser.parse_args()
    summary = evaluate(args.validation_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
