#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run only lightweight rules on completed Beiyi Pose caches."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from _bootstrap import REPO_ROOT, ensure_importable

ensure_importable()

from mouse_behavior import lightweight_behavior_inference as lightweight  # noqa: E402
from mouse_behavior.logging_config import configure_logging  # noqa: E402
from validate_beiyi_extended_ethogram import (  # noqa: E402
    BEIYI_EXPECTED_MICE,
    LABELS,
    DURATION_RULES_SECONDS,
    VIDEO_EXTENSIONS,
    _behavior_hit,
    _contact_hit,
    _duration_audit,
    _matching_event_durations,
    _safe_case_name,
    _video_info,
)


LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True, help="已有缓存的验证目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "profiles" / "beiyi.yaml",
        help="北医短视频默认使用固定笼界 profile。",
    )
    parser.add_argument("--expected-mice", type=int, default=BEIYI_EXPECTED_MICE)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    dataset = args.dataset.resolve()
    source_output = args.source_output.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for video in sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ):
        rel = video.relative_to(dataset)
        expected = LABELS.get((rel.parts[0], rel.parts[1])) if len(rel.parts) >= 3 else None
        if expected is not None:
            cases.append((video, expected))
    rows = []
    started = time.perf_counter()
    for index, (video, expected) in enumerate(cases, start=1):
        case_name = _safe_case_name(index, video, dataset)
        source_case = source_output / case_name
        cache_dir = source_case / "yolo_precompute"
        if not (cache_dir / "yolo_results_status.json").exists():
            raise FileNotFoundError(cache_dir)
        case_dir = output / case_name
        if case_dir.exists():
            shutil.rmtree(case_dir)
        analysis_dir = case_dir / "analysis"
        fps, total_frames = _video_info(video)
        lightweight.analyze(
            video,
            cache_dir,
            args.config,
            analysis_dir,
            expected_mice=max(int(args.expected_mice), 2),
            sample_stride=max(int(args.sample_stride), 1),
            fps_override=fps,
        )
        events_path = analysis_dir / "lightweight_behavior_events.csv"
        contacts_path = analysis_dir / "lightweight_contact_events.csv"
        metadata_path = analysis_dir / "lightweight_analysis_metadata.json"
        events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
        contacts = pd.read_csv(contacts_path) if contacts_path.exists() else pd.DataFrame()
        hit = (
            _contact_hit(contacts, expected)
            if expected in {"nose_head", "nose_tail"}
            else _behavior_hit(events, expected)
        )
        matched_durations = _matching_event_durations(events, contacts, expected)
        duration_audit = _duration_audit(
            expected=expected,
            video_duration_s=total_frames / max(float(fps), 1e-9),
            target_event_found=bool(hit),
            durations=matched_durations,
        )
        rows.append(
            {
                "case_index": index,
                "video": str(video),
                "relative_video": str(video.relative_to(dataset)),
                "category": video.relative_to(dataset).parts[0],
                "label_folder": video.relative_to(dataset).parts[1],
                "expected_behavior": expected,
                "target_event_found": bool(hit),
                "video_fps": float(fps),
                "source_frames": int(total_frames),
                "video_duration_s": float(total_frames / max(float(fps), 1e-9)),
                "target_event_count": int(len(matched_durations)),
                **duration_audit,
                "event_count": int(len(events)),
                "contact_event_count": int(len(contacts)),
                "behavior_counts_json": json.dumps(
                    events["behavior"].astype(str).value_counts().to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if "behavior" in events
                else "{}",
                "contact_counts_json": json.dumps(
                    contacts["contact_type"].astype(str).value_counts().to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if "contact_type" in contacts
                else "{}",
                "analysis_elapsed_s": float(
                    json.loads(metadata_path.read_text(encoding="utf-8")).get("elapsed_s", 0.0)
                ),
                "case_output": str(case_dir),
            }
        )
        LOGGER.info(
            "[%02d/%d] %s %s %s",
            index,
            len(cases),
            expected,
            "HIT" if hit else "MISS",
            video.name,
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "beiyi_video_validation.csv", index=False, encoding="utf-8-sig")
    aggregate = (
        result.groupby("expected_behavior", sort=True)
        .agg(
            video_count=("video", "count"),
            detected_video_count=("target_event_found", "sum"),
            total_event_count=("event_count", "sum"),
            total_contact_event_count=("contact_event_count", "sum"),
        )
        .reset_index()
    )
    aggregate["video_coverage"] = aggregate["detected_video_count"] / aggregate["video_count"]
    duration_summary = (
        result.groupby("expected_behavior", sort=True)
        .agg(
            duration_context_sufficient_count=("duration_context_sufficient", "sum"),
            duration_rule_met_count=("duration_rule_met", "sum"),
            duration_unverifiable_count=(
                "duration_validation_status",
                lambda values: int((values == "clip_too_short_to_verify").sum()),
            ),
        )
        .reset_index()
    )
    aggregate = aggregate.merge(duration_summary, on="expected_behavior", how="left")
    aggregate.to_csv(output / "beiyi_behavior_coverage.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset": str(dataset),
        "source_pose_cache": str(source_output),
        "model": "weights/pose/best.pt (local Release asset, if available)",
        "pose_only": True,
        "obb_used": False,
        "multi_mouse_scene": True,
        "video_count": int(len(result)),
        "all_target_videos_hit": bool(result["target_event_found"].all()),
        "duration_rule_seconds": DURATION_RULES_SECONDS,
        "duration_rule_met_video_count": int(result["duration_rule_met"].fillna(False).sum()),
        "duration_unverifiable_video_count": int(
            (result["duration_validation_status"] == "clip_too_short_to_verify").sum()
        ),
        "elapsed_s": float(time.perf_counter() - started),
        "coverage_csv": str(output / "beiyi_behavior_coverage.csv"),
        "case_csv": str(output / "beiyi_video_validation.csv"),
        "limitation": "北医标签为视频/文件夹级示例，不是逐帧真值；video_coverage 不能替代 Precision、Recall、F1 或 actor/target accuracy。",
    }
    (output / "beiyi_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("summary=\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
