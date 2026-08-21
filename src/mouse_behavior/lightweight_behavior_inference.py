#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mouse Behavior Lightweight Inference entry point.

Analyze one completed YOLO cache without rerunning the heavy tracker.

This is intentionally a bounded, single-video fallback for long Windows
videos.  It reads only ``yolo_precompute`` records for the requested video,
keeps at most ``expected_mice`` tracks with position+keypoint matching, builds
the pair-wise kinematics required by the v1.43 standard behavior engine, and
then runs the same standard chase/attack FSM and thresholds offline.

It does not claim to replace the full occlusion/ReID pipeline.  The output
metadata explicitly records that limitation so a detected event can be
reviewed separately from a full-pipeline result.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np
import yaml

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - a deterministic greedy fallback is below
    linear_sum_assignment = None

from . import adaptive_arena_boundary as arena_boundary
from .parallel_behavior_fsm import ParallelBehaviorFSM
from . import standard_behavior_engine as behavior_engine  # noqa: F401
from .annotation_website_export import export_complete_video_package
from .config import load_config
from .logging_config import configure_logging
from .utils.rolling import rolling_corr as _rolling_corr  # noqa: F401
from .utils.rolling import rolling_sum as _rolling_sum  # noqa: F401
from .utils.timer import Timer


from .preprocessing.constants import (  # noqa: F401
    BEHAVIOR_NAMES_ZH,
    EXTENDED_BEHAVIORS,
    FOUR_CLASS_NAMES,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    KEYPOINTS,
    KP_LEFT_EAR,
    KP_LEFT_HIP,
    KP_NECK,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_HIP,
    KP_TAIL,
    PROJECT_NAME,
    SKELETON_EDGES,
    SOCIAL_BEHAVIORS,
)
from .preprocessing.geometry import (
    _angle_deg,
    _box_iou,
    _cosine,
    _finite_point,
    _unit,
    _weighted_mean,
)  # noqa: F401
from .tracking.cache import (
    _assign_tracks,
    _cache_total_frames,
    _deduplicate,
    _inside_arena,
    _iter_cache_records,
    _payload_detection,
    _track_cache,
)  # noqa: F401
from .preprocessing.kinematics import (
    _ema_smooth,
    _ema_smooth_keypoints,
    _kinematics,
    _pose_deformation_energy,
    _rolling_quantile,
)  # noqa: F401
from .preprocessing.pair_features import (
    _PairWorkset,
    _boolean_runs,
    _boolean_runs_with_gap,
    _interaction_radius,
    _pair_dataframe,
    _pair_metrics,
    _pair_prefilter,
    _pair_window_mask,
    _prepare_pair_workset,
)  # noqa: F401
from .behavior.ethogram import (
    CONTACT_EVENT_COLUMNS,
    _contact_components,
    _contact_distance,
    _event_rows_from_mask,
    _extended_behavior_config,
    _extended_individual_and_group_events,
    _extended_pair_events,
    _extended_short_clip_pair_events,
    _extract_contact_events,
)  # noqa: F401
from .behavior.pair_analysis import (
    _PairAnalysisResult,
    _analyze_candidate_pairs,
    _finalize_event_records_in_place,
)  # noqa: F401
from .visualization.rendering import (
    extract_behavior_clips,
    extract_four_class_clips,
    render_behavior_video,
)  # noqa: F401
from .io.csv import _write_csv  # noqa: F401

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PROJECT_NAME",
    "SOCIAL_BEHAVIORS",
    "GROUP_BEHAVIORS",
    "INDIVIDUAL_BEHAVIORS",
    "EXTENDED_BEHAVIORS",
    "BEHAVIOR_NAMES_ZH",
    "KP_NOSE",
    "KP_LEFT_EAR",
    "KP_RIGHT_EAR",
    "KP_NECK",
    "KP_LEFT_HIP",
    "KP_RIGHT_HIP",
    "KP_TAIL",
    "KEYPOINTS",
    "SKELETON_EDGES",
    "FOUR_CLASS_NAMES",
    "_finite_point",
    "_unit",
    "_cosine",
    "_angle_deg",
    "_weighted_mean",
    "_box_iou",
    "_payload_detection",
    "_deduplicate",
    "_iter_cache_records",
    "_cache_total_frames",
    "_inside_arena",
    "_assign_tracks",
    "_track_cache",
    "_ema_smooth",
    "_ema_smooth_keypoints",
    "_pose_deformation_energy",
    "_kinematics",
    "_rolling_quantile",
    "_pair_dataframe",
    "_pair_metrics",
    "_boolean_runs_with_gap",
    "_boolean_runs",
    "_interaction_radius",
    "_pair_prefilter",
    "_pair_window_mask",
    "_PairWorkset",
    "_prepare_pair_workset",
    "_event_rows_from_mask",
    "_extended_behavior_config",
    "_extended_short_clip_pair_events",
    "_extended_pair_events",
    "_extended_individual_and_group_events",
    "_contact_distance",
    "_contact_components",
    "_extract_contact_events",
    "CONTACT_EVENT_COLUMNS",
    "_PairAnalysisResult",
    "_analyze_candidate_pairs",
    "_finalize_event_records_in_place",
    "render_behavior_video",
    "extract_four_class_clips",
    "extract_behavior_clips",
    "_write_csv",
    "_rolling_corr",
    "_rolling_sum",
    "ParallelBehaviorFSM",
    "behavior_engine",
    "analyze",
    "main",
]


def _resolve_boundary_reuse_path(value: Any, output_dir: Path, config_path: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    output_candidate = output_dir / path
    if output_candidate.exists():
        return output_candidate
    return config_path.parent / path


def _prepare_video_arena_boundary(
    video_path: Path,
    cache_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    width: int,
    height: int,
    max_frames: int | None = None,
) -> tuple[arena_boundary.ArenaBoundaryResult | None, np.ndarray | None]:
    """Learn/reuse this video's boundary and persist auditable artifacts.

    The default is always per-video learning.  A JSON can only be reused when
    the caller explicitly sets ``reuse_boundary_json``; the boundary module
    then checks the saved source path, resolution, and file fingerprint.
    """

    arena_cfg = dict(config.get("adaptive_arena", {}))
    if not bool(arena_cfg.get("enabled", True)):
        return None, None
    configured_polygon = (
        dict(config.get("detector_first", {})).get("arena_mask", {}).get("polygon", [])
    )
    reuse_value = str(arena_cfg.get("reuse_boundary_json", "") or "").strip()
    if reuse_value:
        reuse_path = _resolve_boundary_reuse_path(reuse_value, output_dir, config_path)
        result = arena_boundary.load_boundary_json(
            reuse_path,
            width=width,
            height=height,
            source_video=video_path,
            require_video_match=bool(arena_cfg.get("reuse_require_video_match", True)),
        )
        heatmap = np.zeros(
            (
                max(int(math.ceil(height / max(result.heatmap_cell_px, 1))), 1),
                max(int(math.ceil(width / max(result.heatmap_cell_px, 1))), 1),
            ),
            dtype=np.float32,
        )
    else:
        records: Iterable[Mapping[str, Any]] = _iter_cache_records(cache_dir)
        if max_frames is not None:
            frame_limit = max(int(max_frames), 1)
            source_records = records

            def limited_records() -> Iterator[Mapping[str, Any]]:
                for record in source_records:
                    frame = int(record.get("frame", -1))
                    if frame >= frame_limit:
                        break
                    yield record

            records = limited_records()
        result, heatmap = arena_boundary.learn_from_yolo_records(
            records,
            width=width,
            height=height,
            config=arena_cfg,
            configured_polygon=configured_polygon,
            source_video=video_path,
        )

    json_path = output_dir / "阶段一_自适应笼界.json"
    png_path = output_dir / "阶段一_运动热力图与笼界.png"
    comparison_path = output_dir / "阶段一_原视频帧叠加笼界.png"
    arena_boundary.save_boundary_artifacts(
        result,
        heatmap,
        json_path,
        png_path,
        comparison_path,
    )
    return result, heatmap


def analyze(
    video_path: Path,
    cache_dir: Path,
    config_path: Path,
    output_dir: Path,
    expected_mice: int = 20,
    max_frames: int | None = None,
    sample_stride: int = 1,
    fps_override: float | None = None,
) -> Path:
    started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    setup_timer = Timer(
        "setup_and_video_probe",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    configured_fps = config.pop("_fps_override", 29.329)
    source_fps = float(configured_fps if fps_override is None else fps_override)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"视频 FPS 必须是正数，实际为：{source_fps}")
    sample_stride = max(int(sample_stride), 1)
    total_frames = _cache_total_frames(cache_dir)
    if max_frames is not None:
        total_frames = min(total_frames, max(int(max_frames), 1))
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    video_cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"无法读取源视频尺寸: {video_path}")
    setup_timer.stop()

    arena_timer = Timer(
        "arena_boundary",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    arena_result, _arena_heatmap = _prepare_video_arena_boundary(
        video_path,
        cache_dir,
        output_dir,
        config_path,
        config,
        width,
        height,
        max_frames=total_frames,
    )
    arena_polygon = (
        np.asarray(arena_result.polygon, dtype=np.float64) if arena_result is not None else None
    )
    arena_tolerance = float(
        dict(config.get("adaptive_arena", {})).get("hard_gate_tolerance_px", 2.0)
    )
    arena_timer.stop()

    track_timer = Timer(
        "track_cache",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    tracks, tracking_stats = _track_cache(
        cache_dir,
        total_frames,
        expected_mice,
        arena_polygon=arena_polygon,
        arena_tolerance_px=arena_tolerance,
    )
    # Preserve full-resolution source-frame tracks for the annotation website.
    # Behavior analysis may sample this array below; the export contract may not.
    source_tracks = tracks
    track_timer.stop()

    kinematics_timer = Timer(
        "kinematics",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    if sample_stride > 1:
        tracks = {
            key: (
                value[::sample_stride]
                if isinstance(value, np.ndarray) and value.ndim > 0
                else value
            )
            for key, value in tracks.items()
        }
    analysis_frames = int(tracks["valid"].shape[0])
    fps = source_fps / sample_stride
    kin = _kinematics(tracks, fps=fps)
    kinematics_timer.stop()
    pair_workset = _prepare_pair_workset(
        kin,
        fps,
        config,
        stage_timings=stage_timings,
    )
    pair_analysis = _analyze_candidate_pairs(
        pair_workset,
        kin,
        fps=fps,
        source_fps=source_fps,
        sample_stride=sample_stride,
        video_path=video_path,
        config=config,
        stage_timings=stage_timings,
    )
    events = pair_analysis.events
    contact_events = pair_analysis.contact_events
    extended_events = pair_analysis.extended_events
    pair_summaries = pair_analysis.pair_summaries
    top_evidence = pair_analysis.top_evidence
    pair_fsm_coordinator = pair_analysis.fsm_coordinator

    interaction_radius = pair_workset.interaction_radius
    prefilter = pair_workset.prefilter
    all_pair_i = pair_workset.all_pair_i
    candidate_pair_indices = pair_workset.candidate_pair_indices
    candidate_frame_mask = pair_workset.candidate_frame_mask
    pair_window_stats = pair_workset.pair_window_stats
    metrics = pair_workset.metrics
    pair_i = pair_workset.pair_i
    pair_j = pair_workset.pair_j

    global_events_timer = Timer(
        "global_events_and_finalization",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    if bool(_extended_behavior_config(config).get("enabled", True)):
        extended_events.extend(
            _extended_individual_and_group_events(
                kin,
                pair_metrics=metrics,
                pair_i=pair_i,
                pair_j=pair_j,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
            )
        )
    events.extend(extended_events)
    _finalize_event_records_in_place(events, contact_events, source_fps)
    global_events_timer.stop()

    website_frame_count = (
        int(video_frame_count) if int(video_frame_count) > 0 else max(int(total_frames), 1)
    )
    website_fps = (
        float(video_fps) if np.isfinite(video_fps) and float(video_fps) > 0.0 else float(source_fps)
    )
    extended_cfg = _extended_behavior_config(config)
    website_timer = Timer(
        "website_export",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    website_export = export_complete_video_package(
        source_video=video_path,
        output_dir=output_dir,
        behavior_events=events,
        contact_events=contact_events,
        tracks=source_tracks,
        fps=website_fps,
        frame_count=website_frame_count,
        width=width,
        height=height,
        skeleton_edges=SKELETON_EDGES,
        cm_per_pixel=float(kin["cm_per_pixel"]),
        huddle_distance_cm=float(
            dict(extended_cfg.get("group", {})).get("huddle_distance_cm", 9.0)
        ),
        tracker_params={
            "expected_mice": int(expected_mice),
            "sample_stride": int(sample_stride),
            "behavior_analysis_fps": float(source_fps),
            "video_container_fps": float(website_fps),
            "analysis_mode": "lightweight_cache_tracking",
            "full_pipeline_not_run": True,
        },
    )
    website_timer.stop()

    csv_timer = Timer(
        "csv_output",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    _write_csv(output_dir / "lightweight_behavior_events.csv", events)
    _write_csv(
        output_dir / "lightweight_contact_events.csv",
        contact_events,
        columns=CONTACT_EVENT_COLUMNS,
    )
    _write_csv(output_dir / "lightweight_pair_summary.csv", pair_summaries)
    _write_csv(output_dir / "lightweight_top_evidence.csv", top_evidence)
    fsm_evaluated_pair_frames = int(
        sum(int(summary.get("fsm_evaluated_frames", 0)) for summary in pair_summaries)
    )
    fsm_candidate_timeline_frames = int(analysis_frames * len(candidate_pair_indices))
    fsm_skipped_pair_frames = max(
        fsm_candidate_timeline_frames - fsm_evaluated_pair_frames,
        0,
    )
    csv_timer.stop()
    metadata = {
        "source_video": str(video_path),
        "yolo_cache": str(cache_dir),
        "config": str(config_path),
        "analysis_mode": "lightweight_cache_tracking",
        "full_pipeline_not_run": True,
        "tracker": "position_plus_keypoint_hungarian",
        "expected_mice": int(expected_mice),
        "source_frames": int(total_frames),
        "analysis_frames": int(analysis_frames),
        "source_fps": float(source_fps),
        "analysis_fps": float(fps),
        "sample_stride": int(sample_stride),
        "duration_s": float(total_frames / source_fps),
        "cm_per_pixel": float(kin["cm_per_pixel"]),
        "reference_body_px": float(kin["reference_body_px"]),
        "tracking": tracking_stats,
        "arena_boundary": (asdict(arena_result) if arena_result is not None else None),
        "event_counts": {
            f"{level}_{behavior}": int(
                sum(
                    1
                    for event in events
                    if event.get("candidate_level") == level and event.get("behavior") == behavior
                )
            )
            for level in ("weak", "strong")
            for behavior in ("chase", "attack")
        },
        "extended_behavior_counts": {
            behavior: int(sum(event.get("behavior") == behavior for event in events))
            for behavior in EXTENDED_BEHAVIORS
        },
        "extended_behavior_scopes": {
            scope: int(sum(event.get("event_scope") == scope for event in events))
            for scope in ("pair", "individual", "group")
        },
        "contact_event_counts": {
            contact_type: int(
                sum(event.get("contact_type") == contact_type for event in contact_events)
            )
            for contact_type in (
                "nose_head",
                "nose_tail",
                "nose_head_and_nose_tail",
            )
        },
        "contact_event_csv": str(output_dir / "lightweight_contact_events.csv"),
        "annotation_website_export": website_export,
        "notes": [
            "仅读取指定视频的完整 YOLO 预推理缓存，没有读取其他行为目录。",
            "该结果用于当前长视频的快速行为筛查；它不包含完整流水线的遮挡簇 ReID、ROI Pose 恢复和伪掩码身份保护。",
            "行为事件 CSV 同时包含 legacy chase/attack 与扩展 ethogram 标签；鼻头/鼻尾接触写入独立 lightweight_contact_events.csv，接触本身不会单独升级为 attack。",
            "候选鼠对的昂贵鼻体几何、滚动轨迹特征和标准行为连续证据只在距离/朝向窗口及其上下文 padding 内计算；窗口外帧保留在输出时间轴中并作为FSM硬否决/状态重置行。",
            "新视频没有人工行为标签，因此不能据此计算 Precision、Recall、F1 或 actor/target accuracy；事件中的角色 ID 和 role confidence 仅是模型诊断。",
        ],
        "interaction_radius_cm": float(interaction_radius),
        "candidate_pair_count": int(len(candidate_pair_indices)),
        "total_pair_count": int(len(all_pair_i)),
        "standard_behavior_engine": {
            "skip_inactive_rows": bool(
                dict(config.get("standard_behavior_engine", {})).get("skip_inactive_rows", True)
            ),
            "evaluated_pair_frame_count": fsm_evaluated_pair_frames,
            "skipped_pair_frame_count": fsm_skipped_pair_frames,
            "candidate_timeline_pair_frame_count": fsm_candidate_timeline_frames,
            "evaluated_pair_frame_fraction": float(
                fsm_evaluated_pair_frames / fsm_candidate_timeline_frames
            )
            if fsm_candidate_timeline_frames
            else 0.0,
        },
        "parallel_behavior_fsm": {
            "enabled": bool(pair_fsm_coordinator.enabled),
            "mode": str(pair_fsm_coordinator.mode),
            "version": ParallelBehaviorFSM.VERSION,
            "collect_diagnostics": bool(pair_fsm_coordinator.collect_diagnostics),
            "execution_semantics": (
                "active_temporal_regions"
                if pair_fsm_coordinator.enabled
                else "disabled_no_parallel_events"
            ),
            "regions": {
                "individual": "stationary|walking|running per mouse",
                "pair": "together|approach|chase|avoidance|attack per pair",
                "contact": "nose_head|nose_tail|combined per pair",
                "group": "huddle|isolation per video",
            },
        },
        "pair_prefilter": {
            "enabled": bool(prefilter["enabled"]),
            "close_distance_cm": float(prefilter["close_distance_cm"]),
            "min_heading_cosine": float(prefilter["min_heading_cosine"]),
            "valuable_frame_count": int(np.asarray(prefilter["valuable_frame"], dtype=bool).sum()),
            "valuable_frame_fraction": float(
                np.asarray(prefilter["valuable_frame"], dtype=bool).mean()
            )
            if np.asarray(prefilter["valuable_frame"]).size
            else 0.0,
        },
        "pair_window": {
            **pair_window_stats,
            "candidate_active_frame_count": int(candidate_frame_mask.sum()),
            "candidate_active_frame_fraction": float(candidate_frame_mask.mean())
            if candidate_frame_mask.size
            else 0.0,
        },
        "stage_timings_s": stage_timings,
        "elapsed_s": float(time.perf_counter() - started),
    }
    with (output_dir / "lightweight_analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=False)
    parser.add_argument("--expected-mice", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=3,
        help="Analyze every Nth cached frame; FPS is reduced by the same factor.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只读取已有事件 CSV，输出一个带框/骨架/行为标签的 MP4，不生成事件片段。",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="已有 lightweight_behavior_events.csv；render-only 时默认从 output-dir 读取。",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=None,
        help="render-only 的唯一 MP4 输出路径。",
    )
    parser.add_argument(
        "--extract-four-class-clips",
        action="store_true",
        help="兼容旧接口：只从已有事件 CSV 裁剪四类原始视频，不生成渲染视频。",
    )
    parser.add_argument(
        "--extract-behavior-clips",
        action="store_true",
        help="按 lightweight_behavior_events.csv 中的行为名称分别裁剪原始视频。",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="行为切片名称；默认输出事件 CSV 中出现的全部行为。",
    )
    parser.add_argument(
        "--behavior-level",
        choices=("all", "weak", "strong"),
        default="all",
        help="行为切片保留的 candidate_level；默认 all，也包含扩展 ethogram 行。",
    )
    parser.add_argument(
        "--behavior-clips-output",
        type=Path,
        default=None,
        help="按行为切片的输出目录；默认 output-dir/behavior_clips。",
    )
    parser.add_argument(
        "--behavior-clip-seconds",
        type=float,
        default=5.0,
        help="每个行为切片的长度，默认 5 秒。",
    )
    parser.add_argument(
        "--max-clips-per-behavior",
        type=int,
        default=200,
        help="每种行为最多输出的片段数，默认 200。",
    )
    parser.add_argument(
        "--clip-level",
        choices=("weak", "strong"),
        default="strong",
        help="四类裁剪使用 weak 或 strong 事件层，默认 strong。",
    )
    parser.add_argument(
        "--clips-output",
        type=Path,
        default=None,
        help="四类视频输出目录；默认 output-dir/四类视频。",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="每个原始视频片段的长度，默认 5 秒。",
    )
    parser.add_argument(
        "--max-clips-per-class",
        type=int,
        default=200,
        help="每类最多输出的片段数，默认 200。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="日志级别，默认 INFO。",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        render_output = args.render_output or (args.output_dir / "part001_追逐攻击渲染.mp4")
        render_behavior_video(
            args.video,
            args.yolo_cache,
            events_path,
            render_output,
            expected_mice=max(int(args.expected_mice), 2),
            max_frames=args.max_frames,
        )
        LOGGER.info("render_output=%s", render_output)
        return 0

    if args.extract_behavior_clips:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        clips_output = args.behavior_clips_output or (args.output_dir / "behavior_clips")
        extract_behavior_clips(
            args.video,
            events_path,
            clips_output,
            behavior_names=args.behaviors,
            event_level=args.behavior_level,
            clip_seconds=max(float(args.behavior_clip_seconds), 0.1),
            max_clips_per_behavior=max(int(args.max_clips_per_behavior), 1),
        )
        LOGGER.info("behavior_clips_output=%s", clips_output)
        return 0

    if args.extract_four_class_clips:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        clips_output = args.clips_output or (args.output_dir / "四类视频")
        extract_four_class_clips(
            args.video,
            events_path,
            clips_output,
            expected_level=args.clip_level,
            clip_seconds=max(float(args.clip_seconds), 0.1),
            max_clips_per_class=max(int(args.max_clips_per_class), 1),
        )
        LOGGER.info("clips_output=%s", clips_output)
        return 0

    if args.config is None or args.fps is None:
        parser.error(
            "普通分析模式需要同时提供 --config 和 --fps；渲染已有结果请使用 --render-only。"
        )
    # Keep the function signature self-contained while passing the video FPS.
    config = load_config(args.config)
    config["_fps_override"] = float(args.fps)
    temp_config = args.output_dir / ".lightweight_runtime_config.yaml"
    with temp_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    try:
        result_dir = analyze(
            args.video,
            args.yolo_cache,
            temp_config,
            args.output_dir,
            expected_mice=max(int(args.expected_mice), 2),
            max_frames=args.max_frames,
            sample_stride=max(int(args.sample_stride), 1),
        )
        metadata_path = result_dir / "lightweight_analysis_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["config"] = str(args.config)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    finally:
        temp_config.unlink(missing_ok=True)
    LOGGER.info("output_dir=%s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
