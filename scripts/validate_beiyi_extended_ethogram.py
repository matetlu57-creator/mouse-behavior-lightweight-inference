#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the lightweight extended ethogram over every Beiyi example video.

The Beiyi folders are video-level examples, not frame-level annotations.  This
script therefore reports *video coverage* (whether the expected label appears
in its labelled video) and does not claim Precision/Recall/F1.  It processes
all visible mice in each cage and never assumes a single- or two-mouse clip.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO

from _bootstrap import REPO_ROOT
from mouse_behavior import lightweight_behavior_inference as lightweight
from mouse_behavior.config import load_config
from mouse_behavior import pose_cache
from mouse_behavior.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".wmv", ".m4v"}

LABELS = {
    ("社交行为", "1.一起"): "together",
    ("社交行为", "2.接近-被接近"): "approach",
    ("社交行为", "3.追逐-被追逐"): "chase",
    ("社交行为", "4.回避-被回避"): "avoidance",
    ("社交行为", "5.攻击行为"): "attack",
    ("社交行为", "6.鼻头接触"): "nose_head",
    ("社交行为", "7.鼻尾接触"): "nose_tail",
    ("群体行为", "1.扎堆行为"): "huddle",
    ("群体行为", "2.孤立行为"): "isolation",
    ("个体行为", "1.奔跑"): "running",
    ("个体行为", "2.行走"): "walking",
    ("个体行为", "3.静止"): "stationary",
}


def _video_info(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开北医视频: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"视频元数据无效: fps={fps}, frames={frames}, path={path}")
    return fps, frames


def _safe_case_name(index: int, video: Path, dataset: Path) -> str:
    relative = video.relative_to(dataset)
    parts = [part.replace(" ", "_") for part in relative.parts]
    return f"{index:03d}_" + "__".join(parts).replace(video.suffix, "")


def _contact_hit(contact_df: pd.DataFrame, expected: str) -> bool:
    if contact_df.empty or "contact_type_components" not in contact_df:
        return False
    needle = "nose_head" if expected == "nose_head" else "nose_tail"
    return bool(
        contact_df["contact_type_components"]
        .astype(str)
        .map(lambda value: needle in value.split(";"))
        .any()
    )


def _behavior_hit(events_df: pd.DataFrame, expected: str) -> bool:
    if events_df.empty or "behavior" not in events_df:
        return False
    return bool(events_df["behavior"].astype(str).eq(expected).any())


def _load_or_build_cache(
    video: Path,
    cache_dir: Path,
    model: YOLO,
    model_path: Path,
    *,
    batch_size: int,
    chunk_frames: int,
    imgsz: int,
    device: str,
) -> tuple[float, int, bool]:
    fps, total_frames = _video_info(video)
    status_path = cache_dir / "yolo_results_status.json"
    reusable = False
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            reusable = (
                status.get("status") == "complete"
                and int(status.get("total_frames", -1)) == total_frames
                and bool(status.get("pose_only", False))
                and not bool(status.get("obb_used", True))
                and Path(str(status.get("model", ""))).resolve() == model_path.resolve()
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            reusable = False
    if not reusable:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pose_cache.build_cache(
            video,
            cache_dir,
            model_path,
            batch_size=batch_size,
            chunk_frames=chunk_frames,
            imgsz=imgsz,
            device=device,
            model=model,
        )
    return fps, total_frames, reusable


def run_validation(
    dataset: Path,
    output: Path,
    config_path: Path,
    model_path: Path,
    *,
    batch_size: int = 8,
    chunk_frames: int = 300,
    imgsz: int = 768,
    device: str = "0",
    expected_mice: int = 20,
    sample_stride: int = 1,
    force: bool = False,
) -> Path:
    dataset = dataset.resolve()
    output = output.resolve()
    config_path = config_path.resolve()
    model_path = model_path.resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    videos = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    unknown = []
    cases: list[tuple[Path, str]] = []
    for video in videos:
        relative = video.relative_to(dataset)
        if len(relative.parts) < 3:
            unknown.append(str(video))
            continue
        expected = LABELS.get((relative.parts[0], relative.parts[1]))
        if expected is None:
            unknown.append(str(video))
            continue
        cases.append((video, expected))
    if unknown:
        raise ValueError("存在未映射的北医视频目录:\n" + "\n".join(unknown))
    if not cases:
        raise ValueError(f"未找到北医视频: {dataset}")

    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    config["_fps_override"] = 30.0
    runtime_config = output / ".beiyi_runtime_config.yaml"
    runtime_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    model = YOLO(str(model_path))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, (video, expected) in enumerate(cases, start=1):
            case_name = _safe_case_name(index, video, dataset)
            case_dir = output / case_name
            cache_dir = case_dir / "yolo_precompute"
            analysis_dir = case_dir / "analysis"
            if force and case_dir.exists():
                shutil.rmtree(case_dir)
            fps, total_frames, cache_reused = _load_or_build_cache(
                video,
                cache_dir,
                model,
                model_path,
                batch_size=batch_size,
                chunk_frames=chunk_frames,
                imgsz=imgsz,
                device=device,
            )
            lightweight.analyze(
                video,
                cache_dir,
                runtime_config,
                analysis_dir,
                expected_mice=max(int(expected_mice), 2),
                sample_stride=max(int(sample_stride), 1),
            )
            events_path = analysis_dir / "lightweight_behavior_events.csv"
            contacts_path = analysis_dir / "lightweight_contact_events.csv"
            metadata_path = analysis_dir / "lightweight_analysis_metadata.json"
            events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
            contacts_df = pd.read_csv(contacts_path) if contacts_path.exists() else pd.DataFrame()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            hit = (
                _contact_hit(contacts_df, expected)
                if expected in {"nose_head", "nose_tail"}
                else _behavior_hit(events_df, expected)
            )
            behavior_counts = (
                events_df["behavior"].astype(str).value_counts().to_dict()
                if "behavior" in events_df
                else {}
            )
            contact_counts = (
                contacts_df["contact_type"].astype(str).value_counts().to_dict()
                if "contact_type" in contacts_df
                else {}
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
                    "cache_reused": bool(cache_reused),
                    "event_count": int(len(events_df)),
                    "contact_event_count": int(len(contacts_df)),
                    "behavior_counts_json": json.dumps(
                        behavior_counts, ensure_ascii=False, sort_keys=True
                    ),
                    "contact_counts_json": json.dumps(
                        contact_counts, ensure_ascii=False, sort_keys=True
                    ),
                    "analysis_elapsed_s": float(metadata.get("elapsed_s", 0.0)),
                    "case_output": str(case_dir),
                }
            )
            LOGGER.info(
                "[%02d/%d] %18s %4s events=%3d contacts=%3d %s",
                index,
                len(cases),
                expected,
                "HIT" if hit else "MISS",
                len(events_df),
                len(contacts_df),
                video.name,
            )
    finally:
        runtime_config.unlink(missing_ok=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output / "beiyi_video_validation.csv", index=False, encoding="utf-8-sig")
    aggregate = (
        result_df.groupby("expected_behavior", sort=True)
        .agg(
            video_count=("video", "count"),
            detected_video_count=("target_event_found", "sum"),
            total_event_count=("event_count", "sum"),
            total_contact_event_count=("contact_event_count", "sum"),
        )
        .reset_index()
    )
    aggregate["video_coverage"] = aggregate["detected_video_count"] / aggregate["video_count"]
    aggregate.to_csv(output / "beiyi_behavior_coverage.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset": str(dataset),
        "model": str(model_path),
        "pose_only": True,
        "obb_used": False,
        "multi_mouse_scene": True,
        "video_count": int(len(result_df)),
        "all_target_videos_hit": bool(result_df["target_event_found"].all()),
        "elapsed_s": float(time.perf_counter() - started),
        "coverage_csv": str(output / "beiyi_behavior_coverage.csv"),
        "case_csv": str(output / "beiyi_video_validation.csv"),
        "limitation": "北医标签是视频/文件夹级示例，不是逐帧真值；video_coverage 不能替代 Precision、Recall、F1 或 actor/target accuracy。",
    }
    (output / "beiyi_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("summary=\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "weights" / "pose" / "best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-mice", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    run_validation(
        args.dataset,
        args.output,
        args.config,
        args.model,
        batch_size=args.batch_size,
        chunk_frames=args.chunk_frames,
        imgsz=args.imgsz,
        device=str(args.device),
        expected_mice=args.expected_mice,
        sample_stride=args.sample_stride,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
