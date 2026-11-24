#!/usr/bin/env python3
"""End-to-end group-behavior cropping precision validator and clip exporter for long videos.

Supports both:
1. Pre-inferred delivered directories containing ``tracks.json`` and ``metadata.json``
   (imported via ``scripts/infer_from_delivered_yolo_json.py:import_delivered`` without YOLO re-inference).
2. Un-inferred raw ``.mp4`` videos (runs GPU Ultralytics YOLO26 Pose cache generation in the
   configured CUDA Python environment ``D:\\Anaconda3\\envs\\yolo26\\python.exe``).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT  # noqa: F401
from infer_from_delivered_yolo_json import import_delivered
from mouse_behavior.canonical_video_pipeline import (
    TrackStore,
    bind_cache,
    config_seed_parameters,
    infer_video_events,
    policy_receipt,
    rule_execution_binding,
    video_info,
)
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.group_cropping_validation import (
    CachedTrackStore,
    load_historical_review_decisions,
    refine_all_canonical_and_group_events,
    select_group_review_clips,
    summarize_group_cropping_validation,
    write_pose_cache_source_receipt,
)
from mouse_behavior.evaluation.segment_export import _sha256, _write_json
from mouse_behavior.visualization.behavior_packages import export_packages, write_manifests

LOGGER = logging.getLogger(__name__)
DEFAULT_YOLO_PYTHON = Path(r"D:\Anaconda3\envs\yolo26\python.exe")
DEFAULT_MODEL_PATH = REPO_ROOT / "weights" / "pose" / "best.pt"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _detect_clip_codec() -> str:
    import imageio_ffmpeg

    smoke = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=30",
            "-frames:v",
            "3",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=30,
    )
    return "h264_nvenc" if smoke.returncode == 0 else "libx264"


def _prepare_cache(
    *,
    video: Path | None,
    delivered_dir: Path | None,
    pose_cache_dir: Path,
    model_path: Path,
    yolo_python: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    tracker: str,
) -> tuple[Path, Path, Path, str]:
    """Ensure a valid complete pose cache + ``source_receipt.json`` exist."""
    if delivered_dir is not None:
        meta = json.loads((delivered_dir / "metadata.json").read_text(encoding="utf-8-sig"))
        source_video = delivered_dir / Path(meta["source"]).name
        status_file = pose_cache_dir / "yolo_results_status.json"
        receipt_file = pose_cache_dir / "source_receipt.json"
        if not (status_file.is_file() and receipt_file.is_file()):
            stage_base = pose_cache_dir.parent / pose_cache_dir.name.removesuffix("_import")
            imported = import_delivered(delivered_dir, stage_base)
            return (
                Path(imported["video_path"]),
                Path(imported["cache_directory"]),
                Path(imported["source_receipt"]),
                "delivered_tracks_import",
            )
        return source_video, pose_cache_dir, receipt_file, "delivered_tracks_import_reused"

    if video is None:
        raise ValueError("either --video or --delivered-dir must be provided")
    resolved_video = video.resolve()
    status_file = pose_cache_dir / "yolo_results_status.json"
    if not status_file.is_file():
        pose_cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if not yolo_python.is_file():
            raise FileNotFoundError(f"YOLO26 Python interpreter not found: {yolo_python}")
        cmd = [
            str(yolo_python),
            str(REPO_ROOT / "scripts" / "build_lightweight_pose_cache.py"),
            "--video",
            str(resolved_video),
            "--output",
            str(pose_cache_dir),
            "--model",
            str(model_path.resolve()),
            "--imgsz",
            str(imgsz),
            "--conf",
            str(conf),
            "--iou",
            str(iou),
            "--max-det",
            str(max_det),
            "--device",
            str(device),
            "--tracker",
            str(tracker),
        ]
        LOGGER.info("Running GPU YOLO26 Pose cache build: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        inference_source = "yolo26_gpu_new_inference"
    else:
        LOGGER.info("Reusing existing complete pose cache: %s", pose_cache_dir)
        inference_source = "yolo26_gpu_cached"
    receipt_file = write_pose_cache_source_receipt(resolved_video, pose_cache_dir, model_path)
    return resolved_video, pose_cache_dir, receipt_file, inference_source


def run_long_video_group_validation(
    *,
    video_key: str,
    title: str,
    output_dir: Path,
    video: Path | None = None,
    delivered_dir: Path | None = None,
    pose_cache_dir: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    yolo_python: Path = DEFAULT_YOLO_PYTHON,
    historical_review_json: Path | None = None,
    workers: int = 4,
    clip_workers: int = 2,
    max_clips_per_behavior: int | None = None,
    clip_context_seconds: float = 0.0,
    skip_clips: bool = False,
    device: str = "0",
    imgsz: int = 768,
    conf: float = 0.15,
    iou: float = 0.50,
    max_det: int = 100,
    tracker: str = "bytetrack.yaml",
) -> dict[str, Any]:
    """Execute pose cache preparation, full canonical inference, group cropping audit, and clip export."""
    resolved_out = output_dir.resolve()
    resolved_out.mkdir(parents=True, exist_ok=True)
    default_cache = (
        resolved_out / "pose_stage_import"
        if delivered_dir is not None
        else resolved_out / "pose_cache"
    )
    resolved_cache = (pose_cache_dir or default_cache).resolve()

    source_video, cache_dir, source_receipt, inference_source = _prepare_cache(
        video=video,
        delivered_dir=delivered_dir,
        pose_cache_dir=resolved_cache,
        model_path=model_path,
        yolo_python=yolo_python,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        tracker=tracker,
    )

    inf_dir = resolved_out / "inference"
    db_path = inf_dir / "tracks.sqlite"
    binding_path = inf_dir / "input_binding.json"
    scale_path = inf_dir / "scale_receipt.json"
    pre_path = inf_dir / "pre_crop_events.json"
    post_path = inf_dir / "events.json"

    params = config_seed_parameters(load_config(config_path))

    if db_path.is_file() and binding_path.is_file() and scale_path.is_file() and pre_path.is_file():
        LOGGER.info("Reusing pre-crop inference results in %s", inf_dir)
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        scale = json.loads(scale_path.read_text(encoding="utf-8"))
        pre_events = json.loads(pre_path.read_text(encoding="utf-8"))
        store = TrackStore(db_path)
    else:
        if db_path.exists():
            db_path.unlink()
        inf_dir.mkdir(parents=True, exist_ok=True)
        binding, scale = bind_cache(source_video, cache_dir, db_path, source_receipt=source_receipt)
        store = TrackStore(db_path)
        binding.update(rule_execution_binding(params, "calibrated"))
        pre_events, _pre_audit = infer_video_events(
            store,
            binding,
            scale,
            params,
            core_seconds=20,
            context_seconds=10,
            workers=workers,
            posture_mode="calibrated",
            enable_fine_cropping=False,
            max_events_per_behavior=None,
        )
        _write_json(binding_path, binding)
        _write_json(scale_path, scale)
        _write_json(pre_path, pre_events)

    cached_store = CachedTrackStore(store)
    try:
        post_events, refine_meta = refine_all_canonical_and_group_events(
            pre_events,
            cached_store,
            binding,
            scale,
            params,
        )
        if post_path.exists():
            post_path.unlink()
        _write_json(post_path, post_events)

        run_receipt = {
            "status": "complete",
            "video_key": video_key,
            "inference_source": inference_source,
            "posture_mode": "calibrated",
            "enable_fine_cropping": True,
            "enable_huddle_sliding": True,
            "all_behaviors_counts": dict(Counter(e["canonical_behavior"] for e in post_events)),
            "policy": policy_receipt(),
            "seeded_parameters": asdict(params),
        }
        receipt_file = inf_dir / "run_receipt.json"
        if receipt_file.exists():
            receipt_file.unlink()
        _write_json(receipt_file, run_receipt)

        hist = load_historical_review_decisions(historical_review_json)
        summary_stats, per_event_audit = summarize_group_cropping_validation(
            pre_events,
            post_events,
            cached_store,
            binding,
            scale,
            params,
            historical_reviews=hist,
        )

        vinfo = video_info(source_video)
        summary = {
            "video_key": video_key,
            "title": title,
            "video_path": str(source_video),
            "video_sha256": binding.get("source_video", {}).get("sha256") or _sha256(source_video),
            "fps": vinfo["fps"],
            "frame_count": vinfo["frame_count"],
            "duration_s": round(vinfo["frame_count"] / vinfo["fps"], 3),
            "inference_source": inference_source,
            "scale_cm_per_pixel": scale["cm_per_pixel"],
            "scale_reference_body_px": scale.get("reference_body_px"),
            "all_behaviors_counts": run_receipt["all_behaviors_counts"],
            "aggregate": summary_stats["aggregate"],
            "by_behavior": summary_stats["by_behavior"],
            "refinement_meta_counts": {
                "huddle_audit_count": len(refine_meta.get("refinement_audit", {}).get("huddle_audit", [])),
                "selection_audit": refine_meta.get("selection_audit", {}),
            },
        }

        selected = select_group_review_clips(
            post_events,
            cached_store,
            binding,
            per_event_audit=per_event_audit,
            max_per_behavior=(
                None
                if (max_clips_per_behavior is None or max_clips_per_behavior <= 0)
                else max_clips_per_behavior
            ),
        )
    finally:
        cached_store.close()

    review_candidates: list[dict[str, Any]] = []
    if selected and not skip_clips:
        codec = _detect_clip_codec()
        raw_dir = resolved_out / "clips_raw"
        overlay_dir = resolved_out / "clips_overlay"
        if not raw_dir.exists():
            raw_rows = export_packages(
                raw_dir,
                source_video,
                store,
                selected,
                binding,
                scale,
                workers=clip_workers,
                clip_preset="veryfast",
                clip_context_seconds=clip_context_seconds,
                original_video=True,
                clip_codec=codec,
            )
            write_manifests(raw_dir, raw_rows)
        else:
            raw_rows = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))["clips"]

        if not overlay_dir.exists():
            overlay_rows = export_packages(
                overlay_dir,
                source_video,
                store,
                selected,
                binding,
                scale,
                workers=clip_workers,
                clip_preset="veryfast",
                clip_context_seconds=clip_context_seconds,
                target_overlay=True,
                clip_codec=codec,
            )
            write_manifests(overlay_dir, overlay_rows)
        else:
            overlay_rows = json.loads((overlay_dir / "manifest.json").read_text(encoding="utf-8"))["clips"]

        raw_map = {str(r["event_id"]): r["path"] for r in raw_rows}
        ov_map = {str(r["event_id"]): r["path"] for r in overlay_rows}
        fps = vinfo["fps"]
        for ev in selected:
            eid = str(ev["event_id"])
            review_candidates.append(
                {
                    "event_id": eid,
                    "behavior": ev["behavior"],
                    "canonical_behavior": ev["canonical_behavior"],
                    "mouse_ids": ev["mouse_ids"],
                    "start_frame": ev["start_frame"],
                    "end_frame": ev["end_frame"],
                    "frame_count": ev["end_frame"] - ev["start_frame"],
                    "start_time_s": round(ev["start_frame"] / fps, 3),
                    "end_time_s": round(ev["end_frame"] / fps, 3),
                    "duration_s": round((ev["end_frame"] - ev["start_frame"]) / fps, 3),
                    "raw_package_dir": raw_map[eid],
                    "overlay_package_dir": ov_map[eid],
                    "audit": per_event_audit.get(eid, {}),
                }
            )

    summary_file = resolved_out / "validation_summary.json"
    if summary_file.exists():
        summary_file.unlink()
    _write_json(summary_file, summary)

    cand_file = resolved_out / "review_candidates.json"
    if cand_file.exists():
        cand_file.unlink()
    _write_json(cand_file, review_candidates)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--delivered-dir", type=Path, default=None)
    parser.add_argument("--pose-cache-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--yolo-python", type=Path, default=DEFAULT_YOLO_PYTHON)
    parser.add_argument("--historical-review-json", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-workers", type=int, default=2)
    parser.add_argument(
        "--max-clips-per-behavior",
        type=int,
        default=0,
        help="Optional per-behavior clip export cap (<=0 means unlimited; default: 0)",
    )
    parser.add_argument("--clip-context-seconds", type=float, default=0.0)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_long_video_group_validation(
        video_key=args.video_key,
        title=args.title,
        output_dir=args.output_dir,
        video=args.video,
        delivered_dir=args.delivered_dir,
        pose_cache_dir=args.pose_cache_dir,
        model_path=args.model,
        config_path=args.config,
        yolo_python=args.yolo_python,
        historical_review_json=args.historical_review_json,
        workers=args.workers,
        clip_workers=args.clip_workers,
        max_clips_per_behavior=args.max_clips_per_behavior,
        clip_context_seconds=args.clip_context_seconds,
        skip_clips=args.skip_clips,
        device=args.device,
    )
    print(json.dumps({"status": "ok", "video_key": summary["video_key"], "aggregate": summary["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
