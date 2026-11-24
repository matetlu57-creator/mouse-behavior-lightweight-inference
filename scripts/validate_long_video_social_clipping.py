#!/usr/bin/env python3
"""End-to-end social-behavior cropping validation and MP4 review clip exporter for long videos."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.canonical_video_pipeline import (
    TrackStore,
    config_seed_parameters,
    policy_receipt,
    rule_execution_binding,
    video_info,
)
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.segment_export import _sha256, _write_json
from mouse_behavior.evaluation.social_cropping_validation import (
    CachedTrackStore,
    refine_all_social_events,
    select_social_review_clips,
    summarize_social_cropping_validation,
)
from mouse_behavior.visualization.behavior_packages import export_packages, write_manifests

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "long_video_social_validation_20261123"


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


def run_long_video_social_validation(
    *,
    video_key: str,
    title: str,
    video: Path,
    inference_dir: Path,
    replay_json: Path | None,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    clip_workers: int = 4,
    max_clips_per_behavior: int | None = None,
    clip_context_seconds: float = 0.0,
    skip_clips: bool = False,
    clean_clips: bool = False,
) -> dict[str, Any]:
    """Replay cached full-video inference with upgraded social clipping and export review clips."""
    resolved_out = output_dir.resolve()
    resolved_out.mkdir(parents=True, exist_ok=True)
    source_video = video.resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"source video not found: {source_video}")

    db_path = inference_dir / "tracks.sqlite"
    binding_path = inference_dir / "input_binding.json"
    scale_path = inference_dir / "scale_receipt.json"
    pre_path = inference_dir / "pre_crop_events.json"
    for req in (db_path, binding_path, scale_path):
        if not req.is_file():
            raise FileNotFoundError(f"missing cached inference artifact: {req}")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    params = config_seed_parameters(load_config(config_path))
    # Refresh rule binding with current modular social clipping provenance.
    binding.update(rule_execution_binding(params, "calibrated"))

    if replay_json is not None and replay_json.is_file():
        replay_data = json.loads(replay_json.read_text(encoding="utf-8"))
        pre_events = replay_data.get("raw_events", [])
        inference_source = f"replay_20261123_social_from_{replay_json.name}"
    elif pre_path.is_file():
        pre_events = json.loads(pre_path.read_text(encoding="utf-8"))
        inference_source = f"replay_20261123_social_from_{pre_path.name}"
    else:
        raise FileNotFoundError(f"missing pre_crop_events / replay JSON for {video_key}")

    store = TrackStore(db_path)
    cached_store = CachedTrackStore(store)
    try:
        post_events, refine_meta = refine_all_social_events(
            pre_events,
            cached_store,
            binding,
            scale,
            params,
        )
        post_path = resolved_out / "social_events.json"
        _write_json(post_path, post_events)

        summary_stats, per_event_audit = summarize_social_cropping_validation(
            pre_events,
            post_events,
            cached_store,
            binding,
            params,
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
            "social_behavior_counts": dict(Counter(e["canonical_behavior"] for e in post_events)),
            "aggregate": summary_stats["aggregate"],
            "by_behavior": summary_stats["by_behavior"],
            "refinement_meta_counts": {
                "approach_audit_count": len(refine_meta.get("refinement_audit", {}).get("approach_audit", [])),
                "selection_audit": refine_meta.get("selection_audit", {}),
            },
            "policy": policy_receipt(),
            "seeded_parameters": asdict(params),
        }

        selected = select_social_review_clips(
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
        if clean_clips:
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            if overlay_dir.exists():
                shutil.rmtree(overlay_dir)
        if not (raw_dir / "manifest.json").is_file():
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

        if not (overlay_dir / "manifest.json").is_file():
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
                    "actor_id": ev.get("actor_id", -1),
                    "target_id": ev.get("target_id", -1),
                    "initiator_id": ev.get("initiator_id"),
                    "receiver_id": ev.get("receiver_id"),
                    "role_status": ev.get("role_status", "not_applicable"),
                    "roles": ev.get("roles"),
                    "has_visual_id_switch": bool(ev.get("has_visual_id_switch", False)),
                    "id_switch_note": ev.get("id_switch_note"),
                    "discovery_source": ev.get("discovery_source"),
                    "observed_onset_pair": ev.get("observed_onset_pair"),
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
    parser.add_argument("--run-both", action="store_true", help="Run default Video Project 22 and 大旷场1 validation")
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--inference-dir", type=Path, default=None)
    parser.add_argument("--replay-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--clip-workers", type=int, default=4)
    parser.add_argument(
        "--max-clips-per-behavior",
        type=int,
        default=0,
        help="Optional per-behavior clip export cap (<=0 means unlimited; default: 0)",
    )
    parser.add_argument("--clip-context-seconds", type=float, default=0.0)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--clean-clips", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.run_both:
        runs = [
            {
                "video_key": "video22",
                "title": "Video Project 22 (10分钟 · 2026-11-24 社交裁剪升级)",
                "video": Path(r"F:\小鼠\Video Project 22_三文件\Video Project 22.mp4"),
                "inference_dir": REPO_ROOT / "outputs" / "long_video_group_validation_20261118" / "video22_verified" / "inference",
                "replay_json": REPO_ROOT / "outputs" / "long_video_group_validation_20261122" / "video22_replay.json",
                "output_dir": DEFAULT_OUTPUT_ROOT / "video22_verified",
            },
            {
                "video_key": "dakuangchang1",
                "title": "大旷场1 (60分钟 · 2026-11-24 社交裁剪升级)",
                "video": Path(r"F:\北医标注-行为例子\大旷场1.mp4"),
                "inference_dir": REPO_ROOT / "outputs" / "long_video_group_validation_20261118" / "dakuangchang1_verified" / "inference",
                "replay_json": REPO_ROOT / "outputs" / "long_video_group_validation_20261122" / "dakuangchang1_replay.json",
                "output_dir": DEFAULT_OUTPUT_ROOT / "dakuangchang1_verified",
            },
        ]
        results = []
        for cfg in runs:
            LOGGER.info("Validating social clipping on %s...", cfg["video_key"])
            res = run_long_video_social_validation(
                **cfg,
                config_path=args.config,
                clip_workers=args.clip_workers,
                max_clips_per_behavior=args.max_clips_per_behavior,
                clip_context_seconds=args.clip_context_seconds,
                skip_clips=args.skip_clips,
                clean_clips=args.clean_clips,
            )
            results.append({"video_key": res["video_key"], "aggregate": res["aggregate"]})
        print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False, indent=2))
        return 0

    if not all((args.video_key, args.title, args.video, args.inference_dir, args.output_dir)):
        parser.error("Provide --run-both or all of --video-key, --title, --video, --inference-dir, --output-dir")

    summary = run_long_video_social_validation(
        video_key=args.video_key,
        title=args.title,
        video=args.video,
        inference_dir=args.inference_dir,
        replay_json=args.replay_json,
        output_dir=args.output_dir,
        config_path=args.config,
        clip_workers=args.clip_workers,
        max_clips_per_behavior=args.max_clips_per_behavior,
        clip_context_seconds=args.clip_context_seconds,
        skip_clips=args.skip_clips,
        clean_clips=args.clean_clips,
    )
    print(json.dumps({"status": "ok", "video_key": summary["video_key"], "aggregate": summary["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
