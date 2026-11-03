#!/usr/bin/env python3
"""Run canonical rules, full-video overlays and behavior-specific four-file packages."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ensure_importable

ensure_importable()

from mouse_behavior.canonical_video_pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New run directory; existing targets are refused")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--source-receipt", type=Path, help="Required for inference: source_video.path/sha256 and model.path/sha256")
    parser.add_argument("--source-sha256", help="Additional source consistency check; a model-bound source receipt is still required")
    parser.add_argument("--events-from", type=Path, help="Complete prior inference run; reuse events and trajectories without YOLO")
    parser.add_argument("--mode", choices=("all", "infer", "full", "clips"), default="all")
    parser.add_argument("--render-full", action="store_true", help="Request full render (alias for --mode full)")
    parser.add_argument("--export-clips", action="store_true", help="Request clips; full render is their reusable dependency")
    parser.add_argument("--core-seconds", type=float, default=20)
    parser.add_argument("--context-seconds", type=float, default=10)
    parser.add_argument("--posture-mode", choices=("disabled", "v3_review", "calibrated"), default="disabled",
                        help="Posture mode: disabled, v3_review (V3.1 review candidates), or calibrated (all 4 individual rules)")
    parser.add_argument("--workers", type=int, default=1, help="Independent context/core processes; default serial")
    parser.add_argument("--clip-workers", type=int, default=1, help="Independent event exports; each FFmpeg uses one thread, default serial")
    parser.add_argument("--clip-preset", choices=("veryfast", "ultrafast"), default="veryfast",
                        help="Clip H.264 encoding preset; CRF remains 21, full render is unchanged")
    parser.add_argument("--clip-context-seconds", type=float, default=0., help="Optional review context; default exports only the recognized event; context never qualifies duration")
    parser.add_argument("--original-video", action="store_true", help="Export original pixels with optional source audio encoded AAC")
    parser.add_argument("--target-overlay", action="store_true", help="Render each clip directly from the source, highlighting only that event's measured participants")
    parser.add_argument("--clip-codec", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--comparison-reviews", type=Path, help="Explicit source-bound human comparator review manifest")
    parser.add_argument("--comparison-sample-id", help="Exact reviewed sample ID; never applied across samples")
    parser.add_argument("--comparison-tracks", type=Path, help="Reviewed standardized tracks; must match every bound cache observation")
    args = parser.parse_args()
    mode = args.mode
    if args.render_full or args.export_clips:
        mode = "all" if args.export_clips else "full"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receipt = run_pipeline(args.video, args.pose_cache, args.output, args.config, mode=mode,
                           source_receipt=args.source_receipt, expected_source_sha256=args.source_sha256,
                           events_from=args.events_from, core_seconds=args.core_seconds,
                           context_seconds=args.context_seconds, workers=args.workers, clip_workers=args.clip_workers,
                           clip_preset=args.clip_preset, posture_mode=args.posture_mode,
                           clip_context_seconds=args.clip_context_seconds, original_video=args.original_video, clip_codec=args.clip_codec,
                           comparison_reviews=args.comparison_reviews, comparison_sample_id=args.comparison_sample_id,
                           comparison_tracks=args.comparison_tracks, target_overlay=args.target_overlay)
    logging.info("Complete: %s | events=%d clips=%d", args.output.resolve(), receipt["event_count"], receipt["clip_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
