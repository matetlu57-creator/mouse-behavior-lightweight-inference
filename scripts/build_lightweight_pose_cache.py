#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for building the seven-keypoint YOLO Pose cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT
from mouse_behavior.logging_config import configure_logging
from mouse_behavior.pose_cache import VIDEO_EXTENSIONS, build_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="输出 yolo_precompute 目录"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "weights" / "pose" / "best.pt",
        help="七关键点 YOLO Pose 权重；不使用 OBB",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if args.video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的视频扩展名: {args.video.suffix}")
    if not args.model.exists():
        raise FileNotFoundError(f"Pose 权重不存在: {args.model}")
    build_cache(
        args.video,
        args.output,
        args.model,
        batch_size=args.batch_size,
        chunk_frames=args.chunk_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
