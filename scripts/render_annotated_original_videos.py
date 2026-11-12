"""Batch render full-length original behavior videos with detection bounding boxes & roles.

Usage:
    python scripts/render_annotated_original_videos.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.clip_overlay_renderer import (
    ROLE_COLORS,  # noqa: F401
    normalize_roles_map as _normalize_roles_map,  # noqa: F401
    render_annotated_clip_from_dir as render_single_annotated_original_clip,
)
from mouse_behavior.visualization.comparison_site import (
    DEFAULT_ATTACK_DIR,
    DEFAULT_CHASING_DIR,
    DEFAULT_EVALUATION_JSON,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ISOLATION_DIR,
    DEFAULT_LOCOMOTION_DIR,
    DEFAULT_ORIG_ANNOTATED_DIR,
    DEFAULT_STATIC_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render full-length original behavior videos with detection boxes & roles.",
    )
    parser.add_argument("--evaluation-file", type=Path, default=DEFAULT_EVALUATION_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ORIG_ANNOTATED_DIR)
    parser.add_argument("--attack-dir", type=Path, default=DEFAULT_ATTACK_DIR)
    parser.add_argument("--chasing-dir", type=Path, default=DEFAULT_CHASING_DIR)
    parser.add_argument("--isolation-dir", type=Path, default=DEFAULT_ISOLATION_DIR)
    parser.add_argument("--group-locomotion-dir", type=Path, default=DEFAULT_GROUP_LOCOMOTION_DIR)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--locomotion-dir", type=Path, default=DEFAULT_LOCOMOTION_DIR)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _resolve_clip_dir(
    clip_id: str,
    behavior: str,
    behavior_dirs: dict[str, Path],
) -> Path | None:
    preferred = behavior_dirs.get(behavior)
    if preferred is not None:
        candidate = preferred / clip_id
        if candidate.is_dir():
            return candidate
    for base_dir in behavior_dirs.values():
        candidate = base_dir / clip_id
        if candidate.is_dir():
            return candidate
    return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    behavior_dirs: dict[str, Path] = {
        "Attack": args.attack_dir,
        "Chasing": args.chasing_dir,
        "Isolation": args.isolation_dir,
        "Group locomotion": args.group_locomotion_dir,
        "Static": args.static_dir,
        "Locomotion": args.locomotion_dir,
    }

    tasks: list[dict[str, Any]] = []
    if args.evaluation_file.is_file():
        eval_data = json.loads(args.evaluation_file.read_text(encoding="utf-8"))
        for clip in eval_data.get("per_clip_results", []):
            cid = clip["clip_id"]
            beh = clip.get("ground_truth_behavior", "")
            clip_path = _resolve_clip_dir(cid, beh, behavior_dirs)
            if clip_path is not None:
                tasks.append({
                    "clip_dir": clip_path,
                    "target_ids": set(clip.get("target_mouse_ids", [])),
                    "roles_map": clip.get("semantic_roles", {}),
                    "gt_behavior": beh,
                })
    else:
        for beh, base_dir in behavior_dirs.items():
            if not base_dir.is_dir():
                continue
            for d in sorted(base_dir.glob("*")):
                if d.is_dir() and (d / "clip.mp4").is_file():
                    tasks.append({"clip_dir": d, "gt_behavior": beh})

    logging.info("Found %d clips to render annotated original videos.", len(tasks))
    t0 = time.time()
    success_count = 0

    def _worker(item: dict[str, Any]) -> tuple[str, bool]:
        ok = render_single_annotated_original_clip(
            clip_dir=item["clip_dir"],
            output_dir=args.output_dir,
            ffmpeg_exe=ffmpeg_exe,
            target_ids=item.get("target_ids"),
            roles_map=item.get("roles_map"),
            gt_behavior=item.get("gt_behavior"),
        )
        return item["clip_dir"].name, ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            cid, ok = future.result()
            if ok:
                success_count += 1
                logging.info("[%d/%d] Rendered: %s", success_count, len(tasks), cid)
            else:
                logging.warning("Failed to render: %s", cid)

    elapsed = time.time() - t0
    logging.info(
        "Rendered %d/%d annotated original videos in %.2fs -> %s",
        success_count,
        len(tasks),
        elapsed,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
