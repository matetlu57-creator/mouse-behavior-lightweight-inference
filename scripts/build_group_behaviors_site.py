"""Build the standalone, self-contained local website bundle for the three group behaviors.

Usage:
    python scripts/build_group_behaviors_site.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.group_behaviors_site import (
    DEFAULT_GROUP_EVAL_JSON,
    DEFAULT_SITE_DIR,
    build_standalone_group_website,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standalone local website directory for Dispersal, Group locomotion, and Social clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=DEFAULT_GROUP_EVAL_JSON,
        help="Evaluation JSON file produced by scripts/evaluate_group_behaviors.py",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        help="Target standalone website directory",
    )
    parser.add_argument(
        "--skip-video-copy",
        action="store_true",
        help="Write HTML/JSON/server files without copying MP4 files",
    )
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    result = build_standalone_group_website(
        evaluation_file=args.evaluation_file,
        site_dir=args.site_dir,
        copy_videos=not args.skip_video_copy,
    )
    print("=" * 72, flush=True)
    print("  独立群体行为本地网站构建完成 (Standalone Group Behaviors Website)", flush=True)
    print("=" * 72, flush=True)
    print(f"  ● 网站目录: {result['site_dir']}", flush=True)
    print(f"  ● 纳入片段: {result['total_clips']} 段 (Dispersal / Group locomotion / Social clustering)", flush=True)
    print(f"  ● 复制视频: {result['copied_videos']}", flush=True)
    print("  ● 启动命令: python scripts/serve_group_behaviors_site.py --port 8780", flush=True)
    print("  ● 也可直接双击目录中的 start_local_site.bat 或 index.html 打开", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
