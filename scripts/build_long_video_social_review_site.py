#!/usr/bin/env python3
"""Build the standalone local manual review website from long-video social validation outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.long_video_social_review_site import (
    DEFAULT_SOCIAL_REVIEW_PORT,
    DEFAULT_SOCIAL_REVIEW_SITE_DIR,
    build_long_video_social_review_site,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SOCIAL_REVIEW_SITE_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_SOCIAL_REVIEW_PORT)
    parser.add_argument("--skip-video-copy", action="store_true")
    parser.add_argument("--clean-videos", action="store_true")
    args = parser.parse_args()

    result = build_long_video_social_review_site(
        run_dirs=args.run_dirs,
        site_dir=args.site_dir,
        copy_videos=not args.skip_video_copy,
        port=args.port,
        clean_videos=args.clean_videos,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f'启动命令: D:\\Anaconda3\\python.exe scripts/serve_long_video_social_review_site.py --site-dir "{result["site_dir"]}" --port {args.port}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
