#!/usr/bin/env python3
"""Serve the local manual review website for long-video group behavior validation."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.long_video_group_review_site import (
    DEFAULT_REVIEW_SITE_DIR,
    DEFAULT_REVIEW_SITE_PORT,
    create_long_video_group_review_server,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_REVIEW_SITE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_REVIEW_SITE_PORT)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"长视频群体行为人工审核网站已启动: {url}", flush=True)
    print(f"网站目录: {args.site_dir.resolve()}", flush=True)
    print(
        f"可复现启动命令: python scripts/serve_long_video_group_review_site.py --site-dir \"{args.site_dir}\" --port {args.port}",
        flush=True,
    )
    server = create_long_video_group_review_server(
        site_dir=args.site_dir,
        port=args.port,
        host=args.host,
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
