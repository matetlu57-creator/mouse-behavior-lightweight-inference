#!/usr/bin/env python3
"""Serve the local Mouse Behavior Review & Interactive Clip Collection website."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.mouse_clip_cropper import is_ffmpeg_available
from mouse_behavior.visualization.mouse_clip_review_site import (
    DEFAULT_PORT,
    create_mouse_clip_review_server,
)
from mouse_behavior.visualization.mouse_clip_review_store import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Read-only source directory containing behavior clip folders (default: F:\\mouse)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Writable output directory strictly outside source-dir for reviews & collected clips",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Loopback bind port (default: 8795)")
    parser.add_argument("--open-browser", action="store_true", help="Open the local URL in default browser")
    args = parser.parse_args()

    server = create_mouse_clip_review_server(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )
    bound_host = f"[{args.host}]" if ":" in args.host else args.host
    bound_port = server.server_address[1]
    url = f"http://{bound_host}:{bound_port}/"

    print(f"小鼠行为审核与手动片段采集工作台已启动: {url}", flush=True)
    print(f"只读素材目录 (source-dir): {server.store.source_dir}", flush=True)
    print(f"独立输出目录 (output-dir): {server.store.output_dir}", flush=True)
    print(f"FFmpeg H.264 编码可用: {is_ffmpeg_available()}", flush=True)
    print(
        "可复现启动命令: "
        f'D:\\Anaconda3\\python.exe scripts/serve_mouse_clip_review_site.py '
        f'--source-dir "{args.source_dir}" --output-dir "{args.output_dir}" '
        f'--host {args.host} --port {bound_port}',
        flush=True,
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
