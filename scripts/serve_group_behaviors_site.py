"""CLI script to serve the standalone local website for the three group behaviors:
Dispersal (群体分散), Group locomotion (群体同步运动), and Social clustering (社会聚集).

Usage:
    python scripts/serve_group_behaviors_site.py --port 8780
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.group_behaviors_site import (
    DEFAULT_GROUP_EVAL_JSON,
    DEFAULT_SITE_DIR,
    DEFAULT_SITE_PORT,
    build_standalone_group_website,
    create_group_behaviors_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the standalone local website for Dispersal, Group locomotion, and Social clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (loopback only for local security)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SITE_PORT,
        help="Port number to listen on",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        help="Standalone website directory containing index.html, data/, and videos/",
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=None,
        help="Optional override path for evaluation JSON (defaults to <site-dir>/data/evaluation.json)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild standalone site bundle from outputs/group_behaviors_evaluation.json before starting",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open default browser to the local website",
    )
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    site_dir: Path = args.site_dir
    if args.rebuild or (not (site_dir / "index.html").is_file() and DEFAULT_GROUP_EVAL_JSON.is_file()):
        build_standalone_group_website(
            evaluation_file=DEFAULT_GROUP_EVAL_JSON,
            site_dir=site_dir,
            copy_videos=True,
        )

    eval_file = args.evaluation_file or (site_dir / "data" / "evaluation.json")
    url = f"http://{args.host}:{args.port}/"

    print("=" * 76, flush=True)
    print("  三种群体行为独立本地网站 (Dispersal / Group locomotion / Social clustering)", flush=True)
    print("=" * 76, flush=True)
    print(f"  ● 访问地址: {url}", flush=True)
    print(f"  ● 独立目录: {site_dir} (存在: {site_dir.is_dir()})", flush=True)
    print(f"  ● 评测数据: {eval_file} (存在: {eval_file.is_file()})", flush=True)
    print(f"  ● 本地视频: {site_dir / 'videos'} (存在: {(site_dir / 'videos').is_dir()})", flush=True)
    print("=" * 76, flush=True)
    print("  单条可复现启动命令:", flush=True)
    print(f"    python scripts/serve_group_behaviors_site.py --port {args.port}", flush=True)
    print("=" * 76, flush=True)
    print("  独立本地网站服务已启动并持续运行...", flush=True)

    server = create_group_behaviors_server(
        site_dir=site_dir,
        evaluation_file=eval_file,
        port=args.port,
        host=args.host,
    )

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止独立本地网站服务器...")
    finally:
        server.server_close()
        print("独立本地网站服务器已安全关闭。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
