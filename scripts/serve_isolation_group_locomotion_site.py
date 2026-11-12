"""CLI script to serve the Isolation and Group Locomotion comparison verification site.

Usage:
    python scripts/serve_isolation_group_locomotion_site.py --port 8769
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.comparison_site import (
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ISOLATION_DIR,
    DEFAULT_ISO_GLOCO_EVALUATION_JSON,
    DEFAULT_ORIG_ANNOTATED_DIR,
    create_comparison_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Isolation and Group Locomotion behavior cropping comparison website.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (loopback only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8769,
        help="Port number to listen on",
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=DEFAULT_ISO_GLOCO_EVALUATION_JSON,
        help="Path to visualized evaluation JSON report",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing original 605-segment dataset behaviors",
    )
    parser.add_argument(
        "--isolation-dir",
        type=Path,
        default=DEFAULT_ISOLATION_DIR,
        help="Directory containing original Isolation clips",
    )
    parser.add_argument(
        "--group-locomotion-dir",
        type=Path,
        default=DEFAULT_GROUP_LOCOMOTION_DIR,
        help="Directory containing original Group locomotion clips",
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="Directory containing generated visualized cropped video clips",
    )
    parser.add_argument(
        "--orig-annotated-dir",
        type=Path,
        default=DEFAULT_ORIG_ANNOTATED_DIR,
        help="Directory containing rendered full-length annotated original video clips",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open default browser to the comparison site",
    )
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    url = f"http://{args.host}:{args.port}/"

    print("=" * 76, flush=True)
    print("  孤立 (Isolation) 与 群体同步运动 (Group Locomotion) 精准裁剪对比系统", flush=True)
    print("=" * 76, flush=True)
    print(f"  ● 访问地址: {url}", flush=True)
    print(f"  ● 评测数据: {args.evaluation_file} (存在: {args.evaluation_file.is_file()})", flush=True)
    print(f"  ● 裁剪视频: {args.crops_dir} (存在: {args.crops_dir.is_dir()})", flush=True)
    print(f"  ● 原画标注: {args.orig_annotated_dir} (存在: {args.orig_annotated_dir.is_dir()})", flush=True)
    print(f"  ● 孤立原片: {args.isolation_dir} (存在: {args.isolation_dir.is_dir()})", flush=True)
    print(f"  ● 同步原片: {args.group_locomotion_dir} (存在: {args.group_locomotion_dir.is_dir()})", flush=True)
    print("=" * 76, flush=True)
    print("  单条可复现启动命令:", flush=True)
    print(f"    python scripts/serve_isolation_group_locomotion_site.py --port {args.port}", flush=True)
    print("=" * 76, flush=True)
    print("  对比服务已启动并持续运行...", flush=True)

    server = create_comparison_server(
        evaluation_file=args.evaluation_file,
        dataset_dir=args.dataset_dir,
        isolation_dir=args.isolation_dir,
        group_locomotion_dir=args.group_locomotion_dir,
        crops_dir=args.crops_dir,
        orig_annotated_dir=args.orig_annotated_dir,
        port=args.port,
        host=args.host,
    )

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止对比服务器...")
    finally:
        server.server_close()
        print("对比服务器已安全关闭。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
