#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render all persisted Beiyi analysis cases with the review side panel."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_importable

ensure_importable()

from mouse_behavior.logging_config import configure_logging  # noqa: E402
from mouse_behavior.visualization.rendering import render_behavior_video  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def render_beiyi_cases(
    analysis_output: Path,
    render_output: Path,
    manifest: Path,
    *,
    expected_mice: int = 20,
    force: bool = False,
    font_path: Path | None = None,
) -> Path:
    """Copy fresh analysis files and render one MP4 for every manifest row."""

    analysis_output = analysis_output.resolve()
    render_output = render_output.resolve()
    manifest = manifest.resolve()
    if not analysis_output.is_dir():
        raise FileNotFoundError(f"分析输出目录不存在: {analysis_output}")
    if not manifest.is_file():
        raise FileNotFoundError(f"分析清单不存在: {manifest}")
    rows = pd.read_csv(manifest)
    if rows.empty:
        raise ValueError(f"分析清单为空: {manifest}")
    render_output.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows.to_dict("records"), start=1):
        case_name = Path(str(row.get("case_output", ""))).name
        if not case_name or case_name in {".", ".."}:
            raise ValueError(f"案例目录无效: {row.get('case_output')}")
        source_case = analysis_output / case_name
        destination_case = render_output / case_name
        if not _inside(destination_case, render_output):
            raise ValueError(f"拒绝访问输出根目录之外的案例路径: {destination_case}")
        source_analysis = source_case / "analysis"
        if not source_analysis.is_dir():
            raise FileNotFoundError(f"新分析目录不存在: {source_analysis}")
        video = Path(str(row.get("video", ""))).resolve()
        cache_dir = source_case / "yolo_precompute"
        if not video.is_file():
            raise FileNotFoundError(f"源视频不存在: {video}")
        if not (cache_dir / "yolo_results_status.json").is_file():
            raise FileNotFoundError(f"YOLO 缓存状态不存在: {cache_dir}")

        if destination_case.exists():
            if not force:
                raise FileExistsError(
                    f"渲染案例已存在；如需覆盖请显式传 --force: {destination_case}"
                )
            shutil.rmtree(destination_case)
        destination_case.mkdir(parents=True, exist_ok=True)
        destination_analysis = destination_case / "analysis"
        shutil.copytree(source_analysis, destination_analysis)
        events_path = destination_analysis / "lightweight_behavior_events.csv"
        render_path = destination_analysis / "轻量行为推理_渲染.mp4"
        LOGGER.info("[%02d/%d] render %s", index, len(rows), case_name)
        render_behavior_video(
            video,
            cache_dir,
            events_path,
            render_path,
            expected_mice=max(int(expected_mice), 2),
            font_path=font_path,
        )

    for filename in (
        "beiyi_video_validation.csv",
        "beiyi_behavior_coverage.csv",
        "beiyi_validation_summary.json",
    ):
        shutil.copy2(analysis_output / filename, render_output / filename)
    LOGGER.info("render batch completed: %d cases -> %s", len(rows), render_output)
    return render_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--render-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-mice", type=int, default=20)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="删除并覆盖 render-output 下已有的同名案例目录。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    render_beiyi_cases(
        args.analysis_output,
        args.render_output,
        args.manifest,
        expected_mice=args.expected_mice,
        force=args.force,
        font_path=args.font_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
