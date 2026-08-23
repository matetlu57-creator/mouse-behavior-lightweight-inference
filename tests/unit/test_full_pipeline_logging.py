from __future__ import annotations

import argparse
import logging

import pytest

from mouse_behavior.full_pipeline import high_recall


class _Parser:
    def parse_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            verbose=False,
            model="weights/best.pt",
            video="sample.mp4",
            output="outputs/sample",
            stage="all",
            config="missing.yaml",
            calibration=None,
            manual_annotations=None,
            behavior_from_cache=False,
        )


def _stop_after_startup(_value: object) -> None:
    raise RuntimeError("stop after startup logging")


def test_main_logs_startup_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(high_recall, "build_parser", _Parser)
    monkeypatch.setattr(high_recall.base, "setup_logging", lambda _verbose: None)
    monkeypatch.setattr(high_recall, "resolve_runtime_path", _stop_after_startup)
    caplog.set_level(logging.INFO)

    assert high_recall.main() == 1

    messages = [record.getMessage() for record in caplog.records]
    assert f"程序版本：{high_recall.PROGRAM_VERSION}" in messages
    assert "使用模型：weights/best.pt" in messages
    assert "输入视频：sample.mp4" in messages
    assert "输出目录：outputs/sample" in messages
    assert "运行阶段：all" in messages
