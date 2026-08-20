from __future__ import annotations

import logging

import pytest

from mouse_behavior.utils import timer as timer_module
from mouse_behavior.utils.timer import Timer


def test_timer_records_named_stage_in_sink(monkeypatch: pytest.MonkeyPatch):
    readings = iter([10.0, 12.75])
    monkeypatch.setattr(timer_module.time, "perf_counter", lambda: next(readings))
    timings: dict[str, float] = {}

    with Timer("pair_metrics", logger=logging.getLogger("test"), sink=timings) as timer:
        pass

    assert timer.elapsed_s == pytest.approx(2.75)
    assert timings == {"pair_metrics": pytest.approx(2.75)}


def test_timer_supports_explicit_start_and_stop(monkeypatch: pytest.MonkeyPatch):
    readings = iter([4.0, 4.125])
    monkeypatch.setattr(timer_module.time, "perf_counter", lambda: next(readings))
    timer = Timer("csv_output").start()

    assert timer.stop() == pytest.approx(0.125)
    with pytest.raises(RuntimeError, match="has not been started"):
        timer.stop()
