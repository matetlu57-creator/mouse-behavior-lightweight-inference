from __future__ import annotations

from . import regression_performance_test as suite


def test_result_preserving_optimizations() -> None:
    suite.main()
