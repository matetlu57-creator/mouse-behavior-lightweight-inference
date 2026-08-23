from __future__ import annotations

import logging

import pytest

from mouse_behavior.logging_config import configure_logging


def test_configure_logging_preserves_pytest_handler_and_records_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("mouse_behavior.tests.analysis")
    caplog.set_level(logging.INFO)
    handlers_before = tuple(logging.getLogger().handlers)

    configure_logging("INFO")
    logger.info("analysis_started video=%s", "sample.mp4")

    assert tuple(logging.getLogger().handlers) == handlers_before
    record = next(record for record in caplog.records if record.name == logger.name)
    assert record.levelno == logging.INFO
    assert record.getMessage() == "analysis_started video=sample.mp4"


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown logging level"):
        configure_logging("NOT_A_LEVEL")
