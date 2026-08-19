"""Shared logging setup for library modules and command-line scripts."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _level_number(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"unknown logging level: {level!r}")
    return value


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure one consistent process-wide logging policy.

    ``force=False`` preserves handlers installed by applications, notebooks,
    and pytest.  A normal command-line process with no handlers receives one
    timestamped stream handler.  Callers can opt into ``force=True`` when a
    standalone process must replace an inherited logging configuration.
    """

    numeric_level = _level_number(level)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    if force:
        logging.basicConfig(
            level=numeric_level,
            format=DEFAULT_LOG_FORMAT,
            stream=stream or sys.stderr,
            force=True,
        )
        return
    if root.handlers:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a reusable module."""

    return logging.getLogger(name)
