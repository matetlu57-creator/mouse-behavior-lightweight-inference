"""Public logging facade."""

from __future__ import annotations

import logging

from ..logging_config import configure_logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without changing global handlers."""

    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
