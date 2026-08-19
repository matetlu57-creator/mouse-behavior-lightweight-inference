"""Small logging timer for pipeline stages."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Measure a stage and log its duration when used as a context manager."""

    label: str
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    started: float | None = field(default=None, init=False)
    elapsed_s: float | None = field(default=None, init=False)

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.started is not None:
            self.elapsed_s = time.perf_counter() - self.started
            self.logger.info("stage=%s elapsed_s=%.3f", self.label, self.elapsed_s)
