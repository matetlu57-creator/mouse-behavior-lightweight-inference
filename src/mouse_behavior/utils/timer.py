"""Small logging timer for pipeline stages."""
from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Measure a stage and log its duration when used as a context manager."""

    label: str
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    sink: MutableMapping[str, float] | None = None
    started: float | None = field(default=None, init=False)
    elapsed_s: float | None = field(default=None, init=False)

    def start(self) -> "Timer":
        """Start or restart the timer and return it for fluent use."""

        self.started = time.perf_counter()
        self.elapsed_s = None
        return self

    def stop(self) -> float:
        """Stop the timer, log the duration and optionally store it by label."""

        if self.started is None:
            raise RuntimeError(f"Timer {self.label!r} has not been started")
        self.elapsed_s = time.perf_counter() - self.started
        self.started = None
        if self.sink is not None:
            self.sink[self.label] = float(self.elapsed_s)
        self.logger.info("stage=%s elapsed_s=%.3f", self.label, self.elapsed_s)
        return float(self.elapsed_s)

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.started is not None:
            self.stop()
