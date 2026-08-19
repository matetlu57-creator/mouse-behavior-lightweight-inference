"""Canonical output-directory layout for one analysis run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunDirectories:
    """Paths that keep tracking, behavior, visualization and reports separate."""

    root: Path

    @property
    def tracking(self) -> Path:
        return self.root / "tracking"

    @property
    def behavior(self) -> Path:
        return self.root / "behavior"

    @property
    def visualization(self) -> Path:
        return self.root / "visualization"

    @property
    def report(self) -> Path:
        return self.root / "report"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> "RunDirectories":
        for path in (self.root, self.tracking, self.behavior, self.visualization, self.report, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self
