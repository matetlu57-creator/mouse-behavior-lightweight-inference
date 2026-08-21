"""Behavior-analysis interfaces.

The public facade is loaded lazily so importing a focused behavior submodule
does not create a cycle through the standard-engine compatibility facade.
Attribute access remains backward compatible for existing callers.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ParallelBehaviorFSM",
    "apply_standard_behavior_engine",
    "extract_standard_behavior_events",
]


def __getattr__(name: str) -> Any:
    if name == "ParallelBehaviorFSM":
        from .engine import ParallelBehaviorFSM

        return ParallelBehaviorFSM
    if name in {"apply_standard_behavior_engine", "extract_standard_behavior_events"}:
        from .engine import (
            apply_standard_behavior_engine,
            extract_standard_behavior_events,
        )

        return {
            "apply_standard_behavior_engine": apply_standard_behavior_engine,
            "extract_standard_behavior_events": extract_standard_behavior_events,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
