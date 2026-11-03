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
    "endpoint_observations",
    "endpoints_fully_observed",
    "extract_standard_behavior_events",
    "is_process_boundary_closed",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_behavior_segments",
    "refine_dispersal_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_rearing_segments",
    "select_behavior_segments",
    "update_segment_bounds",
]

_SEGMENT_CLIPPING_EXPORTS = {
    "endpoint_observations",
    "endpoints_fully_observed",
    "is_process_boundary_closed",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_behavior_segments",
    "refine_dispersal_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_rearing_segments",
    "select_behavior_segments",
    "update_segment_bounds",
}


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
    if name in _SEGMENT_CLIPPING_EXPORTS:
        from . import segment_clipping

        return getattr(segment_clipping, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
