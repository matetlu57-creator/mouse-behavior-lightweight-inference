"""Parallel and hierarchical finite-state machines for the ethogram.

The lightweight pipeline has several behavior scopes that are allowed to be
true at the same time:

* an individual mouse can be running while its pair is in a social state;
* a pair can be chasing while a nose-head contact is present;
* a local huddle can coexist with individual stationary states.

Consequently, the ethogram is represented as orthogonal FSM regions instead
of one mutually-exclusive 12-label state.  The detector-specific thresholds
remain in :mod:`lightweight_behavior_inference`; this module only converts
the resulting evidence masks/categories into auditable temporal states.

The boolean region intentionally preserves the legacy lightweight event
semantics exactly: internal false gaps up to ``max_gap_frames`` are filled,
and a run is emitted only when it reaches ``min_duration_frames``.  The
difference is that the temporal operation is now explicit and reusable as a
finite-state machine, with ``CANDIDATE``, ``ACTIVE``, ``RECOVERY`` and
``IDLE`` states.  This makes the migration behavior-preserving while giving
future rules a place to add state-specific hysteresis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Mapping

import numpy as np


BOOLEAN_STATES = ("IDLE", "CANDIDATE", "ACTIVE", "RECOVERY")


@dataclass(frozen=True)
class BooleanFSMSpan:
    """One finalized boolean FSM event span in analysis-frame coordinates."""

    start: int
    end: int


@dataclass(frozen=True)
class BooleanFSMResult:
    """Result of one independent boolean FSM region."""

    active_mask: np.ndarray
    state: np.ndarray
    effective_mask: np.ndarray
    spans: tuple[BooleanFSMSpan, ...]


@dataclass(frozen=True)
class CategoricalFSMSpan:
    """One run of the same categorical state."""

    start: int
    end: int
    key: Hashable


@dataclass(frozen=True)
class CategoricalFSMResult:
    """Result of one categorical FSM region."""

    state: tuple[Any, ...]
    spans: tuple[CategoricalFSMSpan, ...]


def _fill_internal_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill only internal false runs whose length is within the gap budget."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    effective = values.copy()
    gap = max(int(max_gap_frames), 0)
    if gap == 0 or not effective.size:
        return effective

    size = len(effective)
    false_starts, false_ends = _span_bounds(~effective)
    internal = (
        (false_starts > 0)
        & (false_ends < size - 1)
        & ((false_ends - false_starts + 1) <= gap)
    )
    for start, end in zip(false_starts[internal], false_ends[internal]):
        effective[int(start) : int(end) + 1] = True
    return effective


def _span_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized contiguous-true bounds used by all FSM regions."""

    values = np.asarray(values, dtype=bool).reshape(-1)
    if values.size == 0:
        empty = np.asarray([], dtype=int)
        return empty, empty
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    return starts, ends


def _true_spans(values: np.ndarray) -> list[BooleanFSMSpan]:
    """Return contiguous true spans without relying on pandas or scipy."""

    starts, ends = _span_bounds(values)
    return [
        BooleanFSMSpan(start=int(start), end=int(end))
        for start, end in zip(starts, ends)
    ]


class BooleanBehaviorFSM:
    """A behavior channel with candidate, active and recovery substates.

    ``effective_mask`` is the observation timeline after the configured
    recovery-gap policy.  The emitted spans and ``active_mask`` are based on
    that timeline, which is the compatibility contract with the previous
    mask-run implementation.  ``state`` still preserves whether a frame was
    a raw positive observation or an accepted short recovery gap.
    """

    def __init__(self, *, min_duration_frames: int, max_gap_frames: int) -> None:
        self.min_duration_frames = max(int(min_duration_frames), 1)
        self.max_gap_frames = max(int(max_gap_frames), 0)

    def run(
        self,
        mask: Iterable[bool],
        *,
        collect_diagnostics: bool = True,
    ) -> BooleanFSMResult:
        raw = np.asarray(list(mask) if not isinstance(mask, np.ndarray) else mask, dtype=bool).reshape(-1)
        effective = _fill_internal_gaps(raw, self.max_gap_frames)
        spans = tuple(
            span
            for span in _true_spans(effective)
            if span.end - span.start + 1 >= self.min_duration_frames
        )

        # Production event extraction only needs finalized spans.  Keeping the
        # optional per-frame arrays available for diagnostics makes the FSM
        # inspectable without paying that allocation/copy cost on every pair.
        if not collect_diagnostics:
            return BooleanFSMResult(
                active_mask=np.empty(0, dtype=bool),
                state=np.empty(0, dtype=object),
                effective_mask=np.empty(0, dtype=bool),
                spans=spans,
            )

        state = np.full(raw.shape, "IDLE", dtype=object)
        active_mask = np.zeros(raw.shape, dtype=bool)

        # First walk every observed frame as a finite-state channel.  A run
        # is promoted from CANDIDATE to ACTIVE only after confirmation.
        for span in _true_spans(effective):
            length = span.end - span.start + 1
            if length < self.min_duration_frames:
                state[span.start : span.end + 1] = np.where(
                    raw[span.start : span.end + 1], "CANDIDATE", "IDLE"
                )
                continue

            active_mask[span.start : span.end + 1] = True
            state[span.start : span.end + 1] = "ACTIVE"
            # An accepted false gap is an explicit recovery state.  It remains
            # active in the compatibility mask but is distinguishable for
            # diagnostics and later state-specific rules.
            recovery = ~raw[span.start : span.end + 1]
            if np.any(recovery):
                state[span.start : span.end + 1][recovery] = "RECOVERY"

        return BooleanFSMResult(
            active_mask=active_mask,
            state=state,
            effective_mask=effective,
            spans=spans,
        )


class CategoricalBehaviorFSM:
    """A categorical region that remains in one state until it changes."""

    def run(
        self,
        states: Iterable[Any],
        *,
        state_key: Callable[[Any], Hashable | None] | None = None,
    ) -> CategoricalFSMResult:
        values = tuple(states)
        key_function = state_key or (lambda value: value)
        spans: list[CategoricalFSMSpan] = []
        index = 0
        while index < len(values):
            key = key_function(values[index])
            if key is None:
                index += 1
                continue
            start = index
            while index + 1 < len(values) and key_function(values[index + 1]) == key:
                index += 1
            spans.append(CategoricalFSMSpan(start=start, end=index, key=key))
            index += 1
        return CategoricalFSMResult(state=values, spans=tuple(spans))


class ParallelBehaviorFSM:
    """Coordinator for independent individual, pair and group FSM regions.

    The coordinator is deliberately lightweight: each region owns its own
    state machine, while this object records the region identity for audit
    and future hierarchy-aware diagnostics.  No region overwrites another
    region's state.
    """

    VERSION = "parallel_hierarchical_fsm_v1"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        configured = dict(config or {})
        self.enabled = bool(configured.get("enabled", True))
        self.mode = str(configured.get("mode", "active")).strip().lower()
        if self.mode != "active":
            raise ValueError("parallel_behavior_fsm.mode currently supports only 'active'")
        self.collect_diagnostics = bool(configured.get("collect_diagnostics", False))
        self.regions: dict[str, BooleanFSMResult | CategoricalFSMResult] = {}

    def run_boolean_region(
        self,
        *,
        scope: str,
        region_id: str,
        behavior: str,
        mask: Iterable[bool],
        min_duration_frames: int,
        max_gap_frames: int,
    ) -> BooleanFSMResult:
        """Run one orthogonal boolean channel and retain its diagnostic state."""

        key = f"{scope}:{region_id}:{behavior}"
        if self.enabled:
            result = BooleanBehaviorFSM(
                min_duration_frames=min_duration_frames,
                max_gap_frames=max_gap_frames,
            ).run(mask, collect_diagnostics=self.collect_diagnostics)
        else:
            raw = np.asarray(
                list(mask) if not isinstance(mask, np.ndarray) else mask,
                dtype=bool,
            ).reshape(-1)
            result = BooleanFSMResult(
                active_mask=(
                    np.zeros(raw.shape, dtype=bool)
                    if self.collect_diagnostics
                    else np.empty(0, dtype=bool)
                ),
                state=(
                    np.full(raw.shape, "IDLE", dtype=object)
                    if self.collect_diagnostics
                    else np.empty(0, dtype=object)
                ),
                effective_mask=(
                    np.zeros(raw.shape, dtype=bool)
                    if self.collect_diagnostics
                    else np.empty(0, dtype=bool)
                ),
                spans=(),
            )
        if self.collect_diagnostics:
            self.regions[key] = result
        return result

    def run_categorical_region(
        self,
        *,
        scope: str,
        region_id: str,
        states: Iterable[Any],
        state_key: Callable[[Any], Hashable | None] | None = None,
    ) -> CategoricalFSMResult:
        """Run one categorical contact/interaction region."""

        key = f"{scope}:{region_id}:categorical"
        if self.enabled:
            result = CategoricalBehaviorFSM().run(states, state_key=state_key)
        else:
            result = CategoricalFSMResult(state=tuple(states), spans=())
        if self.collect_diagnostics:
            self.regions[key] = result
        return result

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable coordinator summary."""

        boolean_regions = sum(isinstance(value, BooleanFSMResult) for value in self.regions.values())
        categorical_regions = sum(
            isinstance(value, CategoricalFSMResult) for value in self.regions.values()
        )
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "version": self.VERSION,
            "boolean_region_count": int(boolean_regions),
            "categorical_region_count": int(categorical_regions),
        }


__all__ = [
    "BOOLEAN_STATES",
    "BooleanBehaviorFSM",
    "BooleanFSMResult",
    "BooleanFSMSpan",
    "CategoricalBehaviorFSM",
    "CategoricalFSMResult",
    "CategoricalFSMSpan",
    "ParallelBehaviorFSM",
]
