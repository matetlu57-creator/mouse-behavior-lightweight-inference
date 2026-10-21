"""Clip-local half-open frame support shared by events and evaluators."""

from __future__ import annotations

from numbers import Integral
from typing import Iterable
import numpy as np

FrameIntervals = tuple[tuple[int, int], ...]


def validate_intervals(intervals: FrameIntervals, frame_count: int | None = None) -> None:
    previous = 0
    for pair in intervals:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("frame interval must be an immutable (start, stop) pair")
        start, stop = pair
        if (
            any(isinstance(v, bool) or not isinstance(v, Integral) for v in pair)
            or start < previous
            or start < 0
            or stop <= start
            or (frame_count is not None and stop > frame_count)
        ):
            raise ValueError(
                "frame intervals must be ordered, disjoint, positive and inside the clip"
            )
        previous = stop


def mask_intervals(mask: Iterable[bool]) -> FrameIntervals:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("event frame mask must be one-dimensional")
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        )
    )


def interval_frame_count(intervals: FrameIntervals, frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count <= 0:
        raise ValueError("clip frame count must be a positive integer")
    validate_intervals(intervals, frame_count)
    return sum(stop - start for start, stop in intervals)
