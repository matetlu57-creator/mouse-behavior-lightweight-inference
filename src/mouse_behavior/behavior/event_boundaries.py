"""One source-frame contract for event qualification, annotation and playback.

All machine boundaries are half-open. The annotation's legacy inclusive end
is a display projection of this contract, never another source of boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class EventBoundary:
    start: int
    end: int
    crop_start: int
    crop_end: int
    fps: float

    def __post_init__(self):
        bounds = (self.start, self.end, self.crop_start, self.crop_end)
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in bounds):
            raise ValueError("event and playback bounds must be integer source frames")
        if not 0 <= self.crop_start <= self.start < self.end <= self.crop_end:
            raise ValueError("playback must contain the positive half-open event interval")
        if isinstance(self.fps, bool) or not isinstance(self.fps, Real) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("boundary fps must be finite and positive")

    @classmethod
    def for_event(cls, event, info, context_seconds=0.):
        if (isinstance(context_seconds, bool) or not isinstance(context_seconds, Real)
                or not math.isfinite(context_seconds) or context_seconds < 0):
            raise ValueError("clip context seconds must be finite and nonnegative")
        start, end, total = event["start_frame"], event["end_frame"], info["frame_count"]
        if isinstance(total, bool) or not isinstance(total, Integral):
            raise ValueError("event exceeds the source video frame range")
        # Validate before arithmetic, including fractional frames and invalid fps.
        cls(start, end, start, end, info["fps"])
        if end > total:
            raise ValueError("event exceeds the source video frame range")
        context = int(math.ceil(context_seconds * info["fps"] - 1e-9))
        canonical = event.get("canonical_behavior", event.get("behavior", "")).lower()
        after = 0 if canonical == "approach" else context
        return cls(start, end, max(0, start-context), min(total, end+after), info["fps"])

    def as_dict(self):
        before, after = self.start-self.crop_start, self.crop_end-self.end
        scope = ("pure_event" if not before and not after else
                 "event_with_leading_context" if not after else "event_with_context")
        return dict(boundary_schema_version="1.0", coordinate_system="source_video_half_open_frames",
            source_event_half_open=dict(start=int(self.start), end=int(self.end)),
            playback_crop_half_open=dict(start=int(self.crop_start), end=int(self.crop_end)),
            event_offset_in_playback_half_open=dict(start=int(before), end=int(self.end-self.crop_start)),
            event_frame_count=int(self.end-self.start), playback_frame_count=int(self.crop_end-self.crop_start),
            context_before_frames=int(before), context_after_frames=int(after), fps=float(self.fps),
            event_duration_s=(self.end-self.start)/self.fps,
            playback_duration_s=(self.crop_end-self.crop_start)/self.fps,
            playback_scope=scope, qualification_basis="source_event_half_open_only")
