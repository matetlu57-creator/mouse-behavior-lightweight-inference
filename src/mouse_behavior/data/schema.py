"""Small, dependency-light contracts for generated event files."""

from __future__ import annotations

from collections.abc import Iterable

BEHAVIOR_EVENTS_FILENAME = "lightweight_behavior_events.csv"
CONTACT_EVENTS_FILENAME = "lightweight_contact_events.csv"
REQUIRED_EVENT_COLUMNS = (
    "behavior",
    "candidate_level",
    "event_scope",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
)


def validate_event_columns(columns: Iterable[str]) -> list[str]:
    """Return missing required columns; an empty list means the schema passes."""

    available = set(columns)
    return [column for column in REQUIRED_EVENT_COLUMNS if column not in available]
