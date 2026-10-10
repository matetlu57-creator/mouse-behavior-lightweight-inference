"""Read the normalized identity output of an external mTrack run.

The upstream mTrack project owns RFID acquisition and RFID--vision matching.
This package deliberately does not vendor or reimplement that algorithm.  It
consumes a small, explicit interchange CSV so behavior analysis can remain
independent of RFID hardware and of mTrack's internal files.

Accepted CSV forms are either one exact frame per row::

    frame,visual_track_id,animal_id,identity_confidence

or one inclusive frame interval per row::

    start_frame,end_frame,visual_track_id,animal_id,identity_confidence

``track_id``/``cv_track_id`` and ``tag_id``/``rfid_id``/``epc`` are accepted
as column aliases.  Conflicting overlapping identities are rejected instead
of being resolved heuristically.
"""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_VISUAL_ID_COLUMNS = ("visual_track_id", "cv_track_id", "track_id")
_ANIMAL_ID_COLUMNS = ("animal_id", "tag_id", "rfid_id", "epc")
_CONFIDENCE_COLUMNS = ("identity_confidence", "confidence", "match_confidence")


def _first_value(row: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _integer(value: str | None, *, field: str, row_number: int) -> int:
    if value is None:
        raise ValueError(f"identity CSV row {row_number} is missing {field}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"identity CSV row {row_number} has invalid {field}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"identity CSV row {row_number} has negative {field}: {parsed}")
    return parsed


def _confidence(row: Mapping[str, str], row_number: int) -> float:
    raw = _first_value(row, _CONFIDENCE_COLUMNS)
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"identity CSV row {row_number} has invalid identity confidence: {raw!r}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"identity CSV row {row_number} identity confidence must be in [0, 1]"
        )
    return value


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    """One authoritative RFID identity assignment for a visual tracklet."""

    start_frame: int
    end_frame: int
    visual_track_id: int
    animal_id: str
    confidence: float = 1.0
    source: str = "rfid_mtrack"

    def contains(self, frame: int) -> bool:
        return self.start_frame <= int(frame) <= self.end_frame


class IdentityMap:
    """Validated, frame-addressable mTrack identity assignments."""

    def __init__(
        self,
        assignments: Iterable[IdentityAssignment],
        *,
        source_path: Path | None = None,
    ) -> None:
        grouped: dict[int, list[IdentityAssignment]] = {}
        unique: set[tuple[int, int, int, str]] = set()
        for assignment in assignments:
            if assignment.end_frame < assignment.start_frame:
                raise ValueError("identity assignment end_frame precedes start_frame")
            if not assignment.animal_id.strip():
                raise ValueError("identity assignment animal_id must not be empty")
            key = (
                int(assignment.start_frame),
                int(assignment.end_frame),
                int(assignment.visual_track_id),
                str(assignment.animal_id),
            )
            if key in unique:
                continue
            unique.add(key)
            grouped.setdefault(int(assignment.visual_track_id), []).append(assignment)

        for visual_id, rows in grouped.items():
            rows.sort(key=lambda item: (item.start_frame, item.end_frame, item.animal_id))
            previous: IdentityAssignment | None = None
            for current in rows:
                if (
                    previous is not None
                    and current.start_frame <= previous.end_frame
                    and current.animal_id != previous.animal_id
                ):
                    raise ValueError(
                        "conflicting identity interval overlap for visual track "
                        f"{visual_id}: {previous.animal_id!r} and {current.animal_id!r}"
                    )
                if previous is None or current.end_frame > previous.end_frame:
                    previous = current

        self._by_visual_id = {key: tuple(value) for key, value in grouped.items()}
        self._starts = {
            key: tuple(item.start_frame for item in value)
            for key, value in self._by_visual_id.items()
        }
        ordered_animals = sorted(
            {
                (item.start_frame, item.animal_id)
                for rows in self._by_visual_id.values()
                for item in rows
            },
            key=lambda value: (value[0], value[1]),
        )
        self.animal_ids = tuple(dict.fromkeys(animal for _frame, animal in ordered_animals))
        self.source_path = Path(source_path) if source_path is not None else None

    def __bool__(self) -> bool:
        return bool(self._by_visual_id)

    def resolve(self, frame: int, visual_track_id: int) -> IdentityAssignment | None:
        rows = self._by_visual_id.get(int(visual_track_id))
        if not rows:
            return None
        starts = self._starts[int(visual_track_id)]
        index = bisect_right(starts, int(frame)) - 1
        if index < 0:
            return None
        candidate = rows[index]
        if candidate.contains(frame):
            return candidate
        # Same-animal overlapping intervals are legal.  If bisect selected a
        # shorter interval, the preceding interval can still cover this frame.
        for previous in reversed(rows[:index]):
            if previous.contains(frame):
                return previous
        return None


def load_identity_map(path: str | Path) -> IdentityMap:
    """Load a normalized mTrack identity mapping CSV."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    assignments: list[IdentityAssignment] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"identity CSV has no header: {csv_path}")
        fields = {str(name).strip() for name in reader.fieldnames if name is not None}
        if not fields.intersection(_VISUAL_ID_COLUMNS):
            raise ValueError(
                "identity CSV requires visual_track_id (or track_id/cv_track_id)"
            )
        if not fields.intersection(_ANIMAL_ID_COLUMNS):
            raise ValueError("identity CSV requires animal_id (or tag_id/rfid_id/epc)")
        exact_frames = "frame" in fields
        if not exact_frames and not {"start_frame", "end_frame"}.issubset(fields):
            raise ValueError(
                "identity CSV requires frame or both start_frame and end_frame"
            )

        for row_number, raw_row in enumerate(reader, start=2):
            row = {str(key).strip(): str(value or "").strip() for key, value in raw_row.items()}
            if not any(row.values()):
                continue
            visual_id = _integer(
                _first_value(row, _VISUAL_ID_COLUMNS),
                field="visual_track_id",
                row_number=row_number,
            )
            if exact_frames:
                start_frame = end_frame = _integer(
                    row.get("frame"), field="frame", row_number=row_number
                )
            else:
                start_frame = _integer(
                    row.get("start_frame"), field="start_frame", row_number=row_number
                )
                end_frame = _integer(
                    row.get("end_frame"), field="end_frame", row_number=row_number
                )
            animal_id = _first_value(row, _ANIMAL_ID_COLUMNS)
            if animal_id is None:
                raise ValueError(f"identity CSV row {row_number} is missing animal_id")
            assignments.append(
                IdentityAssignment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    visual_track_id=visual_id,
                    animal_id=animal_id,
                    confidence=_confidence(row, row_number),
                )
            )

    if not assignments:
        raise ValueError(f"identity CSV contains no assignments: {csv_path}")
    return IdentityMap(assignments, source_path=csv_path)


__all__ = ["IdentityAssignment", "IdentityMap", "load_identity_map"]
