"""Attach stable animal labels to behavior records that use array-slot IDs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any


_SCALAR_ID_FIELDS = (
    "actor_id",
    "target_id",
    "contact_actor_id",
    "contact_target_id",
    "mouse_id",
    "id_a",
    "id_b",
)
_MEMBER_ID_FIELDS = ("members", "member_ids", "resolved_mouse_ids")


def identity_label(value: Any, labels: Sequence[str]) -> str | None:
    try:
        slot = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if slot < 0 or slot >= len(labels):
        return None
    label = str(labels[slot])
    return None if label.startswith("unassigned:") else label


def _member_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [part.strip() for part in text.split(",") if part.strip()]
        value = parsed
    if isinstance(value, Mapping):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def annotate_identity_labels(
    records: Iterable[MutableMapping[str, Any]],
    labels: Sequence[str],
) -> None:
    """Add label columns without changing numeric slot fields used internally."""

    normalized = tuple(str(label) for label in labels)
    for record in records:
        for field in _SCALAR_ID_FIELDS:
            if field not in record:
                continue
            record[f"{field}_label"] = identity_label(record.get(field), normalized) or ""
        for field in _MEMBER_ID_FIELDS:
            if field not in record:
                continue
            resolved = [
                label
                for value in _member_values(record.get(field))
                if (label := identity_label(value, normalized)) is not None
            ]
            record[f"{field}_labels"] = resolved


__all__ = ["annotate_identity_labels", "identity_label"]
