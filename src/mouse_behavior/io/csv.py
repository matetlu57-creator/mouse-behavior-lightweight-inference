"""CSV serialization helpers for analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write a UTF-8 CSV while preserving the schema for empty results."""
    if not rows:
        pd.DataFrame(columns=list(columns or [])).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        return
    frame = pd.DataFrame(rows)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame.loc[:, list(columns)]
    frame.to_csv(path, index=False, encoding="utf-8-sig")
