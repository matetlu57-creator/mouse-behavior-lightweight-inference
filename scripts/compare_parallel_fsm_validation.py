#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two lightweight validation directories at event level.

The comparison intentionally ignores generated event IDs and elapsed time.
It checks the behavior/event content that can affect downstream annotation:
scope, roles, analysis/source frame boundaries, scores and contact geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_ID_COLUMNS = {"light_event_id", "contact_event_id"}


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else round(value, 10)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    columns = [column for column in frame.columns if column not in EVENT_ID_COLUMNS]
    records = [
        {column: _json_value(row[column]) for column in columns}
        for _, row in frame[columns].iterrows()
    ]
    return sorted(
        records,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True),
    )


def _validation_rows(path: Path) -> dict[str, dict[str, Any]]:
    csv_path = path / "beiyi_video_validation.csv"
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    return {
        str(row["relative_video"]): {
            "target_event_found": bool(row["target_event_found"]),
            "event_count": int(row["event_count"]),
            "contact_event_count": int(row["contact_event_count"]),
            "behavior_counts_json": str(row["behavior_counts_json"]),
            "contact_counts_json": str(row["contact_counts_json"]),
            "analysis_elapsed_s": float(row["analysis_elapsed_s"]),
            "case_output": str(row["case_output"]),
        }
        for _, row in frame.iterrows()
    }


def compare(old_dir: Path, new_dir: Path) -> dict[str, Any]:
    old_rows = _validation_rows(old_dir)
    new_rows = _validation_rows(new_dir)
    videos = sorted(set(old_rows) | set(new_rows))
    cases: list[dict[str, Any]] = []
    event_exact_count = 0
    contact_exact_count = 0
    summary_exact_count = 0

    for relative_video in videos:
        old_case = old_rows.get(relative_video, {})
        new_case = new_rows.get(relative_video, {})
        old_output = Path(str(old_case.get("case_output", "")))
        new_output = Path(str(new_case.get("case_output", "")))
        old_analysis = old_output / "analysis"
        new_analysis = new_output / "analysis"
        old_events = _records(old_analysis / "lightweight_behavior_events.csv")
        new_events = _records(new_analysis / "lightweight_behavior_events.csv")
        old_contacts = _records(old_analysis / "lightweight_contact_events.csv")
        new_contacts = _records(new_analysis / "lightweight_contact_events.csv")
        event_exact = old_events == new_events
        contact_exact = old_contacts == new_contacts
        summary_exact = (
            old_case.get("target_event_found") == new_case.get("target_event_found")
            and old_case.get("event_count") == new_case.get("event_count")
            and old_case.get("contact_event_count") == new_case.get("contact_event_count")
            and old_case.get("behavior_counts_json") == new_case.get("behavior_counts_json")
            and old_case.get("contact_counts_json") == new_case.get("contact_counts_json")
        )
        event_exact_count += int(event_exact)
        contact_exact_count += int(contact_exact)
        summary_exact_count += int(summary_exact)
        cases.append(
            {
                "relative_video": relative_video,
                "event_exact": event_exact,
                "contact_exact": contact_exact,
                "summary_exact": summary_exact,
                "old_event_count": len(old_events),
                "new_event_count": len(new_events),
                "old_contact_event_count": len(old_contacts),
                "new_contact_event_count": len(new_contacts),
                "old_target_event_found": old_case.get("target_event_found"),
                "new_target_event_found": new_case.get("target_event_found"),
                "old_elapsed_s": old_case.get("analysis_elapsed_s"),
                "new_elapsed_s": new_case.get("analysis_elapsed_s"),
                "elapsed_delta_s": (
                    float(new_case.get("analysis_elapsed_s", 0.0))
                    - float(old_case.get("analysis_elapsed_s", 0.0))
                ),
            }
        )

    old_elapsed = np.asarray(
        [float(row.get("analysis_elapsed_s", 0.0)) for row in old_rows.values()],
        dtype=float,
    )
    new_elapsed = np.asarray(
        [float(row.get("analysis_elapsed_s", 0.0)) for row in new_rows.values()],
        dtype=float,
    )
    return {
        "old_validation_dir": str(old_dir),
        "new_validation_dir": str(new_dir),
        "video_count": len(videos),
        "event_exact_video_count": event_exact_count,
        "contact_exact_video_count": contact_exact_count,
        "summary_exact_video_count": summary_exact_count,
        "all_event_records_exact": event_exact_count == len(videos),
        "all_contact_records_exact": contact_exact_count == len(videos),
        "all_video_summaries_exact": summary_exact_count == len(videos),
        "old_target_video_count": int(
            sum(bool(row.get("target_event_found")) for row in old_rows.values())
        ),
        "new_target_video_count": int(
            sum(bool(row.get("target_event_found")) for row in new_rows.values())
        ),
        "old_elapsed_total_s": float(old_elapsed.sum()) if old_elapsed.size else 0.0,
        "new_elapsed_total_s": float(new_elapsed.sum()) if new_elapsed.size else 0.0,
        "old_elapsed_mean_s": float(old_elapsed.mean()) if old_elapsed.size else 0.0,
        "new_elapsed_mean_s": float(new_elapsed.mean()) if new_elapsed.size else 0.0,
        "elapsed_delta_total_s": float(new_elapsed.sum() - old_elapsed.sum()),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.old.resolve(), args.new.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report["cases"]).to_csv(
        args.output.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
