"""Join verified screening cores and quarantined contact windows for review."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from screen_behavior_segments import review_page, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    render = json.loads((root / "render_summary.json").read_text(encoding="utf-8"))
    contact_path = root / "contact_overlap_review/summary.json"
    contacts = json.loads(contact_path.read_text(encoding="utf-8")) if contact_path.exists() else {
        "errors": [], "verified_contact_review_crops": 0}
    if render["errors"] or contacts["errors"]:
        raise ValueError("finish failed video exports before joining the package")
    original = json.loads((root / "review_manifest.json").read_text(encoding="utf-8"))
    rows = copy.deepcopy(original)
    by_id = {r["record"]["sample_id"]: r for r in rows}
    contact_rows = json.loads((root / "contact_overlap_review/manifest.json").read_text(encoding="utf-8")) if contact_path.exists() else []
    core_behaviors = {w["behavior"] for r in original for w in r["selected_windows"]}
    for row in contact_rows:
        by_id[row["source_sample_id"]]["selected_windows"].append(row["candidate"])
    review_page(root, rows)
    write_json(root / "combined_review_manifest.json", rows)
    counts = Counter(w["behavior"] for r in rows for w in r["selected_windows"])
    status = Counter(w["status"] for r in rows for w in r["selected_windows"])
    summary = {"input_clips": len(rows), "verified_core_exports": render["verified_segments"],
        "verified_contact_quarantine_exports": contacts["verified_contact_review_crops"],
        "verified_total_exports": sum(counts.values()), "exports_by_behavior": dict(counts),
        "status_counts": dict(status),
        "source_clips_with_output": sum(bool(r["selected_windows"]) for r in rows),
        "source_clips_without_output": sum(not r["selected_windows"] for r in rows),
        "screening_errors": [], "render_errors": [],
        "zero_core_behaviors": [label for label in ("Chasing", "Snout-head_contact", "Snout-rear_contact")
                                if label not in core_behaviors],
        "quarantine_note": ("Contact geometry supported, attack gate overlapping: needs human review; not training truth"
                            if contact_rows else "No separate contact quarantine; contact source labels screened directly")}
    write_json(root / "package_summary.json", summary)
    fields = ["sample_id", "source_behavior", "candidate_behavior", "mouse_ids", "start_frame",
              "end_frame", "duration_s", "status", "reason", "export_path"]
    with (root / "combined_screening_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = row["record"]
            for window in row["selected_windows"] or [None]:
                writer.writerow({"sample_id": record["sample_id"], "source_behavior": record["behavior"],
                    "candidate_behavior": window["behavior"] if window else "", "mouse_ids": str(window["mouse_ids"]) if window else str(record["mouse_ids"]),
                    "start_frame": window["start_frame"] if window else "", "end_frame": window["end_frame"] if window else "",
                    "duration_s": (window["end_frame"]-window["start_frame"])/record["fps"] if window else "",
                    "status": window["status"] if window else "needs_review_or_suspended",
                    "reason": window["reason"] if window else row["reason"], "export_path": window.get("export_path", "") if window else ""})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
