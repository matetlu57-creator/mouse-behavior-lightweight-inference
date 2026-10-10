#!/usr/bin/env python3
"""Create a full-ethogram session split without the old 50-sample filter."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _bootstrap import REPO_ROOT
from mouse_behavior.evaluation.exported_dataset import (
    RETAINED_BEHAVIORS,
    discover_samples,
    split_by_recording_session,
    write_split_manifest,
)


VALIDATION_SESSIONS = (
    "large_arena_2.mp4",
    "WIN_20260821_14_20_49_Pro.mp4",
    "WIN_20260822_18_03_51_Pro.mp4",
    "WIN_20260822_18_53_17_Pro.mp4",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "hierarchical_heuristics_v1" / "dataset_split",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing split: {args.output_dir}")

    records = discover_samples(args.dataset_root)
    labels = {record.behavior for record in records}
    if labels != set(RETAINED_BEHAVIORS):
        raise ValueError(
            "the dataset must contain exactly the configured 16 behavior labels; "
            f"missing={sorted(set(RETAINED_BEHAVIORS) - labels)}, "
            f"unexpected={sorted(labels - set(RETAINED_BEHAVIORS))}"
        )
    train, validation = split_by_recording_session(
        records, validation_sessions=VALIDATION_SESSIONS
    )
    counts = Counter(record.behavior for record in records)
    manifest = write_split_manifest(
        args.output_dir,
        all_records=records,
        train_records=train,
        validation_records=validation,
        counts=counts,
        excluded_counts={},
        minimum_count_inclusive=None,
    )
    print(
        f"created {args.output_dir.resolve()} | all={len(records)} "
        f"train={len(train)} validation={len(validation)}"
    )
    for behavior in RETAINED_BEHAVIORS:
        print(
            f"{behavior}: all={counts[behavior]} "
            f"train={manifest['counts']['train_by_behavior'][behavior]} "
            f"validation={manifest['counts']['validation_by_behavior'][behavior]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
