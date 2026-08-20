#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.42.1 end-to-end regression comparator.

Run the same representative video once with the baseline build and once with
v1.42.1, then compare the two output directories.  This script checks the
scientific outputs rather than encoded MP4 bytes.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

TRACK_MAP = "检测轨迹对应表.csv"
IDENTITY_DEBUG = "身份分配调试记录.csv"
EVENT_FILES = ("行为事件_弱候选.csv", "行为事件_强候选.csv")
CSV_SCHEMA_FILES = (
    "逐帧检测流缓存.csv",
    TRACK_MAP,
    IDENTITY_DEBUG,
    "逐帧行为汇总.csv",
    *EVENT_FILES,
    "视频四分类结果.csv",
)


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def formal_track_rows(df: pd.DataFrame) -> pd.DataFrame:
    lid = pd.to_numeric(df.get("logical_id"), errors="coerce")
    # Cluster anonymous IDs start at 100000; Pxx display rows use negative IDs.
    return df[lid.notna() & (lid >= 0) & (lid < 100000)].copy()


def track_metrics(root: Path) -> dict[str, Any]:
    track = formal_track_rows(read_csv(root, TRACK_MAP))
    track["logical_id"] = pd.to_numeric(track["logical_id"], errors="coerce").astype(int)
    track["frame"] = pd.to_numeric(track["frame"], errors="coerce").astype(int)
    lengths = track.groupby("logical_id")["frame"].nunique().astype(int).to_dict()

    debug = read_csv(root, IDENTITY_DEBUG)
    switches = 0
    if {"raw_track_id", "logical_id", "frame"}.issubset(debug.columns):
        d = debug[["raw_track_id", "logical_id", "frame"]].copy()
        d["raw_track_id"] = pd.to_numeric(d["raw_track_id"], errors="coerce")
        d["logical_id"] = pd.to_numeric(d["logical_id"], errors="coerce")
        d["frame"] = pd.to_numeric(d["frame"], errors="coerce")
        d = d.dropna().sort_values(["raw_track_id", "frame"])
        for _, group in d.groupby("raw_track_id"):
            values = group["logical_id"].astype(int).to_numpy()
            if len(values) > 1:
                switches += int(np.count_nonzero(values[1:] != values[:-1]))
    return {
        "formal_id_count": int(track["logical_id"].nunique()),
        "trajectory_lengths": {int(k): int(v) for k, v in lengths.items()},
        "raw_to_logical_switch_count": int(switches),
    }


def trajectory_agreement(a: dict[int, int], b: dict[int, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(int(a.get(k, 0)), int(b.get(k, 0))) for k in keys)
    denominator = sum(max(int(a.get(k, 0)), int(b.get(k, 0))) for k in keys)
    return float(numerator / denominator) if denominator else 1.0


def event_signature(df: pd.DataFrame) -> list[tuple[int, int, int, int, int]]:
    needed = ["label_id", "actor_id", "target_id", "start_frame", "end_frame"]
    if not set(needed).issubset(df.columns):
        return []
    work = df[needed].copy()
    for column in needed:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(-1).astype(int)
    return sorted(tuple(map(int, row)) for row in work.to_numpy())


def event_agreement(
    a: list[tuple[int, int, int, int, int]], b: list[tuple[int, int, int, int, int]]
) -> float:
    if not a and not b:
        return 1.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return float(intersection / union) if union else 1.0


def json_key_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_key_tree(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [json_key_tree(v) for v in value[:1]]  # schema of list items only
    return type(value).__name__


def compare_schema(baseline: Path, optimized: Path) -> list[str]:
    failures: list[str] = []
    for name in CSV_SCHEMA_FILES:
        left, right = baseline / name, optimized / name
        if not left.exists() or not right.exists():
            failures.append(f"missing CSV schema file: {name}")
            continue
        lh = list(pd.read_csv(left, encoding="utf-8-sig", nrows=0).columns)
        rh = list(pd.read_csv(right, encoding="utf-8-sig", nrows=0).columns)
        if lh != rh:
            failures.append(f"CSV field/order mismatch: {name}")
    # Compare JSON structures for files present on both sides.
    for left in baseline.glob("*.json"):
        right = optimized / left.name
        if not right.exists():
            continue
        try:
            lobj = json.loads(left.read_text(encoding="utf-8"))
            robj = json.loads(right.read_text(encoding="utf-8"))
        except Exception:
            continue
        if json_key_tree(lobj) != json_key_tree(robj):
            failures.append(f"JSON structure mismatch: {left.name}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--tracking-min", type=float, default=0.99)
    parser.add_argument("--behavior-min", type=float, default=0.99)
    args = parser.parse_args()
    baseline, optimized = args.baseline.resolve(), args.optimized.resolve()

    bt, ot = track_metrics(baseline), track_metrics(optimized)
    traj_score = trajectory_agreement(bt["trajectory_lengths"], ot["trajectory_lengths"])
    id_count_equal = bt["formal_id_count"] == ot["formal_id_count"]
    switch_equal = bt["raw_to_logical_switch_count"] == ot["raw_to_logical_switch_count"]

    event_scores = {}
    for name in EVENT_FILES:
        event_scores[name] = event_agreement(
            event_signature(read_csv(baseline, name)),
            event_signature(read_csv(optimized, name)),
        )
    behavior_score = min(event_scores.values()) if event_scores else 1.0
    schema_failures = compare_schema(baseline, optimized)

    print("TRACKING")
    print(
        f"  formal_id_count: {bt['formal_id_count']} -> {ot['formal_id_count']} | equal={id_count_equal}"
    )
    print(
        f"  raw_to_logical_switch_count: {bt['raw_to_logical_switch_count']} -> {ot['raw_to_logical_switch_count']} | equal={switch_equal}"
    )
    print(f"  trajectory_agreement: {traj_score:.6f}")
    print("BEHAVIOR")
    for name, score in event_scores.items():
        print(f"  {name}: event_signature_agreement={score:.6f}")
    print(f"  minimum_behavior_agreement: {behavior_score:.6f}")
    print("SCHEMA")
    print("  PASS" if not schema_failures else "  " + "\n  ".join(schema_failures))

    passed = (
        id_count_equal
        and switch_equal
        and traj_score >= float(args.tracking_min)
        and behavior_score >= float(args.behavior_min)
        and not schema_failures
    )
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
