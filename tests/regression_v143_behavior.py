#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.42.1 -> v1.43 A/B comparator.

v1.43 intentionally changes behavior decisions, so baseline behavior equality is
reported rather than required by default.  Tracking/identity is still expected
to remain unchanged.  The script also validates the new standard ethogram and
that legacy CSV schemas are preserved as subsets of v1.43 outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

TRACK_MAP = "检测轨迹对应表.csv"
IDENTITY_DEBUG = "身份分配调试记录.csv"
EVENT_FILES = ("行为事件_弱候选.csv", "行为事件_强候选.csv")
ETHOGRAM = "标准行为事件_时序引擎.csv"
SCHEMA_FILES = (
    "逐帧检测流缓存.csv", TRACK_MAP, IDENTITY_DEBUG,
    "逐帧行为汇总.csv", *EVENT_FILES, "视频四分类结果.csv",
)


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def formal_track_rows(df: pd.DataFrame) -> pd.DataFrame:
    lid = pd.to_numeric(df.get("logical_id"), errors="coerce")
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
        for col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna().sort_values(["raw_track_id", "frame"])
        for _, group in d.groupby("raw_track_id"):
            values = group["logical_id"].astype(int).to_numpy()
            switches += int(np.count_nonzero(values[1:] != values[:-1])) if len(values) > 1 else 0
    return {
        "formal_id_count": int(track["logical_id"].nunique()),
        "trajectory_lengths": {int(k): int(v) for k, v in lengths.items()},
        "raw_to_logical_switch_count": int(switches),
    }


def trajectory_agreement(a: dict[int, int], b: dict[int, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(int(a.get(k, 0)), int(b.get(k, 0))) for k in keys)
    den = sum(max(int(a.get(k, 0)), int(b.get(k, 0))) for k in keys)
    return float(num / den) if den else 1.0


def event_signature(df: pd.DataFrame) -> list[tuple[int, int, int, int, int]]:
    cols = ["label_id", "actor_id", "target_id", "start_frame", "end_frame"]
    if not set(cols).issubset(df.columns):
        return []
    work = df[cols].copy()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(-1).astype(int)
    return sorted(tuple(map(int, row)) for row in work.to_numpy())


def jaccard_multiset(a, b) -> float:
    if not a and not b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return float(sum((ca & cb).values()) / union) if union else 1.0


def schema_additive_failures(baseline: Path, v143: Path) -> list[str]:
    failures: list[str] = []
    for name in SCHEMA_FILES:
        left, right = baseline / name, v143 / name
        if not left.exists() or not right.exists():
            failures.append(f"missing output: {name}")
            continue
        old_cols = list(pd.read_csv(left, encoding="utf-8-sig", nrows=0).columns)
        new_cols = list(pd.read_csv(right, encoding="utf-8-sig", nrows=0).columns)
        missing = [col for col in old_cols if col not in new_cols]
        if missing:
            failures.append(f"removed legacy fields in {name}: {missing}")
    return failures


def validate_ethogram(root: Path) -> list[str]:
    path = root / ETHOGRAM
    if not path.exists():
        return [f"missing {ETHOGRAM}"]
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "standard_event_id", "behavior_engine", "candidate_level", "behavior",
        "pair_key", "actor_id", "target_id", "role_ambiguous",
        "start_frame", "peak_frame", "end_frame", "duration_s",
        "mean_score", "peak_score", "mean_behavior_confidence",
        "mean_role_confidence",
    }
    missing = sorted(required - set(df.columns))
    failures = [f"ethogram missing fields: {missing}"] if missing else []
    if not df.empty:
        valid_behavior = df["behavior"].astype(str).isin(["chase", "attack"]).all()
        if not valid_behavior:
            failures.append("ethogram contains behavior outside chase/attack")
        start = pd.to_numeric(df["start_frame"], errors="coerce")
        peak = pd.to_numeric(df["peak_frame"], errors="coerce")
        end = pd.to_numeric(df["end_frame"], errors="coerce")
        if not ((start <= peak) & (peak <= end)).all():
            failures.append("ethogram violates start<=peak<=end")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_v1421", type=Path)
    parser.add_argument("v143", type=Path)
    parser.add_argument("--tracking-min", type=float, default=0.99)
    parser.add_argument("--require-legacy-behavior-agreement", action="store_true")
    parser.add_argument("--legacy-behavior-min", type=float, default=0.99)
    args = parser.parse_args()
    baseline, v143 = args.baseline_v1421.resolve(), args.v143.resolve()

    bt, nt = track_metrics(baseline), track_metrics(v143)
    traj = trajectory_agreement(bt["trajectory_lengths"], nt["trajectory_lengths"])
    id_equal = bt["formal_id_count"] == nt["formal_id_count"]
    switch_equal = bt["raw_to_logical_switch_count"] == nt["raw_to_logical_switch_count"]

    behavior_scores = {}
    for name in EVENT_FILES:
        behavior_scores[name] = jaccard_multiset(
            event_signature(read_csv(baseline, name)),
            event_signature(read_csv(v143, name)),
        )
    behavior_min = min(behavior_scores.values()) if behavior_scores else 1.0
    failures = schema_additive_failures(baseline, v143) + validate_ethogram(v143)

    print("TRACKING (must remain equivalent)")
    print(f"  formal_id_count: {bt['formal_id_count']} -> {nt['formal_id_count']} | equal={id_equal}")
    print(f"  raw_to_logical_switch_count: {bt['raw_to_logical_switch_count']} -> {nt['raw_to_logical_switch_count']} | equal={switch_equal}")
    print(f"  trajectory_agreement: {traj:.6f}")
    print("BEHAVIOR (expected to change; report unless explicitly gated)")
    for name, score in behavior_scores.items():
        print(f"  {name}: legacy_event_signature_agreement={score:.6f}")
    print(f"  minimum_legacy_behavior_agreement: {behavior_min:.6f}")
    print("V1.43 STRUCTURE")
    print("  PASS" if not failures else "  " + "\n  ".join(failures))

    passed = id_equal and switch_equal and traj >= args.tracking_min and not failures
    if args.require_legacy_behavior_agreement:
        passed = passed and behavior_min >= args.legacy_behavior_min
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
