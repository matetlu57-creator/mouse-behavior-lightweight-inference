#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline video-level calibration for the v1.43 standard behavior engine.

The extractor output is replayed pair-by-pair, so each pair gets a fresh FSM
state.  This makes it safe to compare thresholds without rerunning pose
inference.  The folder label is intentionally treated as a *video-level*
truth label.  Event-level Precision/Recall/F1 and actor/target accuracy need
frame boundaries and role annotations, which this input format does not carry.

Example::

    python scripts/calibrate_standard_behavior.py \
      --dataset chase=D:\\data\\threshold_calibration_chase \
      --dataset attack=D:\\data\\threshold_calibration_attack \
      --dataset none=D:\\data\\threshold_calibration_social6 \
      --output-dir D:\\reports\\threshold_calibration
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT
from mouse_behavior import standard_behavior_engine as engine
from mouse_behavior.config import load_config
from mouse_behavior.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


BEHAVIORS = ("chase", "attack")
LEVELS = ("weak", "strong")


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _truth_flags(label: str) -> dict[str, bool]:
    normalized = label.strip().lower()
    return {
        "chase": normalized in {"chase", "both"},
        "attack": normalized in {"attack", "both"},
    }


def _safe_metric(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _binary_metrics(truth: Iterable[bool], prediction: Iterable[bool]) -> dict[str, Any]:
    y_true = np.asarray(list(truth), dtype=bool)
    y_pred = np.asarray(list(prediction), dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = _safe_metric(tp, tp + fp)
    recall = _safe_metric(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else float(2.0 * precision * recall / (precision + recall))
    )
    return {
        "n_videos": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _pair_csvs(result_root: Path) -> list[Path]:
    return sorted(
        path / "成对行为标签.csv"
        for path in result_root.iterdir()
        if path.is_dir() and (path / "成对行为标签.csv").is_file()
    )


def _apply_per_pair(
    df: pd.DataFrame, config: dict[str, Any], fps: float
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if "pair_key" not in df.columns:
        df = df.copy()
        df["pair_key"] = "__single_pair__"
    frames: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    for pair_key, pair in df.groupby("pair_key", sort=False, dropna=False):
        pair = pair.sort_values("frame").reset_index(drop=True)
        enriched = engine.apply_standard_behavior_engine(pair, fps, config)
        frames.append(enriched)
        for level in LEVELS:
            events.extend(
                {
                    **event,
                    "level": level,
                }
                for event in engine.extract_standard_behavior_events(
                    enriched, fps, level, pair_key=str(pair_key)
                )
            )
    if not frames:
        return pd.DataFrame(), events
    return pd.concat(frames, ignore_index=True), events


def _video_result(
    pair_csv: Path,
    truth_label: str,
    config: dict[str, Any],
    fps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = pd.read_csv(pair_csv)
    enriched, events = _apply_per_pair(df, config, fps)
    row: dict[str, Any] = {
        "video": pair_csv.parent.name,
        "truth_label": truth_label,
        "pair_csv": str(pair_csv),
        "source_result_root": str(pair_csv.parent.parent),
        "input_rows": int(len(df)),
        "input_frames": int(df["frame"].nunique()) if "frame" in df else 0,
        "pair_count": int(df["pair_key"].nunique()) if "pair_key" in df else 1,
    }
    truth = _truth_flags(truth_label)
    for level in LEVELS:
        for behavior in BEHAVIORS:
            mask_col = f"{level}_standard_final_{behavior}"
            actor_col = f"{level}_standard_{behavior}_actor_id"
            target_col = f"{level}_standard_{behavior}_target_id"
            score_col = f"{level}_standard_{behavior}_score"
            if mask_col not in enriched:
                active = pd.Series(False, index=enriched.index)
            else:
                active = enriched[mask_col].fillna(False).astype(bool)
            predicted_frames = (
                int(enriched.loc[active, "frame"].nunique()) if "frame" in enriched else 0
            )
            row[f"{level}_{behavior}_pred"] = bool(predicted_frames > 0)
            row[f"{level}_{behavior}_frames"] = predicted_frames
            row[f"{level}_{behavior}_event_count"] = int(
                sum(
                    event.get("behavior") == behavior and event.get("level") == level
                    for event in events
                )
            )
            if score_col in enriched:
                row[f"{level}_{behavior}_max_score"] = float(enriched[score_col].max())
            else:
                row[f"{level}_{behavior}_max_score"] = 0.0
            if active.any() and actor_col in enriched and target_col in enriched:
                known = (
                    pd.to_numeric(enriched.loc[active, actor_col], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    .to_numpy()
                    >= 0
                ) & (
                    pd.to_numeric(enriched.loc[active, target_col], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    .to_numpy()
                    >= 0
                )
                row[f"{level}_{behavior}_role_known_rate"] = float(np.mean(known))
            else:
                row[f"{level}_{behavior}_role_known_rate"] = None
            fallback_col = f"{level}_standard_chase_role_fallback"
            if behavior == "chase" and active.any() and fallback_col in enriched:
                row[f"{level}_{behavior}_role_fallback_rate"] = float(
                    enriched.loc[active, fallback_col].fillna(False).astype(bool).mean()
                )
            else:
                row[f"{level}_{behavior}_role_fallback_rate"] = 0.0
            row[f"{level}_{behavior}_truth"] = truth[behavior]
    for event in events:
        event["video"] = pair_csv.parent.name
        event["truth_label"] = truth_label
        event["pair_csv"] = str(pair_csv)
    return row, events


def evaluate(
    datasets: list[tuple[str, Path]],
    config: dict[str, Any],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    video_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for truth_label, result_root in datasets:
        if not result_root.is_dir():
            raise FileNotFoundError(f"result root does not exist: {result_root}")
        pair_csvs = _pair_csvs(result_root)
        if not pair_csvs:
            raise FileNotFoundError(f"no pair CSV found below: {result_root}")
        for pair_csv in pair_csvs:
            row, events = _video_result(pair_csv, truth_label, config, fps)
            video_rows.append(row)
            event_rows.extend(events)

    metrics: dict[str, Any] = {}
    for level in LEVELS:
        for behavior in BEHAVIORS:
            truth = [bool(row[f"{level}_{behavior}_truth"]) for row in video_rows]
            prediction = [bool(row[f"{level}_{behavior}_pred"]) for row in video_rows]
            metrics[f"{level}_{behavior}"] = _binary_metrics(truth, prediction)
    return video_rows, event_rows, metrics


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Standard behavior threshold calibration",
        "",
        f"- Unit: `{report['protocol']['unit']}`",
        f"- Videos: `{len(report['videos'])}`",
        f"- Role ground truth: `{report['protocol']['role_ground_truth']}`",
        "",
        "| Level | Behavior | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metric in report["metrics"].items():
        level, behavior = key.split("_", 1)
        values = [metric.get(name) for name in ("precision", "recall", "f1")]
        display = ["N/A" if value is None else f"{value:.3f}" for value in values]
        lines.append(
            f"| {level} | {behavior} | {display[0]} | {display[1]} | {display[2]} | "
            f"{metric['tp']} | {metric['fp']} | {metric['fn']} | {metric['tn']} |"
        )
    lines.extend(
        [
            "",
            "Metrics are video-level because the supplied folder labels do not",
            "contain event boundaries. Actor/target accuracy is intentionally not",
            "reported without role annotations; role_known_rate is only a diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="LABEL=RESULT_ROOT",
        help="Repeat for chase, attack, none, or both. RESULT_ROOT contains one subdirectory per video.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-selected-role-fallback",
        action="store_true",
        help="Disable the legacy-schema role hint for a strict baseline comparison.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    config = copy.deepcopy(config)
    if args.no_selected_role_fallback:
        config.setdefault("standard_behavior_engine", {}).setdefault("selected_role_fallback", {})[
            "enabled"
        ] = False

    datasets: list[tuple[str, Path]] = []
    for item in args.dataset:
        label, separator, root = item.partition("=")
        if not separator or not label or not root:
            raise ValueError(f"invalid --dataset value: {item!r}; expected LABEL=RESULT_ROOT")
        datasets.append((label.strip().lower(), Path(root)))

    videos, events, metrics = evaluate(datasets, config, args.fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "video_results.csv"
    event_path = args.output_dir / "standard_events.csv"
    report_path = args.output_dir / "calibration_report.json"
    pd.DataFrame(videos).to_csv(video_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(events).to_csv(event_path, index=False, encoding="utf-8-sig")
    report = {
        "protocol": {
            "unit": "video",
            "label_source": "result-root dataset labels",
            "event_metrics": "not computed; frame boundaries are not supplied",
            "role_ground_truth": "not supplied; actor/target accuracy is N/A",
            "fps": args.fps,
            "datasets": [{"label": label, "result_root": str(root)} for label, root in datasets],
            "selected_role_fallback": config.get("standard_behavior_engine", {}).get(
                "selected_role_fallback", {}
            ),
        },
        "metrics": metrics,
        "videos": videos,
        "events": events,
    }
    report_path.write_text(
        json.dumps(_json_value(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(args.output_dir / "calibration_summary.md", report)
    LOGGER.info("metrics=\n%s", json.dumps(_json_value(metrics), ensure_ascii=False, indent=2))
    LOGGER.info("videos=%s", video_path)
    LOGGER.info("events=%s", event_path)
    LOGGER.info("report=%s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
