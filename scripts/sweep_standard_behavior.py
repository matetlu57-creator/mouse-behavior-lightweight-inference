#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search selected v1.43 FSM thresholds using cached pair features."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT
from mouse_behavior import standard_behavior_engine as engine
from mouse_behavior.config import load_config
from mouse_behavior.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


LEVELS = ("weak", "strong")


def parse_datasets(items: list[str]) -> list[tuple[str, Path]]:
    datasets: list[tuple[str, Path]] = []
    for item in items:
        label, separator, root = item.partition("=")
        if not separator or not label or not root:
            raise ValueError(f"invalid dataset: {item!r}")
        datasets.append((label.lower().strip(), Path(root)))
    return datasets


def load_videos(datasets: list[tuple[str, Path]], config: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    for truth, root in datasets:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            pair_csv = child / "成对行为标签.csv"
            if not pair_csv.is_file():
                continue
            df = pd.read_csv(pair_csv)
            pairs: list[pd.DataFrame] = []
            for _, pair in df.groupby("pair_key", sort=False, dropna=False):
                pairs.append(
                    engine.apply_standard_behavior_engine(
                        pair.sort_values("frame").reset_index(drop=True), fps, config
                    )
                )
            videos.append({"name": child.name, "truth": truth, "pairs": pairs})
            LOGGER.info("loaded %s: %s (%d pairs)", truth, child.name, len(pairs))
    if not videos:
        raise FileNotFoundError("no pair CSV found below the dataset roots")
    return videos


def hard_veto(frame_table: pd.DataFrame) -> np.ndarray:
    valid = frame_table["valid_pair"].fillna(False).astype(bool).to_numpy()
    wall = frame_table.get(
        "pair_wall_jump_excluded", pd.Series(False, index=frame_table.index)
    ).fillna(False).astype(bool).to_numpy()
    candidate = frame_table["standard_interaction_candidate"].fillna(False).astype(bool).to_numpy()
    return (~valid) | wall | (~candidate)


def truth_value(truth: str, behavior: str) -> bool:
    return truth in {behavior, "both"}


def metrics(videos: list[dict[str, Any]], predictions: list[bool], behavior: str) -> dict[str, Any]:
    truth = np.asarray([truth_value(video["truth"], behavior) for video in videos], dtype=bool)
    prediction = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def chase_predictions(videos: list[dict[str, Any]], level: str, fsm_cfg: dict[str, Any], fps: float) -> list[bool]:
    predictions: list[bool] = []
    for video in videos:
        present = False
        for table in video["pairs"]:
            result = engine._run_chase_fsm(
                table[f"{level}_standard_chase_score"].to_numpy(float),
                table[f"{level}_standard_approach_score"].to_numpy(float),
                table["standard_behavior_quality"].to_numpy(float),
                table[f"{level}_standard_chase_role_confidence"].to_numpy(float),
                hard_veto(table),
                fps,
                fsm_cfg,
            )
            present = present or bool(result.mask.any())
        predictions.append(present)
    return predictions


def attack_predictions(videos: list[dict[str, Any]], level: str, fsm_cfg: dict[str, Any], fps: float) -> list[bool]:
    predictions: list[bool] = []
    for video in videos:
        present = False
        for table in video["pairs"]:
            def gate_column(name: str, default: bool = True) -> np.ndarray:
                if name not in table:
                    return np.full(len(table), default, dtype=bool)
                return table[name].fillna(default).astype(bool).to_numpy()

            result = engine._run_attack_fsm(
                table[f"{level}_standard_initiation_score"].to_numpy(float),
                table[f"{level}_standard_contact_score"].to_numpy(float),
                table[f"{level}_standard_reaction_score"].to_numpy(float),
                table[f"{level}_standard_dynamic_attack_score"].to_numpy(float),
                table[f"{level}_standard_grapple_score"].to_numpy(float),
                table[f"{level}_standard_occlusion_score"].to_numpy(float),
                table["standard_behavior_quality"].to_numpy(float),
                table[f"{level}_standard_attack_role_confidence"].to_numpy(float),
                hard_veto(table),
                gate_column(f"{level}_standard_attack_dynamic_gate"),
                gate_column(f"{level}_standard_attack_stationary_gate"),
                gate_column(f"{level}_standard_attack_context_gate"),
                gate_column(f"{level}_standard_attack_impulse_gate"),
                fps,
                fsm_cfg,
            )
            present = present or bool(result.mask.any())
        predictions.append(present)
    return predictions


def sweep_chase(videos: list[dict[str, Any]], config: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        base = copy.deepcopy(config["standard_behavior_engine"][level]["chase_fsm"])
        for enter in np.arange(0.60, 1.001, 0.05):
            for confirm in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20):
                fsm = copy.deepcopy(base)
                fsm["enter_score"] = round(float(enter), 2)
                fsm["confirm_seconds"] = confirm
                result = metrics(videos, chase_predictions(videos, level, fsm, fps), "chase")
                rows.append({
                    "behavior": "chase",
                    "level": level,
                    "parameter": "enter_score+confirm_seconds",
                    "enter_score": fsm["enter_score"],
                    "confirm_seconds": confirm,
                    **result,
                })
    return rows


def sweep_attack(videos: list[dict[str, Any]], config: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        base = copy.deepcopy(config["standard_behavior_engine"][level]["attack_fsm"])
        for dynamic in (0.62, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88):
            for grapple in (0.66, 0.70, 0.74, 0.78, 0.82, 0.86):
                for occlusion in (0.64, 0.68, 0.72, 0.74, 0.76, 0.80, 0.84):
                    fsm = copy.deepcopy(base)
                    fsm["dynamic_confirm_score"] = dynamic
                    fsm["grapple_confirm_score"] = grapple
                    fsm["occlusion_confirm_score"] = occlusion
                    result = metrics(videos, attack_predictions(videos, level, fsm, fps), "attack")
                    rows.append({
                        "behavior": "attack",
                        "level": level,
                        "parameter": "dynamic+grapple+occlusion",
                        "dynamic_confirm_score": dynamic,
                        "grapple_confirm_score": grapple,
                        "occlusion_confirm_score": occlusion,
                        **result,
                    })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dataset", action="append", required=True, metavar="LABEL=RESULT_ROOT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    datasets = parse_datasets(args.dataset)
    videos = load_videos(datasets, config, args.fps)
    rows = sweep_chase(videos, config, args.fps) + sweep_attack(videos, config, args.fps)
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["behavior", "level", "f1", "precision", "recall"],
        ascending=[True, True, False, False, False],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    for behavior in ("chase", "attack"):
        for level in LEVELS:
            top = result[(result.behavior == behavior) & (result.level == level)].head(10)
            LOGGER.info("%s %s\n%s", behavior, level, top.to_string(index=False))
    LOGGER.info("sweep=%s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
