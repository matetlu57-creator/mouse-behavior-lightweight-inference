#!/usr/bin/env python3
"""Retired historical CLI; retain CSV helpers imported by older audits."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import _bootstrap  # noqa: F401 - preserve direct-script src import setup

from mouse_behavior.evaluation.blind_top1 import (
    LabeledClip,
    Top1Prediction,
)
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_predictions(path: Path, rows: list[tuple[str, Top1Prediction]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "predicted_behavior", "predicted_target_ids", "score"),
        )
        writer.writeheader()
        for sample_id, prediction in rows:
            writer.writerow({
                "sample_id": sample_id,
                "predicted_behavior": prediction.behavior or "",
                "predicted_target_ids": ",".join(map(str, prediction.target_ids)),
                "score": prediction.score,
            })


def _read_unlabeled_inputs(path: Path) -> list[tuple[str, str, float]]:
    """Project the val manifest to anonymous paths/FPS without accessing labels."""

    inputs: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError("validation manifest is empty")
        indexes = {name: index for index, name in enumerate(header)}
        required = {"tracks_path", "fps", "split"}
        missing = required - set(indexes)
        if missing:
            raise ValueError(f"validation manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"invalid validation row width at row {row_number + 1}")
            if row[indexes["split"]] not in {"val", "validation"}:
                raise ValueError(f"non-validation row at row {row_number + 1}")
            tracks_path = row[indexes["tracks_path"]].strip()
            try:
                fps = float(row[indexes["fps"]])
            except ValueError as exc:
                raise ValueError(f"invalid FPS at row {row_number + 1}") from exc
            if not tracks_path or fps <= 0:
                raise ValueError(f"invalid trajectory path/FPS at row {row_number + 1}")
            inputs.append((f"v-{row_number:06d}", tracks_path, fps))
    if not inputs:
        raise ValueError("validation manifest has no rows")
    return inputs


def _read_labels_after_receipt(path: Path) -> list[LabeledClip]:
    labels: list[LabeledClip] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tracks_path", "fps", "split", "recording_session",
            "canonical_behavior", "mouse_ids",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"validation labels are missing columns: {sorted(missing)}")
        for index, row in enumerate(reader, start=1):
            if row["split"] not in {"val", "validation"}:
                raise ValueError(f"non-validation label at row {index + 1}")
            behavior = row["canonical_behavior"]
            if behavior not in CANONICAL_BEHAVIORS:
                raise ValueError(f"unsupported validation label at row {index + 1}: {behavior}")
            try:
                fps = float(row["fps"])
                target_ids = tuple(sorted({
                    int(value) for value in row["mouse_ids"].split(",") if value.strip()
                }))
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"invalid validation IDs/FPS at row {index + 1}") from exc
            labels.append(LabeledClip(
                sample_id=f"v-{index:06d}",
                tracks_path=row["tracks_path"],
                fps=fps,
                recording_session=row["recording_session"],
                behavior=behavior,
                target_ids=target_ids,
            ))
    if not labels:
        raise ValueError("validation labels are empty")
    return labels


def _load_predictions_file(path: Path, expected_hash: str) -> dict[str, Top1Prediction]:
    if _sha256(path) != expected_hash:
        raise ValueError(f"prediction output changed after receipt: {path.name}")
    predictions: dict[str, Top1Prediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["sample_id"]
            if not sample_id or sample_id in predictions:
                raise ValueError("prediction output has missing or duplicate sample IDs")
            behavior = row["predicted_behavior"] or None
            target_ids = tuple(int(value) for value in row["predicted_target_ids"].split(",") if value)
            predictions[sample_id] = Top1Prediction(behavior, target_ids, float(row["score"]))
    return predictions


def main() -> int:
    print(
        "This historical validation CLI is retired. Run "
        "scripts/evaluate_active_dataset_v17.py --help "
        "for current-rule coverage auditing, or scripts/run_canonical_behavior_video.py --help for tracked-video inference. This retired command does not train or validate current models. "
        "Original source: outputs/current_heuristic_validation_retirement_20261013/"
        "retired_validation_sources.zip.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
