"""Label-free clip prediction and target-ID behavior evaluation.

The exported annotation IDs are clip-local visual track IDs, not stable RFID
animal identities.  A blind predictor must never receive a ``LabeledClip``;
it receives only trajectory-derived evidence and frozen ranking weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exported_dataset import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    PAIR_BEHAVIORS,
)


@dataclass(frozen=True)
class LabeledClip:
    sample_id: str
    tracks_path: str
    fps: float
    recording_session: str
    behavior: str
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class UnlabeledClip:
    sample_id: str
    tracks_path: str
    fps: float


@dataclass(frozen=True)
class Candidate:
    behavior: str
    target_ids: tuple[int, ...]
    evidence: float


@dataclass(frozen=True)
class Top1Prediction:
    behavior: str | None
    target_ids: tuple[int, ...]
    score: float
    secondary_behaviors: tuple[str, ...] = ()


def load_training_manifest(path: str | Path) -> list[LabeledClip]:
    """Accept only explicit training rows; never discover a dataset root."""

    manifest = Path(path)
    required = {
        "sample_id",
        "tracks_path",
        "fps",
        "recording_session",
        "canonical_behavior",
        "mouse_ids",
        "split",
    }
    records: list[LabeledClip] = []
    seen_ids: set[str] = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"training manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if row["split"] != "train":
                raise ValueError(f"non-training split at row {row_number}: {row['split']!r}")
            sample_id = str(row["sample_id"] or "").strip()
            session = str(row["recording_session"] or "").strip()
            behavior = str(row["canonical_behavior"] or "").strip()
            tracks_path = str(row["tracks_path"] or "").strip()
            if not sample_id or not session or not tracks_path:
                raise ValueError(f"incomplete training manifest row {row_number}")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id in training manifest: {sample_id}")
            if behavior not in CANONICAL_BEHAVIORS:
                raise ValueError(f"unsupported training behavior at row {row_number}: {behavior}")
            try:
                fps = float(row["fps"])
                target_ids = tuple(sorted({int(value) for value in row["mouse_ids"].split(",") if value.strip()}))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid FPS or target IDs at row {row_number}") from exc
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"invalid FPS at row {row_number}: {fps}")
            required_count = 1 if behavior in INDIVIDUAL_BEHAVIORS else 2
            if behavior == "isolation":
                required_count = 1
            elif behavior in GROUP_BEHAVIORS:
                required_count = 3
            if behavior in {"isolation", *INDIVIDUAL_BEHAVIORS, *PAIR_BEHAVIORS}:
                count_invalid = len(target_ids) != required_count
            else:
                count_invalid = len(target_ids) < required_count
            if count_invalid:
                raise ValueError(f"invalid target ID count for {behavior} at row {row_number}")
            records.append(
                LabeledClip(
                    sample_id=sample_id,
                    tracks_path=tracks_path,
                    fps=fps,
                    recording_session=session,
                    behavior=behavior,
                    target_ids=target_ids,
                )
            )
            seen_ids.add(sample_id)
    if not records:
        raise ValueError(f"empty training manifest: {manifest}")
    return records


def load_unlabeled_manifest(path: str | Path) -> list[UnlabeledClip]:
    """Reject label-bearing columns at the blind-inference input boundary."""

    records: list[UnlabeledClip] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {"sample_id", "tracks_path", "fps"}:
            raise ValueError("blind manifest must have exactly sample_id, tracks_path, fps")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"extra blind manifest fields at row {row_number}")
            sample_id = str(row["sample_id"] or "").strip()
            tracks_path = str(row["tracks_path"] or "").strip()
            if not sample_id or not tracks_path or sample_id in seen:
                raise ValueError(f"missing or repeated blind sample at row {row_number}")
            try:
                fps = float(row["fps"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid blind FPS at row {row_number}") from exc
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"invalid blind FPS at row {row_number}")
            records.append(UnlabeledClip(sample_id, tracks_path, fps))
            seen.add(sample_id)
    if not records:
        raise ValueError("blind manifest is empty")
    return records


def load_locked_predictions(directory: str | Path) -> dict[str, Top1Prediction]:
    """Verify a blind prediction receipt before any label join is allowed."""

    root = Path(directory)
    receipt = json.loads((root / "prediction_receipt.json").read_text(encoding="utf-8"))
    path = root / "predictions.csv"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != receipt.get("predictions_sha256"):
        raise ValueError("prediction CSV changed after the blind prediction receipt")
    predictions: dict[str, Top1Prediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "predicted_behavior", "predicted_target_ids", "score"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("blind prediction CSV is missing required columns")
        for row in reader:
            sample_id = row["sample_id"]
            if not sample_id or sample_id in predictions:
                raise ValueError(f"missing or repeated predicted sample ID: {sample_id!r}")
            ids = tuple(int(value) for value in row["predicted_target_ids"].split(",") if value)
            predictions[sample_id] = Top1Prediction(
                row["predicted_behavior"] or None,
                ids,
                float(row["score"]),
                tuple(
                    value for value in row.get("secondary_behaviors", "").split(",") if value
                ),
            )
    if len(predictions) != int(receipt["prediction_count"]):
        raise ValueError("prediction count differs from the blind prediction receipt")
    return predictions


def _positive_score(value: Any) -> float:
    score = float(value)
    return score if math.isfinite(score) and score > 0 else 0.0


def candidates_from_evidence(prediction: Mapping[str, Any]) -> tuple[Candidate, ...]:
    """Create behavior/ID candidates solely from trajectory-rule evidence."""

    candidates: list[Candidate] = []
    identity_scores = dict(prediction.get("identity_scores", {}))
    pair_scores = dict(prediction.get("pair_scores", {}))
    for text_id, raw_scores in identity_scores.items():
        track_id = int(text_id)
        values = dict(raw_scores)
        for behavior in INDIVIDUAL_BEHAVIORS:
            score = _positive_score(values.get(behavior, 0.0))
            if score:
                candidates.append(Candidate(behavior, (track_id,), score))
    for text_ids, raw_scores in pair_scores.items():
        ids = tuple(sorted(int(value) for value in text_ids.split(",")))
        if len(ids) != 2 or ids[0] == ids[1]:
            raise ValueError(f"invalid visual-ID pair in prediction: {text_ids!r}")
        values = dict(raw_scores)
        for behavior in PAIR_BEHAVIORS:
            score = _positive_score(values.get(behavior, 0.0))
            if score:
                candidates.append(Candidate(behavior, ids, score))
    if "group_scores" in prediction:
        for text_ids, raw_scores in dict(prediction["group_scores"]).items():
            ids = tuple(sorted(int(value) for value in text_ids.split(",")))
            if len(ids) != len(set(ids)):
                raise ValueError(f"invalid visual-ID group in prediction: {text_ids!r}")
            for behavior in GROUP_BEHAVIORS:
                score = _positive_score(dict(raw_scores).get(behavior, 0.0))
                minimum_size = 1 if behavior == "isolation" else 3
                if score and len(ids) >= minimum_size:
                    candidates.append(Candidate(behavior, ids, score))
    else:
        # Compatibility for older rule evidence without explicit components.
        for behavior in GROUP_BEHAVIORS:
            if behavior == "isolation":
                for track_id, values in identity_scores.items():
                    score = _positive_score(dict(values).get(behavior, 0.0))
                    if score:
                        candidates.append(Candidate(behavior, (int(track_id),), score))
                continue
            positive = sorted(
                (int(track_id), _positive_score(dict(values).get(behavior, 0.0)))
                for track_id, values in identity_scores.items()
            )
            members = [(track_id, score) for track_id, score in positive if score]
            minimum_size = 1 if behavior == "isolation" else 3
            if len(members) >= minimum_size:
                candidates.append(
                    Candidate(behavior, tuple(track_id for track_id, _ in members),
                              min(score for _, score in members))
                )
    return tuple(candidates)


def select_top1(
    candidates: Sequence[Candidate],
    weights: Mapping[str, float] | None = None,
) -> Top1Prediction:
    """Return exactly one prediction without a label or annotated target ID."""

    if not candidates:
        return Top1Prediction(None, (), 0.0)
    weights = weights or {}
    if set(weights) - set(CANONICAL_BEHAVIORS):
        raise ValueError("ranking weights contain unknown behaviors")
    for behavior, weight in weights.items():
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError(f"ranking weight for {behavior} must be positive and finite")
    eligible = hierarchical_candidate_indices(candidates)
    best = min(
        (candidates[index] for index in eligible),
        key=lambda item: (
            -item.evidence * float(weights.get(item.behavior, 1.0)),
            item.behavior,
            item.target_ids,
        ),
    )
    return Top1Prediction(
        best.behavior,
        best.target_ids,
        best.evidence * float(weights.get(best.behavior, 1.0)),
    )


def hierarchical_candidate_indices(candidates: Sequence[Candidate]) -> list[int]:
    """Keep cross-layer behaviors co-occurring for train-ranked Top-1.

    The individual/social/group hierarchy controls output presentation, not
    whether a lower-layer event exists. In particular, a social/group event
    must not erase the annotated target mouse's Walking/Static/Running label.
    The renderer applies category priority while preserving each layer; this
    evaluation selector leaves every candidate available to the train-only
    ranker.
    """

    return list(range(len(candidates)))


def evaluate_top1(
    rows: Sequence[tuple[LabeledClip, Top1Prediction]],
) -> dict[str, Any]:
    """Score Top-1 behavior for the labeled target IDs.

    A prediction is correct when it names the labeled behavior and includes
    every annotated target ID. Extra predicted group members are retained as
    an auxiliary exact-set diagnostic, not treated as target-behavior errors.
    """

    per_behavior: dict[str, dict[str, int]] = {}
    errors: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    joint = compatible_exact = behavior_hits = id_hits = covered = 0
    for record, prediction in rows:
        values = per_behavior.setdefault(
            record.behavior,
            {"support": 0, "joint_correct": 0, "compatible_exact_correct": 0,
             "behavior_correct": 0,
             "id_correct": 0, "exact_id_correct": 0},
        )
        values["support"] += 1
        behavior_ok = prediction.behavior == record.behavior
        ids_ok = set(record.target_ids).issubset(set(prediction.target_ids))
        exact_ids_ok = tuple(sorted(prediction.target_ids)) == record.target_ids
        joint_ok = behavior_ok and ids_ok
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now
        # allows contact without interaction, so the Together-vs-contact
        # alternative no longer exists; compatible_exact mirrors strict exact-ID.
        compatible_ok = exact_ids_ok and behavior_ok
        joint += int(joint_ok)
        compatible_exact += int(compatible_ok)
        behavior_hits += int(behavior_ok)
        id_hits += int(ids_ok)
        covered += int(prediction.behavior is not None)
        values["joint_correct"] += int(joint_ok)
        values["compatible_exact_correct"] += int(compatible_ok)
        values["behavior_correct"] += int(behavior_ok)
        values["id_correct"] += int(ids_ok)
        values["exact_id_correct"] += int(exact_ids_ok)
        if not joint_ok:
            errors[
                "wrong_ids" if behavior_ok else "wrong_behavior" if ids_ok else "both_wrong"
            ] += 1
        confusion[(record.behavior, prediction.behavior or "<none>")] += 1
    total = len(rows)
    detailed = {
        behavior: {
            **values,
            "joint_top1_accuracy": values["joint_correct"] / values["support"],
            "compatible_exact_top1_accuracy": (
                values["compatible_exact_correct"] / values["support"]
            ),
            "behavior_top1_accuracy": values["behavior_correct"] / values["support"],
            "id_top1_accuracy": values["id_correct"] / values["support"],
            "exact_id_set_accuracy": values["exact_id_correct"] / values["support"],
        }
        for behavior, values in sorted(per_behavior.items())
    }
    return {
        "n_samples": total,
        "joint_top1_accuracy": joint / total if total else 0.0,
        "compatible_exact_top1_accuracy": compatible_exact / total if total else 0.0,
        "macro_joint_top1_accuracy": (
            sum(value["joint_top1_accuracy"] for value in detailed.values()) / len(detailed)
            if detailed else 0.0
        ),
        "behavior_top1_accuracy": behavior_hits / total if total else 0.0,
        "id_top1_accuracy": id_hits / total if total else 0.0,
        "exact_id_set_accuracy": (
            sum(values["exact_id_correct"] for values in per_behavior.values()) / total
            if total else 0.0
        ),
        "prediction_coverage": covered / total if total else 0.0,
        "per_behavior": detailed,
        "error_breakdown": dict(sorted(errors.items())),
        "confusion": {
            f"{truth}->{predicted}": count
            for (truth, predicted), count in sorted(confusion.items())
        },
    }


def target_layer_candidate_diagnostics(
    rows: Sequence[tuple[LabeledClip, Sequence[Candidate]]],
) -> dict[str, Any]:
    """Summarize clip-level candidate support for annotated target IDs.

    This is not time-aligned co-occurrence or ground truth. Call only after
    label-free candidates have been locked; labels are used only to select the
    annotated target IDs for this diagnostic.
    """

    behavior_layers = {
        "individual": set(INDIVIDUAL_BEHAVIORS),
        "social": set(PAIR_BEHAVIORS),
        "group": set(GROUP_BEHAVIORS),
    }
    per_behavior: dict[str, dict[str, Any]] = {}
    for label, candidates in rows:
        metrics = per_behavior.setdefault(
            label.behavior,
            {
                "support": 0,
                "target_mouse_instances": 0,
                **{
                    layer: {
                        "clips_with_candidate": 0,
                        "target_mouse_instances_with_candidate": 0,
                    }
                    for layer in behavior_layers
                },
            },
        )
        target_ids = set(label.target_ids)
        metrics["support"] += 1
        metrics["target_mouse_instances"] += len(target_ids)
        matched_by_layer = {layer: set() for layer in behavior_layers}
        for candidate in candidates:
            matched_ids = target_ids.intersection(candidate.target_ids)
            if not matched_ids:
                continue
            for layer, behaviors in behavior_layers.items():
                if candidate.behavior in behaviors:
                    matched_by_layer[layer].update(matched_ids)
                    break
        for layer, matched_ids in matched_by_layer.items():
            if matched_ids:
                metrics[layer]["clips_with_candidate"] += 1
                metrics[layer]["target_mouse_instances_with_candidate"] += len(matched_ids)
    return {
        "n_samples": len(rows),
        "scope": "clip-level positive candidate support on labeled target IDs; not time-overlap or ground truth",
        "per_labeled_behavior": per_behavior,
    }
