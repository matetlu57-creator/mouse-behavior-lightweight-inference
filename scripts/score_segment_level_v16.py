#!/usr/bin/env python3
"""Segment-level (coverage) detection scoring for the frozen v16 sample sets.

Acceptance criterion
--------------------
The exported clip *is* the annotated bout, so the fraction of the clip's frames
that a predicted mask covers is the temporal overlap between prediction and
annotation.  A sample counts as DETECTED when that coverage reaches
``--overlap-threshold`` (default 0.70).  Two accuracies are reported:

* ``A1`` - behavior accuracy, ID-agnostic.  The annotated behavior reaches the
  threshold on ANY target.  A wrong, over-inclusive or under-inclusive target
  ID set is forgiven.
* ``A2`` - strict accuracy, with IDs.  The annotated behavior reaches the
  threshold on EXACTLY the annotated target ID set.

``manual_review_queue`` holds the samples where A1 is true and A2 is false,
each classified by how the ID set that earned A1 differs from the annotation:
``predicted_is_superset`` (too many IDs), ``predicted_is_subset`` (too few),
``partial_overlap``, ``disjoint``.

Because an annotation interval can be wider than the physical bout, a 0.70
threshold may be unreachable even for a perfect detector.  Every row therefore
also carries ``coverage_on_true_target`` as a continuous value and a
``coverage_band`` bucket, and the summary carries a per-domain band histogram,
so a near miss at 0.60-0.69 (threshold / annotation-width problem) is separable
from a total miss at 0 (the rule never fires).

Group caveat
------------
The group layer emits DURATIONS, not per-frame masks, so group coverage is a
proxy: ``evidence_seconds / clip_duration_seconds``.  It cannot express WHEN
inside the clip the evidence accumulated, so a group sample can clear the
threshold with evidence located elsewhere than the annotated bout.  This is
flagged in the summary as
``group_coverage_is_evidence_over_clip_duration_proxy: true`` and group numbers
must not be read as frame-level overlap.

Scale verification is per domain and mirrors the frozen validators exactly:
individual uses ``current_validation.verified_source_scale``, social uses
``validate_social_heuristics._verified_frozen_scale`` and group uses
``validate_group_heuristics._verified_fixed_scale``.  Verification failures are
recorded as per-sample errors; they are never bypassed.

This script only reads.  It trains nothing, recalibrates nothing and rewrites
no frozen artifact; it scores the frozen v16 sample sets so the denominators
stay comparable (59 individual / 115 social / 73 group).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import REPO_ROOT  # noqa: E402
from mouse_behavior.behavior.individual_rules import (  # noqa: E402
    DEFAULT_INDIVIDUAL_CONFIG,
    SUSPENDED_POSTURE_BEHAVIORS,
    IndividualHeuristicConfig,
    individual_rule_masks,
)
from mouse_behavior.config import load_config  # noqa: E402
from mouse_behavior.evaluation.confirmed_dataset import file_sha256  # noqa: E402
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks, policy_receipt  # noqa: E402
from mouse_behavior.evaluation.current_validation import (  # noqa: E402
    resolved_input_path,
    verified_source_scale,
)
from mouse_behavior.evaluation.exported_dataset import (  # noqa: E402
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.exported_dataset_models import (  # noqa: E402
    ExportedFeatures,
    HeuristicParameters,
)
from mouse_behavior.evaluation.group_event_rules import (  # noqa: E402
    group_rule_candidates,
    isolation_candidates,
)
from validate_group_heuristics import _verified_fixed_scale  # noqa: E402
from validate_social_heuristics import _verified_frozen_scale  # noqa: E402

DOMAINS: tuple[str, ...] = ("individual", "social", "group")
FROZEN_RUN_NAME = "current_validation_20261015_v16_final"
EXPECTED_ROWS: dict[str, int] = {"individual": 59, "social": 115, "group": 73}
DEFAULT_MAX_TRACKS = 64

# pair_rule_masks uses rule-internal names; the dataset uses canonical names.
SOCIAL_MASK_TO_CANONICAL: dict[str, str] = {
    "approach": "approach",
    "attack": "attack",
    "chasing": "chase",
    "together": "together",
    "head_contact": "nose_head_contact",
    "tail_contact": "nose_tail_contact",
}
CANONICAL_TO_SOCIAL_MASK: dict[str, str] = {
    canonical: mask for mask, canonical in SOCIAL_MASK_TO_CANONICAL.items()
}

COVERAGE_BANDS: tuple[str, ...] = (
    "0", "(0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]",
)

ROW_FIELDS: tuple[str, ...] = (
    "domain", "sample_id", "true_behavior", "true_target_ids", "a1_hit", "a2_hit",
    "coverage_on_true_target", "coverage_band", "coverage_best_any_target",
    "best_target_ids", "id_difference_kind", "winning_behavior_on_true_target",
    "frames", "fps", "status", "error",
    "coverage_on_true_target_any_slot", "true_target_any_exclusive_mask_nonempty",
    "true_ids_missing_from_tracks", "scale_drift",
)

SCALE_VERIFICATION_BY_DOMAIN: dict[str, str] = {
    "individual": "mouse_behavior.evaluation.current_validation.verified_source_scale",
    "social": "scripts/validate_social_heuristics.py::_verified_frozen_scale",
    "group": "scripts/validate_group_heuristics.py::_verified_fixed_scale",
}

@dataclass(frozen=True)
class RunContext:
    """Read-only roots plus the shared hash ledger for one scoring run."""

    data_root: Path
    approved_root: Path
    v16_dir: Path
    val_csv: Path
    rig_path: Path
    max_tracks: int
    hashes: dict[str, str]


def coverage_band(value: float) -> str:
    """Bucket a coverage fraction so near misses are separable from no-fires."""

    if value <= 0.0:
        return "0"
    if value < 0.3:
        return "(0,0.3)"
    if value < 0.5:
        return "[0.3,0.5)"
    if value < 0.7:
        return "[0.5,0.7)"
    if value < 0.9:
        return "[0.7,0.9)"
    return "[0.9,1.0]"


def id_difference_kind(true_ids: Sequence[int], predicted_ids: Sequence[int]) -> str:
    """Classify how an A1-earning ID set differs from the annotated ID set."""

    truth, predicted = set(true_ids), set(predicted_ids)
    if predicted == truth:
        # Unreachable for a genuine manual-review row: an exact ID match that
        # cleared the threshold on A1 also clears A2. Reported, not hidden.
        return "identical"
    if not (truth & predicted):
        return "disjoint"
    if predicted > truth:
        return "predicted_is_superset"
    if predicted < truth:
        return "predicted_is_subset"
    return "partial_overlap"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a frozen manifest/prediction CSV exactly as the validators do."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def target_ids(text: str) -> tuple[int, ...]:
    """Parse a ``mouse_ids`` cell into a sorted, de-duplicated visual-ID set."""

    return tuple(sorted({int(value) for value in str(text).split(",") if value.strip()}))


def slot_lookup(features: ExportedFeatures) -> dict[int, int]:
    """Map visual track IDs to mask column slots."""

    return {int(value): slot for slot, value in enumerate(features.track_ids.tolist())}


def mask_coverage(mask: np.ndarray, slots: Sequence[int], mode: str) -> float:
    """Fraction of ALL clip frames where the mask holds across ``slots``.

    ``mode="all"`` requires every slot (the strict A2 reading of coverage on a
    multi-ID target set); ``mode="any"`` requires at least one and is recorded
    alongside it as a diagnostic.
    """

    frames = int(mask.shape[0])
    if frames <= 0 or not slots:
        return 0.0
    subset = mask[:, list(slots)]
    hits = subset.all(axis=1) if mode == "all" else subset.any(axis=1)
    return float(np.count_nonzero(hits)) / frames


def individual_config(params: HeuristicParameters) -> IndividualHeuristicConfig:
    """Rebuild the individual config exactly as production inference does.

    Mirrors ``exported_dataset_prediction.predict_features``: seven fields are
    overridden from the seeded parameters and every other field keeps its
    ``DEFAULT_INDIVIDUAL_CONFIG`` value.  ``static_detection_gap_max_s`` is
    deliberately NOT passed, so it keeps its 1.0 s default as in production.
    """

    return replace(
        DEFAULT_INDIVIDUAL_CONFIG,
        motion_speed_threshold_cm_s=params.individual_motion_speed_threshold_cm_s,
        static_min_duration_s=params.stationary_min_duration_s,
        locomotion_min_duration_s=params.locomotion_min_duration_s,
        static_max_radius_cm=params.stationary_max_center_radius_cm,
        static_transient_max_s=params.stationary_transient_max_duration_s,
        static_transient_max_displacement_cm=params.stationary_transient_max_displacement_cm,
        locomotion_pause_max_s=params.locomotion_stop_gap_s,
    )


def load_domain_rows(domain: str, ctx: RunContext) -> list[dict[str, str]]:
    """Return the frozen v16 rows for one domain, in frozen order.

    Individual and group rows already carry the full provenance columns.  The
    social artifact only stores predictions, so its sample_id list is joined
    back to the frozen split manifest to recover tracks_path/fps/scale receipt.
    """

    if domain == "individual":
        return read_csv_rows(ctx.v16_dir / "individual" / "individual_validation.csv")
    if domain == "group":
        return read_csv_rows(ctx.v16_dir / "group" / "group_validation.csv")
    if domain != "social":
        raise ValueError(f"unknown domain: {domain}")
    predictions = read_csv_rows(ctx.v16_dir / "social" / "social_validation_predictions.csv")
    joined_by_id = {row["sample_id"]: row for row in read_csv_rows(ctx.val_csv)}
    rows: list[dict[str, str]] = []
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        source = joined_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"social sample_id absent from frozen split manifest: {sample_id}")
        rows.append(dict(source))
    return rows


def new_record(domain: str, row: dict[str, str]) -> dict[str, Any]:
    """Create an all-default output record so errored samples are still counted."""

    ids = target_ids(row.get("mouse_ids", ""))
    return {
        "domain": domain,
        "sample_id": row.get("sample_id", ""),
        "true_behavior": row.get("canonical_behavior", ""),
        "true_target_ids": ",".join(map(str, ids)),
        "a1_hit": False,
        "a2_hit": False,
        "coverage_on_true_target": 0.0,
        "coverage_band": "0",
        "coverage_best_any_target": 0.0,
        "best_target_ids": "",
        "id_difference_kind": "",
        "winning_behavior_on_true_target": "",
        "frames": 0,
        "fps": 0.0,
        "status": "ok",
        "error": "",
        "coverage_on_true_target_any_slot": 0.0,
        "true_target_any_exclusive_mask_nonempty": "",
        "true_ids_missing_from_tracks": False,
        "scale_drift": "",
        "_true_ids": ids,
    }

def finalize_record(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Derive A1/A2, the coverage band and the ID-difference kind for one row."""

    record["a1_hit"] = bool(record["coverage_best_any_target"] >= threshold)
    record["a2_hit"] = bool(record["coverage_on_true_target"] >= threshold)
    record["coverage_band"] = coverage_band(float(record["coverage_on_true_target"]))
    best_ids = target_ids(record["best_target_ids"])
    if record["a1_hit"] and not record["a2_hit"]:
        record["id_difference_kind"] = id_difference_kind(record["_true_ids"], best_ids)
    return record


def score_individual(row: dict[str, str], params: HeuristicParameters,
                     ctx: RunContext) -> dict[str, Any]:
    """Score one individual clip from its per-column rule masks.

    ``valid`` is passed straight through as ``features.valid``: the production
    call site in ``exported_dataset_prediction.predict_features`` does exactly
    that, and the finite-centre intersection happens INSIDE
    ``individual_rule_masks`` (``observed = observed & isfinite(centers)``).
    Re-masking here would diverge from the frozen run.
    """

    record = new_record("individual", row)
    scale, drift = verified_source_scale(
        row, ctx.data_root, ctx.approved_root, ctx.hashes, active_rig_path=ctx.rig_path,
    )
    features = load_exported_features(
        row["tracks_path"], fps=float(row["fps"]), max_tracks=ctx.max_tracks, scale_config=scale,
    )
    masks = individual_rule_masks(
        features.centers_cm, features.valid, features.fps,
        keypoints_cm=features.collective_keypoints_cm,
        reference_body_cm=features.reference_body_cm,
        config=individual_config(params),
    ).masks
    behavior = row["canonical_behavior"]
    if behavior not in masks:
        raise ValueError(f"individual rule layer emits no mask for {behavior!r}")
    mask = np.asarray(masks[behavior], bool)
    frames = int(mask.shape[0])
    lookup = slot_lookup(features)
    true_ids: tuple[int, ...] = record["_true_ids"]
    slots = [lookup[value] for value in true_ids if value in lookup]
    missing = [value for value in true_ids if value not in lookup]
    record["frames"] = frames
    record["fps"] = float(features.fps)
    record["scale_drift"] = "; ".join(drift)
    record["true_ids_missing_from_tracks"] = bool(missing)

    best_coverage, best_ids = 0.0, ()
    if frames > 0 and mask.shape[1] > 0:
        columns = mask.mean(axis=0)
        best_slot = int(np.argmax(columns))
        best_coverage = float(columns[best_slot])
        best_ids = (int(features.track_ids[best_slot]),)
    record["coverage_best_any_target"] = round(best_coverage, 6)
    record["best_target_ids"] = ",".join(map(str, best_ids))

    # An annotated ID that max_tracks dropped cannot be covered; report 0.0 and
    # flag it rather than crediting the remaining subset of the target set.
    if slots and not missing:
        record["coverage_on_true_target"] = round(mask_coverage(mask, slots, "all"), 6)
        record["coverage_on_true_target_any_slot"] = round(mask_coverage(mask, slots, "any"), 6)

    winning, winning_coverage = "", 0.0
    for name, other in masks.items():
        value = mask_coverage(np.asarray(other, bool), slots, "all") if slots else 0.0
        if value > winning_coverage:
            winning, winning_coverage = name, value
    record["winning_behavior_on_true_target"] = winning
    record["true_target_any_exclusive_mask_nonempty"] = bool(winning_coverage > 0.0)
    return record


def score_social(row: dict[str, str], params: HeuristicParameters,
                 ctx: RunContext) -> dict[str, Any]:
    """Score one social clip from the post-waterfall exclusive pair masks.

    The six pair masks are mutually exclusive per edge, so the winning
    behavior on the annotated pair is recorded separately: it shows which rule
    consumed the bout when arbitration, not detection, caused the miss.
    """

    record = new_record("social", row)
    scale, drift = _verified_frozen_scale(row, ctx.data_root, ctx.rig_path, ctx.hashes)
    tracks_path = resolved_input_path(row["tracks_path"], ctx.data_root)
    features = load_exported_features(
        tracks_path, fps=float(row["fps"]), max_tracks=ctx.max_tracks, scale_config=scale,
    )
    masks = pair_rule_masks(features, params)
    behavior = row["canonical_behavior"]
    if behavior not in CANONICAL_TO_SOCIAL_MASK:
        raise ValueError(f"social rule layer emits no mask for {behavior!r}")
    mask = np.asarray(masks[CANONICAL_TO_SOCIAL_MASK[behavior]], bool)
    frames = int(features.valid.shape[0])
    lookup = slot_lookup(features)
    track_ids = [int(value) for value in features.track_ids.tolist()]
    edge_ids = [
        frozenset((track_ids[int(left)], track_ids[int(right)]))
        for left, right in zip(features.pair_i.tolist(), features.pair_j.tolist(), strict=True)
    ]
    true_ids: tuple[int, ...] = record["_true_ids"]
    true_set = frozenset(true_ids)
    missing = [value for value in true_ids if value not in lookup]
    record["frames"] = frames
    record["fps"] = float(features.fps)
    record["scale_drift"] = "; ".join(drift)
    record["true_ids_missing_from_tracks"] = bool(missing)
    true_edge = next((index for index, ids in enumerate(edge_ids) if ids == true_set), None)

    if true_edge is not None and frames > 0:
        coverage = round(float(mask[:, true_edge].mean()), 6)
        record["coverage_on_true_target"] = coverage
        record["coverage_on_true_target_any_slot"] = coverage

    best_coverage, best_ids = 0.0, ()
    if frames > 0 and mask.shape[1] > 0:
        columns = mask.mean(axis=0)
        best_edge = int(np.argmax(columns))
        best_coverage = float(columns[best_edge])
        best_ids = tuple(sorted(edge_ids[best_edge]))
    record["coverage_best_any_target"] = round(best_coverage, 6)
    record["best_target_ids"] = ",".join(map(str, best_ids))

    winning, winning_coverage = "", 0.0
    if true_edge is not None and frames > 0:
        for mask_key, canonical in SOCIAL_MASK_TO_CANONICAL.items():
            value = float(np.asarray(masks[mask_key], bool)[:, true_edge].mean())
            if value > winning_coverage:
                winning, winning_coverage = canonical, value
        record["true_target_any_exclusive_mask_nonempty"] = bool(winning_coverage > 0.0)
    record["winning_behavior_on_true_target"] = winning
    return record


def score_group(row: dict[str, str], params: HeuristicParameters,
                ctx: RunContext) -> dict[str, Any]:
    """Score one group clip with the evidence-over-duration coverage PROXY.

    The group layer returns ``Candidate(behavior, target_ids, evidence_seconds)``
    and no per-frame mask, so coverage here is
    ``evidence / (frames / fps)``.  It measures how much of the clip's duration
    the rule accumulated, not frame-aligned overlap with the annotation.
    """

    record = new_record("group", row)
    scale, drift, scale_hashes = _verified_fixed_scale(row, ctx.data_root, ctx.rig_path)
    ctx.hashes.update(scale_hashes)
    tracks_path = resolved_input_path(row["tracks_path"], ctx.data_root)
    features = load_exported_features(
        tracks_path, fps=float(row["fps"]), max_tracks=ctx.max_tracks, scale_config=scale,
    )
    frames = int(features.valid.shape[0])
    fps = float(features.fps)
    duration = frames / fps if fps > 0 else 0.0
    candidates = (
        tuple(group_rule_candidates(features, params, ()))
        + tuple(isolation_candidates(features, params))
    )
    lookup = slot_lookup(features)
    true_ids: tuple[int, ...] = record["_true_ids"]
    true_set = frozenset(true_ids)
    behavior = row["canonical_behavior"]
    record["frames"] = frames
    record["fps"] = fps
    record["scale_drift"] = "; ".join(drift)
    record["true_ids_missing_from_tracks"] = any(value not in lookup for value in true_ids)

    scored = [
        (candidate, float(candidate.evidence) / duration if duration > 0 else 0.0)
        for candidate in candidates
    ]
    same_behavior = [(candidate, value) for candidate, value in scored if candidate.behavior == behavior]
    exact = [value for candidate, value in same_behavior if frozenset(candidate.target_ids) == true_set]
    if exact:
        record["coverage_on_true_target"] = round(max(exact), 6)
        record["coverage_on_true_target_any_slot"] = round(max(exact), 6)
    if same_behavior:
        best_candidate, best_value = max(same_behavior, key=lambda item: item[1])
        record["coverage_best_any_target"] = round(best_value, 6)
        record["best_target_ids"] = ",".join(map(str, sorted(best_candidate.target_ids)))

    on_true_set = [
        (candidate, value) for candidate, value in scored
        if frozenset(candidate.target_ids) == true_set
    ]
    if on_true_set:
        winning_candidate, winning_value = max(on_true_set, key=lambda item: item[1])
        record["winning_behavior_on_true_target"] = (
            winning_candidate.behavior if winning_value > 0 else ""
        )
        record["true_target_any_exclusive_mask_nonempty"] = bool(winning_value > 0)
    return record


SCORERS = {"individual": score_individual, "social": score_social, "group": score_group}

def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` when there is nothing to average."""

    return round(sum(values) / len(values), 6) if values else None


def accuracy(hits: int, denominator: int) -> float | None:
    """Hit rate over SCORED samples; ``None`` when the domain had none."""

    return round(hits / denominator, 6) if denominator else None


def behavior_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-behavior support, A1/A2 hits and mean coverages."""

    table: dict[str, Any] = {}
    for behavior in sorted({record["true_behavior"] for record in records}):
        subset = [record for record in records if record["true_behavior"] == behavior]
        scored = [record for record in subset if record["status"] == "ok"]
        a1_hits = sum(1 for record in scored if record["a1_hit"])
        a2_hits = sum(1 for record in scored if record["a2_hit"])
        table[behavior] = {
            "support": len(subset),
            "n_scored": len(scored),
            "n_error": len(subset) - len(scored),
            "a1_hits": a1_hits,
            "a2_hits": a2_hits,
            "a1_accuracy": accuracy(a1_hits, len(scored)),
            "a2_accuracy": accuracy(a2_hits, len(scored)),
            "mean_coverage_on_true_target": mean(
                [float(record["coverage_on_true_target"]) for record in scored]
            ),
            "mean_coverage_any_target": mean(
                [float(record["coverage_best_any_target"]) for record in scored]
            ),
        }
    return table


def domain_summary(domain: str, records: list[dict[str, Any]], n_input: int,
                   threshold: float, elapsed: float) -> dict[str, Any]:
    """Aggregate one domain, including the coverage-band histogram."""

    scored = [record for record in records if record["status"] == "ok"]
    errored = [record for record in records if record["status"] == "error"]
    a1_hits = sum(1 for record in scored if record["a1_hit"])
    a2_hits = sum(1 for record in scored if record["a2_hit"])
    queue = [record for record in scored if record["a1_hit"] and not record["a2_hit"]]
    bands = Counter(record["coverage_band"] for record in scored)
    walls = [float(record.get("wall_seconds", 0.0)) for record in records]
    return {
        "n_rows_in_frozen_csv": n_input,
        "n_processed": len(records),
        "n_scored": len(scored),
        "n_error": len(errored),
        "a1_hits": a1_hits,
        "a2_hits": a2_hits,
        "a1_accuracy": accuracy(a1_hits, len(scored)),
        "a2_accuracy": accuracy(a2_hits, len(scored)),
        "manual_review_queue": len(queue),
        "id_difference_kind_counts": dict(sorted(Counter(
            record["id_difference_kind"] for record in queue
        ).items())),
        "coverage_band_histogram": {band: bands.get(band, 0) for band in COVERAGE_BANDS},
        "mean_coverage_on_true_target": mean(
            [float(record["coverage_on_true_target"]) for record in scored]
        ),
        "mean_coverage_any_target": mean(
            [float(record["coverage_best_any_target"]) for record in scored]
        ),
        "true_ids_missing_from_tracks": sum(
            1 for record in records if record["true_ids_missing_from_tracks"]
        ),
        "per_behavior": behavior_summary(records),
        "scale_verification_function": SCALE_VERIFICATION_BY_DOMAIN[domain],
        "coverage_definition": (
            "evidence_seconds / clip_duration_seconds (PROXY, not frame overlap)"
            if domain == "group" else
            "fraction of ALL clip frames where the mask is True on the target"
        ),
        "runtime_seconds": round(elapsed, 2),
        "mean_wall_seconds_per_sample": mean(walls),
        "max_wall_seconds_per_sample": round(max(walls), 3) if walls else None,
        "errors": [
            {"sample_id": record["sample_id"], "message": record["error"]} for record in errored
        ],
        "threshold": threshold,
    }


def print_domain_table(domain: str, summary: dict[str, Any], threshold: float) -> None:
    """Print the compact per-behavior table for one domain as soon as it lands."""

    print(
        f"\n[{domain}] threshold={threshold:.2f} rows={summary['n_processed']} "
        f"scored={summary['n_scored']} errors={summary['n_error']} "
        f"({summary['runtime_seconds']}s, "
        f"{summary['mean_wall_seconds_per_sample']}s/sample)",
        flush=True,
    )
    header = (
        f"  {'behavior':<20}{'support':>8}{'A1':>5}{'A2':>5}"
        f"{'A1%':>8}{'A2%':>8}{'cov_true':>10}{'cov_any':>9}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for behavior, stats in summary["per_behavior"].items():
        print(
            f"  {behavior:<20}{stats['support']:>8}{stats['a1_hits']:>5}{stats['a2_hits']:>5}"
            f"{(stats['a1_accuracy'] or 0.0) * 100:>8.1f}{(stats['a2_accuracy'] or 0.0) * 100:>8.1f}"
            f"{stats['mean_coverage_on_true_target'] or 0.0:>10.4f}"
            f"{stats['mean_coverage_any_target'] or 0.0:>9.4f}",
            flush=True,
        )
    print(
        f"  {'DOMAIN TOTAL':<20}{summary['n_processed']:>8}"
        f"{summary['a1_hits']:>5}{summary['a2_hits']:>5}"
        f"{(summary['a1_accuracy'] or 0.0) * 100:>8.1f}"
        f"{(summary['a2_accuracy'] or 0.0) * 100:>8.1f}"
        f"{summary['mean_coverage_on_true_target'] or 0.0:>10.4f}"
        f"{summary['mean_coverage_any_target'] or 0.0:>9.4f}",
        flush=True,
    )
    print(
        "  coverage bands: "
        + "  ".join(f"{band}={count}" for band, count in summary["coverage_band_histogram"].items()),
        flush=True,
    )
    print(f"  manual review queue: {summary['manual_review_queue']}", flush=True)


def run_domain(domain: str, args: argparse.Namespace, ctx: RunContext,
               params: HeuristicParameters) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every frozen row of one domain; never drop a sample silently."""

    rows = load_domain_rows(domain, ctx)
    n_input = len(rows)
    if args.limit is None and n_input != EXPECTED_ROWS[domain]:
        raise ValueError(
            f"frozen {domain} denominator is {n_input}, expected {EXPECTED_ROWS[domain]}"
        )
    if args.limit is not None:
        rows = rows[: args.limit]
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_started = time.perf_counter()
        try:
            record = SCORERS[domain](row, params, ctx)
        except Exception as exc:
            if args.fail_fast:
                raise
            record = new_record(domain, row)
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        finalize_record(record, args.overlap_threshold)
        record["wall_seconds"] = round(time.perf_counter() - sample_started, 3)
        records.append(record)
        if index % 10 == 0 or index == len(rows):
            print(f"  {domain} {index}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    n_scored = sum(1 for record in records if record["status"] == "ok")
    n_error = len(records) - n_scored
    if n_scored + n_error != len(rows):
        raise AssertionError(
            f"{domain}: n_scored({n_scored}) + n_error({n_error}) != rows({len(rows)})"
        )
    return records, domain_summary(domain, records, n_input, args.overlap_threshold, elapsed)

BASELINE_SOURCE_FILES: tuple[str, ...] = (
    "src/mouse_behavior/behavior/static_rule.py",
    "src/mouse_behavior/behavior/individual_rules.py",
    "src/mouse_behavior/evaluation/confirmed_rules.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v17.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v18.py",
    "src/mouse_behavior/evaluation/event_frames.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v19.py",
    "src/mouse_behavior/evaluation/retrospective_thresholds_v20.py",
    "src/mouse_behavior/evaluation/blind_top1.py",
    "src/mouse_behavior/evaluation/exported_dataset_config.py",
    "src/mouse_behavior/evaluation/group_heuristic_config.py",
    "src/mouse_behavior/evaluation/group_event_rules.py",
    "src/mouse_behavior/evaluation/calibrated_rule_ranker.py",
    "src/mouse_behavior/evaluation/social_heuristic_config.py",
    "src/mouse_behavior/evaluation/exported_dataset_prediction.py",
)


def hash_tree(root: Path) -> dict[str, str]:
    """Hash every ``*.py`` under ``root``, keyed by resolved absolute path."""

    return {str(path.resolve()): file_sha256(path) for path in sorted(root.rglob("*.py"))}


def aggregate_sha256(per_file: dict[str, str]) -> str:
    """Order-stable digest over a ``{path: sha256}`` mapping."""

    blob = "".join(f"{key}\0{value}\0" for key, value in sorted(per_file.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def baseline_source_hashes() -> dict[str, str]:
    """Hash rule, configuration and prediction boundary files BEFORE scoring starts.

    Other agents own ``src/`` now and may edit it while this run is in flight,
    so the baseline is captured at import time to stay attributable.
    """

    hashes: dict[str, str] = {}
    for relative in BASELINE_SOURCE_FILES:
        path = REPO_ROOT / relative
        hashes[relative] = file_sha256(path) if path.is_file() else "MISSING"
    return hashes


def build_provenance(ctx: RunContext, input_csvs: Sequence[Path],
                     params: HeuristicParameters, baseline: dict[str, str],
                     frozen_at: str) -> dict[str, Any]:
    """Everything needed to attribute this baseline to exact bytes on disk."""

    src_files = hash_tree(REPO_ROOT / "src" / "mouse_behavior")
    loading_paths = (
        REPO_ROOT / "scripts" / "_bootstrap.py",
        REPO_ROOT / "scripts" / "validate_social_heuristics.py",
        REPO_ROOT / "scripts" / "validate_group_heuristics.py",
        Path(__file__).resolve(),
        REPO_ROOT / "configs" / "default.yaml",
        ctx.rig_path,
        ctx.v16_dir / "current_validation_summary.json",
    )
    return {
        "source_frozen_at_utc": frozen_at,
        "baseline_source_files_sha256": baseline,
        "input_csv_sha256": {str(path.resolve()): file_sha256(path) for path in input_csvs},
        "src_python_aggregate_sha256": aggregate_sha256(src_files),
        "src_python_file_count": len(src_files),
        "src_python_files": src_files,
        "loading_path_sha256": {
            str(path.resolve()): file_sha256(path) for path in loading_paths if path.is_file()
        },
        "heuristic_parameters_sha256": hashlib.sha256(
            json.dumps(asdict(params), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "individual_config_as_inferred": asdict(individual_config(params)),
        "suspended_posture_behaviors": sorted(SUSPENDED_POSTURE_BEHAVIORS),
        "posture_policy": "manual_review_and_calibration_required_before_canonical_prediction",
        "note": (
            "Scores the frozen v16 sample sets read from "
            f"{ctx.v16_dir.name}; no git metadata is consulted. Denominators are "
            "59 individual / 115 social / 73 group. This is a read-only "
            "measurement of the code as it existed at source_frozen_at_utc."
        ),
    }


def write_json(path: Path, data: Any) -> None:
    """Write JSON exclusively, matching the repository's no-overwrite discipline."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write output rows exclusively, keeping only the declared schema."""

    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI options; every path defaults to a REPO_ROOT-relative derivation."""

    data_root = REPO_ROOT.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Segment-level coverage scoring of the frozen v16 sample sets: a sample counts as "
            "detected when the predicted mask covers >= threshold of the annotated clip's frames."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--overlap-threshold", type=float, default=0.70,
                        help="Coverage fraction of the annotated clip required to count as detected.")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS),
                        help="Which frozen sample sets to score.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; refused if it already exists.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N samples per domain (smoke tests).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Re-raise the first per-sample exception instead of recording it "
                             "as status=error and continuing.")
    parser.add_argument("--data-root", type=Path, default=data_root,
                        help="Read-only root for resolving tracks and frozen scale receipts.")
    parser.add_argument("--v16-dir", type=Path, default=REPO_ROOT / "outputs" / FROZEN_RUN_NAME,
                        help="Frozen v16 validation run directory to read sample sets from.")
    parser.add_argument("--val-csv", type=Path,
                        default=data_root / "outputs/confirmed_calibration_v1_20261010/dataset_v2/val.csv",
                        help="Frozen split manifest used to join social sample_ids.")
    parser.add_argument("--approved-root", type=Path,
                        default=data_root / "outputs/behavior_confirmed_dataset",
                        help="Approved dataset root used by individual source-scale verification.")
    parser.add_argument("--max-tracks", type=int, default=DEFAULT_MAX_TRACKS,
                        help="Track cap passed to load_exported_features.")
    args = parser.parse_args()
    if not 0.0 < args.overlap_threshold <= 1.0:
        parser.error("--overlap-threshold must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_tracks < 1:
        parser.error("--max-tracks must be >= 1")
    return args


def main() -> int:
    """Score the frozen v16 sample sets and write the segment-level artifacts."""

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Captured before any scoring: src/ is owned by other agents from now on and
    # may change on disk while this baseline run is in flight.
    baseline = baseline_source_hashes()
    ctx = RunContext(
        data_root=args.data_root,
        approved_root=args.approved_root,
        v16_dir=args.v16_dir,
        val_csv=args.val_csv,
        rig_path=REPO_ROOT / "configs" / "scale" / "rig_a.json",
        max_tracks=args.max_tracks,
        hashes={},
    )
    input_csvs = [ctx.v16_dir / "individual" / "individual_validation.csv",
                  ctx.v16_dir / "group" / "group_validation.csv",
                  ctx.v16_dir / "social" / "social_validation_predictions.csv", ctx.val_csv]
    input_csvs = [path for path in input_csvs if path.is_file()]
    params = config_seed_parameters(load_config(REPO_ROOT / "configs" / "default.yaml"))
    provenance = build_provenance(ctx, input_csvs, params, baseline, frozen_at)

    print(f"scoring frozen v16 sample sets -> {args.output_dir}", flush=True)
    print(f"threshold={args.overlap_threshold} domains={args.domains} limit={args.limit}", flush=True)
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    run_started = time.perf_counter()
    for domain in args.domains:
        print(f"\n=== domain {domain} ({SCALE_VERIFICATION_BY_DOMAIN[domain]}) ===", flush=True)
        records, summary = run_domain(domain, args, ctx, params)
        all_records.extend(records)
        summaries[domain] = summary
        print_domain_table(domain, summary, args.overlap_threshold)

    scored = [record for record in all_records if record["status"] == "ok"]
    queue = [record for record in scored if record["a1_hit"] and not record["a2_hit"]]
    a1_hits = sum(1 for record in scored if record["a1_hit"])
    a2_hits = sum(1 for record in scored if record["a2_hit"])
    drifted = {
        relative: file_sha256(REPO_ROOT / relative) != digest
        for relative, digest in baseline.items()
        if (REPO_ROOT / relative).is_file()
    }
    summary = {
        "fit_scope": policy_receipt()["fit_scope"],
        "independent_validation": policy_receipt()["independent_validation"],
        "behavior_policy": policy_receipt(),
        "criterion": {
            "detected_when": "predicted coverage of the annotated clip >= overlap_threshold",
            "coverage_equals_temporal_overlap_because": "the exported clip IS the annotated bout",
            "A1_behavior_accuracy_id_agnostic": "threshold met on ANY target",
            "A2_strict_accuracy_with_ids": "threshold met on EXACTLY the annotated target ID set",
            "manual_review_queue": "A1 true and A2 false",
            "accuracy_denominator": "n_scored (status=ok); n_error is reported alongside",
        },
        "threshold": args.overlap_threshold,
        "group_coverage_is_evidence_over_clip_duration_proxy": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_run": str(ctx.v16_dir.resolve()),
        "domains_requested": list(args.domains),
        "limit": args.limit,
        "max_tracks": args.max_tracks,
        "overall": {
            "n_processed": len(all_records),
            "n_scored": len(scored),
            "n_error": len(all_records) - len(scored),
            "a1_hits": a1_hits,
            "a2_hits": a2_hits,
            "a1_accuracy": accuracy(a1_hits, len(scored)),
            "a2_accuracy": accuracy(a2_hits, len(scored)),
            "manual_review_queue": len(queue),
            "id_difference_kind_counts": dict(sorted(Counter(
                record["id_difference_kind"] for record in queue
            ).items())),
            "coverage_band_histogram": {
                band: sum(1 for record in scored if record["coverage_band"] == band)
                for band in COVERAGE_BANDS
            },
        },
        "domains": summaries,
        "errors": [
            {"domain": record["domain"], "sample_id": record["sample_id"], "message": record["error"]}
            for record in all_records if record["status"] == "error"
        ],
        "baseline_source_files_changed_during_run": drifted,
        "runtime_seconds": round(time.perf_counter() - run_started, 2),
        "provenance": provenance,
    }

    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "segment_level_summary.json", summary)
    write_csv(args.output_dir / "segment_level_rows.csv", all_records)
    write_csv(
        args.output_dir / "manual_review_queue.csv",
        sorted(queue, key=lambda record: (record["domain"], record["true_behavior"])),
    )

    print(f"\n=== TOTALS over {len(args.domains)} domain(s) ===", flush=True)
    for domain, stats in summaries.items():
        print(
            f"  {domain:<12} n={stats['n_processed']:>4} scored={stats['n_scored']:>4} "
            f"err={stats['n_error']:>3} A1={stats['a1_hits']:>4} "
            f"({(stats['a1_accuracy'] or 0.0) * 100:5.1f}%)  A2={stats['a2_hits']:>4} "
            f"({(stats['a2_accuracy'] or 0.0) * 100:5.1f}%)  queue={stats['manual_review_queue']:>3}",
            flush=True,
        )
    print(
        f"  {'OVERALL':<12} n={summary['overall']['n_processed']:>4} "
        f"scored={summary['overall']['n_scored']:>4} err={summary['overall']['n_error']:>3} "
        f"A1={a1_hits:>4} ({(summary['overall']['a1_accuracy'] or 0.0) * 100:5.1f}%)  "
        f"A2={a2_hits:>4} ({(summary['overall']['a2_accuracy'] or 0.0) * 100:5.1f}%)  "
        f"queue={len(queue):>3}",
        flush=True,
    )
    print(f"  id_difference_kind: {summary['overall']['id_difference_kind_counts']}", flush=True)
    print(f"  runtime {summary['runtime_seconds']}s -> {args.output_dir}", flush=True)
    if any(drifted.values()):
        print(f"  WARNING baseline source files changed during run: {drifted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
