"""Reconcile a frozen candidate run; preserve V3 and defer Static conflicts.

This is an explicit derived stage, not a rerun of the upstream rule engine.
Only Static, Grooming, Rearing and their conflicts are optionally exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from mouse_behavior.behavior.review_reconciliation import defer_static_posture_conflicts
from mouse_behavior.canonical_video_pipeline import TrackStore, rule_execution_binding
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.segment_export import _sha256, _write_json
from mouse_behavior.visualization.behavior_packages import export_packages, write_manifests


def spread_select(events, limit):
    """Deterministic time-spread selection, without duplicating short queues."""
    groups = {}
    for event in events:
        groups.setdefault(event["canonical_behavior"], []).append(event)
    selected = []
    for rows in groups.values():
        rows.sort(key=lambda row: (row["start_frame"], row["end_frame"], row["event_id"]))
        if len(rows) > limit:
            indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
            rows = [rows[index] for index in indices]
        selected.extend(rows)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--per-behavior", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--codec", choices=("libx264", "h264_nvenc"), default="libx264")
    args = parser.parse_args()
    if args.per_behavior < 1 or args.workers < 1:
        parser.error("per-behavior and workers must be positive")
    upstream = args.inference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    names = ("events.json", "input_binding.json", "scale_receipt.json", "tracks.sqlite")
    original_hashes = {name: _sha256(upstream / name) for name in names}
    receipt_path = upstream / "run_receipt_compact.json"
    if not receipt_path.exists():
        receipt_path = upstream / "run_receipt.json"
    # Never accidentally load an unbounded multi-gigabyte diagnostic receipt.
    if receipt_path.stat().st_size > 20_000_000:
        raise ValueError("Use a compact upstream receipt preserving payload_sha256")
    receipt_hash = _sha256(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name, digest in original_hashes.items():
        if receipt.get("payload_sha256", {}).get(name) != digest:
            raise ValueError(f"upstream payload hash mismatch: {name}")
    binding = json.loads((upstream / "input_binding.json").read_text(encoding="utf-8"))
    if binding.get("posture_mode") != "v3_review":
        raise ValueError("upstream run must explicitly bind original V3 posture candidates")
    scale = json.loads((upstream / "scale_receipt.json").read_text(encoding="utf-8"))
    events = json.loads((upstream / "events.json").read_text(encoding="utf-8"))
    video = Path(binding["source_video"]["path"])
    if _sha256(video) != binding["source_video"]["sha256"]:
        raise ValueError("source video changed")
    params = HeuristicParameters(**binding["effective_parameters"])
    formal, review, audit = defer_static_posture_conflicts(events, binding["source_video"]["fps"], params)
    stage = dict(stage="v32_posture_conflict_reconciliation_only", upstream=str(upstream),
        upstream_payload_sha256=original_hashes, upstream_receipt_sha256=receipt_hash,
        fresh_full_inference=False, preserve_original_v3=True, failed_strict_trial_promoted=False,
        scope=["stationary", "grooming", "rearing", "posture_review"],
        script_sha256=_sha256(Path(__file__)))
    derived = deepcopy(binding)
    # The new snapshot binds the actual reconciliation/export runtime. The old
    # inference identity remains separately preserved, never relabelled as new.
    derived.update(rule_execution_binding(params, "v3_review"))
    derived["screening_stage"] = stage
    output.mkdir(parents=True)
    _write_json(output / "upstream_input_binding.json", binding)
    _write_json(output / "input_binding.json", derived)
    _write_json(output / "scale_receipt.json", scale)
    _write_json(output / "formal_events.json", formal)
    _write_json(output / "posture_review_events.json", review)
    _write_json(output / "conflict_audit.json", audit)
    relevant = [e for e in formal if e["canonical_behavior"] in {"stationary", "grooming", "rearing"}]
    selected = spread_select(relevant + review, args.per_behavior)
    _write_json(output / "selection.json", selected)
    rows = []
    if args.export:
        rows = export_packages(output / "clips", video, TrackStore(upstream / "tracks.sqlite"), selected,
            derived, scale, workers=args.workers, clip_context_seconds=.25, original_video=True,
            clip_codec=args.codec, clip_preset="ultrafast")
    write_manifests(output, rows)
    if original_hashes != {name: _sha256(upstream / name) for name in names}:
        raise ValueError("upstream inputs changed during reconciliation")
    if _sha256(video) != binding["source_video"]["sha256"]:
        raise ValueError("source video changed during export")
    _write_json(output / "run_receipt.json", dict(status="complete", screening_stage=stage,
        counts_before=dict(Counter(e["canonical_behavior"] for e in events)),
        counts_after=dict(Counter(e["canonical_behavior"] for e in formal)),
        review_count=len(review), selected_counts=dict(Counter(e["canonical_behavior"] for e in selected)),
        clip_count=len(rows), source_and_upstream_preserved=True,
        payload_sha256={p.name: _sha256(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(output=str(output), review_count=len(review), clips=len(rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
