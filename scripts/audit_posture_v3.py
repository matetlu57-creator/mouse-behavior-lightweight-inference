#!/usr/bin/env python3
"""Replay a frozen posture manifest without fitting or changing any source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from _bootstrap import ensure_importable

ensure_importable()

from mouse_behavior.behavior.posture_adapter import load_posture_clip  # noqa: E402
from mouse_behavior.behavior.posture_rules import (  # noqa: E402
    DEFAULT_GROOMING_CONFIG, DEFAULT_REARING_CONFIG, POSTURE_REVISION, infer_posture,
)

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(counts):
    tp, fp, fn = (counts.get(key, 0) for key in ("tp", "fp", "fn"))
    return dict(counts, precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen V3 split_manifest_v31.json")
    parser.add_argument("--output", type=Path, required=True, help="New audit directory")
    parser.add_argument("--baseline-results", type=Path, help="Optional frozen *_class_balanced_final.json directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    groups = {key: value for key, value in manifest.items() if isinstance(value, list)}
    records = [(group, record) for group, rows in groups.items() for record in rows]
    for _, record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", sample_id):
            raise ValueError("sample_id must be a safe nonempty file component")
    if not records or len({record["sample_id"] for _, record in records}) != len(records):
        raise ValueError("manifest must contain nonempty, unique sample records")
    args.output.mkdir(parents=True)
    counters, confusion = defaultdict(lambda: defaultdict(Counter)), defaultdict(Counter)
    summaries, parity_errors, input_hashes = [], [], {}
    intervention_changes = Counter()
    for index, (group, record) in enumerate(records):
        directory, sample_id = Path(record["path"]), record["sample_id"]
        for name in ("metadata.json", "tracks.json", "annotation.json"):
            path = directory / name
            input_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        clip = load_posture_clip(directory, record["track_id"])
        prediction = infer_posture(clip)
        write_json(args.output / f"{sample_id}.json", prediction)
        labels = np.asarray(prediction["resolution"]["frame_labels"])
        confusion[group].update({"frames": len(labels), "unknown": int((labels == "Unknown").sum())})
        for behavior in ("Grooming", "Rearing"):
            truth = record["truth"] == behavior
            exportable = np.zeros(len(labels), bool)
            for event in prediction["resolution"]["events"]:
                if event["behavior"] == behavior:
                    exportable[event["start_frame"]:event["end_frame_exclusive"]] = True
            for kind in ("raw", "resolved", "exportable"):
                mask = (np.asarray(prediction["candidates"][behavior]["frame_candidate"], bool) if kind == "raw"
                        else exportable if kind == "exportable" else labels == behavior)
                counters[group][kind+"_"+behavior].update(tp=int(mask.sum()) if truth else 0,
                    fp=int(mask.sum()) if not truth else 0, fn=int((~mask).sum()) if truth else 0,
                    tn=int((~mask).sum()) if not truth else 0)
        if args.baseline_results:
            old = json.loads((args.baseline_results / f"{sample_id}_class_balanced_final.json").read_text(encoding="utf-8"))
            for behavior in ("Grooming", "Rearing"):
                for key in ("frame_candidate", "events", "candidate_frame_count", "clip_candidate"):
                    if old["candidates"][behavior][key] != prediction["candidates"][behavior][key]:
                        parity_errors.append(dict(sample_id=sample_id, behavior=behavior, field=key))
            if old["resolution"]["frame_labels"] != prediction["resolution"]["frame_labels"]:
                parity_errors.append(dict(sample_id=sample_id, field="resolved_frame_labels"))
        # Run both interventions on every positive clip. The complete opposite
        # candidate ledger (including scores/events) must remain byte-identical.
        if record["truth"] in {"Grooming", "Rearing"}:
            strict_r = infer_posture(clip, rearing=replace(DEFAULT_REARING_CONFIG, min_neck_excursion=1e6))
            strict_g = infer_posture(clip, grooming=replace(DEFAULT_GROOMING_CONFIG, min_relative_amplitude=1e6))
            assert strict_r["candidates"]["Grooming"] == prediction["candidates"]["Grooming"], "Rearing parameters leaked into Grooming"
            assert strict_g["candidates"]["Rearing"] == prediction["candidates"]["Rearing"], "Grooming parameters leaked into Rearing"
            for behavior, changed in (("Grooming", strict_g), ("Rearing", strict_r)):
                intervention_changes[behavior] += int(np.count_nonzero(
                    np.asarray(prediction["candidates"][behavior]["frame_candidate"]) !=
                    np.asarray(changed["candidates"][behavior]["frame_candidate"])))
        summaries.append(dict(sample_id=sample_id, split=group, truth=record["truth"],
                              track_id=record["track_id"], label=prediction["resolution"]["label"],
                              candidates={name: value["candidate_frame_count"] for name, value in prediction["candidates"].items()},
                              resolved_event_count=len(prediction["resolution"]["events"])))
        if (index+1) % 20 == 0:
            LOGGER.info("Replayed %d/%d clips", index+1, len(records))
    input_changes = [path for path, digest in input_hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    receipt = dict(revision=POSTURE_REVISION, sample_count=len(records), frame_metrics={
        group: {name: metrics(count) for name, count in values.items()} for group, values in counters.items()},
        unknown=dict(confusion), sample_results=summaries, source_manifest=str(args.manifest.resolve()),
        source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        baseline_parity_error_count=len(parity_errors), baseline_parity_errors=parity_errors,
        independent_intervention_changed_frames=dict(intervention_changes),
        opposite_candidate_ledger_unchanged=True, protected_input_count=len(input_hashes), changed_inputs=input_changes,
        evaluation_scope="frozen full-fit replay, not new out-of-fold or independent validation; not A1/A2",
        status="pass" if not parity_errors and not input_changes else "fail")
    write_json(args.output / "audit_receipt.json", receipt)
    write_json(args.output / "input_sha256.json", input_hashes)
    LOGGER.info("Audit %s: %d samples, %d parity errors, %d changed inputs", receipt["status"], len(records), len(parity_errors), len(input_changes))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
