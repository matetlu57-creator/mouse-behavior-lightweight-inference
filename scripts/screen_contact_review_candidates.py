"""Export geometry-supported original contact labels for attack-overlap review.

This is a quarantine review set, never an automatic replacement for pure-contact
cores. A high acceleration signal alone does not establish an attack truth label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from _bootstrap import REPO_ROOT
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    config_seed_parameters, discover_samples, load_exported_features, predict_features,
)
from mouse_behavior.evaluation.segment_export import export_segment
from mouse_behavior.evaluation.temporal_screen import contiguous_windows
from screen_behavior_segments import target_poster, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="existing parent screening package; this new subdirectory must not exist")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    destination = root / "contact_overlap_review"
    destination.mkdir(exist_ok=args.resume)
    config = load_config(args.config)
    params = config_seed_parameters(config)
    rows = []
    existing = {}
    receipts = destination / "receipts.jsonl"
    if args.resume and receipts.exists():
        for line in receipts.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["source_sample_id"]] = row
    counts = Counter()
    errors = []
    records = [r for r in discover_samples(args.dataset_root)
               if r.behavior in {"Snout-head_contact", "Snout-rear_contact"}]
    with receipts.open("a" if args.resume else "x", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(records, 1):
            try:
                prior = existing.get(record.sample_id)
                if prior:
                    target = Path(prior["receipt"]["output_dir"])
                    for filename, expected in prior["receipt"]["output_sha256"].items():
                        if hashlib.sha256((target / filename).read_bytes()).hexdigest() != expected:
                            raise ValueError(f"existing contact crop changed: {target}")
                    for filename, expected in prior["receipt"]["source_sha256"].items():
                        if hashlib.sha256((Path(record.sample_dir) / filename).read_bytes()).hexdigest() != expected:
                            raise ValueError(f"source contact sample changed: {record.sample_id}")
                    rows.append(prior)
                    counts[record.behavior] += 1
                    continue
                features = load_exported_features(record.tracks_path, fps=record.fps,
                                                  scale_config=config["scale"])
                evidence = predict_features(features, params, temporal_only=True)
                ids = tuple(record.mouse_ids)
                if ids not in evidence["pair_ids"] or len(features.valid) != record.frame_count:
                    continue
                pair = evidence["pair_ids"].index(ids)
                kind = "nose_head_contact" if record.behavior == "Snout-head_contact" else "nose_tail_contact"
                mask = evidence["pair_masks"][kind][:, pair]
                windows = contiguous_windows(mask, record.fps, params.contact_min_cumulative_seconds)
                if not windows:
                    continue
                start, end = max(windows, key=lambda w: (w[1]-w[0], -w[0]))
                attack_overlap = float(evidence["pair_masks"]["attack"][start:end, pair].mean())
                candidate = {"behavior": record.behavior, "mouse_ids": list(ids), "start_frame": start,
                             "end_frame": end, "status": "contact_geometry_attack_overlap_needs_review",
                             "reason": "original_contact_label_and_continuous_contact_geometry; attack_gate_not_human_truth",
                             "attack_gate_overlap_fraction": attack_overlap}
                target = destination / record.behavior / (
                    f'{Path(record.sample_dir).name}__{start:06d}-{end:06d}__ids-' + '-'.join(map(str, ids)))
                receipt = export_segment(record, target, start_frame=start, end_frame=end,
                    behavior=record.behavior, mouse_ids=ids, provenance={"candidate": candidate,
                        "set_status": "quarantine_review_only_not_training_truth", "parameters": params.as_dict(),
                        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                candidate["export_path"] = target.relative_to(root).as_posix()
                candidate["poster_path"] = target_poster(target, set(ids), root)
                row = {"source_sample_id": record.sample_id, "candidate": candidate, "receipt": receipt}
                rows.append(row)
                counts[record.behavior] += 1
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            except Exception as exc:
                errors.append({"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"})
            if index % 25 == 0 or index == len(records):
                print(json.dumps({"contact_sources": index, "total": len(records),
                                  "verified_contact_review_crops": len(rows), "errors": len(errors)}), flush=True)
    write_json(destination / "summary.json", {"input_contact_samples": len(records),
        "verified_contact_review_crops": len(rows), "by_behavior": dict(counts), "errors": errors,
        "status": "quarantine_review_only_not_training_truth"})
    write_json(destination / "manifest.json", rows)
    if errors:
        raise RuntimeError(f"{len(errors)} contact review exports failed")


if __name__ == "__main__":
    main()
