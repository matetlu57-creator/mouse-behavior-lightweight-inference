"""Screen old annotations, then export frame-exact cores into a new dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import REPO_ROOT
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    SampleRecord,
    config_seed_parameters,
    discover_samples,
    exported_dataset_source_digest,
    exported_dataset_source_hashes,
    load_exported_features,
)
from mouse_behavior.evaluation.segment_export import export_segment
from mouse_behavior.evaluation.temporal_screen import SegmentCandidate, screen_record
from mouse_behavior.evaluation.screening_signals import ScreenPolicy, posture_evidence, runs


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_key(sample_id):
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]


def select_windows(windows, all_windows):
    if all_windows:
        return windows
    selected = {}
    # One representative continuous core per behavior/exact ID set. Keep all
    # alternative windows in plan.jsonl so this selection loses no evidence.
    for window in windows:
        key = (window.behavior, window.mouse_ids)
        prior = selected.get(key)
        rank = (window.status == "human_confirmed", window.end_frame-window.start_frame, -window.start_frame)
        if prior is None or rank > prior[0]:
            selected[key] = (rank, window)
    return [row[1] for row in selected.values()]


def posture_episodes(features, policy, config):
    quad = None
    calibration = config.get("scale", {}).get("calibration_file")
    if calibration:
        path = Path(calibration)
        if not path.is_absolute():
            path = REPO_ROOT / path
        geometry = json.loads(path.read_text(encoding="utf-8"))
        if geometry.get("quad_px"):
            quad = np.asarray(geometry["quad_px"]) * features.cm_per_pixel
    masks, metrics = posture_evidence(features, policy, arena_quad_cm=quad)
    result = []
    minimum = max(1, int(np.ceil(policy.posture_min_s * features.fps)))
    for name, values in masks.items():
        for slot, track in enumerate(features.track_ids):
            for start, end in runs(values[:, slot]):
                if end-start < minimum:
                    continue
                if name == "grooming":
                    if metrics["back_groom"][start:end, slot].mean() >= .3:
                        variant = "turned_back_grooming"
                    elif np.nanmedian(metrics["body_length_cm"][start:end, slot]) <= policy.rear_length_fraction * features.reference_body_cm:
                        variant = "partial_standing_foreaft_grooming"
                    else:
                        variant = "foreaft_head_grooming"
                else:
                    wall = metrics.get("wall_distance_cm")
                    variant = "wall_rearing" if wall is not None and np.nanmedian(wall[start:end, slot]) <= features.reference_body_cm * .5 else "open_floor_rearing"
                result.append({"candidate": name, "track_id": int(track), "start_frame": int(start),
                    "end_frame": int(end), "variant": variant, "body_static_verified": True,
                    "median_foreaft_amplitude_cm": float(np.nanmedian(metrics["foreaft_amplitude_cm"][start:end, slot]))})
    return result


def plan(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    params = config_seed_parameters(config)
    policy = ScreenPolicy(**config.get("segment_screening", {}))
    records = discover_samples(args.dataset_root)
    if args.limit:
        records = records[:args.limit]
    audits = {}
    if args.human_audit:
        with args.human_audit.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                audits.setdefault(str(Path(row["clip_path"]).resolve()).casefold(), []).append(row)
    counts = Counter()
    reasons = Counter()
    failures = []
    with (args.output_dir / "plan.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(records, 1):
            episodes = []
            try:
                checks = {name: fingerprint(getattr(record, name)) for name in
                          ("tracks_path", "clip_path", "annotation_path", "metadata_path")}
                if record.behavior in {"Following", "Avoiding"}:
                    windows, reason = [], "behavior_suspended_by_current_rules"
                elif record.behavior in {"Walking", "Running", "Dispersal", "Group locomotion", "Isolation", "Huddling"}:
                    minimum = {"Walking": params.locomotion_min_duration_s,
                               "Running": params.locomotion_min_duration_s,
                               "Dispersal": params.dispersal_min_duration_s,
                               "Group locomotion": params.group_locomotion_min_duration_s,
                               "Isolation": params.isolation_min_duration_s,
                               "Huddling": params.huddle_min_duration_s}[record.behavior]
                    if record.frame_count / record.fps + 1e-9 < minimum:
                        windows, reason = [], "original_duration_below_current_minimum_needs_review"
                    else:
                        label = "Locomotion" if record.behavior in {"Walking", "Running"} else record.behavior
                        windows = [SegmentCandidate(label, tuple(record.mouse_ids), 0, record.frame_count,
                                                    "unchanged_semantics_full_clip", "original_label_preserved")]
                        reason = "preserved"
                else:
                    features = load_exported_features(record.tracks_path, fps=record.fps,
                                                      scale_config=config["scale"])
                    episodes = []
                    audit = []
                    if record.behavior == "Static":
                        episodes = posture_episodes(features, policy, config)
                        audit = audits.get(str(Path(record.sample_dir).resolve()).casefold(), [])
                    windows, reason = screen_record(record, features, params,
                                                     posture_episodes=episodes, audit_rows=audit, policy=policy)
                # Explicit existing human decisions take precedence for any
                # source label; mining is not limited to old Static folders.
                for audited in audits.get(str(Path(record.sample_dir).resolve()).casefold(), []):
                    if audited["decision"] == "confirmed" and audited["observed_behavior"] in {"grooming", "rearing"}:
                        candidate = SegmentCandidate(audited["observed_behavior"].title(), (int(audited["track_id"]),),
                            int(audited["start_frame"]), int(audited["end_frame"]),
                            "human_audit_observed_behavior_" + audited["review_id"], "human_confirmed")
                        if not any(w == candidate for w in windows):
                            windows.append(candidate)
                chosen = select_windows(windows, args.all_windows)
                row = {"record": asdict(record), "source_fingerprints": checks, "reason": reason,
                       "posture_probe_windows": episodes,
                       "all_windows": [w.as_row() for w in windows],
                       "selected_windows": [w.as_row() for w in chosen]}
                for window in chosen:
                    counts[window.behavior] += 1
                reasons[reason] += 1
            except Exception as exc:
                row = {"record": asdict(record), "reason": "screening_error", "all_windows": [],
                       "selected_windows": [], "error": f"{type(exc).__name__}: {exc}"}
                failures.append({"sample_id": record.sample_id, "error": row["error"]})
                reasons["screening_error"] += 1
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(records):
                print(json.dumps({"screened": index, "total": len(records),
                                  "selected": sum(counts.values()), "errors": len(failures)}), flush=True)
    provenance = {"dataset_root": str(args.dataset_root.resolve()), "config": str(args.config.resolve()),
                  "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "parameters": params.as_dict(), "parameter_units": config.get("behavior_parameter_units"),
                  "selection": "all_contiguous_windows" if args.all_windows else "longest_core_per_behavior_exact_ID_set",
                  "boundary_convention": "clip-local half-open; no context padding",
                  "human_audit": str(args.human_audit) if args.human_audit else None,
                  "human_audit_sha256": hashlib.sha256(args.human_audit.read_bytes()).hexdigest() if args.human_audit else None,
                  "screening_revision": "v7_actual_approach_first_frame_3p5cm_tail_contact_chase_human_posture",
                  "screen_policy": asdict(policy),
                  "source_code_sha256": {
                      **{str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__), REPO_ROOT / "scripts/audit_grooming_rearing_primitives.py",
                                      REPO_ROOT / "src/mouse_behavior/evaluation/temporal_screen.py",
                                      REPO_ROOT / "src/mouse_behavior/evaluation/screening_signals.py",
                                      REPO_ROOT / "src/mouse_behavior/evaluation/segment_export.py")},
                      **exported_dataset_source_hashes(REPO_ROOT),
                  },
                  "exported_dataset_sha256": exported_dataset_source_digest(REPO_ROOT),
                  "calibration_sha256": hashlib.sha256((REPO_ROOT / config["scale"]["calibration_file"]).read_bytes()).hexdigest(),
                  "status": "initial_screen_candidates_not_accuracy_evaluation"}
    write_json(args.output_dir / "provenance.json", provenance)
    write_json(args.output_dir / "screening_summary.json", {"input_clips": len(records),
               "input_by_behavior": dict(Counter(r.behavior for r in records)),
               "selected_by_behavior": dict(counts), "selected_segments": sum(counts.values()),
               "sample_reasons": dict(reasons), "errors": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} screening errors; inspect the summary before rendering")


def review_page(output_dir, rows):
    from mouse_behavior.evaluation.manual_review_page import render_review_page

    render_review_page(output_dir, rows)


def target_poster(target, ids, output_dir):
    existing = target / "target_poster.jpg"
    if existing.is_file():
        if cv2.imread(str(existing)) is None:
            raise ValueError(f"existing target poster cannot be decoded: {existing}")
        return existing.relative_to(output_dir).as_posix()
    tracks = json.loads((target / "tracks.json").read_text(encoding="utf-8"))
    if not tracks:
        return None
    middle = len(tracks) // 2
    # Prefer a real frame showing all target IDs; otherwise the most complete
    # observed frame. No missing animal is inferred or drawn.
    def quality(entry):
        seen = {int(d.get("track_id", -1)) for d in entry.get("detections", [])}
        return (len(seen.intersection(ids)), -abs(entry["frame"] - middle))
    entry = max(tracks, key=quality)
    capture = cv2.VideoCapture(str(target / "clip.mp4"))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, entry["frame"])
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot read poster frame: {target}")
    finally:
        capture.release()
    for detection in entry.get("detections", []):
        track = int(detection.get("track_id", -1))
        if track not in ids:
            continue
        points = np.asarray(detection.get("keypoints", []), dtype=float)
        if points.shape != (7, 3):
            continue
        observed = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0), :2]
        if not len(observed):
            continue
        x, y = np.min(observed, axis=0).astype(int)
        x2, y2 = np.max(observed, axis=0).astype(int)
        cv2.rectangle(frame, (x-10, y-10), (x2+10, y2+10), (0, 220, 255), 3)
        cv2.putText(frame, f"track {track}", (max(x-10, 0), max(y-14, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 255), 2)
        for point in observed.astype(int):
            cv2.circle(frame, tuple(point), 4, (0, 220, 255), -1)
    cv2.putText(frame, f"Local frame {entry['frame']}; visual IDs only", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 255), 2)
    poster = target / "target_poster.jpg"
    if not cv2.imwrite(str(poster), frame):
        raise ValueError(f"cannot write poster: {poster}")
    return poster.relative_to(output_dir).as_posix()


def render(args):
    rows = [json.loads(line) for line in (args.output_dir / "plan.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = json.loads((args.output_dir / "provenance.json").read_text(encoding="utf-8"))
    # Human-confirmed posture windows must survive an automatic top-64 track
    # screening limit. Verify the actual original detection ID in the reviewed
    # window; this supplements Static mining without replacing its target.
    applied_audit = []
    audit_path = Path(provenance["human_audit"]) if provenance.get("human_audit") else None
    if audit_path:
        if hashlib.sha256(audit_path.read_bytes()).hexdigest() != provenance["human_audit_sha256"]:
            raise ValueError("human audit changed since screening")
        with audit_path.open(encoding="utf-8-sig", newline="") as audit_stream:
            audit_rows = list(csv.DictReader(audit_stream))
        for row in rows:
            record = row["record"]
            matched = [a for a in audit_rows if str(Path(a["clip_path"]).resolve()).casefold()
                       == str(Path(record["sample_dir"]).resolve()).casefold()
                       and a["decision"] == "confirmed" and a["observed_behavior"] in {"grooming", "rearing"}]
            if not matched:
                continue
            tracks = json.loads(Path(record["tracks_path"]).read_text(encoding="utf-8"))
            for audited in matched:
                start, end, track = int(audited["start_frame"]), int(audited["end_frame"]), int(audited["track_id"])
                if not 0 <= start < end <= record["frame_count"] or not any(
                    start <= entry["frame"] < end and any(int(d.get("track_id", -1)) == track
                    for d in entry.get("detections", [])) for entry in tracks):
                    raise ValueError(f"audited visual ID/window absent from source: {record['sample_id']}")
                candidate = SegmentCandidate(audited["observed_behavior"].title(), (track,), start, end,
                    "human_audit_observed_behavior_" + audited["review_id"], "human_confirmed").as_row()
                candidates = [SegmentCandidate(**{**w, "mouse_ids": tuple(w["mouse_ids"])}) for w in row["all_windows"]]
                if not any(w["behavior"] == candidate["behavior"] and tuple(w["mouse_ids"]) == tuple(candidate["mouse_ids"])
                           and w["start_frame"] == start and w["end_frame"] == end
                           and w["status"] == "human_confirmed" for w in row["all_windows"]):
                    candidates.append(SegmentCandidate(**candidate))
                row["all_windows"] = [w.as_row() for w in candidates]
                row["selected_windows"] = [w.as_row() for w in select_windows(candidates,
                    provenance["selection"] == "all_contiguous_windows")]
                applied_audit.append({"sample_id": record["sample_id"], "review_id": audited["review_id"],
                                      "behavior": candidate["behavior"], "mouse_ids": [track]})
    write_json(args.output_dir / "applied_human_audit.json", applied_audit)
    write_json(args.output_dir / "effective_screening_summary.json", {
        "input_clips": len(rows), "selected_segments": sum(len(r["selected_windows"]) for r in rows),
        "selected_by_behavior": dict(Counter(w["behavior"] for r in rows for w in r["selected_windows"])),
        "status_counts": dict(Counter(w["status"] for r in rows for w in r["selected_windows"])),
        "supplemental_human_audit_windows": len(applied_audit)})
    provenance["render_code_sha256"] = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), REPO_ROOT / "src/mouse_behavior/evaluation/segment_export.py")}
    dataset = args.output_dir / "dataset"
    dataset.mkdir(exist_ok=True)
    sources = args.output_dir / "review_sources"
    sources.mkdir(exist_ok=True)
    receipts = args.output_dir / "render_receipts.jsonl"
    existing = {}
    if receipts.exists():
        for line in receipts.read_text(encoding="utf-8").splitlines():
            receipt = json.loads(line)
            existing[receipt["output_dir"]] = receipt
    failures = []
    rendered = 0
    with receipts.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, 1):
            raw_record = row["record"]
            raw_record["mouse_ids"] = tuple(raw_record["mouse_ids"])
            record = SampleRecord(**raw_record)
            for name, expected in row.get("source_fingerprints", {}).items():
                if fingerprint(getattr(record, name)) != expected:
                    raise ValueError(f"source changed since screening: {record.sample_id} {name}")
            original_copy = sources / (source_key(record.sample_id) + ".mp4")
            if not original_copy.exists():
                shutil.copy2(record.clip_path, original_copy)
            for window in row["selected_windows"]:
                name = f'{Path(record.sample_dir).name}__{window["start_frame"]:06d}-{window["end_frame"]:06d}__ids-' + '-'.join(map(str, window["mouse_ids"]))
                target = dataset / window["behavior"] / name
                try:
                    receipt = existing.get(str(target.resolve()))
                    if receipt:
                        for filename, expected in receipt["output_sha256"].items():
                            if hashlib.sha256((target / filename).read_bytes()).hexdigest() != expected:
                                raise ValueError(f"existing crop changed: {target}")
                    else:
                        receipt = export_segment(record, target, start_frame=window["start_frame"],
                            end_frame=window["end_frame"], behavior=window["behavior"],
                            mouse_ids=tuple(window["mouse_ids"]), provenance={"screen": provenance,
                                "candidate": window, "original_label": record.behavior})
                        stream.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        existing[str(target.resolve())] = receipt
                    window["export_path"] = target.relative_to(args.output_dir).as_posix()
                    window["poster_path"] = target_poster(target, set(window["mouse_ids"]), args.output_dir)
                    rendered += 1
                except Exception as exc:
                    window["render_error"] = f"{type(exc).__name__}: {exc}"
                    failures.append({"sample_id": record.sample_id, "window": window,
                                     "error": window["render_error"]})
            if index % 25 == 0 or index == len(rows):
                print(json.dumps({"rendered_sources": index, "total": len(rows),
                                  "verified_segments": rendered, "errors": len(failures)}), flush=True)
    write_json(args.output_dir / "render_summary.json", {"verified_segments": rendered, "errors": failures})
    write_json(args.output_dir / "review_manifest.json", rows)
    fields = ["sample_id", "source_behavior", "candidate_behavior", "mouse_ids", "start_frame", "end_frame",
              "duration_s", "status", "reason", "export_path"]
    with (args.output_dir / "screening_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = row["record"]
            for window in row["selected_windows"] or [None]:
                writer.writerow({"sample_id": record["sample_id"], "source_behavior": record["behavior"],
                    "candidate_behavior": window["behavior"] if window else "", "mouse_ids": str(window["mouse_ids"]) if window else str(record["mouse_ids"]),
                    "start_frame": window["start_frame"] if window else "", "end_frame": window["end_frame"] if window else "",
                    "duration_s": (window["end_frame"]-window["start_frame"])/record["fps"] if window else "",
                    "status": window["status"] if window else "needs_review_or_suspended", "reason": window["reason"] if window else row["reason"],
                    "export_path": window.get("export_path", "") if window else ""})
    review_page(args.output_dir, rows)
    if failures:
        raise RuntimeError(f"{len(failures)} segment exports failed; see render_summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    parser.add_argument("--human-audit", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.plan_only and args.render_only:
        parser.error("choose at most one phase flag")
    if not args.render_only:
        plan(args)
    if not args.plan_only:
        render(args)


if __name__ == "__main__":
    main()
