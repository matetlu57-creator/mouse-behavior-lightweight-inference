"""Retrospective directed-pair geometry audit; never called by prediction.

The source index fixes the 755-case cohort and each case's own pixel scale.
Annotations select audit roles only. Runtime rule code never reads this file,
the source index, annotations, or audit outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features  # noqa: E402


def quantiles(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    levels = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
    return {"n_values": len(a), "quantiles": {
        str(q): float(x) for q, x in zip(levels, np.quantile(a, levels))
    }} if len(a) else {"n_values": 0, "quantiles": {}}


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(np.sum(a * b, axis=-1), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=ROOT / "outputs/behavior_result_review_v25_20261019/case_sources.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/directed_roles_v26_20261019")
    args = parser.parse_args()
    source_bytes = args.sources.read_bytes()
    sources = json.loads(source_bytes)
    aggregate = defaultdict(lambda: defaultdict(list))
    metric_samples = defaultdict(lambda: defaultdict(set))
    choices = {
        "approach": {"head_toward_deg": 69.08646603254526, "body_toward_deg": 72.34189423033574},
        "chase": {"head_toward_deg": 53.17451262688489, "body_toward_deg": 60.95376744220115},
    }
    roles = {"approach": ("Approaching", "Approached"), "chase": ("Chasing", "Chased")}
    counts, missing, records, scale_rows = Counter(), Counter(), [], []
    for sample_id, source in sources.items():
        behavior = source["baseline_prediction"]["canonical_behavior"]
        counts[behavior] += 1
        directory = Path(source["directory"])
        annotation = json.loads((directory / "annotation.json").read_text(encoding="utf-8"))
        scale_rows.append({"sample_id": sample_id, "behavior": behavior, **source["scale"]})
        if behavior not in roles:
            continue
        by_role = {r.get("role_name"): r.get("track_ids", [])
                   for r in annotation.get("participants", [])}
        actor_ids, target_ids = (by_role.get(name, []) for name in roles[behavior])
        if len(actor_ids) != 1 or len(target_ids) != 1:
            missing[behavior] += 1
            records.append({"sample_id": sample_id, "behavior": behavior, "role_missing": True})
            continue
        f = load_exported_features(directory / "tracks.json", fps=source["clip"]["fps"],
                                   max_tracks=10000, scale_config=source["scale"])
        slots = {int(track): i for i, track in enumerate(f.track_ids)}
        actor, target = slots.get(actor_ids[0]), slots.get(target_ids[0])
        if actor is None or target is None:
            missing[behavior + "_track_missing"] += 1
            continue
        points = f.collective_keypoints_cm
        offset = f.centers_cm[:, target] - f.centers_cm[:, actor]
        joint = f.valid[:, actor] & f.valid[:, target]
        head = points[:, actor, 0] - points[:, actor, 3]
        body = points[:, actor, 3] - points[:, actor, 6]
        metrics = {
            "head_toward_deg": np.degrees(np.arccos(np.clip(cosine(head, offset), -1, 1))),
            "body_toward_deg": np.degrees(np.arccos(np.clip(cosine(body, offset), -1, 1))),
            "head_body_deg": np.degrees(np.arccos(np.clip(cosine(head, body), -1, 1))),
            "actor_motion_toward_deg": np.degrees(np.arccos(np.clip(cosine(f.velocity_cm_s[:, actor], offset), -1, 1))),
            "target_motion_ahead_deg": np.degrees(np.arccos(np.clip(cosine(f.velocity_cm_s[:, target], offset), -1, 1))),
            "velocity_similarity_deg": np.degrees(np.arccos(np.clip(cosine(f.velocity_cm_s[:, actor], f.velocity_cm_s[:, target]), -1, 1))),
            "actor_speed_cm_s": f.speed_cm_s[:, actor],
            "target_speed_cm_s": f.speed_cm_s[:, target],
            "distance_cm": np.linalg.norm(offset, axis=1),
        }
        row = {"sample_id": sample_id, "behavior": behavior,
               "actor_id": actor_ids[0], "target_id": target_ids[0],
               "cm_per_pixel": float(f.cm_per_pixel), "frames": len(joint),
               "joint_observed_frames": int(joint.sum()),
               "head_known_frames": int((joint & np.isfinite(head).all(axis=1)
                                         & (np.linalg.norm(head, axis=1) > 1e-9)).sum()),
               "body_known_frames": int((joint & np.isfinite(body).all(axis=1)
                                         & (np.linalg.norm(body, axis=1) > 1e-9)).sum()),
               "metrics": {}}
        row["pose_support"] = {}
        for name, values in metrics.items():
            finite = np.asarray(values)[joint & np.isfinite(values)]
            row["metrics"][name] = quantiles(finite)
            aggregate[behavior][name].extend(finite.tolist())
            if len(finite):
                metric_samples[behavior][name].add(sample_id)
                metric_samples[behavior][name + "_clip_median"].add(sample_id)
                aggregate[behavior][name + "_clip_median"].append(float(np.median(finite)))
            if name in choices[behavior]:
                cap = choices[behavior][name]
                accepted = int((finite <= cap + 1e-9).sum())
                row["pose_support"][name] = {
                    "cap_deg": cap, "n_finite": len(finite), "n_support": accepted,
                    "fraction_of_finite": accepted / len(finite) if len(finite) else None,
                    "fraction_of_joint_observed": accepted / int(joint.sum()) if joint.any() else None,
                }
        records.append(row)
    report = {
        "revision": "2026-10-19-v26-observed-directed-role-audit",
        "fit_scope": "all755_retrospective_parameter_selection_not_independent_validation",
        "source_index_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sample_count": len(sources), "behavior_counts": dict(counts),
        "role_missing": {behavior: missing[behavior] for behavior in roles},
        "annotated_role_track_missing": {behavior: missing[behavior + "_track_missing"] for behavior in roles},
        "role_audited_count": len(records),
        "point_contract": "nose0 ears1,2 neck3 hips4,5 tailroot6; confidence>=0.25; finite actual observed points only",
        "heading_fallback_used": False,
        "frame_selection": "all exported source-case frames; both annotated role visual IDs valid; finite nondegenerate measured operands per metric; no imputation; raw adjacent center motion; no prediction-based frame selection; frame-pooled quantiles and equally weighted clip-median quantiles reported separately",
        "scales_cm_per_pixel": quantiles([r["cm_per_pixel"] for r in scale_rows]),
        "geometry": {behavior: {name: {**quantiles(values), "n_samples": len(metric_samples[behavior][name])}
                                for name, values in group.items()}
                     for behavior, group in aggregate.items()},
        "samples": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "geometry_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "sample_scales.json").write_text(
        json.dumps(scale_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {
        "revision": "2026-10-19-v26-observed-directed-pair-role-support",
        "fit_scope": report["fit_scope"],
        "source_index_sha256": report["source_index_sha256"],
        "geometry_statistics_sha256": hashlib.sha256((args.output / "geometry_statistics.json").read_bytes()).hexdigest(),
        "selection_method": "pooled observed annotated actor-frame99th percentile for each of four measured facing support caps; engineering support envelope, no classification-optimality claim and no independent validation",
        "choices_deg": choices,
        "chasing_role_turn_grace_s": {"value": .5, "source": "retained existing ScreenPolicy.chase_heading_lag_s, not refitted"},
        "unknown_contract": "Missing measured head direction never acquires a role from velocity. Motion can retain an undirected behavior candidate. Measured reverse orientation is contradictory; short interior chase head turns retain behavior but those contradictory rows have unknown roles.",
        "metric_contract": "A1/A2 existing target-ID set metrics do not score actor-target ordering. Directed-role metrics must separately report annotated-role sample support, predicted-role availability/unknown rate, and exact ordered actor,target correctness.",
        "pose_support_by_sample": [{k: row[k] for k in ("sample_id", "behavior", "joint_observed_frames", "head_known_frames", "body_known_frames", "pose_support")}
                                   for row in records if "pose_support" in row],
    }
    (args.output / "parameter_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample_count", "behavior_counts", "role_missing",
                                            "role_audited_count", "scales_cm_per_pixel")}, indent=2))
    for behavior, values in report["geometry"].items():
        print(behavior, json.dumps({k: v for k, v in values.items() if "deg" in k}, indent=2))


if __name__ == "__main__":
    main()
