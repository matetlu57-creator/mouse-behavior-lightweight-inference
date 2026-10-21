"""Read-only, full-dataset retrospective heuristic measurements (v17).

No threshold is selected here. Full extrema include observed zeros; split
statistics are descriptive, not a claim of held-out threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mouse_behavior.evaluation.confirmed_dataset import source_reference_pixels  # noqa: E402
from mouse_behavior.evaluation.confirmed_rules import _local_quadratic_point_kinematics  # noqa: E402
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features  # noqa: E402
from mouse_behavior.evaluation.exported_dataset_models import LABEL_TO_CANONICAL  # noqa: E402
from mouse_behavior.evaluation.scale_migration import (  # noqa: E402
    NEW_BODY_LENGTH_CM, NEW_BODY_WIDTH_CM, rescale_frozen_source_scale,
)
from mouse_behavior.evaluation.social_event_geometry import (  # noqa: E402
    endpoint_distances, snout_surface_distances,
)
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG  # noqa: E402

METRIC_BEHAVIORS = {"Approach", "Attack", "Chasing", "Snout-head_contact",
                    "Snout-rear_contact", "Together", "Isolation"}
ROLES = {"Approach": ("Approaching", "Approached"), "Attack": ("Attacker", "Attacked"),
         "Chasing": ("Chasing", "Chased"),
         "Snout-head_contact": ("Snout_contacting", "Head_contacted"),
         "Snout-rear_contact": ("Snout_contacting", "Rear_contacted")}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.mkdir(parents=True, exist_ok=False)


def split_rows(train: Path, val: Path) -> dict[str, dict[str, str]]:
    """Exact review-ID join; source keys are deliberately not join keys."""
    result = {}
    for path, split in ((train, "train"), (val, "val")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["sample_id"]
                if key in result:
                    raise ValueError(f"Duplicate/overlapping split sample_id: {key}")
                if row["split"] not in ({"train"} if split == "train" else {"val", "validation"}):
                    raise ValueError(f"Incorrect split in {path}: {key}")
                result[key] = {**row, "split": split}
    return result


def exact_boundary(values: np.ndarray, frame: int) -> float:
    """Missing boundary remains missing, even if a nearby frame is observed."""
    if not 0 <= frame < len(values):
        return float("nan")
    return float(values[frame])


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if row["value"] is not None]
    values = np.asarray([row["value"] for row in finite], float)
    result = {"total_count": len(rows), "finite_count": len(finite),
              "missing_count": len(rows) - len(finite),
              "zero_count": int(np.count_nonzero(values == 0)),
              "positive_count": int(np.count_nonzero(values > 0))}
    for key in ("min", "max", "p1", "p5", "p50", "p95", "p99", "positive_min"):
        result[key] = None
    if len(values):
        result.update(min=float(values.min()), max=float(values.max()))
        result.update({f"p{p}": float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)})
        positive = values[values > 0]
        result["positive_min"] = float(positive.min()) if len(positive) else None
        result["min_witness"] = min(finite, key=lambda row: row["value"])
        result["max_witness"] = max(finite, key=lambda row: row["value"])
    # Separate diagnostic: it must never replace the global finite minimum.
    peaks = defaultdict(list)
    for row in finite:
        peaks[row["sample_id"]].append(row["value"])
    result["minimum_of_per_sample_peaks"] = min(map(max, peaks.values())) if peaks else None
    result["samples_with_finite_values"] = len(peaks)
    return result


def directions(annotation: dict, ids: list[int]) -> tuple[list[tuple[int, int]], str]:
    roles = ROLES.get(annotation["behavior"])
    if roles:
        participants = annotation.get("participants", [])
        matches = [[int(i) for p in participants if p.get("role_name") == role
                    for i in p.get("track_ids", [])] for role in roles]
        if all(len(set(items)) == 1 for items in matches):
            actor, target = matches[0][0], matches[1][0]
            if actor != target and {actor, target} == set(ids):
                return [(actor, target)], "unique_annotated_roles"
    return [(ids[0], ids[1]), (ids[1], ids[0])], "ambiguous_roles_best_of_both"


def observations(entry: dict, annotation: dict, metadata: dict, features: Any,
                 split: str) -> list[dict[str, Any]]:
    """Emit each measurement with a local/source frame and visual-ID witness."""
    result = []
    sample, behavior = entry["review_id"], entry["behavior"]
    count = int(metadata["clip"]["frame_count"])
    slots = {int(value): index for index, value in enumerate(features.track_ids)}
    ids = [int(value) for value in entry["mouse_ids"]]

    def emit(metric, unit, frame, value, actor=None, target=None, **extra):
        result.append({"sample_id": sample, "source_key": entry["source_key"],
                       "behavior": behavior, "split": split, "metric": metric, "unit": unit,
                       "frame": int(frame),
                       "source_frame": int(metadata["source"]["start_frame"]) + int(frame),
                       "actor_visual_track_id": actor, "target_visual_track_id": target,
                       "value": float(value) if np.isfinite(value) else None, **extra})

    if behavior not in METRIC_BEHAVIORS:
        return result
    if behavior == "Isolation":
        if len(ids) != 1:
            raise ValueError(f"Isolation needs exactly one annotated target: {sample}")
        target = ids[0]
        for frame in range(count):
            distances = []
            if target in slots and frame < len(features.valid) and features.valid[frame, slots[target]]:
                for other, slot in slots.items():
                    if other != target and features.valid[frame, slot]:
                        distance = np.linalg.norm(features.centers_cm[frame, slots[target]] - features.centers_cm[frame, slot])
                        if np.isfinite(distance):
                            distances.append((float(distance), other))
            distance, other = min(distances) if distances else (float("nan"), None)
            emit("isolation_nearest_other_distance", "cm", frame, distance, target, other)
        return result
    if len(ids) != 2:
        raise ValueError(f"Pair behavior needs exactly two annotated IDs: {sample}")
    pairs = [edge for edge, (i, j) in enumerate(zip(features.pair_i, features.pair_j))
             if {int(features.track_ids[i]), int(features.track_ids[j])} == set(ids)]
    edge = pairs[0] if pairs else None
    pair_valid = np.zeros(len(features.valid), bool)
    if all(value in slots for value in ids):
        pair_valid = features.valid[:, slots[ids[0]]] & features.valid[:, slots[ids[1]]]
    selected, policy = directions(annotation, ids)

    def directed(array_pair, actor, target, frame):
        if edge is None:
            return float("nan")
        left = int(features.track_ids[features.pair_i[edge]])
        return exact_boundary(array_pair[0 if actor == left else 1], frame)

    if behavior == "Approach":
        frame = count - 1
        distance = exact_boundary(features.pair_distance_cm[:, edge], frame) if edge is not None else float("nan")
        emit("approach_last_center_distance", "cm", frame, distance, *ids)
        surfaces = snout_surface_distances(features, edge) if edge is not None else None
        chosen = []
        for actor, target in ((ids[0], ids[1]), (ids[1], ids[0])):
            value = directed(surfaces, actor, target, frame)
            emit("approach_last_snout_surface_both_directions", "cm", frame, value, actor, target,
                 role_policy=policy, annotated_direction=(actor, target) in selected)
            if (actor, target) in selected:
                chosen.append((value, actor, target))
        valid_chosen = [item for item in chosen if np.isfinite(item[0])]
        value, actor, target = min(valid_chosen) if valid_chosen else (float("nan"), None, None)
        emit("approach_last_snout_surface_selected", "cm", frame, value, actor, target, role_policy=policy)
    elif behavior in {"Snout-head_contact", "Snout-rear_contact"}:
        endpoint = "head" if behavior == "Snout-head_contact" else "tail"
        measured = endpoint_distances(features, edge, endpoint) if edge is not None else None
        values = []
        for actor, target in ((ids[0], ids[1]), (ids[1], ids[0])):
            value = directed(measured, actor, target, 0)
            emit(f"nose_{endpoint}_first_both_directions", "cm", 0, value, actor, target,
                 role_policy=policy, annotated_direction=(actor, target) in selected)
            if (actor, target) in selected:
                values.append((value, actor, target))
        finite = [item for item in values if np.isfinite(item[0])]
        value, actor, target = min(finite) if finite else (float("nan"), None, None)
        emit(f"nose_{endpoint}_first_selected", "cm", 0, value, actor, target, role_policy=policy)
        both = [(directed(measured, a, b, 0), a, b) for a, b in ((ids[0], ids[1]), (ids[1], ids[0]))]
        finite = [item for item in both if np.isfinite(item[0])]
        value, actor, target = min(finite) if finite else (float("nan"), None, None)
        emit(f"nose_{endpoint}_first_best_direction_diagnostic", "cm", 0, value, actor, target)
    elif behavior == "Together":
        for frame in range(count):
            value = exact_boundary(features.pair_distance_cm[:, edge], frame) if edge is not None else float("nan")
            emit("together_pair_center_distance", "cm", frame, value, *ids)
    else:
        for actor in ids:
            target = ids[1] if actor == ids[0] else ids[0]
            for frame in range(count):
                # Loader's speed=0 at non-motion-valid frames is a sentinel,
                # not an observed zero velocity. Preserve genuine stationary zeros.
                observed = (actor in slots and frame < len(pair_valid) and pair_valid[frame]
                            and features.motion_valid is not None and features.motion_valid[frame, slots[actor]])
                value = features.speed_cm_s[frame, slots[actor]] if observed else float("nan")
                emit("attack_body_speed" if behavior == "Attack" else "chase_body_speed",
                     "cm/s", frame, value, actor, target)
                pair_observed = actor in slots and frame < len(pair_valid) and pair_valid[frame]
                raw = features.speed_cm_s[frame, slots[actor]] if pair_observed else float("nan")
                emit("attack_body_speed_raw_loader" if behavior == "Attack" else "chase_body_speed_raw_loader",
                     "cm/s", frame, raw, actor, target,
                     motion_observed=bool(observed),
                     loader_unmeasured_zero_sentinel=bool(pair_observed and not observed and raw == 0))
            if behavior != "Attack":
                continue
            if actor not in slots:
                for point_name in ("nose", "head_centroid"):
                    for suffix, unit in (("velocity", "cm/s"), ("acceleration", "cm/s^2")):
                        for frame in range(count):
                            emit(f"attack_{point_name}_{suffix}", unit, frame, float("nan"), actor, target)
                continue
            head = features.collective_keypoints_cm[:, slots[actor], :3]
            complete = np.isfinite(head).all(axis=(1, 2))
            head_center = np.where(complete[:, None], head.mean(axis=1), np.nan)
            for point_name, positions in (("nose", head[:, 0]), ("head_centroid", head_center)):
                cfg = DEFAULT_SOCIAL_HEURISTIC_CONFIG
                velocity, acceleration = _local_quadratic_point_kinematics(
                    positions, features.valid[:, slots[actor]], features.fps,
                    half_window_s=cfg.attack_acceleration_fit_half_window_s,
                    max_jump_speed_cm_s=cfg.attack_max_track_jump_speed_cm_s)
                for suffix, unit, vectors in (("velocity", "cm/s", velocity), ("acceleration", "cm/s^2", acceleration)):
                    norms = np.linalg.norm(vectors, axis=1)
                    norms[~pair_valid] = np.nan
                    for frame in range(count):
                        emit(f"attack_{point_name}_{suffix}", unit, frame,
                             exact_boundary(norms, frame), actor, target)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    new_output(args.output_dir)
    hashes = {}

    def protect(path):
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise RuntimeError(f"Protected input changed between uses: {path}")
        hashes.setdefault(str(path), digest)
        return path

    manifest_path = protect(args.dataset_root / "dataset_manifest.json")
    entries = read_json(manifest_path)
    if len({entry["review_id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate manifest review_id")
    splits = split_rows(protect(args.train_csv), protect(args.val_csv))
    rig = protect(args.rig)
    for path in sorted((ROOT / "src").rglob("*.py")):
        protect(path)
    protect(Path(__file__))
    for path in sorted((ROOT / "configs").rglob("*")):
        if path.is_file():
            protect(path)
    all_rows, sample_rows, scales, warnings = [], [], [], []
    for index, entry in enumerate(entries):
        sample = entry["review_id"]
        directory = Path(entry["dataset_path"])
        annotation, metadata = None, None
        for name in ("tracks.json", "annotation.json", "metadata.json"):
            path = protect(directory / name)
            if hashes[str(path)] != entry["dataset_sha256"][name]:
                raise ValueError(f"Manifest asset hash mismatch: {path}")
            if name == "annotation.json":
                annotation = read_json(path)
            elif name == "metadata.json":
                metadata = read_json(path)
        receipt_path = directory / "review_receipt.json"
        if receipt_path.exists():
            protect(receipt_path)
            if hashes[str(receipt_path.resolve())] != entry["dataset_sha256"]["review_receipt.json"]:
                raise ValueError(f"Manifest receipt hash mismatch: {sample}")
        else:
            warnings.append({"sample_id": sample, "warning": "review_receipt_missing_in_packaged_dataset"})
        if (entry.get("decision") != "approved" or annotation.get("manual_review_id") != sample
                or annotation["behavior"] != entry["behavior"]
                or sorted(annotation["mouse_ids"]) != sorted(entry["mouse_ids"])):
            raise ValueError(f"Manifest/annotation join mismatch: {sample}")
        frames, fps = int(metadata["clip"]["frame_count"]), float(metadata["clip"]["fps"])
        if frames <= 0 or not math.isfinite(fps) or fps <= 0 or annotation["frame_range"] != {"start": 0, "end": frames - 1}:
            raise ValueError(f"Invalid crop frame range/fps: {sample}")
        original = splits.get(sample)
        split = original["split"] if original else "unassigned"
        if original and (original["canonical_behavior"] != LABEL_TO_CANONICAL[entry["behavior"]]
                         or original["source_key"] != entry["source_key"]):
            raise ValueError(f"Exact split join disagrees with manifest: {sample}")
        key = entry["source_key"]
        frozen = args.scales_dir / f"{key}.json"
        if frozen.exists():
            receipt = read_json(protect(frozen))
            config, migration = rescale_frozen_source_scale(receipt["scale_config"], rig)
            source_path = Path(receipt["source_tracks_path"])
            if not source_path.is_absolute():
                source_path = args.project_root / source_path
            protect(source_path)
            if hashes[str(source_path.resolve())] != receipt["source_tracks_sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {key}")
            scale_source = "frozen_source_receipt_migrated_74cm"
        else:
            source_path = args.backups_dir / key / "sample" / "tracks.json"
            if not source_path.is_file():
                raise FileNotFoundError(f"No frozen scale or original source backup for {sample}: {source_path}")
            protect(source_path)
            reference = source_reference_pixels(source_path)
            config = {"mode": "fixed", "cm_per_pixel": NEW_BODY_LENGTH_CM / reference,
                      "assumed_mouse_body_length_cm": NEW_BODY_LENGTH_CM,
                      "assumed_mouse_body_width_cm": NEW_BODY_WIDTH_CM}
            migration = {"reference_body_px": reference, "frozen_receipt_available": False}
            scale_source = "original_source_backup_body_length_7.918cm"
        if hashes[str(source_path.resolve())] != metadata["provenance"]["source_sha256"]["tracks.json"]:
            raise ValueError(f"Original source/metadata hash mismatch: {sample}")
        scales.append({"sample_id": sample, "source_key": key, "scale_source": scale_source,
                       "scale_config": config, "migration": migration, "source_tracks": str(source_path.resolve())})
        rows = []
        if entry["behavior"] in METRIC_BEHAVIORS:
            features = load_exported_features(directory / "tracks.json", fps=fps, max_tracks=10000, scale_config=config)
            rows = observations(entry, annotation, metadata, features, split)
            all_rows.extend(rows)
        sample_rows.append({"sample_id": sample, "behavior": entry["behavior"], "split": split,
                            "frame_count": frames, "fps": fps, "measurement_count": len(rows),
                            "scale_source": scale_source})
        if (index + 1) % 100 == 0:
            print(f"measured/validated {index + 1}/{len(entries)}", flush=True)
    after = {path: sha256(Path(path)) for path in hashes}
    changed = [path for path in hashes if hashes[path] != after[path]]
    json_write(args.output_dir / "protected_hashes.json", {"before": hashes, "after": after, "changed": changed})
    if changed:
        raise RuntimeError(f"Protected inputs changed during measurement: {changed}")
    summary = {}
    metric_units = {row["metric"]: row["unit"] for row in all_rows}
    for split in ("all", "train", "val", "unassigned"):
        grouped = defaultdict(list)
        for row in all_rows:
            if split == "all" or row["split"] == split:
                grouped[row["metric"]].append(row)
        summary[split] = {metric: {"unit": unit, **describe(grouped[metric]),
                                   "unmeasured_loader_zero_sentinel_count": sum(bool(row.get("loader_unmeasured_zero_sentinel")) for row in grouped[metric])}
                          for metric, unit in metric_units.items()}
    report = {"schema": "dataset_heuristic_statistics_v17", "fit_scope": "full_dataset_retrospective_extrema",
              "thresholds_selected": False, "split_role": "descriptive_only",
              "manifest_sample_count": len(entries), "unique_source_count": len({e["source_key"] for e in entries}),
              "requested_metric_sample_count": sum(e["behavior"] in METRIC_BEHAVIORS for e in entries),
              "other_behavior_manifest_only_count": sum(e["behavior"] not in METRIC_BEHAVIORS for e in entries),
              "behavior_counts": dict(Counter(e["behavior"] for e in entries)),
              "split_counts": dict(Counter(row["split"] for row in sample_rows)),
              "scale_source_counts": dict(Counter(row["scale_source"] for row in scales)),
              "boundary_policy": "exact crop first/last frame; missing remains missing",
              "body_speed_policy": "adjacent observed pair frames and actor motion_valid; genuine zeros retained",
              "raw_loader_body_speed_policy": "all pair-valid frames; first/no-adjacent zero sentinels explicitly flagged as unmeasured",
              "kinematic_fit": {"half_window_s": DEFAULT_SOCIAL_HEURISTIC_CONFIG.attack_acceleration_fit_half_window_s,
                                "max_jump_speed_cm_s": DEFAULT_SOCIAL_HEURISTIC_CONFIG.attack_max_track_jump_speed_cm_s},
              "warnings": warnings, "statistics": summary, "protected_inputs_unchanged": True}
    json_write(args.output_dir / "statistics.json", report)
    json_write(args.output_dir / "per_sample.json", sample_rows)
    per_sample_metrics = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        per_sample_metrics[row["sample_id"]][row["metric"]].append(row)
    json_write(args.output_dir / "per_sample_metric_statistics.json", {
        sample: {metric: {"unit": metric_units[metric], **describe(rows)} for metric, rows in metrics.items()}
        for sample, metrics in per_sample_metrics.items()})
    json_write(args.output_dir / "source_scales.json", scales)
    with (args.output_dir / "measurements.jsonl").open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def active_dataset_root(repository: Path = ROOT) -> Path:
    """Resolve and hash-check the active membership manifest, never scan assets."""
    pointer = repository / "configs/datasets/current_confirmed_dataset.json"
    if pointer.is_file():
        config = read_json(pointer)
        relative = Path(config["manifest_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("active manifest must be a repository-relative path")
        manifest = repository / relative
        if manifest.name != "dataset_manifest.json" or sha256(manifest) != config["manifest_sha256"]:
            raise ValueError("active dataset manifest hash/name mismatch")
        return manifest.parent
    packaged = repository / "behavior_confirmed_dataset_cleaned_20261014_original_video_v3"
    return packaged if packaged.is_dir() else repository.parent.parent / "outputs" / packaged.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = ROOT.parent.parent
    dataset = active_dataset_root()
    frozen = project / "outputs" / "confirmed_calibration_v1_20261010" / "dataset_v2"
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--dataset-root", type=Path, default=dataset)
    parser.add_argument("--train-csv", type=Path, default=frozen / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=frozen / "val.csv")
    parser.add_argument("--scales-dir", type=Path, default=frozen / "scales")
    parser.add_argument("--backups-dir", type=Path, default=project / "outputs" / "behavior_confirmed_dataset" / "backups")
    parser.add_argument("--rig", type=Path, default=ROOT / "configs" / "scale" / "rig_a.json")
    args = parser.parse_args()
    output_existed = args.output_dir.exists()
    try:
        report = run(args)
    except Exception as exc:
        if not output_existed and args.output_dir.is_dir() and not (args.output_dir / "error.json").exists() and not (args.output_dir / "statistics.json").exists():
            json_write(args.output_dir / "error.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "samples": report["manifest_sample_count"],
                      "protected_inputs_unchanged": report["protected_inputs_unchanged"]}))


if __name__ == "__main__":
    main()
