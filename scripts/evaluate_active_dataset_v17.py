"""Coverage audit of every active confirmed crop; event rules, never Top-1 ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from score_segment_level_v16 import (  # noqa: E402
    CANONICAL_TO_SOCIAL_MASK, coverage_band, id_difference_kind, individual_config,
)
from mouse_behavior.behavior.individual_rules import individual_rule_masks  # noqa: E402
from mouse_behavior.config import load_config  # noqa: E402
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks, policy_receipt  # noqa: E402
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters  # noqa: E402
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features  # noqa: E402
from mouse_behavior.evaluation.exported_dataset_models import LABEL_TO_CANONICAL  # noqa: E402
from mouse_behavior.evaluation.group_event_rules import group_rule_candidates, isolation_candidates  # noqa: E402
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG  # noqa: E402
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG  # noqa: E402

INDIVIDUAL = {"stationary", "locomotion", "grooming", "rearing"}
SOCIAL = set(CANONICAL_TO_SOCIAL_MASK)
REJECTED = "f33eff112c9c720c1b974f84"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def coverage_acceptance_threshold(behavior, default_threshold=.5, attack_threshold=.3):
    """Behavior-specific scoring only; never modifies predicted event masks."""
    for value in (default_threshold, attack_threshold):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("coverage thresholds must be finite and in (0, 1]")
    return float(attack_threshold if behavior == "attack" else default_threshold)


def validate_membership(entries, pointer):
    ids = [r["review_id"] for r in entries]
    if len(set(ids)) != len(ids) or len(entries) != int(pointer["sample_count"]):
        raise ValueError("Active manifest membership count/uniqueness mismatch")
    if REJECTED in ids or sum(r["behavior"] == "Isolation" for r in entries) != pointer["isolation_count"]:
        raise ValueError("Active v4 exclusion/Isolation count mismatch")


def validate_evaluation_selection(entries, selected):
    """A derived cohort may remove source rows but must never rewrite them."""
    source = {r['review_id']: r for r in entries}
    ids = [r['review_id'] for r in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Evaluation selection must be nonempty and unique')
    if any(source.get(r['review_id']) != r for r in selected):
        raise ValueError('Evaluation selection contains a foreign or modified source row')
    return selected


def validate_v28_selection_contract(entries, selected, contract, source_sha, selection_sha, protect):
    """Bind the requested 30/746 cohort to the original755 and V27 exclusion proof."""
    if contract.get('schema') != 'v28_isolation_selection_contract':
        raise ValueError('Unknown derived cohort contract')
    if contract['source_manifest_sha256'] != source_sha or contract['selection_sha256'] != selection_sha:
        raise ValueError('Cohort source/selection hash mismatch')
    derived = read(protect(contract['derived_manifest_path'], contract['derived_manifest_sha256']))
    proof = read(protect(contract['exclusion_receipt_path'], contract['exclusion_receipt_sha256']))
    validate_evaluation_selection(entries, derived)
    if (len(entries) != 755 or len(derived) != 746
            or sum(r['behavior'] == 'Isolation' for r in derived) != 30
            or proof['source_manifest_sha256'] != source_sha):
        raise ValueError('Cohort original755/derived746/Isolation30 lineage mismatch')
    retained = {r['review_id'] for r in derived}
    removed = [r for r in entries if r['review_id'] not in retained]
    if (len(removed) != 9 or any(r['behavior'] != 'Isolation' for r in removed)
            or {r['review_id'] for r in removed} != {r['sample_id'] for r in proof['excluded']}):
        raise ValueError('Derived membership disagrees with the nine Isolation exclusion receipts')
    kind = contract['selection_kind']
    expected = derived if kind == 'derived746' else [r for r in derived if r['behavior'] == 'Isolation']
    if kind not in ('derived746', 'isolation30') or selected != expected:
        raise ValueError('Selection must exactly equal the bound746 cohort or its complete30 Isolation subset')
    return {'selection_kind': kind, 'source_samples': 755, 'derived_samples': 746,
            'retained_isolation_samples': 30, 'excluded_isolation_samples': 9,
            'selection_samples': len(selected), 'lineage_verified': True}


def padded_features(features, frame_count):
    """Preserve every metadata crop bin; missing trailing detections stay invalid."""
    loaded = len(features.valid)
    if loaded > frame_count:
        raise ValueError("Tracks extend beyond metadata crop frame_count")
    if loaded == frame_count:
        return features
    updates = {}
    for field in fields(features):
        value = getattr(features, field.name)
        if field.name in {"track_ids", "pair_i", "pair_j", "body_length_ref_cm", "body_width_ref_cm", "body_area_ref_cm2", "track_presence_fraction"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == loaded:
            fill = False if value.dtype.kind == "b" else np.nan
            updates[field.name] = np.concatenate((value, np.full((frame_count - loaded, *value.shape[1:]), fill, dtype=value.dtype)))
    return replace(features, **updates)


def mask_candidates(mask, target_sets, metadata_frames):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.shape[0] > metadata_frames or mask.shape[1] != len(target_sets):
        raise ValueError("Mask shape exceeds or disagrees with metadata/target sets")
    return [{"target_ids": list(map(int, ids)), "coverage": float(mask[:, index].sum()) / metadata_frames,
             "evidence_frames": int(mask[:, index].sum())} for index, ids in enumerate(target_sets)]


def group_coverage(evidence_seconds, metadata_frames, fps):
    duration = metadata_frames / fps
    value = float(evidence_seconds) / duration
    if not np.isfinite(value) or value < 0 or value > 1 + 1e-9:
        raise ValueError(f"Invalid group duration proxy {value}; evidence={evidence_seconds}, clip={duration}")
    return value


def group_candidate_coverage(candidate, metadata_frames, fps):
    """Prefer clip-local event frames; retain score-only legacy compatibility."""
    from mouse_behavior.evaluation.event_frames import interval_frame_count
    intervals = getattr(candidate, "frame_intervals", None)
    count = interval_frame_count(intervals, metadata_frames) if intervals is not None else None
    legacy = group_coverage(candidate.evidence, metadata_frames, fps)
    row = {"target_ids": list(map(int, candidate.target_ids)),
           "evidence_seconds": float(candidate.evidence),
           "legacy_evidence_coverage": legacy}
    if intervals is None:
        return {**row, "coverage": legacy, "coverage_source": "legacy_evidence_proxy"}
    row.update(coverage=count / metadata_frames, coverage_source="event_frames",
               event_frame_count=count, event_frame_intervals=intervals)
    support_intervals = getattr(candidate, "support_frame_intervals", None)
    if support_intervals is not None:
        support = interval_frame_count(support_intervals, metadata_frames)
        row.update(support_frame_count=support,
                   support_frame_intervals=support_intervals)
    return row


def score_targets(candidates, expected_ids, threshold, individual=False):
    """Truth is used only here, after label-free rule prediction."""
    truth = set(map(int, expected_ids))
    eligible = [r for r in candidates if r["coverage"] >= threshold]
    qualifying = [sorted(set(r["target_ids"])) for r in eligible]
    union = sorted({i for ids in qualifying for i in ids})
    exact = [r["coverage"] for r in candidates if set(r["target_ids"]) == truth]
    strict = bool(exact and max(exact) >= threshold)
    true_coverage = max(exact, default=0.0)
    best = max(candidates, key=lambda r: r["coverage"], default={"coverage": 0.0, "target_ids": []})
    # Non-annotated sets are off-target bookkeeping, not verified negatives.
    # A clip may contain another unannotated instance of the same behavior.
    off_target = [r for r in candidates if set(r["target_ids"]) != truth]
    return {"a1_hit": bool(eligible), "a2_hit": strict, "qualifying_target_sets": qualifying,
            "non_annotated_qualifying_sets": sum(1 for r in off_target if r["coverage"] >= threshold),
            "non_annotated_candidate_sets": len(off_target),
            "event_presence_a1_hit": any(r["coverage"] > 0 for r in candidates),
            "event_presence_a2_hit": any(r["coverage"] > 0 and set(r["target_ids"]) == truth for r in candidates),
            "qualifying_union_ids": union, "a2_union_ids_exact_diagnostic": bool(eligible) and set(union) == truth,
            "coverage_on_true_target": true_coverage, "coverage_best_any_target": best["coverage"],
            "best_target_ids": best["target_ids"], "coverage_band": coverage_band(true_coverage),
            "id_difference_kind": id_difference_kind(sorted(truth), union) if eligible and not strict else ""}


def off_target_candidate_rejection_fraction(qualifying, candidates):
    """Candidate-set diagnostic; no true-negative or specificity interpretation."""
    if not candidates:
        return None
    return 1.0 - qualifying / candidates


def within_clip_specificity(qualifying, candidates):
    """Historical function alias; returns a candidate fraction, NOT specificity."""
    return off_target_candidate_rejection_fraction(qualifying, candidates)


def summarize(records):
    support = len(records)
    scored = [r for r in records if r["status"] == "ok"]
    a1, a2 = sum(r["a1_hit"] for r in scored), sum(r["a2_hit"] for r in scored)
    wrong_qualifying = sum(int(r.get("non_annotated_qualifying_sets", 0)) for r in scored)
    wrong_candidates = sum(int(r.get("non_annotated_candidate_sets", 0)) for r in scored)
    return {"support": support, "scored": len(scored), "errors": support - len(scored),
            "a1_hits": a1, "a2_hits": a2, "a1_rate_all_input": a1 / support if support else None,
            "a2_rate_all_input": a2 / support if support else None,
            "non_annotated_qualifying_sets_total": wrong_qualifying,
            "non_annotated_candidate_sets_total": wrong_candidates,
            "off_target_candidate_rejection_fraction": off_target_candidate_rejection_fraction(wrong_qualifying, wrong_candidates),
            "within_clip_specificity": None,
            "specificity_available": False,
            "samples_with_any_wrong_target_qualifying": sum(
                int(r.get("non_annotated_qualifying_sets", 0)) > 0 for r in scored),
            "retired_diagnostic_any_nonzero_coverage_a1_hits": sum(bool(r.get("event_presence_a1_hit")) for r in scored),
            "retired_diagnostic_any_nonzero_coverage_a2_hits": sum(bool(r.get("event_presence_a2_hit")) for r in scored),
            "retired_diagnostic_any_nonzero_coverage_a1_recall_on_labeled_positives": sum(bool(r.get("event_presence_a1_hit")) for r in scored) / support if support else None,
            "retired_diagnostic_any_nonzero_coverage_a2_recall_on_labeled_positives": sum(bool(r.get("event_presence_a2_hit")) for r in scored) / support if support else None,
            "sample_duration_seconds_min": min((r["clip_duration_seconds"] for r in scored if "clip_duration_seconds" in r), default=None),
            "sample_duration_seconds_median": float(np.median([r["clip_duration_seconds"] for r in scored if "clip_duration_seconds" in r])) if any("clip_duration_seconds" in r for r in scored) else None,
            "sample_duration_seconds_max": max((r["clip_duration_seconds"] for r in scored if "clip_duration_seconds" in r), default=None),
            "samples_shorter_than_rule_minimum": sum(
                r.get("clip_duration_seconds", float("inf")) <= r.get("rule_min_duration_seconds", 0)
                if r.get("rule_min_duration_strict", False) else
                r.get("clip_duration_seconds", float("inf")) < r.get("rule_min_duration_seconds", 0)
                for r in scored),
            "mean_true_target_coverage_scored": float(np.mean([r["coverage_on_true_target"] for r in scored])) if scored else None}


def minimum_duration(behavior, params, config):
    social, group = DEFAULT_SOCIAL_HEURISTIC_CONFIG, DEFAULT_GROUP_HEURISTIC_CONFIG
    return {"stationary": config.static_min_duration_s, "locomotion": config.locomotion_min_duration_s,
            "grooming": config.grooming_min_duration_s, "rearing": config.rearing_min_duration_s,
            "attack": social.attack_min_duration_s, "approach": social.approach_min_duration_s,
            "chase": social.chasing_min_duration_s, "together": social.together_min_duration_s,
            "nose_head_contact": min(social.nose_contact_min_duration_s, social.contact_head_extension_min_duration_s),
            "nose_tail_contact": min(social.nose_tail_min_duration_s, social.contact_tail_extension_min_duration_s),
            "isolation": max(params.isolation_min_duration_s, group.isolation_min_duration_s),
            "huddle": params.huddle_min_duration_s,
            "group_locomotion": max(params.group_locomotion_min_duration_s, group.locomotion_simultaneous_min_s),
            "social_clustering": group.clustering_final_stability_s,
            "dispersal": (.15 if group.dispersal_extend_actual_motion else params.dispersal_min_duration_s)}.get(behavior, 0.)


def reviewed_features(features, sample, duplicate_pairs):
    """Human-assisted comparator-only diagnostic, without annotation target input."""
    nearest = np.asarray(features.nearest_distance_cm, float).copy()
    ids = list(map(int, features.track_ids))
    masked_comparisons = 0
    for slot, target in enumerate(ids):
        forbidden = {other for s, pair in duplicate_pairs if s == sample and target in pair for other in pair if other != target}
        if not forbidden:
            continue
        for frame in range(len(features.valid)):
            if not features.valid[frame, slot]:
                continue
            values = []
            for other_slot, other in enumerate(ids):
                if other == target or not features.valid[frame, other_slot]:
                    continue
                if other in forbidden:
                    masked_comparisons += 1
                    continue
                value = np.linalg.norm(features.centers_cm[frame, slot] - features.centers_cm[frame, other_slot])
                if np.isfinite(value):
                    values.append(float(value))
            nearest[frame, slot] = min(values) if values else np.nan
    return replace(features, nearest_distance_cm=nearest), masked_comparisons


def configuration_snapshot(params, config):
    """Distinguish serialized legacy mirrors from the configs the rules read."""
    return {
        "parameters": asdict(params),
        "parameters_semantics": (
            "Raw config_seed_parameters snapshot; contains legacy unused social mirror fields. "
            "Social thresholds come from canonical_social_config; group rules read both "
            "parameters and canonical_group_config. Retired social fields are listed in policy_receipt."
        ),
        "individual_config": asdict(config),
        "canonical_social_config": asdict(DEFAULT_SOCIAL_HEURISTIC_CONFIG),
        "canonical_group_config": asdict(DEFAULT_GROUP_HEURISTIC_CONFIG),
        "canonical_config_sources": {
            "social": "mouse_behavior.evaluation.social_heuristic_config.DEFAULT_SOCIAL_HEURISTIC_CONFIG",
            "group": "mouse_behavior.evaluation.group_heuristic_config.DEFAULT_GROUP_HEURISTIC_CONFIG",
            "parameters": "config_seed_parameters(load_config(configs/default.yaml))",
        },
    }


def evaluate(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def protect(path, expected=None):
        path = Path(path).resolve()
        value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError(f"Expected hash mismatch: {path}")
        if str(path) in hashes and hashes[str(path)] != value:
            raise ValueError(f"Protected input changed during read: {path}")
        hashes.setdefault(str(path), value)
        return path

    pointer = read(protect(args.pointer))
    manifest_path = Path(pointer["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    entries = read(protect(manifest_path, pointer["manifest_sha256"]))
    validate_membership(entries, pointer)
    source_count = len(entries)
    selection_path = getattr(args, 'selection_manifest', None)
    selection_hash = None
    selection_lineage = None
    if selection_path is not None:
        selection_path = protect(selection_path)
        selection_hash = hashes[str(selection_path)]
        selected_entries = validate_evaluation_selection(entries, read(selection_path))
        if getattr(args, 'selection_contract', None) is None:
            raise ValueError('Derived V28 cohorts require --selection-contract')
        contract = read(protect(args.selection_contract))
        selection_lineage = validate_v28_selection_contract(
            entries, selected_entries, contract, hashes[str(manifest_path.resolve())], selection_hash, protect)
        entries = selected_entries
    for path in sorted((ROOT / "src").rglob("*.py")):
        protect(path)
    for path in sorted((ROOT / "configs").rglob("*")):
        if path.is_file():
            protect(path)
    protect(Path(__file__))
    protect(ROOT / "scripts/score_segment_level_v16.py")
    protect(manifest_path.parent / "dataset_revision_receipt.json")
    scale_rows = read(protect(args.provenance_root / "dataset_statistics_v2/source_scales.json"))
    scale_by_id = {r["sample_id"]: r for r in scale_rows}
    original_hashes = read(protect(args.provenance_root / "dataset_statistics_v2/protected_hashes.json"))["before"]
    splits = {r["sample_id"]: r["split"] for r in read(protect(args.provenance_root / "dataset_v4_statistics/per_sample.json"))}
    decisions = read(protect(args.provenance_root / "isolation_manual_review/user_review_decisions.json"))
    duplicate_pairs = {(r["sample_id"], frozenset((r["target_id"], r["duplicate_id"]))) for r in decisions["confirmed_duplicate_pairs"]}
    assert len(duplicate_pairs) == 8
    params = config_seed_parameters(load_config(ROOT / "configs/default.yaml"))
    config = individual_config(params)
    if config.enable_posture_screen:
        raise ValueError("Canonical Grooming/Rearing must remain disabled")
    records, secondary, predictions = [], [], []
    start = time.perf_counter()
    selected = entries[:args.limit] if args.limit else entries
    with (args.output_dir / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for index, entry in enumerate(selected):
            sample, behavior = entry["review_id"], LABEL_TO_CANONICAL[entry["behavior"]]
            acceptance = coverage_acceptance_threshold(
                behavior, args.threshold, getattr(args, "attack_threshold", .3))
            domain = "individual" if behavior in INDIVIDUAL else "social" if behavior in SOCIAL else "group"
            record = {"sample_id": sample, "behavior": entry["behavior"], "canonical_behavior": behavior,
                      "domain": domain, "split": splits[sample], "expected_visual_ids": sorted(entry["mouse_ids"]),
                      "acceptance_threshold": acceptance,
                      "status": "error", "error": "", "a1_hit": False, "a2_hit": False,
                      "group_duration_proxy": False, "coverage_on_true_target": 0.0,
                      "canonical_attempted": behavior not in {"grooming", "rearing"},
                      "disabled_reason": "manual_posture_calibration_pending" if behavior in {"grooming", "rearing"} else ""}
            before = time.perf_counter()
            try:
                directory = Path(entry["dataset_path"])
                for name in ("tracks.json", "metadata.json", "annotation.json"):
                    protect(directory / name, entry["dataset_sha256"][name])
                meta, annotation = read(directory / "metadata.json"), read(directory / "annotation.json")
                if annotation["manual_review_id"] != sample or annotation["behavior"] != entry["behavior"] or sorted(annotation["mouse_ids"]) != sorted(entry["mouse_ids"]):
                    raise ValueError("Annotation/manifest join mismatch")
                frames, fps = int(meta["clip"]["frame_count"]), float(meta["clip"]["fps"])
                if frames <= 0 or fps <= 0 or not np.isfinite(fps) or annotation["frame_range"] != {"start": 0, "end": frames - 1}:
                    raise ValueError("Invalid metadata crop extent/FPS")
                scale = scale_by_id[sample]
                source = Path(scale["source_tracks"])
                protect(source, original_hashes[str(source.resolve())])
                if hashes[str(source.resolve())] != meta["provenance"]["source_sha256"]["tracks.json"]:
                    raise ValueError("Source scale provenance mismatch")
                old_scale = ROOT.parent.parent / "outputs/confirmed_calibration_v1_20261010/dataset_v2/scales" / f'{entry["source_key"]}.json'
                if old_scale.exists():
                    protect(old_scale, original_hashes[str(old_scale.resolve())])
                features = load_exported_features(directory / "tracks.json", fps=fps, max_tracks=10000, scale_config=scale["scale_config"])
                loaded = len(features.valid)
                features = padded_features(features, frames)
                ids = list(map(int, features.track_ids))
                if domain == "individual":
                    masks = individual_rule_masks(features.centers_cm, features.valid, fps,
                        keypoints_cm=features.collective_keypoints_cm, reference_body_cm=features.reference_body_cm, config=config).masks
                    candidates = mask_candidates(masks[behavior], [[track] for track in ids], frames)
                elif domain == "social":
                    masks = pair_rule_masks(features, params)
                    targets = [[ids[int(i)], ids[int(j)]] for i, j in zip(features.pair_i, features.pair_j)]
                    candidates = mask_candidates(masks[CANONICAL_TO_SOCIAL_MASK[behavior]], targets, frames)
                else:
                    generated = (isolation_candidates(features, params) if behavior == "isolation" else
                                 group_rule_candidates(features, params, (), behaviors=(behavior,)))
                    candidates = [group_candidate_coverage(c, frames, fps)
                                  for c in generated if c.behavior == behavior]
                    record["group_duration_proxy"] = any(c["coverage_source"] == "legacy_evidence_proxy" for c in candidates)
                    legacy = [{**c, "coverage": c["legacy_evidence_coverage"]} for c in candidates]
                    record["legacy_evidence_coverage_on_true_target"] = score_targets(
                        legacy, entry["mouse_ids"], args.threshold)["coverage_on_true_target"]
                prediction = {"sample_id": sample, "canonical_behavior": behavior,
                              "acceptance_threshold": acceptance, "candidates": candidates}
                predictions.append(prediction)
                record.update(score_targets(candidates, entry["mouse_ids"], acceptance, domain == "individual"))
                record.update(status="ok", metadata_frames=frames, loader_frames=loaded, fps=fps,
                              clip_duration_seconds=frames / fps, rule_min_duration_seconds=minimum_duration(behavior, params, config),
                              true_ids_missing_from_tracks=sorted(set(entry["mouse_ids"]) - set(ids)),
                              elapsed_seconds=time.perf_counter() - before)
                if behavior == "group_locomotion":
                    record.update(
                        rule_min_duration_strict=params.group_locomotion_min_duration_s >= DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_simultaneous_min_s,
                        simultaneous_support_min_duration_seconds=DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_simultaneous_min_s,
                        process_min_duration_seconds=params.group_locomotion_min_duration_s,
                        process_min_duration_comparison=">",
                    )
                elif behavior == "dispersal":
                    record.update(process_min_duration_seconds=params.dispersal_min_duration_s,
                                  process_min_duration_comparison=">",
                                  confirmation_min_delay_seconds=5.0,
                                  confirmation_delay_comparison=">",
                                  confirmation_requires_measured_departed_members=True)
                elif behavior == "social_clustering":
                    record.update(final_spatial_stability_min_seconds=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_final_stability_s,
                                  final_state_uses_instantaneous_quiet_speed=False,
                                  requires_observed_actor_arrival=True,
                                  actor_min_mean_speed_cm_s=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_speed_cm_s,
                                  actor_min_path_cm=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_min_path_cm,
                                  actor_path_comparison='>',
                                  actor_min_net_displacement_cm=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_min_displacement_cm)
                    record.update(
                        formation_unknown_memory_s=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_unknown_gap_s,
                        formation_motion_lookback_s=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_motion_lookback_s,
                        formation_min_toward_fraction=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_min_toward_fraction,
                        formation_min_observed_fraction=DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_min_observed_fraction,
                        final_stability_requires_contiguous_observed_run=True,
                        formation_outputs_measured_prefix=True,
                        unknown_frames_contribute_event_or_motion=False)
                elif behavior in ('nose_head_contact', 'nose_tail_contact'):
                    kind = 'head' if behavior == 'nose_head_contact' else 'tail'
                    record.update(
                        baseline_min_duration_seconds=DEFAULT_SOCIAL_HEURISTIC_CONFIG.contact_minimum_s(kind),
                        short_extension_min_duration_seconds=(DEFAULT_SOCIAL_HEURISTIC_CONFIG.contact_head_extension_min_duration_s
                            if kind == 'head' else DEFAULT_SOCIAL_HEURISTIC_CONFIG.contact_tail_extension_min_duration_s),
                        rule_min_duration_semantics='minimum of enabled independently qualified baseline and strict short-extension channels')
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}", a1_hit=False, a2_hit=False,
                              elapsed_seconds=time.perf_counter() - before)
            if behavior == "isolation":
                # A diagnostic failure must never invalidate a successful raw prediction.
                reviewed = {"sample_id": sample, "behavior": entry["behavior"], "split": record["split"],
                            "status": "error", "error": "", "a1_hit": False, "a2_hit": False,
                            "coverage_on_true_target": 0.0, "human_assisted_reviewed_comparator_only": True}
                if record["status"] != "ok":
                    reviewed.update(error=record["error"], error_stage="primary_unavailable")
                else:
                    try:
                        assisted, masked = reviewed_features(features, sample, duplicate_pairs)
                        generated = isolation_candidates(assisted, params)
                        corrected = [group_candidate_coverage(c, frames, fps) for c in generated]
                        reviewed.update(status="ok", symmetric_directed_comparisons_masked=masked,
                                        **score_targets(corrected, entry["mouse_ids"], args.threshold))
                    except Exception as exc:
                        reviewed.update(error=f"{type(exc).__name__}: {exc}", error_stage="secondary_reviewed_comparator")
                secondary.append(reviewed)
            records.append(record)
            progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            if (index + 1) % 25 == 0:
                print(json.dumps({"processed": index + 1, "total": len(selected), "elapsed_s": round(time.perf_counter() - start, 1),
                                  "errors": sum(r["status"] == "error" for r in records)}), flush=True)
    after = {path: sha256(path) for path in hashes}
    changed = [path for path in hashes if hashes[path] != after[path]]
    summaries = {}
    for split in ("all", "train", "val", "unassigned"):
        subset = [r for r in records if split == "all" or r["split"] == split]
        summaries[split] = {"overall": summarize(subset),
                            "per_behavior": {name: summarize([r for r in subset if r["behavior"] == name]) for name in sorted({r["behavior"] for r in entries})},
                            "per_domain": {name: summarize([r for r in subset if r["domain"] == name]) for name in ("individual", "social", "group")}}
    report = {"schema": "active_dataset_v17_event_coverage", "manifest_path": str(manifest_path.resolve()),
              "manifest_sha256": hashes[str(manifest_path.resolve())], "active_manifest_samples": source_count, "processed_samples": len(records),
              "evaluation_manifest_samples": len(entries),
              "selection_manifest_path": str(selection_path) if selection_path else None,
              "selection_manifest_sha256": selection_hash,
              "selection_is_exact_unmodified_subset": selection_path is not None,
              "selection_lineage": selection_lineage,
              "threshold": args.threshold, "threshold_is_global": False,
              "attack_threshold": getattr(args, "attack_threshold", .3),
              "threshold_by_behavior": {label: coverage_acceptance_threshold(canonical, args.threshold,
                  getattr(args, "attack_threshold", .3)) for label, canonical in LABEL_TO_CANONICAL.items()},
              "threshold_is_acceptance_criterion": True,
              "acceptance_criterion": "a qualified event whose coverage of the annotated segment is >= threshold for that behavior: Attack defaults to0.30, all other behaviors to0.50; A1 ignores target-ID errors, A2 requires the exact annotated target set; a changed threshold is not an algorithm gain",
              "retired_metric_note": "the former event_presence(coverage>0) vs coverage>=0.70 duality is retired as an acceptance criterion on 2026-10-18 by user decision; the any-nonzero-coverage counts are kept as a diagnostic only and must not be quoted as accuracy",
              "specificity_scope": "unavailable: non-annotated target sets are not verified negatives; the off-target candidate rejection fraction is bookkeeping only, not population specificity or a false-positive rate",
              "is_top1_ranking": False, "coverage_type": "predicted event frames over annotated crop frames; legacy evidence proxy only for score-only candidates",
              "group_coverage_is_evidence_over_metadata_duration_proxy": any(r["group_duration_proxy"] for r in records),
              "group_frame_contract": "half-open crop-local frame intervals: huddle counts compact support, isolation coobserved support; clustering emits observed formation prefix plus measured dense-state support, unknown-only memory contributes no event frames, travel, time or cross-gap displacement; group motion retains confirmed process intervals with simultaneous support exported separately; dispersal exports actual departure phase. These are predictions, not manual frame truth or IoU",
              "group_values_over_one_are_errors_never_capped": True,
              "all_error_samples_retained_in_denominators": True, "primary_A2_all_domains": "a qualifying candidate has exactly annotated target set; inherited frozen247 target-candidate convention",
              "A2_union_exact_secondary_diagnostic": "union of all qualifying candidate IDs equals annotated ID set; stricter than primary A2 and distinct from event presence",
              "retired_diagnostic_any_nonzero_coverage_metric": "RETIRED as an acceptance criterion: positive nonzero already-qualified mask/evidence, i.e. recall on these labeled positive samples only; kept for artifact continuity, never to be quoted as accuracy",
              "within_clip_specificity_metric": "retired misleading name; always null because verified negatives are unavailable",
              "off_target_candidate_rejection_fraction_metric": "1 - (non-annotated sets reaching threshold / non-annotated candidate sets); no verified negative class; not specificity",
              "annotation_used_only_after_generation_for_scoring": True, "canonical_posture_enabled": config.enable_posture_screen,
              **configuration_snapshot(params, config), "policy_receipt": policy_receipt(),
              "samples_shorter_than_rule_minimum_semantics": "insufficient crop duration; includes equality for strict process-span minima; group_locomotion requires simultaneous support >= its separate minimum and process span > its process minimum",
              "summary": summaries,
              "attempt_policy_summaries": {
                  "primary_all_inputs": summarize(records),
                  "enabled_behaviors_only_secondary": summarize([r for r in records if r["canonical_attempted"]]),
                  "disabled_posture_separate": summarize([r for r in records if not r["canonical_attempted"]]),
                  "note": f"{len(entries)} evaluation-manifest denominator; source approved manifest retains {source_count}; disabled posture remains separately labeled and is not silently excluded"},
              "isolation_secondary_human_assisted_reviewed_comparator_only": summarize(secondary),
              "runtime_seconds": time.perf_counter() - start, "protected_inputs_unchanged": not changed,
              "changed_protected_paths": changed, "errors": [r for r in records if r["status"] == "error"],
              "isolation_secondary_errors": [r for r in secondary if r["status"] == "error"],
              "full_dataset_run": not args.limit and selection_path is None,
              "full_evaluation_manifest_run": not args.limit,
              "split_counts": dict(Counter(r["split"] for r in records))}
    write_json(args.output_dir / "summary.json", report)
    write_json(args.output_dir / "records.json", records)
    write_json(args.output_dir / "predicted_target_coverage.json", predictions)
    write_json(args.output_dir / "isolation_secondary_reviewed.json", secondary)
    write_json(args.output_dir / "protected_hashes.json", {"before": hashes, "after": after, "changed": changed})
    print(json.dumps({"output": str(args.output_dir.resolve()), "summary": summaries["all"], "runtime_seconds": report["runtime_seconds"], "changed_sources": changed}), flush=True)
    return report


def build_parser():
    """Attack30 and other-label50 coverage acceptance, independently overridable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, default=ROOT / "configs/datasets/current_confirmed_dataset.json")
    parser.add_argument("--provenance-root", type=Path, default=ROOT / "outputs/heuristics_v17_20261016_run01")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--attack-threshold", type=float, default=.3,
                        help="Attack-only acceptance coverage; use0.5 to reproduce the previous scoring policy")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; never described as a full-dataset result")
    parser.add_argument("--selection-manifest", type=Path,
                        help="Derived evaluation cohort; rows must exactly match the protected approved manifest")
    parser.add_argument("--selection-contract", type=Path, help="Frozen30/746 cohort lineage and hash contract")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (not 0 < args.threshold <= 1 or not 0 < args.attack_threshold <= 1
            or args.limit < 0):
        parser.error("coverage thresholds must be (0,1], limit non-negative")
    report = evaluate(args)
    if report["changed_protected_paths"] or report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
