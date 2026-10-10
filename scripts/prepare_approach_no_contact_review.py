"""Prepare human-review-only, contact-free recrops for rejected Approach cards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    SampleRecord,
    config_seed_parameters,
    load_exported_features,
)
from mouse_behavior.evaluation.manual_review import review_catalog, source_key
from mouse_behavior.evaluation.review_transaction import tree_hashes, write_json
from mouse_behavior.evaluation.screening_signals import ScreenPolicy, pair_phases, runs
from mouse_behavior.evaluation.segment_export import export_segment


BATCH_ID = "approach_no_contact_from_rejected_v1"
STATUS = "approach_no_contact_recut_review_candidate"
FAMILY = "approach_no_contact"
PAYLOADS = ("clip.mp4", "tracks.json", "annotation.json", "metadata.json")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _source_for(row: dict, confirmed_root: Path, source_root: Path) -> Path:
    record = row["record"]
    key = source_key(record["sample_id"])
    backup = confirmed_root / "backups" / key
    if backup.exists():
        receipt_path = backup / "receipt.json"
        sample = backup / "sample"
        if not receipt_path.is_file() or not sample.is_dir():
            raise ValueError(f"incomplete immutable source backup: {backup}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("sample_id") != record["sample_id"] or tree_hashes(sample) != receipt.get(
            "sha256"
        ):
            raise ValueError(f"immutable source backup changed: {backup}")
        return sample

    sample = Path(record["sample_dir"]).resolve()
    if not sample.is_relative_to(source_root.resolve()):
        raise ValueError(f"source sample is outside the configured source dataset: {sample}")
    fingerprints = row.get("source_fingerprints", {})
    for field in ("clip_path", "tracks_path", "annotation_path", "metadata_path"):
        path = Path(record[field])
        expected = fingerprints.get(field)
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f"missing source fingerprint or file: {path}")
        stat = path.stat()
        if expected != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError(f"source changed since screening: {path}")
    return sample


def _record_at_source(raw: dict, source: Path) -> SampleRecord:
    value = copy.deepcopy(raw)
    value.update(
        sample_dir=str(source),
        clip_path=str(source / "clip.mp4"),
        tracks_path=str(source / "tracks.json"),
        annotation_path=str(source / "annotation.json"),
        metadata_path=str(source / "metadata.json"),
        mouse_ids=tuple(raw["mouse_ids"]),
    )
    return SampleRecord(**value)


def _poster(video: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f"cannot create review poster: {destination}")
    finally:
        capture.release()


def _review_id(sample_id: str, window: dict) -> str:
    identity = {
        "sample_id": sample_id,
        "window": {
            key: window[key] for key in ("behavior", "mouse_ids", "start_frame", "end_frame")
        },
        "candidate_family": FAMILY,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _previous_supplemental(state: dict, confirmed_root: Path, manifest_sha: str) -> list[dict]:
    pointer = state.get("supplemental_review_candidates")
    if not pointer:
        return []
    digest = pointer.get("sha256") if isinstance(pointer, dict) else None
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"path", "sha256"}
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(value not in "0123456789abcdef" for value in digest)
        or pointer["path"] != f"candidate_updates/supplemental-{digest}.json"
    ):
        raise ValueError("invalid active supplemental-candidate pointer")
    path = confirmed_root / pointer["path"]
    if _digest(path) != pointer["sha256"]:
        raise ValueError("active supplemental-candidate sidecar changed")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or document.get("base_manifest_sha256") != manifest_sha
        or not isinstance(document.get("cards"), list)
    ):
        raise ValueError("active supplemental-candidate sidecar does not match this review package")
    return copy.deepcopy(document["cards"])


def prepare(
    package_root: Path,
    confirmed_root: Path,
    source_root: Path,
    output_root: Path,
    config_path: Path,
) -> dict:
    package_root = package_root.resolve()
    confirmed_root = confirmed_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite task output: {output_root}")

    manifest_path = package_root / "combined_review_manifest.json"
    decisions_path = confirmed_root / "decisions.json"
    manifest_bytes = manifest_path.read_bytes()
    decisions_bytes = decisions_path.read_bytes()
    manifest_sha = _digest_bytes(manifest_bytes)
    decisions_sha = _digest_bytes(decisions_bytes)
    rows = json.loads(manifest_bytes)
    state = json.loads(decisions_bytes)
    if state.get("manifest_sha256") != manifest_sha:
        raise ValueError("review decisions do not reference the frozen manifest")
    if type(state.get("revision")) is not int or not isinstance(state.get("decisions"), dict):
        raise ValueError("invalid manual-review state")

    config = load_config(config_path)
    parameters = config_seed_parameters(config)
    policy = ScreenPolicy(**config.get("segment_screening", {}))
    row_by_key = {source_key(row["record"]["sample_id"]): row for row in rows}
    base_cards = {card["review_id"]: card for card in review_catalog(rows)}
    prior_cards = _previous_supplemental(state, confirmed_root, manifest_sha)

    task_root = output_root
    media_root = package_root / "dataset" / "Approach" / "approach_no_contact_review_v1"
    task_root.mkdir(parents=True, exist_ok=False)
    created_targets: list[Path] = []
    prepared: list[dict] = []
    omitted: list[dict] = []
    source_checks: dict[str, dict] = {}
    try:
        rejected = [
            (review_id, decision)
            for review_id, decision in state["decisions"].items()
            if decision.get("decision") == "rejected" and decision.get("behavior") == "Approach"
        ]
        for review_id, decision in sorted(rejected):
            card = base_cards.get(review_id)
            row = row_by_key.get(decision.get("source_key"))
            if (
                card is None
                or row is None
                or card["source_key"] != decision.get("source_key")
                or card["sample_id"] != row["record"]["sample_id"]
            ):
                omitted.append(
                    {"source_review_id": review_id, "reason": "frozen_source_identity_mismatch"}
                )
                continue
            if "身体接触" not in decision.get("note", ""):
                omitted.append(
                    {
                        "source_review_id": review_id,
                        "reason": "rejection_not_marked_as_body_contact",
                    }
                )
                continue
            ids = sorted({int(value) for value in decision.get("mouse_ids", [])})
            bounds = (decision.get("start_frame"), decision.get("end_frame"))
            if (
                len(ids) != 2
                or type(bounds[0]) is not int
                or type(bounds[1]) is not int
                or not 0 <= bounds[0] < bounds[1] <= card["frame_count"]
            ):
                omitted.append(
                    {"source_review_id": review_id, "reason": "invalid_rejected_pair_or_range"}
                )
                continue

            source = _source_for(row, confirmed_root, source_root)
            record = _record_at_source(row["record"], source)
            features = load_exported_features(
                record.tracks_path, fps=record.fps, scale_config=config["scale"]
            )
            by_track = {int(track): index for index, track in enumerate(features.track_ids)}
            if any(track not in by_track for track in ids):
                omitted.append(
                    {
                        "source_review_id": review_id,
                        "reason": "target_visual_id_missing_from_tracks",
                    }
                )
                continue
            pair = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(features.pair_i, features.pair_j))
                    if tuple(
                        sorted(
                            (
                                int(features.track_ids[int(left)]),
                                int(features.track_ids[int(right)]),
                            )
                        )
                    )
                    == tuple(ids)
                ),
                None,
            )
            if pair is None:
                omitted.append(
                    {"source_review_id": review_id, "reason": "target_pair_missing_from_tracks"}
                )
                continue

            phases = pair_phases(features, pair, parameters, policy)
            mask = phases["approach"] & ~phases["guarded_contact"]
            valid_windows = [
                (start, end) for start, end in runs(mask) if bounds[0] <= start < end <= bounds[1]
            ]
            if not valid_windows:
                omitted.append(
                    {
                        "source_review_id": review_id,
                        "reason": "no_complete_qualifying_no_contact_approach",
                    }
                )
                continue
            start, end = max(
                valid_windows, key=lambda bounds_: (bounds_[1] - bounds_[0], -bounds_[0])
            )
            if phases["contact"][start:end].any() or phases["guarded_contact"][start:end].any():
                raise ValueError(f"contact or unsafe geometry inside selected crop: {review_id}")
            endpoint_cm = float(phases["nearest_surface_cm"][end - 1])
            if (
                not abs(endpoint_cm - policy.approach_stop_distance_cm)
                <= policy.approach_endpoint_tolerance_cm + 1e-9
            ):
                raise ValueError(
                    f"selected crop endpoint is outside the configured distance band: {review_id}"
                )

            window = {
                "behavior": "Approach",
                "mouse_ids": ids,
                "start_frame": int(start),
                "end_frame": int(end),
            }
            new_id = _review_id(record.sample_id, window)
            target = media_root / new_id
            if target.exists():
                raise FileExistsError(f"refusing to overwrite candidate media: {target}")
            candidate_reason = (
                "原接近片段因出现身体接触未通过；此候选按当前接近证据重裁至身体表面间距 "
                f"{endpoint_cm:.3f} cm，候选范围内无接触且无最短时长门槛。"
            )
            exported = export_segment(
                record,
                target,
                start_frame=int(start),
                end_frame=int(end),
                behavior="Approach",
                mouse_ids=tuple(ids),
                provenance={
                    "candidate": {"status": STATUS},
                    "source_rejection": {
                        "review_id": review_id,
                        "decision": "rejected",
                        "note": decision.get("note", ""),
                        "decision_revision": state["revision"],
                    },
                    "screening": {
                        "batch_id": BATCH_ID,
                        "method": "pair_phases.approach with guarded_contact excluded",
                        "approach_stop_distance_cm": policy.approach_stop_distance_cm,
                        "approach_endpoint_tolerance_cm": policy.approach_endpoint_tolerance_cm,
                        "surface_contact_width_fraction": policy.surface_contact_width_fraction,
                        "reference_mouse_width_cm": float(features.mouse_width_cm),
                        "endpoint_nearest_surface_cm": endpoint_cm,
                        "interval_semantics": "source-clip-local half-open [start, end)",
                        "minimum_duration": None,
                    },
                },
            )
            created_targets.append(target)
            _poster(target / "clip.mp4", target / "target_poster.jpg")
            relative = (Path("dataset") / "Approach" / media_root.name / new_id).as_posix()
            prepared.append(
                {
                    "review_id": new_id,
                    "source_review_id": review_id,
                    "source_key": card["source_key"],
                    "sample_id": record.sample_id,
                    "behavior": "Approach",
                    "mouse_ids": ids,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "fps": float(record.fps),
                    "frame_count": int(record.frame_count),
                    "duration_seconds": (int(end) - int(start)) / float(record.fps),
                    "endpoint_nearest_surface_cm": endpoint_cm,
                    "export_path": relative,
                    "poster_path": (Path(relative) / "target_poster.jpg").as_posix(),
                    "candidate_status": STATUS,
                    "candidate_reason": candidate_reason,
                    "output_sha256": tree_hashes(target),
                    "export_receipt": exported,
                }
            )
            source_checks[review_id] = exported["source_sha256"]
            # A source that changed during export invalidates the complete batch.
            _source_for(row, confirmed_root, source_root)

        if not prepared:
            raise ValueError(
                "no rejected Approach clip has a complete, qualifying contact-free approach window"
            )
        if _digest(decisions_path) != decisions_sha:
            raise ValueError("review decisions changed while candidates were being exported")
        if _digest(manifest_path) != manifest_sha:
            raise ValueError("frozen review manifest changed while candidates were being exported")

        supplemental_document = {
            "schema_version": 1,
            "batch_id": BATCH_ID,
            "base_manifest_sha256": manifest_sha,
            "source_decisions_revision": state["revision"],
            "source_decisions_sha256": decisions_sha,
            "cards": prior_cards
            + [
                {
                    key: item[key]
                    for key in (
                        "review_id",
                        "source_key",
                        "sample_id",
                        "behavior",
                        "mouse_ids",
                        "start_frame",
                        "end_frame",
                        "fps",
                        "frame_count",
                        "export_path",
                        "poster_path",
                        "candidate_status",
                        "candidate_reason",
                        "output_sha256",
                    )
                }
                for item in prepared
            ],
        }
        write_json(task_root / "supplemental_review_candidates.json", supplemental_document)
        report = {
            "schema_version": 1,
            "batch_id": BATCH_ID,
            "status": "pending_human_review",
            "automatic_detector_enabled": False,
            "base_manifest_sha256": manifest_sha,
            "source_decisions_revision": state["revision"],
            "source_decisions_sha256": decisions_sha,
            "config_sha256": _digest(config_path),
            "screening_revision": "v7_actual_approach_first_frame_3p5cm_no_contact",
            "parameters": parameters.as_dict(),
            "screen_policy": asdict(policy),
            "source_checks": source_checks,
            "candidate_count": len(prepared),
            "candidate_duration_seconds": [item["duration_seconds"] for item in prepared],
            "cards": prepared,
            "omitted_count": len(omitted),
            "omitted_candidates": omitted,
            "frame_interval": "source-clip-local half-open [start_frame, end_frame)",
        }
        write_json(task_root / "candidate_manifest.json", report)
        return report
    except BaseException:
        for target in reversed(created_targets):
            if target.is_dir() and target.resolve().is_relative_to(media_root.resolve()):
                shutil.rmtree(target)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--confirmed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    args = parser.parse_args()
    report = prepare(
        args.package_root,
        args.confirmed_root,
        args.source_root,
        args.output_root,
        args.config,
    )
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "omitted_count": report["omitted_count"],
                "candidate_manifest": str(args.output_root / "candidate_manifest.json"),
                "supplemental_document": str(
                    args.output_root / "supplemental_review_candidates.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
