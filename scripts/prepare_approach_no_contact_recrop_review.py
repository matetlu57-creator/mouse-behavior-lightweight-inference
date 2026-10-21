"""Register the verified 60-clip no-contact Approach recrop batch for review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.manual_review import PAYLOADS, ReviewStore, source_key


RECROP_STATUS = "approach_no_contact_recrop_v1_review_candidate"
BATCH_ID = "approach_no_contact_from_rejected_v2"
PUBLISHED_PAYLOADS = (*PAYLOADS, "target_poster.jpg")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace a different review artifact: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def _verify_item(item, store, state, recrop_root: Path):
    current = state.get("decisions", {}).get(item["review_id"])
    if (
        not current
        or current.get("decision") != "rejected"
        or current.get("behavior") != "Approach"
        or current.get("note", "") != item.get("decision_note", "")
        or current.get("updated_at") != item.get("decision_updated_at")
    ):
        raise ValueError(f"source rejection changed since recropping: {item['review_id']}")

    key = item["source_key"]
    source = store.sources.get(key)
    if (
        not source
        or source["record"]["sample_id"] != item["sample_id"]
        or key != source_key(item["sample_id"])
    ):
        raise ValueError(
            f"source sample is absent from the active review package: {item['sample_id']}"
        )
    raw = source["record"]
    source_candidate = (
        store.package / "dataset" / "Approach" / "approach_contact_review_v1" / item["review_id"]
    )
    candidate_hashes = item.get("source_sha256", item.get("source_hashes", {}))
    for name, expected in candidate_hashes.items():
        path = source_candidate / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"frozen contact-review input changed: {path}")

    candidate_metadata = _read_json(source_candidate / "metadata.json")
    candidate_provenance = candidate_metadata.get("provenance", {})
    frozen_source_hashes = candidate_provenance.get("source_sha256", {})
    if candidate_provenance.get("source_sample_id") != item["sample_id"]:
        raise ValueError(f"frozen source identity mismatch: {item['review_id']}")
    backup_dir = store.root / "backups" / key / "sample"
    selected_source_dir = backup_dir if backup_dir.is_dir() else Path(raw["sample_dir"])
    for name in PAYLOADS:
        expected = frozen_source_hashes.get(name)
        path = selected_source_dir / name
        if not expected or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"source/backup no longer matches the frozen review input: {path}")

    if item["status"] != "exported":
        return None
    if (
        item["physical_contact_frames_in_crop"] != 0
        or item["target_box_overlap_frames_in_crop"] != 0
    ):
        raise ValueError(f"contact appears in the candidate crop: {item['review_id']}")
    if item["nearest_surface_distance_end_cm"] < 3.5:
        raise ValueError(f"candidate ends inside the 3.5 cm cutoff: {item['review_id']}")

    start, end = item["recropped_interval_original_clip_local_half_open"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= raw["frame_count"]
        or item["frame_count"] != end - start
        or item["fps"] != raw["fps"]
    ):
        raise ValueError(f"invalid source-frame range or timeline: {item['review_id']}")

    source_candidate = recrop_root / item["output_path"]
    expected_outputs = item["output_sha256"]
    for name, expected in expected_outputs.items():
        path = source_candidate / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"recrop output hash mismatch: {path}")

    annotation = _read_json(source_candidate / "annotation.json")
    metadata = _read_json(source_candidate / "metadata.json")
    tracks = _read_json(source_candidate / "tracks.json")
    provenance = metadata.get("provenance", {})
    offsets = provenance.get("offsets", {}).get("source_clip_local_half_open")
    source_timeline = metadata.get("source", {})
    frozen_source_metadata = _read_json(selected_source_dir / "metadata.json")
    frozen_source_start = frozen_source_metadata.get("source", {}).get("start_frame")
    absolute_start, absolute_end = item["recording_interval_absolute_half_open"]
    if (
        annotation.get("behavior") != "Approach"
        or annotation.get("mouse_ids") != item["mouse_ids"]
        or annotation.get("frame_range") != {"start": 0, "end": item["frame_count"] - 1}
        or len(tracks) != item["frame_count"]
        or [frame.get("frame") for frame in tracks] != list(range(item["frame_count"]))
        or metadata.get("clip", {}).get("frame_count") != item["frame_count"]
        or provenance.get("source_sample_id") != item["sample_id"]
        or offsets
        not in (
            {"start": 0, "end": item["frame_count"]},
            {"start": start, "end": end},
        )
        or type(frozen_source_start) is not int
        or frozen_source_start + start != absolute_start
        or source_timeline.get("start_frame") != absolute_start
        or source_timeline.get("end_frame") != absolute_end - 1
        or not provenance.get("target_id_overlay")
    ):
        raise ValueError(f"recrop provenance or ID overlay is incomplete: {item['review_id']}")

    identity = {
        "sample_id": item["sample_id"],
        "window": {
            "behavior": "Approach",
            "mouse_ids": item["mouse_ids"],
            "start_frame": start,
            "end_frame": end,
        },
        "candidate_family": "approach_no_contact",
    }
    review_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    duration = (end - start) / item["fps"]
    reason = (
        f"从接近拒绝片段重裁；时长 {duration:.3f} 秒，终点表面间距 "
        f"{item['nearest_surface_distance_end_cm']:.2f} cm；裁剪内身体接触帧 0、目标框重叠帧 0。"
        "目标 ID 框已绘入候选视频；短暂插值框以虚线和 * 标记，tracks.json 保留原始轨迹。"
    )
    return {
        "review_id": review_id,
        "source_key": key,
        "sample_id": item["sample_id"],
        "behavior": "Approach",
        "mouse_ids": item["mouse_ids"],
        "start_frame": start,
        "end_frame": end,
        "fps": raw["fps"],
        "frame_count": raw["frame_count"],
        "candidate_status": RECROP_STATUS,
        "candidate_reason": reason,
        "source_candidate_dir": source_candidate,
        "payload_hashes": {name: expected_outputs[name] for name in PUBLISHED_PAYLOADS},
    }


def _publish_media(cards, store):
    batch_dir = store.package / "dataset" / "Approach" / "approach_no_contact_recrop_v1"
    batch_dir.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = {card["review_id"] for card in cards}
    if batch_dir.exists():
        if {child.name for child in batch_dir.iterdir() if child.is_dir()} != expected_ids:
            raise FileExistsError(
                f"review media directory already exists with different contents: {batch_dir}"
            )
        for card in cards:
            target = batch_dir / card["review_id"]
            for name, expected in card["payload_hashes"].items():
                path = target / name
                if not path.is_file() or _sha256_file(path) != expected:
                    raise FileExistsError(f"existing review media differs: {path}")
    else:
        with tempfile.TemporaryDirectory(
            prefix=".approach-recrops-", dir=batch_dir.parent
        ) as temporary:
            staged = Path(temporary) / batch_dir.name
            staged.mkdir()
            for card in cards:
                target = staged / card["review_id"]
                target.mkdir()
                for name in PUBLISHED_PAYLOADS:
                    shutil.copyfile(card["source_candidate_dir"] / name, target / name)
            staged.rename(batch_dir)

    result = []
    for card in cards:
        relative = Path("dataset") / "Approach" / batch_dir.name / card["review_id"]
        target = batch_dir / card["review_id"]
        hashes = {name: _sha256_file(target / name) for name in PUBLISHED_PAYLOADS}
        if hashes != card["payload_hashes"]:
            raise ValueError(f"published review media failed hash validation: {card['review_id']}")
        result.append(
            {
                key: value
                for key, value in card.items()
                if key not in {"source_candidate_dir", "payload_hashes"}
            }
            | {
                "export_path": relative.as_posix(),
                "poster_path": (relative / "target_poster.jpg").as_posix(),
                "output_sha256": hashes,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--confirmed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recrop-root", type=Path, required=True)
    parser.add_argument(
        "--publish", action="store_true", help="publish the candidates to the active review store"
    )
    args = parser.parse_args()

    recrop_root = args.recrop_root.resolve()
    manifest_path = recrop_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    exported = [item for item in manifest.get("items", []) if item.get("status") == "exported"]
    unresolved = [
        item
        for item in manifest.get("items", [])
        if item.get("status") == "unresolved_no_safe_approach_crop"
    ]
    if (
        manifest.get("schema_version") != "1.0"
        or len(exported) != 60
        or len(unresolved) != 3
        or manifest.get("human_decisions_changed") is not False
        or manifest.get("original_sources_changed") is not False
    ):
        raise ValueError("the recrop manifest is not the expected verified 60+3 batch")

    store = ReviewStore(args.package_root, args.confirmed_root, args.source_root)
    try:
        state_bytes = store.state_path.read_bytes()
        state = json.loads(state_bytes)
        items = manifest["items"]
        prepared = []
        for item in items:
            verified = _verify_item(item, store, state, recrop_root)
            if verified is not None:
                prepared.append(verified)
        if len({card["review_id"] for card in prepared}) != 60:
            raise ValueError("candidate review IDs are not unique")
        collisions = {card["review_id"] for card in prepared} & (
            set(store.base_cards) | set(store.supplemental_cards)
        )
        if collisions:
            raise ValueError(
                f"candidate review ID collides with an existing card: {sorted(collisions)[:3]}"
            )

        old_document = {}
        pointer = state.get("supplemental_review_candidates")
        if pointer:
            old_path = store.root / pointer["path"]
            old_bytes = old_path.read_bytes()
            if _sha256_bytes(old_bytes) != pointer["sha256"]:
                raise ValueError("active supplemental candidate document hash mismatch")
            old_document = json.loads(old_bytes)
        old_cards = list(old_document.get("cards", []))
        old_ids = {card["review_id"] for card in old_cards}
        if old_ids != set(store.supplemental_cards):
            raise ValueError("active supplemental candidate cards do not match the review store")

        new_cards = _publish_media(prepared, store)
        document = {
            "schema_version": 1,
            "batch_id": BATCH_ID,
            "base_manifest_sha256": store.manifest_sha,
            "source_decisions_revision": state["revision"],
            "source_decisions_sha256": _sha256_bytes(state_bytes),
            "origin_manifest_sha256": _sha256_bytes(manifest_bytes),
            "origin_decision_snapshot": manifest["decision_snapshot"],
            "cards": old_cards + new_cards,
        }
        candidate_path = (
            recrop_root / f"supplemental_review_candidates_v2_rev{state['revision']}.json"
        )
        _write_json(candidate_path, document)

        if args.publish:
            before_decisions = state["decisions"]
            before_dataset_manifest = (store.root / "dataset_manifest.json").read_bytes()
            snapshot = store.publish_supplemental_candidates(
                document, expected_revision=state["revision"]
            )
            after = json.loads(store.state_path.read_text(encoding="utf-8"))
            if after["decisions"] != before_decisions:
                raise RuntimeError("publishing candidates unexpectedly changed human decisions")
            if (store.root / "dataset_manifest.json").read_bytes() != before_dataset_manifest:
                raise RuntimeError(
                    "publishing candidates unexpectedly changed the confirmed dataset manifest"
                )
            receipt = {
                "status": "published",
                "review_revision": snapshot["revision"],
                "supplemental_sha256": snapshot["supplemental_sha256"],
                "total_supplemental_cards": len(store.supplemental_cards),
                "new_approach_recrop_cards": len(new_cards),
                "preserved_prior_cards": len(old_cards),
                "unresolved_not_published": [item["review_id"] for item in unresolved],
                "candidate_document": str(candidate_path),
            }
            _write_json(
                recrop_root / f"publication_receipt_v2_rev{snapshot['revision']}.json", receipt
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
        else:
            print(
                json.dumps(
                    {
                        "status": "prepared_only",
                        "source_review_revision": state["revision"],
                        "new_approach_recrop_cards": len(new_cards),
                        "preserved_prior_cards": len(old_cards),
                        "candidate_document": str(candidate_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
