"""Extract a hash-bound, labeled posture comparison cache; no production changes."""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re

from _bootstrap import ensure_importable

ensure_importable()
from mouse_behavior.behavior.posture_adapter import posture_clip_from_rows  # noqa: E402
from mouse_behavior.behavior.posture_rules import infer_posture  # noqa: E402
from mouse_behavior.behavior.posture_motion_candidates import motion_windows  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8-sig"))
    protected = {str(manifest_path): sha(manifest_path)}
    records = []
    for i, row in enumerate(manifest):
        d = Path(row["dataset_path"])
        values = {}
        for name in ("tracks.json", "metadata.json", "annotation.json", "review_receipt.json"):
            path = d / name
            digest = sha(path)
            if digest != row["dataset_sha256"][name]:
                raise ValueError(f"Input hash changed: {path}")
            protected[str(path)] = digest
            values[name] = json.loads(path.read_text("utf-8-sig"))
        meta = values["metadata.json"]
        annotation = values["annotation.json"]
        tracks = values["tracks.json"]
        assert annotation["behavior"] == row["behavior"] and set(annotation["mouse_ids"]) == set(
            row["mouse_ids"]
        )
        assert len(tracks) == meta["clip"]["frame_count"]
        source = (
            meta.get("source", {}).get("video_filename")
            or row.get("source_key")
            or row["review_id"]
        )
        group = re.sub(r"_part\d+", "", source, flags=re.I)
        entry = dict(
            sample_id=row["review_id"],
            truth=row["behavior"],
            group=group,
            source=source,
            fps=meta["clip"]["fps"],
            frames=meta["clip"]["frame_count"],
            ids=row["mouse_ids"],
            targets=[],
        )
        for track_id in row["mouse_ids"]:
            clip = posture_clip_from_rows(
                tracks,
                entry["frames"],
                entry["fps"],
                track_id,
                keypoint_names=meta["pose"]["keypoint_names"],
            )
            base = infer_posture(clip, diagnostics=False)
            windows, scale = motion_windows(clip)
            from mouse_behavior.behavior.posture_measurements import quality_frames

            entry["targets"].append(
                dict(
                    track_id=track_id,
                    scale=scale,
                    quality=quality_frames(clip).tolist(),
                    windows=windows,
                    baseline_events=base["resolution"]["events"],
                    baseline_raw={k: v["events"] for k, v in base["candidates"].items()},
                    baseline_label=base["resolution"]["label"],
                )
            )
        records.append(entry)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(manifest)} extracted", flush=True)
    for path, digest in protected.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Input changed during extraction: {path}")
    (args.output / "features.json").write_text(
        json.dumps(records, ensure_ascii=False, allow_nan=False), "utf-8"
    )
    receipt = dict(
        status="complete",
        samples=len(records),
        dataset=str(args.dataset.resolve()),
        manifest_sha256=protected[str(manifest_path)],
        protected_hashes=protected,
        source_sha256={str(Path(__file__).resolve()): sha(Path(__file__))},
        scope="Full latest approved cohort; annotated target IDs only. Other-label clips are weak negative controls, not exhaustively annotated absence.",
        protected_inputs_unchanged=True,
    )
    (args.output / "extraction_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8"
    )
    print("COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
