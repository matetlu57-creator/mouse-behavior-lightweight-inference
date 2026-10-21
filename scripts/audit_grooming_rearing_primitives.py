"""Phase A (Step 5-A) unsupervised audit of grooming / rearing-climbing primitives.

This script never touches the primary labeling path.  It computes the Step-1
pose primitives on real clips under the unified per-video body-length scale,
reports per-track distributions, and emits candidate episode windows for
manual clip review so Phase B thresholds are evidence-based.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401  (makes src importable)
from mouse_behavior.preprocessing import pose_primitives as pp
from mouse_behavior.preprocessing.scale_resolution import (
    default_scale_config,
    resolve_scale,
    robust_reference_body_px,
)

FPS_DEFAULT = 30.0

# Probe gates for candidate mining only; NOT wired into primary labeling.
GROOM_HIND_MAX_CM_S = 2.0
GROOM_TAIL_MAX_CM_S = 2.0
GROOM_BURST_MAX = 3.0
GROOM_HEAD_MIN_CM_S = 1.0
REAR_FORESHORTEN_MAX = 0.75
REAR_HIND_MAX_CM_S = 2.0
REAR_TAIL_MAX_CM_S = 2.0
REAR_HEAD_HIND_RATIO_MIN = 2.0
MIN_EPISODE_SECONDS = 0.5


def _load_tracks_px(path: Path, max_tracks: int = 64):
    raw = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        int(d["track_id"]) for fr in raw for d in fr.get("detections", []) if "track_id" in d
    )
    ids = sorted(t for t, _ in counts.most_common(max(int(max_tracks), 2)))
    index = {t: i for i, t in enumerate(ids)}
    frames = max(int(fr.get("frame", i)) for i, fr in enumerate(raw)) + 1
    kp = np.full((frames, len(ids), 7, 2), np.nan, dtype=float)
    valid = np.zeros((frames, len(ids)), dtype=bool)
    for fr in raw:
        fi = int(fr.get("frame", 0))
        for d in fr.get("detections", []):
            slot = index.get(int(d.get("track_id", -1)))
            pts = np.asarray(d.get("keypoints", []), dtype=float)
            if slot is None or pts.shape != (7, 3):
                continue
            kp[fi, slot] = pts[:, :2]
            valid[fi, slot] = bool(np.mean(pts[:, 2] > 0.0) >= 0.7)
    return kp, valid, ids


def _episodes(mask: np.ndarray, fps: float, min_seconds: float) -> list[tuple[int, int]]:
    out = []
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if (end - start) / max(fps, 1e-6) >= min_seconds:
            out.append((int(start), int(end)))
    return out


def audit_clip(tracks_path: Path, fps: float, *, scale_config=None,
               stillness_floor_cm_s: float = GROOM_HIND_MAX_CM_S) -> dict[str, Any]:
    kp_px, valid, ids = _load_tracks_px(tracks_path)
    body_px = pp.body_length_px(kp_px, valid)
    per_track = [
        float(np.median(body_px[:, slot][valid[:, slot] & np.isfinite(body_px[:, slot])]))
        for slot in range(len(ids))
        if np.any(valid[:, slot] & np.isfinite(body_px[:, slot]))
    ]
    resolved = resolve_scale(scale_config if scale_config is not None else default_scale_config(),
                             reference_body_px=robust_reference_body_px(per_track))
    cpp = resolved.cm_per_pixel
    kp_cm = kp_px * cpp
    window = max(int(round(fps)), 2)

    speeds = pp.keypoint_speed_cm_s(kp_cm, valid, fps)
    head_speed = pp.head_speed_cm_s(speeds)
    hind_speed = pp.hind_speed_cm_s(speeds)
    tail_speed = speeds[:, :, 6]
    fore = pp.foreshorten_ratio(body_px, valid, window=window)
    deformation = pp.rolling_iqr(body_px, window)
    burst = pp.deformation_burst_ratio(deformation, valid, window=window)
    nose_torso = pp.nose_in_torso(kp_cm, valid, inflate_cm=0.5)
    head_hind = pp.ratio_or_nan(head_speed, hind_speed)

    # Phase A mining uses per-track quantile probes: absolute cm/s stillness
    # gates fall below the per-video keypoint jitter noise floor, so "hind/tail
    # still" is expressed relative to each track's own speed distribution.
    groom = np.zeros_like(valid)
    rear = np.zeros_like(valid)
    for slot in range(len(ids)):
        obs = valid[:, slot]
        if not np.any(obs):
            continue
        def q(values: np.ndarray, level: float) -> float:
            finite = values[obs & np.isfinite(values)]
            return float(np.quantile(finite, level)) if finite.size else float("nan")
        hind_q25, tail_q25 = q(hind_speed[:, slot], 0.25), q(tail_speed[:, slot], 0.25)
        head_q60, burst_q75 = q(head_speed[:, slot], 0.60), q(burst[:, slot], 0.75)
        fore_q10, ratio_q75 = q(fore[:, slot], 0.10), q(head_hind[:, slot], 0.75)
        still_hind = np.isfinite(hind_q25) & (hind_speed[:, slot] <= max(hind_q25, stillness_floor_cm_s))
        still_tail = np.isfinite(tail_q25) & (tail_speed[:, slot] <= max(tail_q25, stillness_floor_cm_s))
        groom[:, slot] = obs & (
            (nose_torso[:, slot] | (np.isfinite(burst[:, slot]) & (burst[:, slot] <= burst_q75)))
            & still_hind
            & still_tail
            & (head_speed[:, slot] >= head_q60)
        )
        rear[:, slot] = obs & (
            np.isfinite(fore[:, slot])
            & (fore[:, slot] <= fore_q10)
            & still_hind
            & still_tail
            & (head_hind[:, slot] >= ratio_q75)
        )

    bridge = max(1, int(round(0.15 * fps)))
    tracks_out = []
    episodes_out = []
    for slot, track_id in enumerate(ids):
        obs = valid[:, slot]
        if not np.any(obs):
            continue
        def quant(values: np.ndarray) -> dict[str, float]:
            finite = values[obs & np.isfinite(values)]
            if finite.size == 0:
                return {"p10": None, "p50": None, "p90": None}
            q = np.quantile(finite, [0.10, 0.50, 0.90])
            return {"p10": round(float(q[0]), 4), "p50": round(float(q[1]), 4), "p90": round(float(q[2]), 4)}
        tracks_out.append({
            "track_id": int(track_id),
            "frames": int(obs.sum()),
            "body_px_median": round(float(np.median(body_px[:, slot][obs & np.isfinite(body_px[:, slot])])), 2),
            "head_speed_cm_s": quant(head_speed[:, slot]),
            "hind_speed_cm_s": quant(hind_speed[:, slot]),
            "tail_speed_cm_s": quant(tail_speed[:, slot]),
            "foreshorten_ratio": quant(fore[:, slot]),
            "deformation_burst_ratio": quant(burst[:, slot]),
            "nose_in_torso_fraction": round(float(np.mean(nose_torso[:, slot][obs])), 4),
            "head_hind_speed_ratio": quant(head_hind[:, slot]),
        })
        for name, mask in (("grooming", groom[:, slot]), ("rearing", rear[:, slot])):
            raw_frames = int(mask.sum())
            bridged = mask.copy()
            padded = np.r_[False, bridged, False].astype(np.int8)
            changes = np.diff(padded)
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            for left, right in zip(ends[:-1], starts[1:]):
                if 0 < right - left <= bridge:
                    bridged[left:right] = True
            tracks_out[-1][f"{name}_raw_frames"] = raw_frames
            for start, end in _episodes(bridged, fps, MIN_EPISODE_SECONDS):
                episodes_out.append({
                    "clip": tracks_path.parent.name,
                    "track_id": int(track_id),
                    "candidate": name,
                    "start_frame": start,
                    "end_frame": end,
                    "duration_s": round((end - start) / fps, 3),
                    "mean_head_speed_cm_s": round(float(np.nanmean(head_speed[start:end, slot])), 3),
                    "mean_hind_speed_cm_s": round(float(np.nanmean(hind_speed[start:end, slot])), 3),
                    "mean_foreshorten": round(float(np.nanmean(fore[start:end, slot])), 3),
                    "mean_burst": round(float(np.nanmean(burst[start:end, slot])), 3),
                    "nose_in_torso_fraction": round(float(np.mean(nose_torso[start:end, slot])), 3),
                })
    return {
        "clip": tracks_path.parent.name,
        "cm_per_pixel": round(cpp, 6),
        "reference_body_cm": resolved.reference_body_cm,
        "tracks": tracks_out,
        "candidate_episodes": episodes_out,
    }


def _clip_fps(tracks_path: Path, fallback: float) -> float:
    meta = tracks_path.with_name("metadata.json")
    if meta.exists():
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            return float(raw["clip"]["fps"])
        except (KeyError, TypeError, ValueError):
            return fallback
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, nargs="*", default=[])
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="scan every */*/tracks.json under this root (dataset screening)")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    tracks_paths = [Path(item) for item in args.tracks]
    if args.dataset_root is not None:
        tracks_paths += sorted(args.dataset_root.glob("*/*/tracks.json"))
    if not tracks_paths:
        parser.error("provide --tracks or --dataset-root")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: list[dict] = []
    clip_summaries: list[dict] = []
    for done, path in enumerate(tracks_paths, 1):
        try:
            result = audit_clip(path, _clip_fps(path, args.fps))
        except Exception as exc:  # keep screening going; record the failure
            clip_summaries.append({"clip": path.parent.name, "error": str(exc)})
            continue
        clip_summaries.append({
            "clip": result["clip"],
            "path": str(path.parent),
            "cm_per_pixel": result["cm_per_pixel"],
            "tracks": len(result["tracks"]),
            "grooming_raw_frames": sum(t.get("grooming_raw_frames", 0) for t in result["tracks"]),
            "rearing_raw_frames": sum(t.get("rearing_raw_frames", 0) for t in result["tracks"]),
        })
        for episode in result["candidate_episodes"]:
            episode["clip_path"] = str(path.parent)
            episode["source_folder"] = path.parent.parent.name
            all_episodes.append(episode)
        if done % 100 == 0:
            print(f"screened {done}/{len(tracks_paths)} clips, "
                  f"{len(all_episodes)} candidate episodes so far", flush=True)

    all_episodes.sort(key=lambda ep: (ep["candidate"], -ep["duration_s"]))
    if all_episodes:
        keys = ["candidate", "source_folder", "clip", "clip_path", "track_id",
                "start_frame", "end_frame", "duration_s", "mean_head_speed_cm_s",
                "mean_hind_speed_cm_s", "mean_foreshorten", "mean_burst",
                "nose_in_torso_fraction"]
        lines = [",".join(keys)]
        for ep in all_episodes:
            lines.append(",".join(str(ep.get(key, "")) for key in keys))
        (args.output_dir / "candidates.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "stats.json").write_text(
        json.dumps({"clips": clip_summaries, "episodes": all_episodes},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    from collections import Counter
    by_behavior = Counter(ep["candidate"] for ep in all_episodes)
    by_folder = Counter((ep["candidate"], ep["source_folder"]) for ep in all_episodes)
    print(json.dumps({
        "clips_screened": len(clip_summaries),
        "clips_failed": sum(1 for item in clip_summaries if "error" in item),
        "candidate_episodes": dict(by_behavior),
        "by_source_folder": {f"{b}|{folder}": n for (b, folder), n in sorted(by_folder.items())},
        "outputs": str(args.output_dir),
    }, ensure_ascii=False, indent=1))
    for ep in all_episodes[: args.top]:
        print(f"  {ep['candidate']:8s} {ep['source_folder']:20s} {ep['clip'][:44]:44s} "
              f"track {ep['track_id']:>4} frames {ep['start_frame']}-{ep['end_frame']} "
              f"{ep['duration_s']:.2f}s head {ep['mean_head_speed_cm_s']:.1f} "
              f"hind {ep['mean_hind_speed_cm_s']:.1f} fore {ep['mean_foreshorten']:.2f} "
              f"noseIn {ep['nose_in_torso_fraction']:.2f}")


if __name__ == "__main__":
    main()
