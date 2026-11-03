"""Explicit, source-bound human comparator exclusions, never an ID merge."""
from dataclasses import asdict, dataclass
import json
from numbers import Integral
import re

import numpy as np


def normalized_pairs(pairs):
    result = set()
    for pair in pairs:
        if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in pair):
            raise ValueError("reviewed pairs must contain two nonnegative integer visual IDs")
        left, right = map(int, pair)
        if left == right:
            raise ValueError("a reviewed duplicate pair must contain different visual IDs")
        result.add(tuple(sorted((left, right))))
    return tuple(sorted(result))


@dataclass(frozen=True)
class ReviewedComparisonScope:
    sample_id: str
    tracks_sha256: str
    video_sha256: str
    decisions_sha256: str
    pairs: tuple

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("reviewed comparisons require the exact source sample ID")
        for field in ("tracks_sha256", "video_sha256", "decisions_sha256"):
            if not isinstance(getattr(self, field), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field)):
                raise ValueError(f"reviewed comparisons require a SHA-256 for {field}")
        object.__setattr__(self, "pairs", normalized_pairs(self.pairs))

    def as_dict(self):
        return asdict(self)

    def verify_source(self, *, sample_id, tracks_sha256, video_sha256):
        if (sample_id, tracks_sha256, video_sha256) != (self.sample_id, self.tracks_sha256, self.video_sha256):
            raise ValueError("reviewed comparator scope does not match the exact source sample and assets")


def bound_comparison_review(binding):
    value = binding.get("isolation_comparison_review")
    if value is None:
        return None
    review = ReviewedComparisonScope(**value)
    review.verify_source(sample_id=binding.get("comparison_sample_id"),
        tracks_sha256=binding.get("source_tracks_sha256"), video_sha256=binding["source_video"].get("sha256"))
    return review


def bind_comparison_review(binding, store, review_path, sample_id, tracks_path):
    """Bind an explicitly requested review to the actual normalized cache.

    File hashes alone are insufficient if the cache belongs to another pose
    export of the same video. Compare every observed ID, box and keypoint too.
    """
    from pathlib import Path
    from mouse_behavior.evaluation.segment_export import _sha256
    review_path, tracks_path = Path(review_path), Path(tracks_path)
    review_hash, tracks_hash = _sha256(review_path), _sha256(tracks_path)
    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("ambiguous duplicate keys in comparison review")
            result[key] = value
        return result
    document = json.loads(review_path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)
    if document.get("schema") != "reviewed_comparisons_exact_sample_assets_v1":
        raise ValueError("unsupported comparison review schema")
    value = document.get("scopes", {}).get(sample_id)
    if value is None:
        raise ValueError("no matching sample in the explicitly requested comparison review")
    scope = ReviewedComparisonScope(**value)
    scope.verify_source(sample_id=sample_id, tracks_sha256=tracks_hash,
                        video_sha256=binding["source_video"]["sha256"])
    rows = json.loads(tracks_path.read_text(encoding="utf-8-sig"))
    count = binding["source_video"]["frame_count"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("reviewed tracks differ from bound cache frame count")
    cached_count = 0
    for frame, cached in enumerate(store.rows(0, count)):
        source = rows[frame]
        if source.get("frame") != frame:
            raise ValueError("reviewed tracks differ from bound cache frame origins")
        left = {d["track_id"]: d for d in source["detections"]}
        right = {d["track_id"]: d for d in cached["detections"]}
        if (len(left) != len(source["detections"]) or len(right) != len(cached["detections"])
                or left.keys() != right.keys()):
            raise ValueError("reviewed tracks differ from bound cache visual IDs")
        for track in left:
            for field in ("box", "keypoints"):
                a, b = np.asarray(left[track][field], float), np.asarray(right[track][field], float)
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-5, equal_nan=True):
                    raise ValueError("reviewed tracks differ from bound cache observations")
        cached_count += 1
    if cached_count != count or _sha256(tracks_path) != tracks_hash or _sha256(review_path) != review_hash:
        raise ValueError("comparison review inputs changed during binding")
    binding.update(comparison_sample_id=sample_id, source_tracks_sha256=tracks_hash,
        isolation_comparison_review=scope.as_dict(), comparison_review_manifest_sha256=review_hash)
    return scope


def nearest_comparison_distances(features, excluded_pairs=()):
    """Only the explicitly reviewed edge is excluded; every other ID remains."""
    pairs = normalized_pairs(excluded_pairs)
    if not pairs:
        return np.asarray(features.nearest_distance_cm, float)
    rejected = set(pairs)
    nearest = np.full(features.valid.shape, np.inf)
    for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
        ids = tuple(sorted((int(features.track_ids[left]), int(features.track_ids[right]))))
        if ids in rejected:
            continue
        distance = features.pair_distance_cm[:, edge]
        known = features.valid[:, left] & features.valid[:, right] & np.isfinite(distance)
        measured = np.where(known, distance, np.inf)
        nearest[:, left] = np.minimum(nearest[:, left], measured)
        nearest[:, right] = np.minimum(nearest[:, right], measured)
    return nearest
