"""Post-prediction diagnostics; labels never enter candidate generation."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .blind_top1 import Candidate, LabeledClip, Top1Prediction


def diagnose_predictions(rows: Sequence[tuple[LabeledClip, Sequence[Candidate], Top1Prediction]]) -> dict:
    samples = []
    for label, candidates, prediction in rows:
        target = tuple(sorted(label.target_ids))
        exact = [c for c in candidates if tuple(sorted(c.target_ids)) == target and c.behavior == label.behavior]
        positive = [c for c in exact if c.evidence > 0]
        if prediction.behavior == label.behavior and tuple(sorted(prediction.target_ids)) == target:
            cause = "correct"
        elif not exact:
            visible_ids = {track for c in candidates for track in c.target_ids}
            cause = "target_track_unavailable" if not set(target) <= visible_ids else "exact_candidate_missing"
        elif not positive:
            cause = "no_positive_exact_rule_evidence"
        elif prediction.behavior == label.behavior:
            cause = "within_behavior_member_ranking"
        elif prediction.behavior is None:
            cause = "abstention"
        else:
            cause = "cross_behavior_ranking"
        samples.append({"sample_id": label.sample_id, "behavior": label.behavior,
                        "target_ids": list(target), "prediction": prediction.behavior,
                        "predicted_ids": list(prediction.target_ids), "cause": cause,
                        "exact_enumerated": bool(exact), "exact_positive": bool(positive)})
    return {"scope": "post_receipt_diagnostics_not_optimizer_input",
            "n_samples": len(samples), "counts": dict(Counter(row["cause"] for row in samples)), "samples": samples}
