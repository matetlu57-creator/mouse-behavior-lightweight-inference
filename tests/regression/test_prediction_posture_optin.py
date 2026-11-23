"""The explicit all-behavior path must carry posture evidence into blind ranking."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mouse_behavior.evaluation import exported_dataset_prediction as prediction
from mouse_behavior.evaluation.blind_top1 import candidates_from_evidence
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters


@pytest.mark.parametrize("behavior", ["grooming", "rearing"])
def test_explicit_posture_evidence_reaches_blind_candidates(tmp_path, monkeypatch, behavior):
    path = tmp_path / "tracks.json"
    detection = {
        "track_id": 7, "box": [0, 0, 80, 30],
        "keypoints": [[x, y, 0.99] for x, y in
                      [(70, 15), (60, 10), (60, 20), (50, 15), (20, 10), (20, 20), (10, 15)]],
    }
    path.write_text(json.dumps([
        {"frame": frame, "detections": [detection]} for frame in range(60)
    ]), encoding="utf-8")
    features = load_exported_features(path, fps=30)
    original = prediction.individual_rule_masks

    def with_posture_evidence(*args, **kwargs):
        result = original(*args, **kwargs)
        masks = {name: np.zeros_like(mask) for name, mask in result.masks.items()}
        if kwargs["config"].enable_posture_screen:
            assert kwargs["boxes_cm"] is features.collective_boxes_cm
            masks[behavior][:] = True
        return replace(result, masks=masks)

    # Isolate the integration seam: existing posture-rule tests cover detection;
    # this regression ensures valid evidence is neither disabled nor discarded.
    monkeypatch.setattr(prediction, "individual_rule_masks", with_posture_evidence)
    disabled = prediction.predict_features(features, HeuristicParameters())
    enabled = prediction.predict_features(
        features, HeuristicParameters(), enable_posture_screen=True,
    )
    assert disabled["identity_scores"]["7"][behavior] == 0
    assert disabled["suspended_posture_behaviors"] == ["grooming", "rearing"]
    assert enabled["identity_scores"]["7"][behavior] > 0
    assert enabled["suspended_posture_behaviors"] == []
    assert any(c.behavior == behavior and c.target_ids == (7,)
               for c in candidates_from_evidence(enabled))
