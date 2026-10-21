from __future__ import annotations

import json

import pytest

from scripts.resume_blind_prediction import load_partial_evidence


def _evidence(opaque_id: str) -> dict[str, object]:
    return {
        "opaque_id": opaque_id,
        "identity_scores": {},
        "pair_scores": {},
        "group_scores": {},
        "collective_feature_debug": [],
    }


def test_load_partial_evidence_accepts_an_ordered_manifest_prefix(tmp_path) -> None:
    evidence_path = tmp_path / "trajectory_evidence.partial.jsonl"
    evidence_path.write_text(
        "\n".join(json.dumps(_evidence(value)) for value in ("opaque-a", "opaque-b")),
        encoding="utf-8",
    )

    rows = load_partial_evidence(evidence_path, ["opaque-a", "opaque-b", "opaque-c"])

    assert [row["opaque_id"] for row in rows] == ["opaque-a", "opaque-b"]


def test_load_partial_evidence_rejects_a_non_prefix_or_reordered_row(tmp_path) -> None:
    evidence_path = tmp_path / "trajectory_evidence.partial.jsonl"
    evidence_path.write_text(json.dumps(_evidence("opaque-b")), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest prefix"):
        load_partial_evidence(evidence_path, ["opaque-a", "opaque-b"])


def test_load_partial_evidence_rejects_label_bearing_fields(tmp_path) -> None:
    evidence_path = tmp_path / "trajectory_evidence.partial.jsonl"
    row = _evidence("opaque-a")
    row["canonical_behavior"] = "huddle"
    evidence_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="label-bearing"):
        load_partial_evidence(evidence_path, ["opaque-a"])


def test_load_partial_evidence_rejects_malformed_json(tmp_path) -> None:
    evidence_path = tmp_path / "trajectory_evidence.partial.jsonl"
    evidence_path.write_text('{"opaque_id":', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid partial evidence JSON"):
        load_partial_evidence(evidence_path, ["opaque-a"])
