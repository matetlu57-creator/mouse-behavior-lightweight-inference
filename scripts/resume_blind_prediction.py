#!/usr/bin/env python3
"""Resume frozen label-free inference from a validated partial evidence JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import REPO_ROOT  # noqa: E402,F401
from blind_checkpoint import (  # noqa: E402
    CHECKPOINT_NAME,
    HISTORICAL_POLICY_SCOPE,
    EvidenceCheckpoint,
    checkpoint_binding,
    verify_checkpoint,
)
from predict_exported_blind import _json_default, _sha256, _verified_model  # noqa: E402
from mouse_behavior.evaluation.blind_top1 import (  # noqa: E402
    candidates_from_evidence,
    load_unlabeled_manifest,
    select_top1,
)
from mouse_behavior.evaluation.exported_dataset import (  # noqa: E402
    load_exported_features,
    predict_features,
)


_FORBIDDEN_EVIDENCE_FIELDS = {
    "canonical_behavior",
    "ground_truth",
    "mouse_ids",
    "recording_session",
    "split",
    "tracks_path",
}


def load_partial_evidence(path: Path, expected_sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Load only a valid prefix of label-free evidence in manifest order."""

    if not path.is_file():
        raise FileNotFoundError(f"partial trajectory evidence not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in partial evidence at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid partial evidence JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"partial evidence line {line_number} is not an object")
            if _FORBIDDEN_EVIDENCE_FIELDS.intersection(row):
                raise ValueError(f"label-bearing fields found in partial evidence line {line_number}")
            index = len(rows)
            if index >= len(expected_sample_ids):
                raise ValueError("partial evidence contains more rows than the blind manifest")
            if row.get("opaque_id") != expected_sample_ids[index]:
                raise ValueError(f"partial evidence is not a manifest prefix at line {line_number}")
            if not all(key in row for key in ("identity_scores", "pair_scores", "group_scores")):
                raise ValueError(f"partial evidence is missing behavior scores at line {line_number}")
            if not isinstance(row.get("collective_feature_debug"), list):
                raise ValueError(f"partial evidence is missing debug rows at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("partial evidence is empty; start a fresh blind inference run")
    return rows


def _prediction_row(evidence: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    top1 = select_top1(candidates_from_evidence(evidence), weights)
    pair_key = ",".join(map(str, sorted(top1.target_ids)))
    compatible = evidence.get("pair_compatible_labels", {}).get(pair_key, [])
    secondary = (
        next((name for name in compatible if name != top1.behavior), "")
        if len(compatible) == 2 and top1.behavior in compatible
        else ""
    )
    return {
        "sample_id": evidence["opaque_id"],
        "predicted_behavior": top1.behavior or "",
        "predicted_target_ids": ",".join(map(str, top1.target_ids)),
        "score": top1.score,
        "secondary_behaviors": secondary,
        "selected_track_count": evidence["selected_track_count"],
        "track_selection_truncated": evidence["track_selection_truncated"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-model", action="store_true",
                        help="Explicitly select the historical frozen-model route")
    args = parser.parse_args()
    if not args.historical_model:
        parser.error("this is the historical frozen-model route; use run_canonical_behavior_video.py "
                     "for current rules, or explicitly select --historical-model")
    return args


def main() -> int:
    args = parse_args()
    parameters, weights, provenance = _verified_model(args.model_dir)
    clips = load_unlabeled_manifest(args.blind_manifest)
    binding = checkpoint_binding(args.blind_manifest, args.model_dir, clips, provenance)
    checkpoint_count = verify_checkpoint(args.partial_dir, binding)
    prior_evidence = load_partial_evidence(
        args.partial_dir / "trajectory_evidence.partial.jsonl",
        [clip.sample_id for clip in clips],
    )
    if len(prior_evidence) != checkpoint_count:
        raise ValueError("partial evidence count differs from its inference checkpoint")
    if args.output_dir.exists():
        raise FileExistsError(f"resume output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checkpoint = EvidenceCheckpoint(args.output_dir, binding)
    predictions_partial = args.output_dir / "predictions.partial.csv"
    evidence_partial = args.output_dir / "trajectory_evidence.partial.jsonl"
    debug_partial = args.output_dir / "feature_debug.partial.csv"
    prediction_fields = (
        "sample_id", "predicted_behavior", "predicted_target_ids", "score",
        "secondary_behaviors", "selected_track_count", "track_selection_truncated",
    )

    debug_rows: list[dict[str, Any]] = []
    print(f"resume from label-free trajectory evidence: {len(prior_evidence)}/{len(clips)}")
    with (
        predictions_partial.open("x", encoding="utf-8-sig", newline="") as prediction_handle,
        evidence_partial.open("x", encoding="utf-8", newline="\n") as evidence_handle,
    ):
        writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index, evidence in enumerate(prior_evidence):
            writer.writerow(_prediction_row(evidence, weights))
            serialized_evidence = json.dumps(evidence, ensure_ascii=False, default=_json_default) + "\n"
            evidence_handle.write(serialized_evidence)
            debug_rows.extend(
                {"opaque_id": evidence["opaque_id"], **row}
                for row in evidence["collective_feature_debug"]
            )
            prediction_handle.flush()
            evidence_handle.flush()
            checkpoint.record_flushed_row(serialized_evidence)

        for index in range(len(prior_evidence), len(clips)):
            clip = clips[index]
            features = load_exported_features(
                clip.tracks_path, fps=clip.fps, max_tracks=int(provenance["max_tracks"])
            )
            prediction_evidence = predict_features(features, parameters)
            label_free_evidence = {
                key: value
                for key, value in prediction_evidence.items()
                if key != "_collective_candidate_windows"
            }
            row = {"opaque_id": clip.sample_id, **label_free_evidence}
            writer.writerow(_prediction_row(row, weights))
            serialized_evidence = json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            evidence_handle.write(serialized_evidence)
            debug_rows.extend(
                {"opaque_id": clip.sample_id, **debug}
                for debug in prediction_evidence.get("collective_feature_debug", ())
            )
            prediction_handle.flush()
            evidence_handle.flush()
            checkpoint.record_flushed_row(serialized_evidence)
            completed = index + 1
            if completed % 5 == 0 or completed == len(clips):
                print(f"blind inference progress: {completed}/{len(clips)}", flush=True)

    predictions_path = args.output_dir / "predictions.csv"
    evidence_path = args.output_dir / "trajectory_evidence.jsonl"
    debug_path = args.output_dir / "feature_debug.csv"
    predictions_partial.rename(predictions_path)
    evidence_partial.rename(evidence_path)
    debug_fields = sorted({key for row in debug_rows for key in row})
    with debug_partial.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=debug_fields or ["opaque_id"])
        writer.writeheader()
        writer.writerows(debug_rows)
    debug_partial.rename(debug_path)
    receipt = {
        "policy_scope": HISTORICAL_POLICY_SCOPE,
        "count": len(clips),
        "prediction_count": len(clips),
        "predictions_sha256": _sha256(predictions_path),
        "evidence_sha256": _sha256(evidence_path),
        "parameters_sha256": provenance["model_sha256"],
        "parameters_file": provenance["model_file"],
        "feature_debug_sha256": _sha256(debug_path),
        "feature_debug_rows": len(debug_rows),
        "blind_manifest_sha256": _sha256(args.blind_manifest),
        "model_sha256": provenance["model_sha256"],
        "checkpoint_sha256": _sha256(args.output_dir / CHECKPOINT_NAME),
        "contains_ground_truth": False,
        "contains_labels": False,
        "resumed_evidence_count": len(prior_evidence),
        "resumed_evidence_sha256": _sha256(
            args.partial_dir / "trajectory_evidence.partial.jsonl"
        ),
    }
    (args.output_dir / "prediction_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {len(clips)} blind predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
