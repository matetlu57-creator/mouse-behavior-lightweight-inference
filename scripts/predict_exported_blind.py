#!/usr/bin/env python3
"""Run a historical frozen Top-1 model on a label-free trajectory manifest.

The input CSV schema is exactly ``sample_id,tracks_path,fps``.  This program
cannot accept annotations or ground-truth target IDs.  It refuses to run until
the train-only readiness gate has produced a frozen model. This compatibility
command is not the current v27 canonical detector; select --historical-model
explicitly. Current video inference uses run_canonical_behavior_video.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT
from mouse_behavior.evaluation.blind_top1 import (
    candidates_from_evidence,
    load_unlabeled_manifest,
    select_top1,
)
from mouse_behavior.evaluation.exported_dataset import (
    HeuristicParameters,
    load_exported_features,
    predict_features,
)


def _json_default(value: Any) -> Any:
    """Serialize NumPy scalar/array evidence without leaking label objects."""

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_model(model_dir: Path) -> tuple[HeuristicParameters, dict[str, float], dict[str, Any]]:
    provenance_path = model_dir / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("ready_for_blind") is not True:
        raise ValueError("training readiness gate did not pass; blind inference is blocked")
    if provenance.get("model_file") != "frozen_model.json":
        raise ValueError("provenance does not name a frozen model")
    model_path = model_dir / "frozen_model.json"
    if _sha256(model_path) != provenance.get("model_sha256"):
        raise ValueError("frozen model hash has changed")
    for relative_path, expected in dict(provenance["source_sha256"]).items():
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"frozen source hash has changed: {relative_path}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    parameters = HeuristicParameters(**payload["heuristic_parameters"])
    weights = {name: float(value) for name, value in payload["ranking_weights"].items()}
    return parameters, weights, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-model", action="store_true",
                        help="Explicitly select historical frozen-model inference, not current v27 rules")
    args = parser.parse_args()
    if not args.historical_model:
        parser.error("this is the historical frozen-model route; use run_canonical_behavior_video.py "
                     "for current rules, or explicitly select --historical-model")
    return args


def main() -> int:
    args = parse_args()
    parameters, weights, provenance = _verified_model(args.model_dir)
    clips = load_unlabeled_manifest(args.blind_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    partial_path = args.output_dir / "predictions.partial.csv"
    evidence_path = args.output_dir / "trajectory_evidence.jsonl"
    evidence_partial_path = args.output_dir / "trajectory_evidence.partial.jsonl"
    if any(path.exists() for path in (predictions_path, partial_path, evidence_path, evidence_partial_path)):
        raise FileExistsError(f"blind prediction output already exists in {args.output_dir}")
    debug_rows: list[dict[str, Any]] = []
    with partial_path.open("x", encoding="utf-8-sig", newline="") as handle, evidence_partial_path.open(
        "x", encoding="utf-8"
    ) as evidence_handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id", "predicted_behavior", "predicted_target_ids", "score",
                "secondary_behaviors",
                "selected_track_count", "track_selection_truncated",
            ),
        )
        writer.writeheader()
        for clip in clips:
            features = load_exported_features(
                clip.tracks_path, fps=clip.fps, max_tracks=int(provenance["max_tracks"])
            )
            evidence = predict_features(features, parameters)
            top1 = select_top1(candidates_from_evidence(evidence), weights)
            label_free_evidence = {
                key: value
                for key, value in evidence.items()
                if key != "_collective_candidate_windows"
            }
            evidence_handle.write(
                json.dumps(
                    {"opaque_id": clip.sample_id, **label_free_evidence},
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )
            debug_rows.extend(
                {"opaque_id": clip.sample_id, **debug}
                for debug in evidence.get("collective_feature_debug", ())
            )
            pair_key = ",".join(map(str, sorted(top1.target_ids)))
            compatible = evidence.get("pair_compatible_labels", {}).get(pair_key, [])
            secondary = (
                next((name for name in compatible if name != top1.behavior), "")
                if len(compatible) == 2 and top1.behavior in compatible
                else ""
            )
            writer.writerow(
                {
                    "sample_id": clip.sample_id,
                    "predicted_behavior": top1.behavior or "",
                    "predicted_target_ids": ",".join(map(str, top1.target_ids)),
                    "score": top1.score,
                    "secondary_behaviors": secondary,
                    "selected_track_count": evidence["selected_track_count"],
                    "track_selection_truncated": evidence["track_selection_truncated"],
                }
            )
    partial_path.rename(predictions_path)
    evidence_partial_path.rename(evidence_path)
    debug_path = args.output_dir / "feature_debug.csv"
    debug_fields = sorted({key for row in debug_rows for key in row})
    with debug_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=debug_fields or ["opaque_id"])
        writer.writeheader()
        writer.writerows(debug_rows)
    receipt = {
        "policy_scope": "historical_frozen_model_not_current_canonical_v27",
        "count": len(clips),
        "prediction_count": len(clips),
        "predictions_sha256": _sha256(predictions_path),
        "evidence_sha256": _sha256(evidence_path),
        "parameters_sha256": _sha256(args.model_dir / "tuned_parameters.json"),
        "feature_debug_sha256": _sha256(debug_path),
        "feature_debug_rows": len(debug_rows),
        "blind_manifest_sha256": _sha256(args.blind_manifest),
        "model_sha256": provenance["model_sha256"],
        "contains_ground_truth": False,
        "contains_labels": False,
    }
    (args.output_dir / "prediction_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {len(clips)} blind predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
