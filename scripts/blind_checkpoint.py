"""Bind resumable label-free evidence to its complete frozen inference inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import REPO_ROOT
from mouse_behavior.evaluation.blind_top1 import UnlabeledClip
from mouse_behavior.evaluation.exported_dataset import exported_dataset_source_hashes

CHECKPOINT_NAME = "inference_checkpoint.json"
HISTORICAL_POLICY_SCOPE = "historical_frozen_model_not_current_canonical_v27"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_source_hashes() -> dict[str, str]:
    hashes = exported_dataset_source_hashes(REPO_ROOT)
    for relative in ("scripts/_bootstrap.py", "scripts/blind_checkpoint.py",
                     "scripts/predict_exported_blind.py", "scripts/resume_blind_prediction.py"):
        hashes[relative] = sha256(REPO_ROOT / relative)
    return hashes


def checkpoint_binding(
    manifest: Path, model_dir: Path, clips: Sequence[UnlabeledClip],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash trajectory contents as well as the manifest that assigns opaque IDs."""
    return {
        "schema_version": 1,
        "policy_scope": HISTORICAL_POLICY_SCOPE,
        "blind_manifest_sha256": sha256(manifest),
        "model_sha256": provenance["model_sha256"],
        "model_provenance_sha256": sha256(model_dir / "run_provenance.json"),
        "implementation_source_sha256": implementation_source_hashes(),
        "trajectory_sha256": [
            {"sample_id": clip.sample_id, "sha256": sha256(Path(clip.tracks_path))}
            for clip in clips
        ],
    }


class EvidenceCheckpoint:
    """Advance an atomic sidecar only after each evidence row is flushed."""

    def __init__(self, directory: Path, binding: Mapping[str, Any]):
        self.path = directory / CHECKPOINT_NAME
        self.binding = dict(binding)
        self.digest = hashlib.sha256()
        self.count = 0
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump(self._payload(), handle, ensure_ascii=False, indent=2)

    def _payload(self) -> dict[str, Any]:
        return {"binding": self.binding, "evidence_count": self.count,
                "evidence_sha256": self.digest.hexdigest()}

    def record_flushed_row(self, serialized_row: str) -> None:
        self.digest.update(serialized_row.encode("utf-8"))
        self.count += 1
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self._payload(), handle, ensure_ascii=False, indent=2)
        temporary.replace(self.path)


def verify_checkpoint(directory: Path, expected_binding: Mapping[str, Any]) -> int:
    """Fail closed for old unsealed evidence, drift, or an interrupted row write."""
    path = directory / CHECKPOINT_NAME
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("missing or invalid inference checkpoint; start a fresh run") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("binding") != expected_binding:
        raise ValueError("inference checkpoint inputs or implementation have changed")
    count = checkpoint.get("evidence_count")
    if type(count) is not int or count < 1:
        raise ValueError("inference checkpoint has no completed evidence rows")
    evidence_path = directory / "trajectory_evidence.partial.jsonl"
    if not evidence_path.is_file() or sha256(evidence_path) != checkpoint.get("evidence_sha256"):
        raise ValueError("partial evidence differs from its inference checkpoint")
    return count
