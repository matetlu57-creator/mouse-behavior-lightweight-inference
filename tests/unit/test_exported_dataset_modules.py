from __future__ import annotations

import hashlib
import json
from pathlib import Path
from shutil import copyfile

from mouse_behavior.evaluation import exported_dataset as adapter
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_metrics import classify_target_ids
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters, SampleRecord
from mouse_behavior.evaluation.exported_dataset_prediction import predict_features
from mouse_behavior.evaluation.exported_dataset_records import discover_samples


def test_legacy_adapter_path_reexports_symbols_from_responsibility_modules() -> None:
    assert adapter.HeuristicParameters is HeuristicParameters
    assert adapter.SampleRecord is SampleRecord
    assert adapter.discover_samples is discover_samples
    assert adapter.load_exported_features is load_exported_features
    assert adapter.predict_features is predict_features
    assert adapter.classify_target_ids is classify_target_ids


def test_source_hashes_cover_every_module_and_have_a_stable_aggregate() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    hashes = adapter.exported_dataset_source_hashes(repository_root)

    assert tuple(hashes) == adapter.EXPORTED_DATASET_SOURCE_FILES
    assert len(hashes) == 9
    assert all(
        hashlib.sha256((repository_root / relative_path).read_bytes()).hexdigest() == digest
        for relative_path, digest in hashes.items()
    )
    expected_digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert adapter.exported_dataset_source_digest(repository_root) == expected_digest


def test_source_digest_detects_changes_in_an_internal_implementation_file(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in adapter.EXPORTED_DATASET_SOURCE_FILES:
        source = repository_root / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)

    original_digest = adapter.exported_dataset_source_digest(tmp_path)
    internal_module = tmp_path / adapter.EXPORTED_DATASET_SOURCE_FILES[-1]
    internal_module.write_bytes(internal_module.read_bytes() + b"\n# source change")

    assert adapter.exported_dataset_source_digest(tmp_path) != original_digest
