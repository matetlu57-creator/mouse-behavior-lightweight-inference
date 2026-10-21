"""Every local module capable of changing adapter output must be fingerprinted."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mouse_behavior.evaluation.exported_dataset import EXPORTED_DATASET_SOURCE_FILES


ROOT = Path(__file__).resolve().parents[2]


def test_exported_adapter_hashes_cover_the_static_local_dependency_closure():
    pending = [Path("src/mouse_behavior/evaluation/exported_dataset.py")]
    seen = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        tree = ast.parse((ROOT/relative).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = list(relative.parts[1:-1])
                    module = ".".join(package[:len(package)-node.level+1]+([module] if module else []))
                names = [module]+[module+"."+alias.name for alias in node.names]
            for name in names:
                if name != "mouse_behavior" and not name.startswith("mouse_behavior."):
                    continue
                path = Path("src", *name.split("."))
                for candidate in (path.with_suffix(".py"), path/"__init__.py"):
                    if (ROOT/candidate).is_file():
                        pending.append(candidate)
                        break
    declared = set(EXPORTED_DATASET_SOURCE_FILES)
    assert {path.as_posix() for path in seen} <= declared
    assert len(EXPORTED_DATASET_SOURCE_FILES) == len(declared)


@pytest.mark.parametrize("sample_id", ["../escape", "/absolute", "C:\\escape", "..", "", None])
def test_posture_audit_rejects_unsafe_sample_ids_before_creating_output(tmp_path, monkeypatch, sample_id):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location("posture_audit_cli_under_test", ROOT/"scripts/audit_posture_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(development=[dict(sample_id=sample_id)])), encoding="utf-8")
    output = tmp_path/"new-output"
    monkeypatch.setattr(sys, "argv", ["audit_posture_v3", "--manifest", str(manifest), "--output", str(output)])
    with pytest.raises(ValueError, match="safe nonempty"):
        module.main()
    assert not output.exists()
