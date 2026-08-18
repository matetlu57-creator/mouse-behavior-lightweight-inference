from __future__ import annotations

from pathlib import Path

from mouse_behavior import __version__
from mouse_behavior.data.schema import REQUIRED_EVENT_COLUMNS, validate_event_columns
from mouse_behavior.io.paths import RunDirectories


def test_version_is_available_from_one_canonical_module() -> None:
    assert __version__ == "0.1.0"


def test_event_schema_reports_missing_columns() -> None:
    missing = validate_event_columns(REQUIRED_EVENT_COLUMNS[:-1])
    assert missing == [REQUIRED_EVENT_COLUMNS[-1]]
    assert validate_event_columns(REQUIRED_EVENT_COLUMNS) == []


def test_run_directories_keep_pipeline_outputs_separated(tmp_path: Path) -> None:
    directories = RunDirectories(tmp_path / "run").create()

    assert directories.tracking.is_dir()
    assert directories.behavior.is_dir()
    assert directories.visualization.is_dir()
    assert directories.report.is_dir()
    assert directories.logs.is_dir()
