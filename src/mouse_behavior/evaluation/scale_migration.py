"""Explicit in-memory migration for frozen validation receipts after rig rescaling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


OLD_BODY_LENGTH_CM = 10.7
OLD_BODY_WIDTH_CM = 4.1467
NEW_BODY_LENGTH_CM = 7.918
NEW_BODY_WIDTH_CM = 3.068558
ARENA_SIDE_CM = 74.0


def rescale_frozen_source_scale(
    scale_config: dict[str, Any], rig_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the approved 74 cm arena conversion without editing frozen receipts."""
    rig_path = Path(rig_config_path)
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    arena = rig.get("box_inner_cm", {})
    subject = rig.get("subject_metrics", {})
    hip_width = subject.get("segments_cm", {}).get("hip_width", {}).get("mean_cm")
    if (float(arena.get("width", 0)) != ARENA_SIDE_CM
            or float(arena.get("height", 0)) != ARENA_SIDE_CM
            or not math.isclose(float(subject.get("unified_body_length_cm", 0)),
                                NEW_BODY_LENGTH_CM, abs_tol=1e-12)
            or not math.isclose(float(hip_width or 0), NEW_BODY_WIDTH_CM, abs_tol=1e-12)):
        raise ValueError("active rig does not match the approved 74 x 74 cm mouse scale")
    old_length = float(scale_config.get("assumed_mouse_body_length_cm", 0))
    old_width = float(scale_config.get("assumed_mouse_body_width_cm", 0))
    old_cm_per_pixel = float(scale_config.get("cm_per_pixel", 0))
    if (not math.isclose(old_length, OLD_BODY_LENGTH_CM, abs_tol=1e-12)
            or not math.isclose(old_width, OLD_BODY_WIDTH_CM, abs_tol=1e-12)
            or not math.isfinite(old_cm_per_pixel) or old_cm_per_pixel <= 0):
        raise ValueError("frozen scale receipt is outside the known 10.7 cm source-scale era")
    factor = NEW_BODY_LENGTH_CM / OLD_BODY_LENGTH_CM
    migrated = dict(scale_config)
    migrated["cm_per_pixel"] = old_cm_per_pixel * factor
    migrated["assumed_mouse_body_length_cm"] = NEW_BODY_LENGTH_CM
    migrated["assumed_mouse_body_width_cm"] = NEW_BODY_WIDTH_CM
    migration = {
        "schema": "frozen_source_scale_74cm_migration_v1",
        "factor": factor,
        "arena_cm": [ARENA_SIDE_CM, ARENA_SIDE_CM],
        "body_length_cm": {"from": old_length, "to": NEW_BODY_LENGTH_CM},
        "body_width_cm": {"from": old_width, "to": NEW_BODY_WIDTH_CM},
        "cm_per_pixel": {"from": old_cm_per_pixel, "to": migrated["cm_per_pixel"]},
        "source_receipt_modified": False,
    }
    return migrated, migration
