"""Unified scene-scale resolution for every centimetre consumer.

Historically each consumer derived its own centimetre scale or silently fell
back to an assumed 8 cm mouse body length.  This module is the single policy
point for the whole package:

* ``calibration`` -- read ``cm_per_pixel`` and the measured subject body length
  from an arena calibration file produced by
  :mod:`mouse_behavior.preprocessing.scale_calibration`.  A stale source-video
  fingerprint only warns, so one rig calibration can serve every video recorded
  on that rig.
* ``fixed`` -- explicit ``cm_per_pixel`` override for manual operation.
* ``body_length`` -- the default analysis path: each video is scaled by the
  unified *measured* body length (``subject_metrics.unified_body_length_cm``
  from the arena calibration record, config key as fallback) divided by that
  video's own median nose-to-tail-base pixel length.  The caller must supply
  ``reference_body_px``.

Missing or malformed configuration raises instead of guessing: there is no
implicit 8 cm fallback anywhere on these paths anymore.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mouse_behavior.preprocessing import scale_calibration as sc

_LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"

#: Centimetre scale era of artefacts serialized before ``reference_body_cm``
#: existed.  Only used as a documented dataclass default so historical exports
#: keep describing their own era; new builds always write the resolved value.
LEGACY_REFERENCE_BODY_CM = 7.918

VALID_MODES = ("calibration", "fixed", "body_length")


@dataclass(frozen=True)
class ResolvedScale:
    """Fully resolved centimetre scale plus the audit trail of its origin."""

    cm_per_pixel: float
    reference_body_cm: float
    mode: str
    calibration_path: str | None = None
    warnings: tuple[str, ...] = ()
    # Appended for positional compatibility; legacy instances used 2.4 cm.
    reference_width_cm: float = 2.4


def project_root() -> Path:
    """Repository root that owns ``configs/`` (the src-layout parent)."""

    return _PROJECT_ROOT


def default_config_path() -> Path:
    return _DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def default_scale_config() -> Mapping[str, Any]:
    """Scale block of the repository default configuration.

    Consumers that predate explicit scale plumbing (scripts, review exports)
    resolve through here so every centimetre in the repository shares one
    policy.  Pass an explicit ``scale_config`` to override.
    """

    from mouse_behavior.config import load_config

    return dict(load_config(_DEFAULT_CONFIG_PATH).get("scale", {}))


def log_scale_warnings(resolved: ResolvedScale, logger: logging.Logger | None = None) -> None:
    target = logger or _LOGGER
    for warning in resolved.warnings:
        target.warning("scale resolution: %s", warning)


def resolve_calibration_path(candidate: str | Path) -> Path:
    """Resolve a calibration file against absolute, repo-root, then CWD."""

    raw = Path(str(candidate)).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"scale calibration file not found: {raw}")
    for base in (_PROJECT_ROOT, Path.cwd()):
        probe = base / raw
        if probe.exists():
            return probe
    raise FileNotFoundError(
        f"scale calibration file not found (tried absolute, {_PROJECT_ROOT}, {Path.cwd()}): {raw}"
    )


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a finite positive number, got {parsed!r}")
    return parsed


def _assumed_body_cm(cfg: Mapping[str, Any]) -> float:
    if "assumed_mouse_body_length_cm" not in cfg:
        raise ValueError(
            "scale.assumed_mouse_body_length_cm is required for scale.mode="
            f"{cfg.get('mode')!r}; the historical 8.0 cm implicit fallback was removed"
        )
    return _positive_float(cfg["assumed_mouse_body_length_cm"], "scale.assumed_mouse_body_length_cm")


def _source_video_fingerprint(video: str | Path) -> dict[str, Any]:
    from mouse_behavior.preprocessing.arena_learning import _video_fingerprint

    return dict(_video_fingerprint(video))


def _reference_body_cm(
    cfg: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    warnings: list[str],
) -> float:
    if calibration is not None:
        subject = calibration.get("subject_metrics") or {}
        unified = subject.get("unified_body_length_cm")
        if unified is not None:
            reference = _positive_float(unified, "subject_metrics.unified_body_length_cm")
            if "assumed_mouse_body_length_cm" in cfg:
                assumed = _positive_float(
                    cfg["assumed_mouse_body_length_cm"], "scale.assumed_mouse_body_length_cm"
                )
                if abs(assumed - reference) > 0.02 * reference:
                    warnings.append(
                        f"scale.assumed_mouse_body_length_cm={assumed} deviates more than 2% "
                        f"from the calibration unified body length {reference}; the calibration wins"
                    )
            return reference
        warnings.append(
            "calibration file lacks subject_metrics.unified_body_length_cm; "
            "falling back to scale.assumed_mouse_body_length_cm"
        )
    return _assumed_body_cm(cfg)


def resolve_reference_body_cm(scale_cfg: Mapping[str, Any] | None) -> float:
    """Reference mouse body length in cm without needing pixel evidence."""

    cfg = dict(scale_cfg or {})
    if not cfg:
        raise ValueError("scale configuration block is required (scale.mode must be explicit)")
    mode = str(cfg.get("mode", "")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"scale.mode must be one of {VALID_MODES}, got {cfg.get('mode')!r}")
    warnings: list[str] = []
    calibration = None
    if mode == "calibration":
        raw_path = cfg.get("calibration_file")
        if not raw_path:
            raise ValueError("scale.mode=calibration requires scale.calibration_file")
        calibration = sc.load_calibration(resolve_calibration_path(raw_path))
    return _reference_body_cm(cfg, calibration, warnings)


def _reference_width_cm(
    cfg: Mapping[str, Any], calibration: Mapping[str, Any] | None, reference_body_cm: float,
) -> float:
    """Measured hip width is a fallback, never a replacement for observed hips."""
    if calibration is not None:
        subject = calibration.get("subject_metrics") or {}
        segments = subject.get("segments_cm") or {}
        measured = (segments.get("hip_width") or {}).get("mean_cm")
        if measured is not None:
            return _positive_float(measured, "subject_metrics.segments_cm.hip_width.mean_cm")
    if "assumed_mouse_body_width_cm" in cfg:
        return _positive_float(cfg["assumed_mouse_body_width_cm"], "scale.assumed_mouse_body_width_cm")
    # Explicit legacy/custom configs without a measured width retain their
    # original proportion. Repository defaults always supply the measured width.
    return reference_body_cm * 0.30


def resolve_scale(
    scale_cfg: Mapping[str, Any] | None,
    *,
    reference_body_px: float | None = None,
    source_video: str | Path | None = None,
) -> ResolvedScale:
    """Resolve ``cm_per_pixel`` plus the reference body length from config."""

    cfg = dict(scale_cfg or {})
    if not cfg:
        raise ValueError("scale configuration block is required (scale.mode must be explicit)")
    mode = str(cfg.get("mode", "")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"scale.mode must be one of {VALID_MODES}, got {cfg.get('mode')!r}")
    warnings: list[str] = []

    if mode == "calibration":
        raw_path = cfg.get("calibration_file")
        if not raw_path:
            raise ValueError("scale.mode=calibration requires scale.calibration_file")
        path = resolve_calibration_path(raw_path)
        calibration = sc.load_calibration(path)
        scale_block = calibration.get("scale") or {}
        cm_per_pixel = _positive_float(scale_block.get("cm_per_pixel"), "calibration scale.cm_per_pixel")
        reference_body_cm = _reference_body_cm(cfg, calibration, warnings)
        stored = dict(calibration.get("source_video_fingerprint") or {})
        if source_video is not None and stored:
            current = _source_video_fingerprint(source_video)
            if current != stored:
                warnings.append(
                    f"calibration {path} was built from a different source video; "
                    "same-rig reuse is expected, re-calibrate if the rig or lens changed"
                )
        return ResolvedScale(
            cm_per_pixel, reference_body_cm, mode, str(path), tuple(warnings),
            reference_width_cm=_reference_width_cm(cfg, calibration, reference_body_cm),
        )

    if mode == "fixed":
        if cfg.get("cm_per_pixel") is None:
            raise ValueError("scale.mode=fixed requires a positive scale.cm_per_pixel")
        cm_per_pixel = _positive_float(cfg["cm_per_pixel"], "scale.cm_per_pixel")
        reference_body_cm = _assumed_body_cm(cfg)
        return ResolvedScale(
            cm_per_pixel, reference_body_cm, mode, None, tuple(warnings),
            reference_width_cm=_reference_width_cm(cfg, None, reference_body_cm),
        )

    if reference_body_px is None:
        raise ValueError(
            "scale.mode=body_length requires the caller to supply reference_body_px "
            "(the video's own median nose-to-tail-base pixel length)"
        )
    calibration = None
    raw_path = cfg.get("calibration_file")
    if raw_path:
        try:
            calibration = sc.load_calibration(resolve_calibration_path(raw_path))
        except FileNotFoundError:
            warnings.append(f"scale.calibration_file not found ({raw_path}); using configured body length")
    reference_body_cm = _reference_body_cm(cfg, calibration, warnings)
    pixels = _positive_float(reference_body_px, "reference_body_px")
    return ResolvedScale(
        reference_body_cm / pixels, reference_body_cm, mode, None, tuple(warnings),
        reference_width_cm=_reference_width_cm(cfg, calibration, reference_body_cm),
    )


def robust_reference_body_px(per_track_medians: Sequence[float]) -> float:
    """Outlier-robust per-video pixel body length.

    A single pose-broken track (collapsed tail keypoint, curled posture) must
    not move the video's pixel ruler.  One median-absolute-deviation style pass
    keeps tracks within [0.75, 1.33] of the cross-track median.
    """

    values = np.asarray([float(v) for v in per_track_medians if np.isfinite(v)], dtype=float)
    if values.size == 0:
        raise ValueError("reference_body_px requires at least one finite per-track median")
    median = float(np.median(values))
    if values.size == 1:
        return median
    kept = values[(values >= 0.75 * median) & (values <= 1.33 * median)]
    if kept.size == 0:
        return median
    return float(np.median(kept))


def scale_block_of(config_like: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept either a full config mapping or a bare ``scale`` block."""

    cfg = dict(config_like or {})
    scale = cfg.get("scale")
    if isinstance(scale, Mapping):
        return dict(scale)
    return cfg


__all__ = [
    "LEGACY_REFERENCE_BODY_CM",
    "VALID_MODES",
    "ResolvedScale",
    "default_config_path",
    "default_scale_config",
    "log_scale_warnings",
    "project_root",
    "resolve_calibration_path",
    "robust_reference_body_px",
    "scale_block_of",
    "resolve_reference_body_cm",
    "resolve_scale",
]
