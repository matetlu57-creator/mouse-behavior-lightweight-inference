"""Public compatibility API for exported annotation-dataset analysis.

The implementation is separated by responsibility into models, record discovery,
feature loading, prediction, metrics, and configuration modules. Existing imports
from ``mouse_behavior.evaluation.exported_dataset`` remain supported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mouse_behavior.evaluation.exported_dataset_models import (
    RETAINED_BEHAVIORS as RETAINED_BEHAVIORS,
    LABEL_TO_CANONICAL as LABEL_TO_CANONICAL,
    CANONICAL_BEHAVIORS as CANONICAL_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS as INDIVIDUAL_BEHAVIORS,
    PAIR_BEHAVIORS as PAIR_BEHAVIORS,
    GROUP_BEHAVIORS as GROUP_BEHAVIORS,
    _PART_SUFFIX as _PART_SUFFIX,
    _COLLECTIVE_WINDOW_CACHE as _COLLECTIVE_WINDOW_CACHE,
    _remove_collective_window_cache_entry as _remove_collective_window_cache_entry,
    _cached_collective_windows as _cached_collective_windows,
    SampleRecord as SampleRecord,
    ExportedFeatures as ExportedFeatures,
    HeuristicParameters as HeuristicParameters,
)
from mouse_behavior.evaluation.exported_dataset_records import (
    collective_track_quality_metrics as collective_track_quality_metrics,
    collective_track_quality_failures as collective_track_quality_failures,
    _read_json as _read_json,
    _session_name as _session_name,
    discover_samples as discover_samples,
    filter_behaviors as filter_behaviors,
    split_by_recording_session as split_by_recording_session,
    write_split_manifest as write_split_manifest,
)
from mouse_behavior.evaluation.exported_dataset_features import (
    _safe_mean as _safe_mean,
    _huddle_pair_distance_cm as _huddle_pair_distance_cm,
    load_exported_features as load_exported_features,
)
from mouse_behavior.evaluation.exported_dataset_temporal import (
    _longest_run as _longest_run,
    _bridge_short_false_gaps as _bridge_short_false_gaps,
    _duration_score as _duration_score,
    _max_track_duration as _max_track_duration,
    _pair_any_duration as _pair_any_duration,
    _durations_by_column as _durations_by_column,
    _isolation_cumulative_durations as _isolation_cumulative_durations,
    _qualified_bout_mask as _qualified_bout_mask,
    _qualifying_score as _qualifying_score,
    _pair_key as _pair_key,
    _prioritize_attack_over_approach as _prioritize_attack_over_approach,
)
from mouse_behavior.evaluation.exported_dataset_collective import (
    _neighbor_counts as _neighbor_counts,
    _group_resting_at_onset as _group_resting_at_onset,
    _windowed_body_speed as _windowed_body_speed,
    _group_episode_scores as _group_episode_scores,
    _rolling_drop as _rolling_drop,
    _dispersal_scores as _dispersal_scores,
    _recent_contact_mask as _recent_contact_mask,
    _finite_quantile as _finite_quantile,
    _directional_walker_in_group as _directional_walker_in_group,
    _group_formation_geometry_support as _group_formation_geometry_support,
    _collective_rule_config as _collective_rule_config,
    _replace_collective_group_scores_with_features as _replace_collective_group_scores_with_features,
)
from mouse_behavior.evaluation.exported_dataset_prediction import (
    predict_features as predict_features,
)
from mouse_behavior.evaluation.exported_dataset_metrics import (
    classify_target_ids as classify_target_ids,
    evaluate_predictions as evaluate_predictions,
    evaluate_target_classifications as evaluate_target_classifications,
    calibration_objective as calibration_objective,
    group_rule_calibration_rank as group_rule_calibration_rank,
    strict_top1_regressions as strict_top1_regressions,
    group_rule_parameter_tie_break as group_rule_parameter_tie_break,
    minimum_behavior_accuracy as minimum_behavior_accuracy,
    training_target_reached as training_target_reached,
    calibration_rank as calibration_rank,
)
from mouse_behavior.evaluation.exported_dataset_config import (
    config_seed_parameters as config_seed_parameters,
)

EXPORTED_DATASET_SOURCE_FILES = (
    'src/mouse_behavior/behavior/collective_state.py',
    'src/mouse_behavior/behavior/individual_rules.py',
    'src/mouse_behavior/behavior/static_rule.py',
    'src/mouse_behavior/config.py',
    'src/mouse_behavior/evaluation/__init__.py',
    'src/mouse_behavior/evaluation/blind_top1.py',
    'src/mouse_behavior/evaluation/clustering_formation.py',
    'src/mouse_behavior/evaluation/confirmed_rules.py',
    'src/mouse_behavior/evaluation/event_frames.py',
    'src/mouse_behavior/evaluation/exported_dataset.py',
    'src/mouse_behavior/evaluation/exported_dataset_collective.py',
    'src/mouse_behavior/evaluation/exported_dataset_config.py',
    'src/mouse_behavior/evaluation/exported_dataset_features.py',
    'src/mouse_behavior/evaluation/exported_dataset_metrics.py',
    'src/mouse_behavior/evaluation/exported_dataset_models.py',
    'src/mouse_behavior/evaluation/exported_dataset_prediction.py',
    'src/mouse_behavior/evaluation/exported_dataset_records.py',
    'src/mouse_behavior/evaluation/exported_dataset_temporal.py',
    'src/mouse_behavior/evaluation/group_event_rules.py',
    'src/mouse_behavior/evaluation/group_heuristic_config.py',
    'src/mouse_behavior/evaluation/pair_roles.py',
    'src/mouse_behavior/evaluation/requested_thresholds_v27.py',
    'src/mouse_behavior/evaluation/requested_thresholds_v28.py',
    'src/mouse_behavior/evaluation/retrospective_thresholds_v17.py',
    'src/mouse_behavior/evaluation/retrospective_thresholds_v18.py',
    'src/mouse_behavior/evaluation/retrospective_thresholds_v19.py',
    'src/mouse_behavior/evaluation/retrospective_thresholds_v20.py',
    'src/mouse_behavior/evaluation/screening_signals.py',
    'src/mouse_behavior/evaluation/social_event_geometry.py',
    'src/mouse_behavior/evaluation/social_heuristic_config.py',
    'src/mouse_behavior/preprocessing/__init__.py',
    'src/mouse_behavior/preprocessing/arena_learning.py',
    'src/mouse_behavior/preprocessing/constants.py',
    'src/mouse_behavior/preprocessing/pose_primitives.py',
    'src/mouse_behavior/preprocessing/scale_calibration.py',
    'src/mouse_behavior/preprocessing/scale_resolution.py',
    'src/mouse_behavior/utils/rolling.py',
)

def exported_dataset_source_hashes(repository_root: str | Path) -> dict[str, str]:
    """Hash every implementation file behind the public adapter module."""

    root = Path(repository_root)
    return {
        relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        for relative_path in EXPORTED_DATASET_SOURCE_FILES
    }


def exported_dataset_source_digest(repository_root: str | Path) -> str:
    """Return a stable aggregate hash for all exported-dataset implementation files."""

    payload = json.dumps(
        exported_dataset_source_hashes(repository_root),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

__all__ = (
    "CANONICAL_BEHAVIORS",
    "ExportedFeatures",
    "GROUP_BEHAVIORS",
    "HeuristicParameters",
    "INDIVIDUAL_BEHAVIORS",
    "LABEL_TO_CANONICAL",
    "PAIR_BEHAVIORS",
    "RETAINED_BEHAVIORS",
    "SampleRecord",
    "calibration_rank",
    "calibration_objective",
    "classify_target_ids",
    "config_seed_parameters",
    "discover_samples",
    "evaluate_predictions",
    "evaluate_target_classifications",
    "group_rule_calibration_rank",
    "group_rule_parameter_tie_break",
    "filter_behaviors",
    "load_exported_features",
    "minimum_behavior_accuracy",
    "predict_features",
    "split_by_recording_session",
    "strict_top1_regressions",
    "training_target_reached",
    "write_split_manifest",
    "EXPORTED_DATASET_SOURCE_FILES",
    "exported_dataset_source_hashes",
    "exported_dataset_source_digest",
)
