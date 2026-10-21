"""Fixed group-process defaults, separate from frozen calibration experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .retrospective_thresholds_v18 import GROUP_FITTED
from .retrospective_thresholds_v19 import PROCESS_FITTED


@dataclass(frozen=True)
class GroupHeuristicConfig:
    """Geometry and timing gates for the latest observed group rules."""

    clustering_seed_pair_distance_cm: float = 6.0  # RETIRED v21: final spatial state only
    clustering_quiet_speed_cm_s: float = 4.0  # RETIRED: final spatial state is not speed-gated
    clustering_arrival_region_gap_cm: float = 2.0  # RETIRED v21: final spatial state only
    clustering_final_stability_s: float = 0.5
    clustering_stability_radius_cm: float = 1.3375  # RETIRED v21: no fixed-position or implicit speed cap
    clustering_actor_speed_cm_s: float = 2.0  # Observed actor path/time; recipients have no speed cap.
    clustering_toward_cosine: float = 0.5
    clustering_distance_drop_cm: float = 1.0
    locomotion_simultaneous_min_s: float = PROCESS_FITTED["locomotion_simultaneous_min_s"]
    locomotion_max_pair_distance_cm: float = GROUP_FITTED["locomotion_max_pair_distance_cm"]
    isolation_min_duration_s: float = 10.0
    neighbor_distance_cm: float = 10.0
    huddle_hold_max_speed_cm_s: float = 2.0
    # Each process owns its physical limits. The generic neighbor remains the
    # formation/departure radius; changing a huddle fit cannot change dispersal.
    huddle_neighbor_distance_cm: float = GROUP_FITTED["huddle_neighbor_distance_cm"]
    locomotion_active_speed_cm_s: float = GROUP_FITTED["locomotion_active_speed_cm_s"]
    locomotion_min_direction_similarity: float = GROUP_FITTED["locomotion_min_direction_similarity"]
    locomotion_min_net_displacement_cm: float = GROUP_FITTED["locomotion_min_net_displacement_cm"]
    dispersal_extend_actual_motion: bool = True
    dispersal_min_outward_speed_cm_s: float = GROUP_FITTED["dispersal_min_outward_speed_cm_s"]
    dispersal_pause_max_s: float = GROUP_FITTED["dispersal_pause_max_s"]
    clustering_actor_min_displacement_cm: float = 1.0
    clustering_actor_min_path_cm: float = 10.0  # strict >, measured adjacent steps only
    clustering_unknown_gap_s: float = 0.15  # Memory only; unknown rows supply zero evidence.
    clustering_motion_lookback_s: float = 3.0
    clustering_min_toward_fraction: float = 0.7
    clustering_min_observed_fraction: float = 0.8

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "dispersal_extend_actual_motion":
                if not isinstance(value, bool):
                    raise ValueError("group.dispersal_extend_actual_motion must be bool")
                continue
            if name == "locomotion_min_net_displacement_cm" and not isinstance(value, bool) and value == 0:
                continue  # Explicit historical configurations may disable it.
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"group.{name} must be finite and positive")
        if self.clustering_toward_cosine > 1:
            raise ValueError("group.clustering_toward_cosine cannot exceed one")
        for name in ('clustering_min_toward_fraction', 'clustering_min_observed_fraction'):
            if getattr(self, name) > 1:
                raise ValueError(f"group.{name} cannot exceed one")
        if self.locomotion_min_direction_similarity > 1:
            raise ValueError("group.locomotion_min_direction_similarity cannot exceed one")

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)

    @property
    def retired_fields(self) -> tuple[str, ...]:
        """Serialized compatibility settings that the arrival process does not read."""
        return ("clustering_seed_pair_distance_cm", "clustering_quiet_speed_cm_s", "clustering_stability_radius_cm",
                "clustering_arrival_region_gap_cm")


DEFAULT_GROUP_HEURISTIC_CONFIG = GroupHeuristicConfig()
