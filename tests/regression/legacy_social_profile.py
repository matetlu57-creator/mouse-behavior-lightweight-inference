"""Explicit historical gates for component fixtures, never production defaults."""
from dataclasses import replace

from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG

LEGACY_SOCIAL = replace(DEFAULT_SOCIAL_HEURISTIC_CONFIG, attack_min_duration_s=.15,
                        approach_min_duration_s=.3, approach_min_actor_path_cm=0.,
                        approach_require_no_body_contact=False,
                        chasing_min_mean_speed_cm_s=15.)
