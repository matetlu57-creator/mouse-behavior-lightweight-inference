"""Explicit historical gates for component fixtures, never production defaults."""
from dataclasses import replace

from mouse_behavior.evaluation.retrospective_thresholds_v18 import SOCIAL_FITTED
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG

LEGACY_SOCIAL = replace(DEFAULT_SOCIAL_HEURISTIC_CONFIG, attack_min_duration_s=.15,
                        dist_attack_max_cm=SOCIAL_FITTED["dist_attack_max_cm"],
                        acc_attack_min_cm_s2=SOCIAL_FITTED["acc_attack_min_cm_s2"],
                        vel_attack_min_cm_s=SOCIAL_FITTED["vel_attack_min_cm_s"],
                        approach_min_duration_s=.3, approach_min_actor_path_cm=0.,
                        approach_require_no_body_contact=False,
                        chasing_min_mean_speed_cm_s=15.)
