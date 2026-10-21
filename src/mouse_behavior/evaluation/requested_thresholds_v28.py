"""V28 user overrides and explicitly retrospective, endpoint-specific onsets.

Contact onsets are p90 of each class's first finite directed annotated endpoint
distance (159 head /75 tail clips), audited in contact_onset_statistics.json.
They are independent of center and snout-to-body-surface Approach endpoints.
Annotation is used for this offline fit only, never by runtime prediction.
"""
from .requested_thresholds_v27 import (  # unchanged explicit user policies
    ISOLATION_DISTANCE_CM, TOGETHER_MAX_SPEED_CM_S, CONTACT_MIN_DURATION_S,
    APPROACH_SURFACE_END_MAX_CM,
)

__all__ = [
    'ISOLATION_DISTANCE_CM', 'TOGETHER_MAX_SPEED_CM_S', 'CONTACT_MIN_DURATION_S',
    'APPROACH_SURFACE_END_MAX_CM', 'APPROACH_CENTER_END_MIN_CM',
    'APPROACH_SURFACE_END_MIN_CM', 'CHASING_MIN_MEAN_SPEED_CM_S',
    'CONTACT_HEAD_ONSET_DISTANCE_CM', 'CONTACT_TAIL_ONSET_DISTANCE_CM',
]

APPROACH_CENTER_END_MIN_CM = 4.456491470336914  # restore V26
APPROACH_SURFACE_END_MIN_CM = 0.7240091693751944  # restore V26
CHASING_MIN_MEAN_SPEED_CM_S = 15.
CONTACT_HEAD_ONSET_DISTANCE_CM = 4.37976635219943
CONTACT_TAIL_ONSET_DISTANCE_CM = 4.7678087826183955
