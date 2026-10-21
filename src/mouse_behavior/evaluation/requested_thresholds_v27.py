"""Explicit user policy, 2026-10-20; these are not dataset-fitted extrema.

Frozen v17-v26 fit modules and historical evaluation receipts stay unchanged.
"""

ISOLATION_DISTANCE_CM = 20.0  # Strictly greater; finite coobserved comparisons only.
APPROACH_CENTER_END_MIN_CM = 7.0
APPROACH_SURFACE_END_MIN_CM = 2.0
APPROACH_SURFACE_END_MAX_CM = 6.3939353708899365  # Previous upper bound retained.
TOGETHER_MAX_SPEED_CM_S = 3.0  # Strictly below, independently for each participant.
CONTACT_MIN_DURATION_S = 0.1
CHASING_MIN_MEAN_SPEED_CM_S = 40.0  # Experimental per-participant observed-step mean.
