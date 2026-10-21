"""Full-data participant speed fit; semantic changes separately user-specified."""
SEMANTIC_PARAMETER_REVISION = "2026-10-18-v20-deceleration-departure-spatial-formation"
FIT_SCOPE = "full_dataset_retrospective_behavior_specific"
INDEPENDENT_VALIDATION = False
ACTIVE_DATASET_SAMPLE_COUNT = 755
ACTIVE_DATASET_MANIFEST_SHA256 = "274243ef8bc054d7ac0fad701217b75e114847e540f2ee8cf4e9de32b208206c"
SOCIAL_FITTED = {"vel_together_max_cm_s": 1.6248862444153374}
PARAMETER_SOURCES = {"vel_together_max_cm_s": "strict upper bound above maximum of 12 Together participant-specific p95 smoothed center speeds, 6 crops; same symmetric limit applies independently to each participant"}
SOURCE_ARTIFACT_SHA256 = {"outputs/heuristics_v20_semantics_20261018/together_speed_fit.json": "392397ed6f5f99ad69313d3021796c98654957ca1dcbbe811b2a1b61599d0134"}
