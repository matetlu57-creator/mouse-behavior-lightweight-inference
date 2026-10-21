"""Full-cohort retrospective process timing; no independent validation claim."""

PROCESS_PARAMETER_REVISION = "2026-10-18-v19-process-and-sync-support"
FIT_SCOPE = "full_dataset_retrospective_behavior_specific"
INDEPENDENT_VALIDATION = False
ACTIVE_DATASET_SAMPLE_COUNT = 755
ACTIVE_DATASET_MANIFEST_SHA256 = "274243ef8bc054d7ac0fad701217b75e114847e540f2ee8cf4e9de32b208206c"
PROCESS_FITTED = {"locomotion_simultaneous_min_s": 0.9329504772495598}
PROCESS_PARAMETER_SOURCES = {
    "locomotion_simultaneous_min_s": "minimum observed simultaneous direction/speed/space support over all eight annotated Group locomotion samples; all eight retained; label-free full-proposal evaluation 5/8 to 8/8"
}
SOURCE_ARTIFACT_SHA256 = {
    "outputs/heuristics_v19_process_20261018/group_sync_support_fit.json": "6b94b17745cd163aaf843f6f0d15e61cb73526af6d6f8a88b81eb7c9457d5773"
}
