# Algorithms

The lightweight path is a cache-based analysis path rather than a replacement
for the full occlusion/ReID pipeline.

1. Completed YOLO Pose detections are matched into lightweight tracks.
2. Center, body, heading and pair kinematics are computed only for candidates
   selected by distance/heading prefiltering and padded valuable-frame windows.
3. Contact geometry is emitted independently from behavior labels.
4. Standard chase/attack causal evidence is processed by the standard FSM.
5. Individual, pair, contact and group channels run as parallel FSM regions.
6. Events are exported with frame/time boundaries and provenance metadata.

The parallel FSM deliberately preserves the previous minimum-duration and
short-gap semantics. Any change to thresholds or temporal gates requires a
regression comparison against frozen event outputs.

## Maintained module boundaries

Stable package facades and focused implementation modules are organized as
follows:

| Responsibility | Maintained module |
| --- | --- |
| cache normalization and lightweight tracking | `src/mouse_behavior/tracking/cache.py` |
| geometry and individual kinematics | `src/mouse_behavior/preprocessing/geometry.py`, `kinematics.py` |
| candidate-pair filtering and pair metrics | `src/mouse_behavior/preprocessing/pair_features.py` |
| standard continuous evidence | `src/mouse_behavior/behavior/standard_evidence.py` |
| standard chase/attack transitions | `src/mouse_behavior/behavior/standard_fsm.py` |
| contact and extended ethogram | `src/mouse_behavior/behavior/ethogram.py` |
| candidate-pair orchestration | `src/mouse_behavior/behavior/pair_analysis.py` |
| cage-boundary learning | `src/mouse_behavior/preprocessing/arena_learning.py` |
| cage-boundary persistence and audit images | `src/mouse_behavior/io/arena_boundary.py` |
| rendered videos and behavior clips | `src/mouse_behavior/visualization/rendering.py` |
| full detector/identity/video pipeline | `src/mouse_behavior/full_pipeline/` |

No threshold, schema, CLI argument or output filename changes as part of the
root-entry migration. Repository-root imports are intentionally retired;
scripts and notebooks must import from `mouse_behavior` or use the maintained
CLI entry points documented in ADR-0002.

`parallel_behavior_fsm.enabled` is an execution switch. With the default
`true`, individual, extended pair, contact and group regions use the temporal
FSM exactly as described above. With `false`, those parallel-region events are
not emitted; standard-engine chase/attack remain available because they own a
separate causal FSM. The only currently supported mode is `active`. Unknown
mode values fail during coordinator construction instead of being silently
reported in metadata without affecting execution.
