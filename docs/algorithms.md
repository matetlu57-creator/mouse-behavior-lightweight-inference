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

`parallel_behavior_fsm.enabled` is an execution switch. With the default
`true`, individual, extended pair, contact and group regions use the temporal
FSM exactly as described above. With `false`, those parallel-region events are
not emitted; standard-engine chase/attack remain available because they own a
separate causal FSM. The only currently supported mode is `active`. Unknown
mode values fail during coordinator construction instead of being silently
reported in metadata without affecting execution.
