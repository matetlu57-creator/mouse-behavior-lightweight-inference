# Data and output format

One run is written under an ignored `outputs/<run-id>/` directory. The current
lightweight analyzer writes these compatibility files:

| File | Purpose |
|---|---|
| `lightweight_behavior_events.csv` | chase/attack and extended behavior events |
| `lightweight_contact_events.csv` | nose-head/nose-tail contact events |
| `lightweight_pair_summary.csv` | per-pair computation and FSM diagnostics |
| `lightweight_analysis_metadata.json` | FPS, config, candidate-pair and timing provenance |
| `annotation_website_import/` | complete-video website-compatible package |

The generated metadata records the source video, cache, resolved runtime
settings, software mode and total elapsed time. Its additive
`stage_timings_s` object reports wall-clock seconds for video probing, arena
preparation, cache loading, kinematics, pair filtering, pair metrics, candidate
pair analysis, global event finalization, website export and CSV output. This
helps distinguish algorithm time from export time without changing event
semantics. Do not commit generated files.

The `parallel_behavior_fsm` metadata object reports the effective `enabled`,
`mode` and `execution_semantics` values used by the run. This is execution
provenance rather than a copy of unchecked configuration text.
