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
settings, software mode and elapsed time. Do not commit generated files.
