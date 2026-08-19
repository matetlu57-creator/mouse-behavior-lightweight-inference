# Code review and performance baseline

This document records the first low-risk maintainability and runtime pass on
the lightweight behavior analyzer. The review deliberately preserves behavior
thresholds, FSM transitions, public entry points, CSV schemas and output file
names.

## Implemented in this pass

- Moved causal rolling sums and trajectory correlations from the main analyzer
  into `mouse_behavior.utils.rolling`.
- Replaced the sparse rolling-correlation frame loop with contiguous active-run
  cumulative sums. An inactive frame still resets all rolling history for that
  pair.
- Reused converted `behavior_speed` and body-length distance arrays while
  constructing each pair DataFrame.
- Extended the shared `Timer` with explicit `start()`, `stop()` and an optional
  result sink.
- Added `stage_timings_s` to `lightweight_analysis_metadata.json` so later
  optimization decisions can be based on measured pipeline costs.
- Added reference-based tests for dense and sparse rolling features, active
  window resets, invalid frames, NaNs and timer recording.

## Verification evidence

All figures below were measured on the same Windows workstation. They are
engineering baselines rather than cross-machine performance guarantees.

| Check | Result |
|---|---:|
| Full pytest suite | 52 passed |
| Repository boundary check | passed |
| Existing performance regression suite | passed |
| Compile check for `src`, `scripts` and `tests` | passed |
| 18,321 frames, 48 candidate pairs, 5.15% active rolling-correlation benchmark | 3.866 s to 0.075 s (51.7x local speedup) |
| Maximum absolute old/new rolling-correlation difference | 0 |

An end-to-end A/B run used a 156-frame annotated attack clip, 20 expected mice,
21 retained candidate pairs, `sample_stride=1` and the same YOLO cache and
configuration. The old and new versions produced byte-identical SHA-256 hashes
for all four primary CSV files, arena artifacts, website annotations,
`tracks.jsonl` and the exported video. Total elapsed time changed from 6.628 s
to 6.336 s. This short-run difference is useful compatibility evidence but is
too small to predict the speedup of a ten-minute video.

## Stage timings

The metadata now records wall-clock seconds for:

- `setup_and_video_probe`
- `arena_boundary`
- `track_cache`
- `kinematics`
- `pair_filter_and_windows`
- `pair_metrics`
- `candidate_pair_analysis`
- `global_events_and_finalization`
- `website_export`
- `csv_output`

The sum may be lower than `elapsed_s` because metadata assembly, logging and
small coordination operations are not forced into an artificial stage.

## Remaining review findings

The following work should be addressed in measured, separately validated
changes instead of one large rewrite:

1. `analyze()` still combines input setup, tracking, pair analysis, event
   finalization and output publication. Extracting pure preparation, per-pair
   analysis and output functions would improve maintainability, but first
   requires a checked-in end-to-end golden-cache fixture.
2. Every candidate pair still creates a wide full-timeline Pandas DataFrame.
   The standard engine then copies it and builds Python row dictionaries for
   active evidence. Stage timings should determine whether this is the next
   dominant long-video cost.
3. Contact extraction still materializes the full pair DataFrame as Python
   records. A sparse-window implementation is promising, but must preserve
   categorical FSM span boundaries and actor/target roles.
4. `parallel_behavior_fsm.enabled` and `mode` are reported in metadata but do
   not currently define an execution bypass. Their compatibility semantics
   must be specified and tested before changing behavior.
5. The repository has useful microbenchmarks but no committed benchmark that
   drives the current `analyze()` entry point from a small deterministic cache
   through all CSV outputs.

For future performance work, compare sorted event outputs and stable hashes in
addition to unit tests. Reducing context padding or changing candidate-pair
rules is a scientific recall tradeoff and must be evaluated separately from
pure implementation optimization.
