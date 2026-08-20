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

## Second-pass implementation status

The follow-up refactor addresses or narrows findings 1 through 4 without
shortening the timeline seen by any FSM:

- Pair prefiltering/window construction and candidate-only metrics now form an
  explicit `_PairWorkset` stage. Candidate-index mapping is tested separately
  from all-pair summary order.
- Candidate-pair analysis now has its own `_analyze_candidate_pairs` boundary,
  and stable event ID/time assignment is isolated in a finalization helper.
- Contact extraction no longer converts the complete wide DataFrame to Python
  row dictionaries. It reads eight required arrays and constructs categorical
  states only for actual contact frames while retaining `None` for every other
  timeline position.
- The standard engine narrows row dictionaries to evidence inputs, converts
  level-independent arrays once, vectorizes contact-type assignment, and lets
  the lightweight owner enrich its temporary pair DataFrame in place. The
  public default remains non-mutating.
- `parallel_behavior_fsm.enabled=false` now suppresses parallel-region events,
  while `mode` accepts only the implemented `active` value. Metadata reports
  the effective coordinator values and execution semantics.

The 190-pair distance/heading prefilter remains intentionally vectorized over
all pairs. It is the inexpensive discovery layer that decides which pairs are
worth heavy geometry and FSM work; replacing it with Python spatial indexing
at 20 mice would add complexity without evidence of a useful runtime gain.

Finding 5 is narrowed by `tools/compare_analysis_outputs.py`, a reproducible
release gate that compares every generated artifact and returns failure for a
missing or changed file. It canonicalizes only runtime measurements and paths
rooted in the two output directories. A tiny deterministic cache/video golden
fixture is still not checked into the repository; generated videos and real
YOLO caches remain excluded from Git by the repository boundary policy.

### Second-pass verification

The same 156-frame annotated attack clip and YOLO cache used for the first-pass
golden comparison were run from commit `35d27dd` and the second-pass working
tree. All four primary CSV files, arena artifacts, website annotations,
`tracks.jsonl` and exported video remained byte-identical by SHA-256. The
machine-readable input and output signatures are retained in
`docs/validation/2026-08-19_second_pass_output_equivalence.json`; reruns use
`tools/compare_analysis_outputs.py` as a non-zero-exit release gate.

| Measurement | First pass | Second pass | Change |
|---|---:|---:|---:|
| `candidate_pair_analysis` | 3.569 s | 3.266 s | 8.5% faster |
| Total short-clip elapsed time | 6.129 s | 5.726 s | 6.6% faster |
| Full pytest suite | 52 passed | 67 passed | 15 additional tests |

A synthetic contact benchmark used 18,321 rows, 98 DataFrame columns and six
sparse contact runs. Median contact extraction time changed from 0.319 s to
0.00385 s, an 82.9x local speedup with the same six events. This is a local
hotspot result, not an end-to-end ten-minute-video prediction.

A separate 5,000-row public standard-engine microbenchmark was unchanged
within measurement noise (approximately 1.002 s in both versions). The narrow
record view therefore has no standalone wall-time claim; its immediate value
is lower Python-object materialization and a cleaner input boundary. The
measured end-to-end pair-stage improvement comes from the combined contact,
owned-DataFrame and orchestration changes.
