# Repository instructions

These rules apply to this repository in addition to higher-priority user and
system instructions.

## Stable boundaries

- Reusable production code belongs under `src/mouse_behavior/`.
- User-facing CLI and batch entry points belong under `scripts/`.
- Repository maintenance checks belong under `tools/`.
- Tests belong in `tests/unit`, `tests/integration`, `tests/regression`, or
  `tests/e2e` according to the observable contract they protect.
- Root-level Python and PowerShell files are compatibility entry points. Do not
  add new root scripts or remove an existing compatibility path without a
  migration test and documentation.
- Videos, detector caches, model weights, local annotations, logs, and analysis
  outputs must remain untracked.

## Behavior compatibility

Do not change scientific thresholds, pair-selection semantics, FSM transitions,
event schemas, actor/target roles, or source-frame timing as part of a repository
or performance-only change. Use `tools/compare_analysis_outputs.py` for a fixed
input A/B comparison when a change can affect generated events.

## Canonical checks

Run the project-local quality gate from the repository root:

```text
python scripts/run_quality.py --ci
```

For a focused change, select one or more named steps:

```text
python scripts/run_quality.py --step unit_test --step repository
```

Before a GitHub push, also inspect `git status --short --branch`, the staged
diff, outgoing commits, tracked file sizes, and the configured remote.
