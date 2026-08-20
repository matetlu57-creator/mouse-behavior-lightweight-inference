# Architecture

```text
input video + completed Pose cache
                │
                ▼
        core / pipeline facade
                │
                ▼
 lightweight cache tracking and geometry
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   behavior   contact   group/individual
       │        │         │
       └────────┼─────────┘
                ▼
       events + metadata + export
```

## Boundaries

- `src/mouse_behavior/core/` coordinates runs and does not contain detector or
  behavior thresholds.
- `models/` exposes Pose-cache model interfaces.
- `behavior/` exposes stable behavior-engine facades; the existing standard
  engine and parallel FSM remain independently testable.
- `data/` defines output contracts.
- `io/` defines canonical run-directory naming.
- `scripts/` contains thin command-line and validation entry points.
- `tools/` contains repository and development checks.
- `tests/regression/fixtures/` contains only the minimal frozen legacy code
  required by a regression; all other historical versions live in Git.

The current tracker and lightweight analyzer are intentionally not split into
many copied files. Their future extraction must proceed one responsibility at
a time with regression tests. The maintained CLI wrappers live under `scripts/`;
only the remaining full-pipeline and import-compatibility files stay at the
repository root until their consumers are migrated.
