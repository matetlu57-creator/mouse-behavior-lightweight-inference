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
- `historical_*` and `original/` are audit material, not runtime imports.

The current tracker and lightweight analyzer are intentionally not split into
many copied files. Their future extraction must proceed one responsibility at
a time with regression tests, preserving the root compatibility wrappers.
