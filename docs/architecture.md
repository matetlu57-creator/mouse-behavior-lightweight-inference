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
- `behavior/` contains the behavior facades and focused algorithm layers:
  `standard_evidence.py`, `standard_fsm.py`, `ethogram.py` and
  `pair_analysis.py`.
- `data/` defines output contracts.
- `io/` defines canonical run-directory naming, CSV writing and arena-boundary
  persistence.
- `scripts/` contains thin command-line and validation entry points.
- `tools/` contains repository and development checks.
- `tests/regression/fixtures/` contains only the minimal frozen legacy code
  required by a regression; all other historical versions live in Git.

The lightweight analyzer is an orchestration facade. Its reusable work is
split across `tracking/cache.py`, `preprocessing/geometry.py`,
`preprocessing/kinematics.py`, `preprocessing/pair_features.py`,
`behavior/ethogram.py`, `behavior/pair_analysis.py` and
`visualization/rendering.py`. The standard behavior engine keeps its public
facade in `standard_behavior_engine.py`, while continuous evidence and FSM
transitions live in `behavior/standard_evidence.py` and
`behavior/standard_fsm.py`.

The cage-boundary facade follows the same rule: learning is in
`preprocessing/arena_learning.py`, and JSON/PNG/video audit output is in
`io/arena_boundary.py`. The maintained CLI wrappers live under `scripts`;
remaining root files are compatibility or full-pipeline entry points only.
