# Configuration layout

`configs/default.yaml` is the maintained entry configuration. It inherits the
legacy root configuration so old commands remain compatible while new runs
can use profiles and experiments without editing Python.

```text
configs/
├── default.yaml                  # inherited project default
├── profiles/
│   ├── fast.yaml                 # lower-cost analysis overlay
│   ├── balanced.yaml             # normal analysis overlay
│   ├── high_accuracy.yaml        # high-recall analysis overlay
│   └── beiyi.yaml                # short Beiyi clips without arena gating
└── experiments/
    └── README.md                 # experiment naming and provenance rules
```

Configuration inheritance is resolved by `mouse_behavior.config.load_config`.
Each profile can override only the fields it is testing. The resolved mapping
is copied into every output directory by the running pipeline where applicable.

The root `mouse_chase_attack_config.yaml` remains a compatibility file for the
legacy full pipeline and older notebooks. New code should default to
`configs/default.yaml`.

For short Beiyi example videos, use `configs/profiles/beiyi.yaml`. It sets
`adaptive_arena.mode: disabled`, which skips per-video motion heatmap learning
and disables arena gating entirely. It does not add or apply a fixed cage
polygon. Long ordinary videos keep the default `adaptive_arena.mode: learned`
behavior.
