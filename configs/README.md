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
│   └── high_accuracy.yaml        # high-recall analysis overlay
└── experiments/
    └── README.md                 # experiment naming and provenance rules
```

Configuration inheritance is resolved by `mouse_behavior.config.load_config`.
Each profile can override only the fields it is testing. The resolved mapping
is copied into every output directory by the running pipeline where applicable.

The root `mouse_chase_attack_config.yaml` remains a compatibility file for the
legacy full pipeline and older notebooks. New code should default to
`configs/default.yaml`.
