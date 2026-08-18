# Configuration

Configuration inheritance is explicit YAML composition:

```text
configs/default.yaml
        │
        ├── configs/profiles/fast.yaml
        ├── configs/profiles/balanced.yaml
        └── configs/profiles/high_accuracy.yaml
```

Each profile extends the default and overrides only the parameters relevant to
that operating mode. Experiment files under `configs/experiments/` extend a
profile and record their dataset, hypothesis and seed.

Use `mouse_behavior.config.load_config()` in code that needs resolved values.
Do not call `yaml.safe_load()` directly for a profile, because that would leave
the `extends` indirection unresolved.

The legacy root `mouse_chase_attack_config.yaml` remains available for older
full-pipeline scripts. New lightweight commands should use
`configs/profiles/balanced.yaml` or `configs/default.yaml`.
