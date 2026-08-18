# Experiment configuration rules

Use one file per experiment, for example `exp001_parallel_fsm.yaml`:

```yaml
extends: ../profiles/balanced.yaml
experiment:
  id: EXP001
  description: "Compare the parallel FSM with the current baseline"
  dataset: "path/to/frozen/dataset"
  seed: 143
```

An experiment file must record its hypothesis, dataset split, profile, random
seed where relevant, and expected validation command. Never overwrite a
profile to run one experiment.
