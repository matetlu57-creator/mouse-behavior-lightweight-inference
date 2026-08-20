# Testing

Run the local deterministic quality profile from the repository root:

```powershell
python scripts/run_quality.py
```

Run the same coverage and package-build profile as GitHub Actions before a PR:

```powershell
python scripts/run_quality.py --ci
```

Test responsibilities:

- `tests/unit/`: isolated algorithm and module tests;
- `tests/integration/`: cross-module pipeline tests;
- `tests/regression/`: frozen-result and version comparison tests;
- `tests/e2e/`: package and script CLI smoke tests;
- fixture directories contain small synthetic or source-only frozen inputs,
  never videos, pose caches or model files.

The mypy gate is intentionally incremental. It covers configuration and stable
data/path/timing contracts today; expanding it into the large legacy analyzer
must be done module by module, without blanket ignores or algorithm changes.

For real videos, report the cache, configuration profile, Git commit, sample
stride, timing and the fact that folder-level labels cannot provide event-level
Precision/Recall/F1.
