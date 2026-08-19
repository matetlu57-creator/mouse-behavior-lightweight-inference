# Testing

Run the fast deterministic suite from the repository root:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
python tools/check_repository.py
```

Test responsibilities:

- existing `tests/test_*.py`: unit and compatibility tests;
- `tests/unit/`: new isolated algorithm tests;
- `tests/integration/`: cross-module pipeline tests;
- `tests/regression/`: frozen-result and version comparison tests;
- `tests/fixtures/`: small synthetic inputs only, never videos or model files.

For real videos, report the cache, configuration profile, Git commit, sample
stride, timing and the fact that folder-level labels cannot provide event-level
Precision/Recall/F1.
