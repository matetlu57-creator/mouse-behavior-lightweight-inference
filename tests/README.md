# Test layout

The existing test files remain at this level for backward compatibility with
the current pytest configuration. New tests should be placed in the following
layers:

```text
tests/unit/          pure functions and single-module behavior
tests/integration/  cross-module pipeline contracts
tests/regression/   frozen outputs and version comparisons
tests/fixtures/     small synthetic fixtures only
```
