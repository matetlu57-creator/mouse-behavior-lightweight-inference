# Coding style

- Put reusable logic in `src/mouse_behavior/`.
- Keep CLI parsing and batch orchestration in `scripts/`.
- Use `logging.getLogger(__name__)`; library modules must not use `print()`.
- Keep thresholds in YAML configuration.
- Add type annotations and docstrings at public boundaries.
- Do not add Python modules or CLI wrappers to the repository root; use the
  `mouse_behavior` package and `scripts/` entry points.
