# Coding style

- Put reusable logic in `src/mouse_behavior/`.
- Keep CLI parsing and batch orchestration in `scripts/`.
- Use `logging.getLogger(__name__)`; library modules must not use `print()`.
- Keep thresholds in YAML configuration.
- Add type annotations and docstrings at public boundaries.
- Preserve root compatibility wrappers until a documented major release removes
  them.
