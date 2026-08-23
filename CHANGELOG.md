# Changelog

All notable changes to this project are recorded here. The project follows
semantic-versioning conventions for releases; Git branches and tags are the
source of truth for historical versions.

## [Unreleased]

- Move the complete detector/identity pipeline into
  `mouse_behavior.full_pipeline`, add maintained script/module/console entry
  points, and remove every repository-root Python compatibility file.
- Replace full-pipeline startup `print()` calls with logging and add pytest
  `caplog` coverage for logging configuration and stage timing.
- Document and exercise an external Git worktree workflow so parallel branches
  do not become copied version directories inside the repository.
- Split the lightweight inference, standard behavior and cage-boundary
  implementations into reusable preprocessing, tracking, behavior, I/O and
  visualization modules while preserving compatibility facades and output
  contracts.
- Standardize the repository around `src/mouse_behavior`, `scripts`,
  `configs`, `tests`, `docs`, and `examples`.
- Refresh the GitHub-facing Markdown navigation so the repository homepage,
  algorithm overview, architecture guide, and development guide point to the
  same maintained directory boundaries.
- Add inherited configuration profiles for fast, balanced, and high-accuracy
  runs.
- Add the reusable configuration, pipeline, data-schema, path, logging, and
  timing facades without changing the existing lightweight algorithm.
- Add repository checks, CI, contribution guidance, and a draft-release
  workflow through GitHub pull requests.
- Split tests into unit, integration, regression and end-to-end layers while
  preserving the minimum v1.38 source fixtures required for output regression.
- Add one TOML-driven quality gate for Ruff, incremental mypy, repository
  validation, pytest coverage and source/wheel build inspection.
- Remove copied historical source trees, duplicate review reports and obsolete
  generated validation manifests; Git history and ADR-0001 preserve provenance.
- Remove the obsolete root CLI compatibility wrappers and `_script_compat.py` for
  cache building, Beiyi validation, cache reruns, threshold calibration and
  threshold sweeps; the maintained commands now live only under `scripts/`.

## [0.1.0] - 2026-08-18

- Preserve the v1.43 lightweight cache analysis path and its compatibility
  entry points.
- Preserve the parallel hierarchical behavior FSM and its event-level
  validation artifacts as local experiment history, not as tracked video or
  cache data.
