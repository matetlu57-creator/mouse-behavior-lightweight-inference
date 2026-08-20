# Changelog

All notable changes to this project are recorded here. The project follows
semantic-versioning conventions for releases; Git branches and tags are the
source of truth for historical versions.

## [Unreleased]

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

## [0.1.0] - 2026-08-18

- Preserve the v1.43 lightweight cache analysis path and its compatibility
  entry points.
- Preserve the parallel hierarchical behavior FSM and its event-level
  validation artifacts as local experiment history, not as tracked video or
  cache data.
