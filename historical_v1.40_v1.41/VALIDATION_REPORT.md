# Validation report — v1.40.1-performance-preserving

## Scope

Validated the uploaded main program, base module, and YAML after the result-preserving optimization patch. No full-video claim is made because the MP4, model weights, and project-side recovery/ReID modules were not included in this session.

## Environment

- Python: `3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]`
- Platform: `Linux-6.18.35-x86_64-with-glibc2.41`
- Libraries: `{"cv2": "4.13.0", "numpy": "2.3.5", "pandas": "2.2.3", "pytest": "9.0.2", "scipy": "1.17.0", "yaml": "6.0.3"}`

## Static checks

- `python -m compileall`: PASS
- `git diff --check`: PASS for both Python files
- deterministic regeneration via `apply_optimizations.py`: PASS
- optimized YAML byte-identical to uploaded YAML: PASS

## Regression suite

```text
REGRESSION: PASS
derived_geometry_old_seconds=0.660317
derived_geometry_new_seconds=0.002288
derived_geometry_speedup=288.599390
pair_features_old_seconds=2.398152
pair_features_new_seconds=0.852761
pair_features_speedup=2.812221
pair_store_old_seconds=1.656421
pair_store_new_seconds=0.033418
pair_store_speedup=49.566760
```

```text
.                                                                        [100%]
1 passed in 7.00s
```

The suite compares the original and optimized implementations for:

- public `Detection` dataclass field schema and `asdict()` key order;
- center/body geometry before and after memoization, invalidation, deepcopy/pickle boundaries;
- appearance descriptor, normalized pose, anchor, heading, brightness, and white-score outputs;
- occlusion-cluster context/state/debug output while confirming duplicate evidence evaluation is removed;
- all fields of base and high-recall directional pair-feature dataclasses;
- forward/reverse floating-point ordering, next-frame cache invalidation;
- frame-record field order and values;
- SQLite `add()` versus `add_many()` logical tables;
- Stage 2 `PairDataFrameStore` output for every pair and a missing key;
- checkpoint exclusion of derivable caches.

## SHA-256

### Uploaded originals

- `mouse_chase_attack_config.yaml`: `9fda82030040c79d34a86377af5fac40f56c5aaef791b5b2d9910a497760d0fb`
- `mouse_chase_attack_extractor_base.py`: `bedf1d5ee9409fa461df112279198c9f740d759e2c58e31b15dcdbf3dd0ae915`
- `mouse_chase_attack_high_recall.py`: `cf3620a0626882aa35f80090b6145c997f3f6ef9269cb5a5732de313b8ba3699`

### Optimized release

- `mouse_chase_attack_high_recall.py`: `e239d1b0e3c435e654100fdaaa98cb6971aac22a4f87baef73582baf3a1d2243`
- `mouse_chase_attack_extractor_base.py`: `41881a8f43deb9f815a89b27fc011f5bd3aad1830021473d67cb9bc0218338d5`
- `mouse_chase_attack_config.yaml`: `9fda82030040c79d34a86377af5fac40f56c5aaef791b5b2d9910a497760d0fb`

## Interpretation of microbenchmarks

The reported speedups are isolated hot-loop measurements in this container. They are useful for confirming that the intended repeated work was removed, but they must not be multiplied and are not a substitute for end-to-end profiling on the actual video, GPU, model, storage, and full project dependency set.
