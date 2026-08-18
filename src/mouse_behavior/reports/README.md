# Reports boundary

Reports should be generated from persisted events, metrics, metadata and the
exact configuration snapshot. Report generation must not run model inference
again or modify source CSV files.
