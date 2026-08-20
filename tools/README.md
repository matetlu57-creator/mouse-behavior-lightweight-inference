# Development tools

Tools support repository maintenance and reproducibility checks. They are not
the normal inference entry points; those belong in `scripts/`.

```text
tools/
├── check_repository.py             # tracked-file, required-path and secret-risk checks
├── compare_analysis_outputs.py     # complete baseline/current output equivalence gate
├── inspect_distribution.py         # verify built sdist/wheel contents and metadata
└── README.md
```

Normal development should call `python scripts/run_quality.py` or
`python scripts/run_quality.py --ci`; `scripts/validate_repository.py` is the
stable wrapper used by contributors and CI.

Run the output-equivalence gate after analyzing the same video/cache/config
with a baseline revision and the proposed revision:

```powershell
python .\tools\compare_analysis_outputs.py `
  F:\validation\baseline `
  F:\validation\optimized `
  --report F:\validation\comparison.json
```

The command compares every generated file. It canonicalizes only measured
runtime fields (`elapsed_s`, `stage_timings_s`) and absolute paths rooted in
the two output directories; missing files or any scientific-output change
return a non-zero exit code.
