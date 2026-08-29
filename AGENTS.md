# Repository instructions

These rules apply to this repository in addition to higher-priority user and
system instructions.

## Stable boundaries

- Reusable production code belongs under `src/mouse_behavior/`.
- User-facing CLI and batch entry points belong under `scripts/`.
- Repository maintenance checks belong under `tools/`.
- Tests belong in `tests/unit`, `tests/integration`, `tests/regression`, or
  `tests/e2e` according to the observable contract they protect.
- Root-level Python and PowerShell files are compatibility entry points. Do not
  add new root scripts or remove an existing compatibility path without a
  migration test and documentation.
- Videos, detector caches, model weights, local annotations, logs, and analysis
  outputs must remain untracked.

## Behavior compatibility

Do not change scientific thresholds, pair-selection semantics, FSM transitions,
event schemas, actor/target roles, or source-frame timing as part of a repository
or performance-only change. Use `tools/compare_analysis_outputs.py` for a fixed
input A/B comparison when a change can affect generated events.

## Canonical checks

在 Windows 上优先使用项目本地 `.venv` 运行质量门。仓库提供的 PowerShell 包装器
会自动选择 `.venv\Scripts\python.exe`，避免误用装有不兼容 GPU/DLL 依赖的全局
Anaconda base 环境：

```text
.\scripts\run_quality.ps1 -CI
```

也可以直接运行项目本地解释器：

```text
.\.venv\Scripts\python.exe scripts/run_quality.py --ci
```

For a focused change, select one or more named steps:

```text
.\scripts\run_quality.ps1 -Step unit_test,repository
```

单独运行 pytest 时使用：

```text
.\scripts\run_pytest.ps1 tests/unit
```

YOLO Pose 缓存生成属于另一条运行时边界，应使用已经验证可导入 Torch 的
`yolo26` 环境，不要把 Torch/CUDA 依赖安装进测试 `.venv`。

Before a GitHub push, also inspect `git status --short --branch`, the staged
diff, outgoing commits, tracked file sizes, and the configured remote.
