# 文档总索引

这里是 GitHub 仓库的工程文档入口。根目录的 [README.md](../README.md) 负责
五分钟项目概览；本目录负责可维护的安装、架构、配置、算法和开发流程。

## 第一次阅读

1. [安装与环境](installation.md)
2. [快速运行](quickstart.md)
3. [仓库架构](architecture.md)
4. [配置继承和 profile](configuration.md)
5. [数据、输出和隐私边界](data_format.md)
6. [算法说明](algorithms.md)
7. [测试与开发](development/testing.md)
8. [代码审查与性能基线](code_review.md)

## GitHub 仓库结构

```text
src/mouse_behavior/   可复用 Python 模块和稳定接口
scripts/              CLI、批处理、验证和校准入口
configs/              default、profiles、experiments
tests/                unit、integration、regression、e2e 和小型 fixtures
docs/                 用户文档和开发文档
examples/             最小可运行 API/配置示例
tools/                仓库边界检查和维护工具
.github/              CI、Issue 模板和 PR 模板
```

仓库根目录不包含 Python 模块或 CLI 包装器；正式实现统一在
`src/mouse_behavior/`，仓库运行入口统一在 `scripts/`。历史源码和验证记录由
Git commit/tag 管理；并行开发使用仓库外 worktree，不复制 `v2`、`final2`
目录。仅回归测试所需的旧实现放在测试 fixture 中，迁移决策见
[ADR-0002](adr/0002-package-full-pipeline-and-remove-root-entrypoints.md)。

## 开发入口

- [CONTRIBUTING.md](../CONTRIBUTING.md)：Git workflow、logging、pytest 和 AI
  协作提示词；
- [配置目录说明](../configs/README.md)：profile 和实验配置规则；
- [统一质量入口](../scripts/run_quality.py)：从 `.quality-gate.toml` 执行与 CI
  一致的格式、lint、类型、测试、覆盖率和构建检查；
- [仓库检查工具](../tools/check_repository.py)：拒绝把视频、权重、缓存、
  大文件、凭据风险和结果文件加入 tracked 文件；
- [GitHub Actions](../.github/workflows/test.yml)：每次 push/PR 执行完整 CI profile。
