# 文档总索引

这里是 GitHub 仓库的工程文档入口。根目录的 [README.md](../README.md) 负责
五分钟项目概览，[README_FIRST.md](../README_FIRST.md) 负责 v1.43 行为引擎
和并行 FSM 说明；本目录负责可维护的安装、架构、配置、算法和开发流程。

## 第一次阅读

1. [安装与环境](installation.md)
2. [快速运行](quickstart.md)
3. [仓库架构](architecture.md)
4. [配置继承和 profile](configuration.md)
5. [数据、输出和隐私边界](data_format.md)
6. [算法说明](algorithms.md)
7. [测试与开发](development/testing.md)

## GitHub 仓库结构

```text
src/mouse_behavior/   可复用 Python 模块和稳定接口
scripts/              CLI、批处理、验证和校准入口
configs/              default、profiles、experiments
tests/                unit、integration、regression、fixtures
docs/                 用户文档和开发文档
examples/             最小可运行 API/配置示例
tools/                仓库边界检查和维护工具
.github/              CI、Issue 模板和 PR 模板
```

根目录的同名 Python 文件、`historical_*` 和 `original/` 有明确的兼容或
审计用途。新功能不要复制目录形成 `v2`、`final2` 等版本；使用 Git 分支、
tag 或 worktree 管理版本。

## 开发入口

- [CONTRIBUTING.md](../CONTRIBUTING.md)：Git workflow、logging、pytest 和 AI
  协作提示词；
- [配置目录说明](../configs/README.md)：profile 和实验配置规则；
- [仓库检查工具](../tools/check_repository.py)：拒绝把视频、权重、缓存和
  结果文件加入 tracked 文件；
- [GitHub Actions](../.github/workflows/test.yml)：每次 push/PR 自动执行测试、
  compileall 和仓库边界检查。
