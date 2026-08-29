# 文档总索引

这里是项目的长期维护文档入口。根目录的 README.md 负责项目概览和最短运行路径；
docs/ 负责安装、配置、架构、算法、输出格式和开发流程。

## 推荐阅读顺序

1. [快速开始](quickstart.md)
2. [安装与环境](installation.md)
3. [配置说明](configuration.md)
4. [仓库架构](architecture.md)
5. [完整仓库结构说明](repository_structure.md)
6. [算法说明](algorithms.md)
7. [输出格式](data_format.md)
8. [测试与开发](development/testing.md)
9. [代码审查与性能基线](code_review.md)

## 目录职责

~~~text
src/mouse_behavior/   可复用 Python 模块和稳定接口
scripts/              CLI、批处理、验证和校准入口
configs/              default、profiles 和 experiments
tests/                unit、integration、regression、e2e
docs/                 用户文档和开发文档
examples/             最小可运行 API 与配置示例
tools/                仓库检查、构建检查和结果比较
.github/              CI、Issue 模板和 PR 模板
~~~

根目录不包含新的 Python 模块或命令行包装器。历史版本由 Git commit、branch 和
tag 管理，并行开发使用仓库外 worktree；不要复制 v2、final2 或其他版本目录。
只有回归测试必需的旧实现才保留在 tests/regression/fixtures/。

## 常用入口

- [贡献规范](../CONTRIBUTING.md)：branch、worktree、模块化、logging、pytest 和
  AI 协作约定；
- [配置说明](configuration.md)：默认配置、profile 和实验配置；
- [统一质量门](../scripts/run_quality.py)：格式、lint、类型、测试、覆盖率和构建
  检查的统一入口；
- [仓库检查器](../scripts/validate_repository.py)：检查目录边界、敏感文件和大文件；
- [GitHub Actions](../.github/workflows/test.yml)：push 和 pull request 的 CI。

## 设计原则

- 可复用逻辑进入 src/，脚本只负责入口编排；
- 行为证据、显示上下文和网站导出边界明确区分；
- 日志使用 Python logging，不在可复用模块中使用 print()；
- 每次算法变更都要有 pytest 回归和真实样例验证记录；
- 视频、缓存、私人标注和未登记模型权重不进入 Git；默认 Pose 权重通过 Git LFS 管理。
