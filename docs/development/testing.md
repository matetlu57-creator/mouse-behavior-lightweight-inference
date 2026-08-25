# 测试与质量门

所有命令都从仓库根目录执行。项目使用 pytest 分层测试，并通过
scripts/run_quality.py 统一调用格式、lint、类型、仓库、测试、覆盖率和构建检查。

## 常用命令

~~~powershell
python -m pytest -q
python scripts/validate_repository.py
python scripts/run_quality.py
python scripts/run_quality.py --ci
~~~

修改代码后，先运行能直接反驳本次修改的 targeted tests，再运行完整测试。修改
行为阈值或 FSM 时，还要运行回归样例和北医示例覆盖验证。

## 测试分层

- tests/unit/：几何、FSM、日志和单模块行为；
- tests/integration/：跨模块流程、CSV、网站导出、渲染和仓库契约；
- tests/regression/：冻结输出、历史版本比较和性能回归；
- tests/e2e/：安装后的命令行和端到端冒烟测试；
- tests/regression/fixtures/：小型源代码或合成输入，不放视频、缓存和模型。

日志测试使用 pytest 的 caplog 检查级别、消息和上下文，不通过捕获生产
print() 来证明日志功能。

## 质量门

CI 使用 .quality-gate.toml 中的 ci profile。质量门包含：

- Ruff format check；
- Ruff lint；
- 增量 mypy 类型检查；
- 仓库边界和敏感文件检查；
- pytest coverage；
- source 和 wheel 构建及产物检查。

mypy 当前只覆盖稳定的配置、版本、schema、路径和计时器契约。扩展类型检查应
按模块逐步进行，不能用全局忽略掩盖错误。

## 真实视频验证记录

真实视频验证需要记录：

- 输入视频或数据集；
- YOLO Pose 缓存位置和模型版本；
- 配置 profile、采样步长和预期小鼠数；
- Git commit；
- 总耗时和阶段耗时；
- 视频级覆盖与帧级标注是否存在。

文件夹级行为标签只能用于覆盖检查，不能单独证明事件级 Precision、Recall、F1
或 actor/target 准确率。
