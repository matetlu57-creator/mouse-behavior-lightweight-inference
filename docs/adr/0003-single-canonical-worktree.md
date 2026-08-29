# ADR 0003：统一使用唯一的模块化工作树

## 状态

已接受

## 背景

项目目录曾同时保留两个 Git worktree：旧的算法模块化工作树和当前的根入口模块化工作树。
两个工作树都指向同一个 GitHub 仓库，但分支、渲染器、配置、测试和未提交修改并不一致。
从不同对话进入不同工作树会使同一个视频使用不同的行为事件 CSV、渲染布局和追踪补偿逻辑，
导致无法从输出视频判断实际代码版本。

## 决策

`mouse_behavior_root_entrypoint_modularization` 是唯一的本地开发和运行工作树。

- 可复用代码统一修改 `src/mouse_behavior/`；
- 命令行和批处理入口统一修改 `scripts/`；
- 测试统一修改 `tests/`；
- 配置统一修改 `configs/`；
- 结果、缓存、模型和原始视频放在仓库外或被忽略的输出目录；
- 并行实验必须使用仓库外的 Git worktree，完成后及时移除；
- 不再在当前工作区内保留第二份源码或旧分支工作树；
- 渲染器只读取已经保存的事件 CSV，不根据视频名或目录名重新生成行为。

## 取舍

保留旧 worktree 的完整提交历史，不把旧渲染器复制到当前模块中。旧渲染器中的侧栏、中文
显示、行为层级和接触事件能力已经由当前 `visualization/overlay.py` 与
`visualization/rendering.py` 统一实现；只保留其可验证的测试意图，避免重复实现和后续漂移。

## 验证

每次切换或合并功能后，从唯一工作树执行：

```text
git worktree list
python -m pytest -q
python scripts/run_quality.py --ci
```

输出目录应记录输入事件 CSV、配置和代码提交信息；旧生成物不能作为当前代码的验收证据。
