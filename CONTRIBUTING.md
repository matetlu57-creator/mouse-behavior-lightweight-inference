# 开发约定

这份文档是本仓库的最小工程化约定。目标不是一次性把所有历史代码重写，而是让新功能从一开始就有清晰的模块边界、可重复的测试和可追踪的运行日志。

## 目录职责

```text
src/mouse_behavior/       可复用 Python 模块；导入模块不应启动任务
scripts/                  命令行、批处理和一次性评估入口
tests/                    pytest 单元测试和回归测试
original/                 历史原始版本，仅用于对照
historical_*/             历史验证材料和迁移记录
weights/、outputs/        本地模型和结果，不提交到 Git
```

完整旧管线的 `mouse_chase_attack_high_recall.py` 和底层提取器暂时保留在根目录，是因为它们仍然承担旧安装器/回归测试的兼容入口，并且依赖仓库外的模块。新轻量路径的可复用部分已经放入 `src/mouse_behavior/`；根目录同名 `.py` 文件只是兼容导出层。

## 本地开发命令

在仓库根目录执行：

```powershell
# 创建/激活项目环境后安装运行和开发依赖
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# 可选：以 editable 方式安装，使任意工作目录都能 import 包
python -m pip install -e .

# 运行全部测试
python -m pytest -q

# 运行一个测试文件
python -m pytest tests/test_lightweight_contact_detection.py -q

# 检查模块和脚本语法
python -m compileall -q src scripts
```

## 日志约定

业务代码使用 `logging.getLogger(__name__)`，不要在可复用模块里使用 `print()`。命令行入口提供 `--log-level`，例如：

```powershell
python .\scripts\build_lightweight_pose_cache.py ... --log-level DEBUG
python .\scripts\run_lightweight_behavior_inference.py ... --log-level INFO
```

`INFO` 记录阶段进度和产物位置，`WARNING` 记录可继续运行但需要复核的降级，`ERROR`/`EXCEPTION` 记录失败。测试或上层应用可以自行配置 handler；库模块不主动改写全局日志配置。

## Git workflow

一个功能或实验使用一个分支，结果文件不要靠根目录复制出 `v2`、`final2`、`new` 文件夹来管理：

```powershell
git switch main
git pull --ff-only
git switch -c feat/short-description

python -m pytest -q
git diff --check
git status --short
git add src scripts tests README.md CONTRIBUTING.md pyproject.toml
git commit -m "refactor: separate reusable modules and scripts"
git push -u origin feat/short-description
```

如果需要同时试验两个互不影响的方向，可以使用 `git worktree` 建立第二个工作目录；不要把第二个版本复制进当前仓库根目录。提交前先确认 `git status`，不要把本地视频、缓存、模型和分析结果加入提交。

## 给 Codex/AI 的可维护代码提示词

不要只说“帮我把这个脚本改好”。一个可复用的任务描述至少包括：

1. 目标和不变条件：输入、输出、算法语义、性能要求，哪些行为不能改变；
2. 目录边界：哪些代码必须放 `src/`，哪些只是 `scripts/` CLI，哪些文件不能修改；
3. 接口：函数签名、数据结构、异常行为和日志级别；
4. 测试：先写/补哪些 pytest，正常、空输入、异常输入和回归样例是什么；
5. 验证命令：要求 AI 运行哪些测试、语法检查、`git diff --check`，最后报告未验证的假设。

推荐模板：

```text
请在当前分支实现 <目标>。
约束：<不能改变的行为/输入输出/性能>。
结构：可复用逻辑放 src/mouse_behavior/<module>.py，CLI 放 scripts/<name>.py，禁止在库模块里解析 argv 或调用 print。
日志：使用 logging.getLogger(__name__)；阶段进度 INFO，降级 WARNING，异常 logging.exception。
测试：先补 pytest 覆盖 <case 1/2/3>，保持现有测试通过。
验证：运行 <commands>，检查 git diff --check，并说明未运行的真实数据验证。
```
