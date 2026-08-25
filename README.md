# Mouse Behavior Lightweight Inference

[![Quality](https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

面向多鼠视频的轻量行为推理、事件导出和可视化项目。项目以已经完成的 YOLO
Pose 逐帧缓存为输入，完成小鼠匹配、笼界范围、关键点几何、候选鼠对、接触
检测、行为状态机和结果渲染。

仓库只保存源代码、配置、测试和说明文档，不保存实验视频、YOLO 缓存、模型
权重或生成结果。

## 项目导航

第一次阅读建议按照下面的顺序进行：

1. [快速开始](docs/quickstart.md)
2. [安装与环境](docs/installation.md)
3. [仓库架构](docs/architecture.md)
4. [完整仓库结构说明](docs/repository_structure.md)
5. [算法说明](docs/algorithms.md)
6. [输出格式](docs/data_format.md)
7. [测试与开发](docs/development/testing.md)
8. [贡献规范](CONTRIBUTING.md)

源码入口在 src/mouse_behavior/，命令行和批处理入口在 scripts/，配置在
configs/。根目录不放新的 Python 模块或临时脚本；历史版本通过 Git branch、
tag 和 worktree 管理。

## 项目解决什么问题

长视频的完整检测和身份恢复管线成本较高。轻量路径复用已经生成的 YOLO Pose
缓存，只对有价值的候选鼠对计算后续几何和运动学特征，适合：

- 对多鼠视频进行逐帧或低采样步长的离线行为分析；
- 输出可供统计、复核和标注网站导入的事件文件；
- 生成带有小鼠 ID、骨架和中文行为标签的渲染视频；
- 在修改算法后使用 pytest、北医样例和质量门进行回归验证。

轻量路径不是完整遮挡恢复、Mask、ReID 和 ROI 二次推理管线的替代品。没有
YOLO Pose 缓存时，应先使用上游缓存生成脚本完成预推理。

## 处理流程

~~~text
原始视频与 YOLO Pose 缓存
          ↓
轻量轨迹匹配与笼界范围学习
          ↓
距离、朝向和速度预筛选候选鼠对
          ↓
候选鼠对的运动学、接触几何和滚动特征
          ↓
个体行为、社交行为、群体行为和接触事件并行 FSM
          ↓
事件 CSV、网站导入包、统计报告和渲染视频
~~~

## 行为输出和显示规则

轻量路径将不同职责的结果分开保存：

- chase：追逐 FSM 确认的追逐事件，渲染为“追逐”和“被追逐”；
- attack：满足发起、接触、反应或接触后分离证据的攻击事件，渲染为“攻击”和
  “被攻击”；
- nose_head、nose_tail：独立几何接触事件，不是第三类攻击，也不会仅因为
  接触距离满足就打开攻击 FSM；
- together、approach、avoidance：鼠对级社交行为；
- running、walking、stationary：小鼠个体行为；
- huddle、isolation：群体行为，并携带实际参与的 member_ids。

渲染器按参与者显示行为，而不是用一个全局标签覆盖整帧：

- 群体行为只显示在真实参与群体行为的小鼠框上；
- 社交行为显示在对应鼠对的参与者上；
- 个体行为仍然可以和其他小鼠的社交或群体行为同时出现；
- 同一只小鼠同时拥有多个事件时，攻击、追逐、回避、接近和群体行为按语义
  优先级显示；普通接触不会覆盖攻击标签；
- 没有行为事件的小鼠显示“仅追踪”，不会因为其他小鼠发生群体行为而被统一
  标记为群体行为。

短暂行为保留两组时间边界：

- analysis_start_frame、analysis_peak_frame、analysis_end_frame 是实际几何和
  运动证据帧；
- start_frame、end_frame 是渲染、切片和网站导出的有限上下文区间；
- core_duration_s 表示核心证据持续时间；
- event_recovery 记录短事件恢复原因。

恢复分支只接受距离、朝向、速度、接触几何、角色和事件分数共同满足条件的
短候选，不使用视频名称判断行为。

## 安装

项目要求 Python 3.10 或更高版本。建议使用项目自己的虚拟环境：

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
~~~

从视频生成 Pose 缓存时，继续使用已经安装 PyTorch 和 Ultralytics 的环境。
模型权重不进入普通 Git 历史，详见 weights/README.md。

## 快速开始

### 使用已有 YOLO Pose 缓存分析视频

~~~powershell
python .\scripts\run_lightweight_behavior_inference.py --video "D:\data\part_001.mp4" --yolo-cache "D:\cache\part_001\yolo_precompute" --config .\configs\profiles\balanced.yaml --output-dir .\outputs\part_001 --fps 29.329 --expected-mice 20 --sample-stride 1
~~~

默认会生成行为事件、接触事件、鼠对汇总、证据摘要、网站导出包和运行元数据。
普通分析不会自动渲染视频。

### 从视频生成轻量 Pose 缓存

~~~powershell
python .\scripts\build_lightweight_pose_cache.py --video "D:\data\part_001.mp4" --output "D:\cache\part_001\yolo_precompute" --model ".\weights\pose\best.pt" --device 0
~~~

### 对已有结果渲染视频

~~~powershell
python .\scripts\run_lightweight_behavior_inference.py --video "D:\data\part_001.mp4" --yolo-cache "D:\cache\part_001\yolo_precompute" --output-dir ".\outputs\part_001" --render-only --events ".\outputs\part_001\lightweight_behavior_events.csv" --render-output ".\outputs\part_001\轻量行为推理_渲染.mp4"
~~~

渲染视频会显示小鼠框、ID、七点骨架、当前行为和右侧行为面板。中文字体可自动
查找，也可以使用 --font-path 指定字体。

### 运行完整管线

~~~powershell
python .\scripts\run_full_behavior_pipeline.py --video "D:\data\part_001.mp4" --model ".\weights\best.pt" --config ".\mouse_chase_attack_config.yaml" --output ".\outputs\full_pipeline" --stage stage1
~~~

## 配置

配置入口为 configs/default.yaml，运行 profile 位于 configs/profiles/：

- fast.yaml：降低计算量，适合快速检查；
- balanced.yaml：常规分析的推荐配置；
- high_accuracy.yaml：使用更小采样步长，优先保留短暂行为。

新实验应在 configs/experiments/ 中使用独立文件记录。根目录的
mouse_chase_attack_config.yaml 仍保留给完整管线和旧调用方。

## 仓库结构

~~~text
src/mouse_behavior/       可复用 Python 模块和稳定接口
scripts/                  CLI、批处理、验证和校准入口
configs/                  默认配置、profile 和实验覆盖
tests/                    unit、integration、regression、e2e
docs/                     用户、算法、架构和开发文档
examples/                 最小 API 与配置示例
tools/                    仓库检查、构建检查和输出比较
.github/                  CI、Issue 模板和 PR 模板
~~~

重要职责边界如下：

- src/mouse_behavior/preprocessing/：笼界、几何、运动学和候选鼠对特征；
- src/mouse_behavior/tracking/：Pose 缓存规范化和轻量轨迹匹配；
- src/mouse_behavior/behavior/：标准证据、追逐/攻击 FSM、扩展行为和鼠对编排；
- src/mouse_behavior/visualization/：渲染视频、行为标签和行为片段；
- src/mouse_behavior/annotation_website_export.py：网站导入数据适配器；
- scripts/：只做参数解析、批处理和验证，不复制算法实现。

仓库根目录不放新的 Python 入口，也不通过复制 v2、final2 等目录保存版本。
并行开发使用 Git branch 和仓库外 worktree。

## 输出文件

典型结果目录如下：

~~~text
outputs/part_001/
├─ lightweight_analysis_metadata.json
├─ lightweight_behavior_events.csv
├─ lightweight_contact_events.csv
├─ lightweight_pair_summary.csv
├─ lightweight_top_evidence.csv
├─ 轻量行为推理_渲染.mp4
├─ annotation_website_export_report.json
└─ annotation_website_import/
   └─ <视频名>/
      ├─ video.mp4 或 video.mov
      ├─ annotations.json
      ├─ tracks.jsonl
      └─ metadata.json
~~~

视频、缓存、模型和生成结果由 .gitignore 排除。网站导入包遵循版本为 1.0
的完整视频导入结构，内部事件仍保留主动方和被动方，网站文件只输出排序去重后的
mouse_ids。

## 测试与质量检查

~~~powershell
python -m pytest -q
python scripts/validate_repository.py
python scripts/run_quality.py
python scripts/run_quality.py --ci
~~~

测试分层如下：

- tests/unit/：几何、FSM、日志和单模块行为；
- tests/integration/：CSV、网站导出、渲染和仓库结构契约；
- tests/regression/：历史输出、性能和最小旧实现夹具；
- tests/e2e/：CLI 和端到端冒烟测试。

北医样例的视频级覆盖只能证明示例中出现了期望类别，不能替代带帧级真值的
Precision、Recall、F1 或 actor/target 准确率。

## 故障排查

### 找不到缓存

确认 --yolo-cache 指向完整 Pose 缓存目录，并确认缓存的视频名称、帧数、FPS
与输入视频一致。没有缓存时先运行 Pose 缓存生成脚本。

### 输出没有渲染视频

普通分析默认不渲染。使用 --render-only，并提供已经存在的行为事件 CSV、Pose
缓存和输出路径。

### 行为持续时间很短

查看 CSV 中的 core_duration_s、temporal_padding_frames 和 event_recovery。
核心证据可能只有一到两帧，而渲染区间会增加有限前后文，不要只用 duration_s
判断算法证据持续时间。

### 本地质量门缺少工具

按照 requirements-dev.txt 安装开发依赖。如果当前解释器与 PyTorch 环境不同，
请在运行质量门、测试和真实视频推理时分别明确 Python 路径。

## 贡献和 Git workflow

一个功能使用一个 branch；需要同时试验多个方向时，使用仓库外 worktree。不要
在根目录复制第二份源码，也不要把视频、缓存、权重和生成结果加入提交。

~~~powershell
git switch main
git pull --ff-only
git switch -c feat/short-description
python -m pytest -q
git diff --check
git add <本次修改的明确文件>
git commit -m "docs: clarify lightweight behavior outputs"
git push -u origin feat/short-description
~~~

模块化、日志、pytest、分支和 AI 协作规则见 CONTRIBUTING.md。较大的架构决策
记录在 docs/adr/ 中。

## 许可证、安全和支持

当前仓库尚未选择开源许可证。除非版权持有人另行授权，源码默认不授予复制、修改、
再发布或公开发布许可，详见 LICENSE。

请勿提交视频、私人标注、模型权重、缓存、凭据或 API token。安全问题不要公开发
Issue，请按照 SECURITY.md 通过私密渠道联系维护者。

项目问题、复现信息和功能建议请使用 GitHub Issue；提交代码前请先阅读
CONTRIBUTING.md。
