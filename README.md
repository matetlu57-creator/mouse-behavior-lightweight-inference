# Mouse Behavior Lightweight Inference

[![Quality](https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

面向多鼠视频的轻量行为推理、事件导出和可视化项目。项目以已经完成的 YOLO
Pose 逐帧缓存为输入，完成小鼠匹配、笼界范围、关键点几何、候选鼠对、接触
检测、行为状态机和结果渲染。

仓库保存源代码、配置、测试、说明文档，以及通过 Git LFS 管理的已确认可公开使用的
默认 Pose 权重；不保存实验视频、YOLO 缓存、私人标注或生成结果。

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
- 北医攻击在关键点被遮挡时优先使用重叠框、框中心快速位移、面积变化和 IoU 突变。
  骑乘或扭打造成的短漏检最多使用配置的 12 帧框级桥接，但预测框不会进入普通
  关键点轨迹或群体人数统计；
- nose_head、nose_tail：独立几何接触事件，不是第三类攻击，也不会仅因为
  接触距离满足就打开攻击 FSM；
- together、approach、avoidance：鼠对级社交行为；
- running、walking、stationary：小鼠个体行为；
- huddle、isolation：群体行为，并携带实际参与的 member_ids。

渲染器按参与者显示行为，而不是用一个全局标签覆盖整帧。对同一只小鼠，显示层级
固定为“群体行为 > 社交行为 > 个体行为”：

- 群体行为只显示在真实参与群体行为的小鼠框上；
- huddle 只有至少三只小鼠在局部稳定核心内才成立；每只成员至少要有两个同组邻居，
  稀疏的三鼠串联和成员不断变化的短暂近邻不会升级为扎堆。通用 profile 默认使用
  5 cm 的单条局部邻接边，北医 profile 根据四个 huddle 正向样例的稳定三鼠核心距离
  标定为 11 cm。大型 huddle 不要求所有成员的两两中心距离都小于 11 cm，因此远端
  对角小鼠不会使真实扎堆失效；北医同时关闭体长二次上限，避免白鼠和黑鼠混合时
  把固定空间阈值错误缩小。两只小鼠接近仍属于社交行为；
- 当小鼠参与 huddle 或 isolation 时，该小鼠显示“扎堆”或“孤立”，不再显示它的
  社交或个体行为；
- 没有群体行为覆盖的鼠对，才显示 together、approach、chase、avoidance、attack
  或接触行为；
- 没有更高层级事件覆盖的小鼠，最后才显示 running、walking、stationary；
- 不参与群体事件的小鼠仍可独立显示自己的社交或个体行为；
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

北医 profile 先对齐规范中可从视频时间轴直接验证的持续时间。群体扎堆的距离阈值
已经根据四个北医 huddle 正向样例做了独立标定；其它尚未完成尺度校准的速度数值不
作为本轮验收标准。规范没有明确固定时长的行为在
本 profile 中统一按至少 1 秒确认：接近、回避、攻击和鼻头接触至少 1 秒；规范
明确的时长仍按原定义执行，即奔跑至少 0.5 秒，行走、静止、一起、扎堆至少
1 秒，追逐至少 2 秒，鼻尾接触至少 0.5 秒。孤立行为是当前项目的明确例外，
按至少 3 秒确认。核心证据达不到门槛时不会进入正式行为 CSV、渲染或网站导出，
即使显示区间因复核需要包含前后文，也不把前后文计入行为持续时间。

攻击事件还有两个跨帧可靠性门：至少需要两个分析样本在短时间窗口内共同支持，
并且核心证据至少持续 1 秒。单帧或不足 1 秒的攻击候选不会进入行为 CSV，也不会
在渲染框上显示为攻击；这类候选仍可在调试特征中定位原因。

北医样例渲染可以额外指定重点行为。例如 `--focus-behavior attack` 会在右侧面板
持续显示“攻击/被攻击”重点，便于检查整段视频；这个重点标签是渲染上下文，不能
替代实际事件证据。渲染日志分别报告 `persistent_display_coverage` 和
`evidence_coverage`，前者在指定重点时为整段视频，后者仍由真实事件区间计算。

## 安装

项目要求 Python 3.10 或更高版本。建议使用项目自己的虚拟环境：

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
~~~

在 Windows 上请使用仓库提供的测试入口：

~~~powershell
.\scripts\run_pytest.ps1
.\scripts\run_quality.ps1 -CI
~~~

测试环境和 YOLO 推理环境是有意分开的。测试 `.venv` 只安装仓库声明的轻量
依赖和开发工具，不加载 Torch；从视频生成 YOLO Pose 缓存时使用已验证可用的
`yolo26` 环境，例如：

~~~powershell
D:\Anaconda3\envs\yolo26\python.exe .\scripts\build_lightweight_pose_cache.py --help
~~~

不要把项目测试命令直接交给含有多个旧版科学计算包的全局 Anaconda base 环境。
否则可能出现 `torch\lib\c10.dll` 的 Windows DLL 初始化错误，或 Pandas 的可选
加速包版本警告；这类错误属于运行时环境漂移，不是行为算法或 pytest 断言失败。

从视频生成 Pose 缓存时，继续使用已经安装 PyTorch 和 Ultralytics 的推理环境；
本机可使用上面的 `yolo26` 解释器。默认 Pose 权重通过 Git LFS 提供，先执行
`git lfs pull`，详见 weights/README.md。

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
查找，也可以使用 --font-path 指定字体。北医三类重点样例可以在渲染时追加：

~~~powershell
python .\scripts\run_lightweight_behavior_inference.py --video "D:\data\attack.mp4" --yolo-cache "D:\cache\attack\yolo_precompute" --output-dir ".\outputs\attack" --render-only --events ".\outputs\attack\lightweight_behavior_events.csv" --render-output ".\outputs\attack\攻击重点渲染.mp4" --focus-behavior attack
~~~

`focus-behavior` 只控制渲染复核的持续重点，不会根据视频名或目录名生成行为。
批量北医渲染脚本会把验证清单中的期望行为传给渲染器，因此重点标题会和样例
文件夹保持一致，但推理 CSV 仍然只来自轨迹、几何和时序证据。

### 运行完整管线

~~~powershell
python .\scripts\run_full_behavior_pipeline.py --video "D:\data\part_001.mp4" --model ".\weights\best.pt" --config ".\mouse_chase_attack_config.yaml" --output ".\outputs\full_pipeline" --stage stage1
~~~

## 配置

配置入口为 configs/default.yaml，运行 profile 位于 configs/profiles/：

- fast.yaml：降低计算量，适合快速检查；
- balanced.yaml：常规分析的推荐配置；
- high_accuracy.yaml：使用更小采样步长，优先保留短暂行为。
- beiyi.yaml：北医短视频不使用自适应笼界，也不使用固定 polygon 笼界。北医 RFID-CV
  示例按 10 只小鼠处理，北医验证和渲染脚本默认使用 `expected_mice=10`；通用 20
  只小鼠视频仍需显式传入 `expected_mice=20`。

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
.\scripts\run_pytest.ps1
.\scripts\run_quality.ps1 -Step repository
.\scripts\run_quality.ps1
.\scripts\run_quality.ps1 -CI
~~~

测试分层如下：

- tests/unit/：几何、FSM、日志和单模块行为；
- tests/integration/：CSV、网站导出、渲染和仓库结构契约；
- tests/regression/：历史输出、性能和最小旧实现夹具；
- tests/e2e/：CLI 和端到端冒烟测试。

北医样例的视频级覆盖只能证明示例中出现了期望类别；持续时间审计还会优先读取
事件的 `core_duration_s`，不把渲染前后文算作行为证据。由于目录是视频级分类而
不是逐帧标注，仍然不能替代带帧级真值的 Precision、Recall、F1 或 actor/target
准确率。验证脚本只把目录名作为外部验收清单，不把目录名传入推理规则。

## 故障排查

### 找不到缓存

确认 --yolo-cache 指向完整 Pose 缓存目录，并确认缓存的视频名称、帧数、FPS
与输入视频一致。没有缓存时先运行 Pose 缓存生成脚本。

### 输出没有渲染视频

普通分析默认不渲染。使用 --render-only，并提供已经存在的行为事件 CSV、Pose
缓存和输出路径。

### 行为持续时间很短

查看 CSV 中的 core_duration_s、temporal_padding_frames 和 event_recovery。
通用调试候选可能只有一到两帧，而渲染区间会增加有限前后文。北医 profile 会在
正式导出前额外执行行为持续时间门，因此应优先检查 core_duration_s，再判断是否
达到该 profile 的要求，不要只用包含前后文的 duration_s。

### 本地质量门缺少工具

按照 requirements-dev.txt 安装开发依赖。如果当前解释器与 PyTorch 环境不同，
请使用项目包装器分别运行测试/质量门，并使用 `yolo26` 解释器运行真实视频的
YOLO Pose 缓存生成。若包装器提示找不到 `.venv`，先按“安装”章节创建它。

## 贡献和 Git workflow

一个功能使用一个 branch；需要同时试验多个方向时，使用仓库外 worktree。不要
在根目录复制第二份源码，也不要把视频、缓存、未登记权重和生成结果加入提交。
默认 Pose 权重是唯一登记的 Git LFS 模型文件。

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

请勿提交视频、私人标注、未登记模型权重、缓存、凭据或 API token。安全问题不要公开发
Issue，请按照 SECURITY.md 通过私密渠道联系维护者。

项目问题、复现信息和功能建议请使用 GitHub Issue；提交代码前请先阅读
CONTRIBUTING.md。
