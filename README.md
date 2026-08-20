# Mouse Behavior Lightweight Inference

`mouse-behavior-lightweight-inference` 是一个面向多鼠行为视频的轻量推理与离线行为分析项目。当前默认路径针对已经完成的 YOLO Pose 预推理缓存，执行轻量的个体匹配、关键点几何特征、鼻头/鼻尾接触识别以及追逐/攻击行为判定。

这个仓库保存源代码、配置、测试和运行文档，不保存实验视频、模型权重、YOLO 缓存或大体积分析结果。

## 0. GitHub 仓库导航

如果你是第一次打开 GitHub 仓库，建议按下面的顺序阅读：

1. [docs/index.md](docs/index.md)：工程结构、安装、快速运行、算法和开发文档总索引；
2. [docs/algorithms.md](docs/algorithms.md)：了解标准行为引擎、并行 FSM 和行为输出边界；
3. [docs/architecture.md](docs/architecture.md)：理解 `src/`、`scripts/`、`configs/`、`tests/` 的职责；
4. [configs/profiles/balanced.yaml](configs/profiles/balanced.yaml)：普通轻量分析的推荐配置；
5. [CONTRIBUTING.md](CONTRIBUTING.md)：Git 分支、worktree、pytest、logging 和 AI 协作约定。

GitHub 上的源码入口是 `src/mouse_behavior/`，可执行入口是 `scripts/`，配置入口是 `configs/`。根目录同名 Python 文件只作为旧命令和旧 notebook 的兼容层保留。

## 1. 当前版本和工作边界

当前代码以 v1.43 Standard Behavior Engine 的行为判定逻辑为基础，并增加了 `lightweight_behavior_inference.py` 作为新的主入口。

当前默认执行链为：

```text
已完成的 YOLO Pose 缓存
        ↓
轻量位置 + 关键点匹配
        ↓
Pair 级运动学与接触特征
        ↓
鼻头/鼻尾接触事件 CSV
        ↓
追逐 / 攻击连续证据与时序 FSM
        ↓
独立的 chase / attack 行为事件 CSV
```

轻量路径的设计目标是减少长视频分析中不必要的重型阶段。它不会重新运行完整的遮挡恢复、Mask、ReID、ROI 二次推理和完整视频渲染，也不宣称替代完整追踪管线。输入缓存必须已经存在；如果没有缓存，应先在外部完成 YOLO Pose 预推理。

项目重命名范围如下：

- 项目/仓库标识从旧的版本化目录名统一为 `mouse-behavior-lightweight-inference`。
- 轻量分析实现的正式文件名为 `lightweight_behavior_inference.py`。
- `lightweight_cache_behavior_analysis.py` 保留为兼容入口，旧脚本、旧笔记本和旧命令仍可通过它调用新实现。
- `mouse_chase_attack_high_recall.py`、`mouse_chase_attack_extractor_base.py` 等历史文件名暂时保留，因为测试、旧配置和完整管线仍引用这些名字；这不是重复实现，而是为了避免升级时破坏现有调用关系。

## 2. 行为标签和接触输出

轻量路径的主分类器现在只输出两个行为类别：`chase`（追逐）和 `attack`（攻击）。鼻头/鼻尾接触是独立的几何事件流，不是第三类攻击，也不会因为接触本身打开攻击 FSM：

| 输出 | 含义 |
| --- | --- | --- |
| `chase` | 通过追逐连续证据和追逐 FSM 确认的追逐事件 |
| `attack` | 通过“发起 → 接触 → 反应”因果链和攻击 FSM 确认的攻击事件 |
| `nose_head` | 一只鼠的鼻头接近另一只鼠的头部关键点，达到接触距离 |
| `nose_tail` | 一只鼠的鼻头接近另一只鼠的尾部关键点，达到接触距离 |

主行为事件写入 `lightweight_behavior_events.csv`。除原有 `chase` / `attack` 外，当前轻量入口还按行为粒度输出北医数据集对应的扩展标签：

| 粒度 | 标签 | 依据 |
|---|---|---|
| 社交/鼠对 | `together`、`approach`、`avoidance` | 每视频独立尺度下的鼠对距离、距离变化、稳健速度、追随/逃逸方向 |
| 个体/鼠 ID | `running`、`walking`、`stationary` | 每只鼠的稳健位移速度和连续时间窗 |
| 群体/整帧 | `huddle`、`isolation` | 当前帧有效鼠的最近邻距离分布 |

鼻头接触和鼻尾接触仍写入独立的 `lightweight_contact_events.csv`。若一帧同时满足鼻头和鼻尾距离门，会写为 `nose_head_and_nose_tail`，并在 `contact_type_components` 中保留两个组成类型；接近不是由“接触”直接替代，而是要求距离持续下降且速度模式符合低速接近。攻击回退只接受方向一致的冲击型或接触后分离型时序证据，普通鼻头/鼻尾接触不会仅凭距离门升级为攻击。四分类裁剪函数仍作为旧命令的显式兼容入口保留，但当前轻量默认配置不会再调用它。

## 3. 骨架连接关系

轻量模块使用用户提供参考图对应的 7 点、8 条连线。关键点索引为：

```text
0 nose
1 left ear
2 right ear
3 neck
4 left hip
5 right hip
6 tail
```

连线为：

```text
nose      -> left ear
nose      -> right ear
left ear  -> neck
right ear -> neck
neck      -> left hip
neck      -> right hip
left hip  -> tail
right hip -> tail
```

这组边定义位于 `lightweight_behavior_inference.py` 的 `SKELETON_EDGES` 常量中。渲染函数如后续启用，也使用同一组连接关系，不会另行使用旧的骨架拓扑。

## 4. 仓库结构

```text
.
├─ src/mouse_behavior/                      # 可复用 Python 模块
│  ├─ lightweight_behavior_inference.py     # 轻量单视频分析实现
│  ├─ standard_behavior_engine.py           # 标准追逐/攻击行为引擎
│  ├─ parallel_behavior_fsm.py              # 个体/鼠对/接触/群体并行FSM
│  ├─ core/                                 # 流程编排和 Pipeline facade
│  ├─ models/                               # Pose/cache 模型接口
│  ├─ behavior/                             # 行为引擎稳定导出接口
│  ├─ data/                                 # 事件 CSV schema 和数据契约
│  ├─ io/                                   # 运行目录和输出路径
│  ├─ utils/                                # logging 和计时器
│  ├─ tracking/                             # 追踪职责边界
│  ├─ preprocessing/                        # 预处理职责边界
│  ├─ postprocessing/                       # 后处理职责边界
│  ├─ visualization/                        # 可视化职责边界
│  ├─ evaluation/                           # 评估职责边界
│  ├─ reports/                              # 报告职责边界
│  ├─ adaptive_arena_boundary.py            # 自适应笼界学习
│  ├─ annotation_website_export.py          # 标注网站输出适配器
│  ├─ pose_cache.py                          # Pose cache 写入模块
│  ├─ mask_trigger_controller.py            # Mask 触发决策
│  ├─ nvenc_video_writer.py                 # NVENC/OpenCV writer
│  └─ logging_config.py                     # 统一日志配置
├─ scripts/                                 # CLI、批处理和评估入口
│  ├─ build_lightweight_pose_cache.py       # 只用 Pose 权重生成逐视频缓存
│  ├─ run_lightweight_behavior_inference.py  # 轻量分析 CLI
│  ├─ validate_beiyi_extended_ethogram.py   # 北医示例集验证
│  ├─ calibrate_standard_behavior.py        # 离线阈值校准
│  ├─ sweep_standard_behavior.py            # 阈值扫描
│  ├─ rerun_beiyi_lightweight_rules.py      # 复用缓存重跑规则
│  ├─ compare_parallel_fsm_validation.py    # 并行FSM前后事件级A/B比较
│  ├─ run_lightweight_behavior_inference.ps1
│  └─ run_stage1_stage2.ps1
├─ lightweight_behavior_inference.py       # 根目录兼容 CLI
├─ lightweight_cache_behavior_analysis.py  # 旧模块名兼容层
├─ standard_behavior_engine.py              # 根目录兼容导入层
├─ adaptive_arena_boundary.py               # 根目录兼容导入层
├─ annotation_website_export.py             # 根目录兼容导入层
├─ mask_trigger_controller.py               # 根目录兼容导入层
├─ nvenc_video_writer.py                    # 根目录兼容导入层
├─ mouse_chase_attack_config.yaml           # 行为阈值和运行开关
├─ mouse_chase_attack_high_recall.py       # 完整管线兼容入口/集成入口
├─ mouse_chase_attack_extractor_base.py    # 原有提取器基础代码
├─ tests/                                   # 分层自动化测试
│  ├─ unit/                                 # 纯函数和模块级测试
│  ├─ integration/                          # 跨模块和输出契约测试
│  ├─ regression/                           # 历史行为与性能回归
│  │  └─ fixtures/legacy_v138/              # 最小旧实现对照夹具
│  └─ e2e/                                  # CLI 冒烟测试
├─ configs/                                 # 默认配置、profile 和实验覆盖
├─ docs/                                    # 安装、架构、算法和开发文档
├─ examples/                                # 可复用 API 和配置示例
├─ data/                                    # 本地数据占位目录，不提交数据
├─ outputs/                                 # 本地结果占位目录，不提交结果
├─ tools/                                   # 仓库检查、构建检查和输出比较
├─ scripts/run_quality.py                   # 统一的本地/CI 质量门入口
├─ scripts/validate_repository.py           # 仓库边界验证入口
├─ .github/                                 # CI、Issue 和 PR 模板
├─ .quality-gate.toml                       # 可执行质量门定义
├─ pyproject.toml                           # 包元数据和 pytest 配置
├─ CONTRIBUTING.md                          # Git/模块/日志/测试约定
├─ weights/                                 # 下载 Release 后放置的本地权重目录
│  └─ README.md
├─ requirements.txt                         # 运行依赖
└─ requirements-dev.txt                     # 测试依赖
```

根目录不再保留 Pose 缓存构建、阈值校准、阈值扫描、北医验证和缓存重跑的
兼容 CLI 文件；这些命令的唯一维护入口是 `scripts/` 下的同名脚本。这样可以
保证可复用代码在 `src/mouse_behavior/`、运行入口在 `scripts/`，不会因为旧命令
入口而继续扩大根目录。历史根目录兼容文件仍只为完整旧管线和旧导入路径保留，
后续新增功能不得再放入根目录。

其中，`src/mouse_behavior/` 中的职责目录是稳定的模块边界；当前部分历史实现仍集中在轻量分析模块中，后续拆分必须以 pytest 回归测试为前提，不能通过复制 `v2`、`final2` 等目录维护多个版本。

根目录的同名 Python 文件只为旧命令、旧 notebook 和完整管线保留兼容路径；新代码应从 `mouse_behavior` 包导入，新的命令行入口应放到 `scripts/`。这个分层参考了 [SOAR-PKU/mTrack](https://github.com/SOAR-PKU/mTrack) 中 `mtrack/` 与 `scripts/` 的职责分离方式。

历史版本由 Git commit/tag 保存，不再复制到根目录。回归测试确实需要的两份 v1.38 旧实现已缩减为 `tests/regression/fixtures/legacy_v138/` 夹具；清理决策见 [ADR-0001](docs/adr/0001-use-git-history-instead-of-copied-version-trees.md)。视频、缓存、权重和生成结果由 `.gitignore` 排除；如果确认拥有发布权，模型权重再作为单独的 GitHub Release 附件发布。

## 4.1 模型权重

源码仓库不包含模型权重。只有在版权和发布许可确认后，才应把批准发布的这一份 YOLO Pose 权重作为 GitHub Release 附件提供：

| 文件 | 用途 |
|---|---|
| `pose_best.pt` | 下载后放到 `weights/pose/best.pt`，YOLO Pose 小鼠关键点模型 |

模型权重不随源码提交，也不应把本机权重路径写入配置或结果。若版权和发布许可已经确认，可将批准发布的 Pose 权重作为 GitHub Release 附件，下载后放到 `weights/pose/best.pt`。轻量入口只分析已有 `yolo_precompute` 缓存，因此不会因为读取缓存而重复加载模型；如果需要在上游生成缓存时调用模型，也只使用这份 Pose 权重。OBB 权重不属于当前轻量路径，本仓库不上传也不依赖 OBB 模型。

SHA-256：`AB2F2FBE7A52980DF993FAD1914B630D9004254A9547FA48F245244662A1BED8`。

## 5. 环境安装

推荐使用项目内虚拟环境，避免改变系统 Python 或已有 Conda 环境；Pose 缓存生成也可以继续使用已经安装 PyTorch/YOLO 的独立 Conda 环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

如果环境中已经存在这些包，不需要重复安装。Windows PowerShell 直接运行脚本时，可以使用：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

这只影响当前 PowerShell 进程，不会永久修改系统执行策略。

## 6. 快速运行：轻量行为分析

输入需要两个路径：

1. 原始视频路径；
2. 对应视频的已完成 `yolo_precompute` 缓存目录。

### 6.1 直接调用 Python

```powershell
python .\scripts\run_lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --config .\configs\profiles\balanced.yaml `
  --output-dir .\outputs\part_001 `
  --fps 29.329 `
  --expected-mice 20 `
  --sample-stride 3
```

默认会生成行为事件 CSV、Pair 汇总、证据摘要和元数据 JSON。该命令不会生成渲染视频。

需要逐帧分析、优先提高召回率时，把采样步长改成 1：

```powershell
python .\scripts\run_lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --config .\configs\profiles\high_accuracy.yaml `
  --output-dir .\outputs\part_001_stride1 `
  --fps 29.329 `
  --expected-mice 20 `
  --sample-stride 1
```

### 6.3 直接从视频生成轻量 Pose 缓存

北医示例或其他视频需要先生成缓存时，只加载七关键点 Pose 权重，不加载 OBB：

```powershell
python .\scripts\build_lightweight_pose_cache.py `
  --video "D:\data\beyi_examples\social\approach_001.mov" `
  --output "D:\cache\beyi_examples\approach_001\yolo_precompute" `
  --model ".\weights\pose\best.pt" `
  --device 0
```

然后将这个 `yolo_precompute` 目录交给轻量行为入口。边界学习、尺度换算和扩展行为分类均在本视频内部完成，不复用其他视频的笼子边界。

`sample_stride=3` 是当前速度与召回率的折中默认值；它不是经过所有行为类别人工标注数据最终校准的科学结论。阈值冻结前仍应使用代表性人工标注视频计算 Precision、Recall、F1 和 actor/target accuracy。

### 6.2 使用 PowerShell 包装脚本

包装脚本已经去除本机专属的输入路径，运行时必须显式传入视频和缓存：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_lightweight_behavior_inference.ps1 `
  -Python "D:\Anaconda3\envs\pytorch\python.exe" `
  -Video "D:\data\part_001.mp4" `
  -YoloCache "D:\cache\part_001\yolo_precompute" `
  -Output ".\outputs\part_001" `
  -Fps 29.329 `
  -ExpectedMice 20 `
  -SampleStride 3
```

当前脚本默认只做行为和接触分析，不触发四类裁剪，也不会触发渲染。若需要兼容旧流程的四类原始片段，显式增加 `-ExtractFourClassClips`。

## 7. 兼容性的四类原始视频裁剪

如果事件 CSV 已经存在，可以只执行裁剪：

```powershell
python .\scripts\run_lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --output-dir .\outputs\part_001 `
  --extract-four-class-clips `
  --clip-level strong `
  --clip-seconds 5 `
  --max-clips-per-class 200 `
  --clips-output .\outputs\part_001\four_class_clips
```

`--clip-level` 可以选择 `weak` 或 `strong`。当前默认使用 `strong`，每类最多 200 个片段，默认片段长度 5 秒，并通过最小起始间隔减少同一事件的重复切片。裁剪输出目录可自定义，建议使用 ASCII 目录名以避免旧版 Windows PowerShell 的编码问题。

## 8. 输出文件

典型输出目录如下：

```text
outputs/part_001/
├─ lightweight_analysis_metadata.json
├─ lightweight_behavior_events.csv       # chase/attack + extended ethogram
├─ lightweight_contact_events.csv        # nose_head / nose_tail 接触
├─ lightweight_pair_summary.csv
├─ lightweight_top_evidence.csv
├─ annotation_website_export_report.json  # 网站兼容导出统计与跳过原因
└─ annotation_website_import/
   └─ <视频名>/
      ├─ video.mp4 或 video.mov           # 原始完整视频，优先硬链接
      ├─ annotations.json                 # schema_version 1.0 行为标注
      ├─ tracks.jsonl                     # 0-based 连续逐帧轨迹，含空检测帧
      └─ metadata.json                    # FPS、尺寸、帧数、关键点和骨架
```

只有显式使用 `--extract-four-class-clips` 或 `-ExtractFourClassClips` 时，才会额外生成旧四类原始片段目录。元数据会记录视频、缓存、FPS、采样步长、鼠只数量、分析帧数、运行耗时、启用的行为路径、接触事件统计、是否渲染以及轻量路径的限制。

`annotation_website_import` 是独立的输出适配层，遵循《已标记行为数据导入格式说明》的完整视频导入结构。它不修改轻量追踪、阈值、行为状态机、现有 CSV、渲染或四类裁剪：算法内部事件仍保留 `actor_id/target_id`；网站文件只写升序且去重的 `mouse_ids`，因为网站合同不使用 ID 顺序表达主动方和被动方。类别名仅在网站文件中映射为网站默认名称，其中 `attack/huddle/isolation` 分别写为 `攻击行为/扎堆行为/孤立行为`。需要上传时，把 `annotation_website_import` 内一个或多个视频目录整体打成 ZIP；当前网站文档注明该上传入口仍待实现，因此现阶段生成的是契约兼容数据包。

## 9. 配置开关

`configs/profiles/balanced.yaml` 是当前轻量路径的推荐默认 profile；它继承 `configs/default.yaml`，再由 default 继承根目录兼容配置。实验只需要在 `configs/experiments/` 或独立实验文件中覆盖差异：

```yaml
lightweight_behavior_inference:
  enabled: true
  expected_mice: 20
  sample_stride: 1
  extract_four_class_clips: false
  clip_level: strong
  clip_seconds: 5.0
  max_clips_per_class: 200
  render_video: false
```

重要约束：

- `render_video: false`：当前阶段不渲染视频；
- `extract_four_class_clips: false`：默认不再运行四分类裁剪；需要旧兼容输出时显式开启；
- `sample_stride: 3`：默认每 3 个缓存帧分析 1 帧；逐帧高召回使用 1；
- `expected_mice: 20`：用于限制轻量匹配轨迹数量；
- `--full-behavior`：仅在完整管线所需依赖均已提供且确实需要时使用；
- `--lightweight-behavior`：显式强制轻量路径。

行为阈值仍位于 `standard_behavior_engine` 相关配置中，不能仅凭当前默认值宣称最终准确率。真正的阈值校准流程必须使用用户的追逐、攻击、普通接触、扭打和遮挡人工标签视频。

## 10. 阈值校准和评估

推荐把评估拆成事件检测与角色判断两部分：

```text
事件级：Precision / Recall / F1 / onset offset error
角色级：actor accuracy / target accuracy / ambiguous-role rate
误报分析：普通接触误判攻击、同向共行误判追逐、遮挡期间重复事件
```

`scripts/calibrate_standard_behavior.py` 用于离线校准和生成结构化结果；
`scripts/sweep_standard_behavior.py` 用于在已缓存的 Pair 特征上扫描选定阈值。
使用时应固定视频划分、模型、FPS、采样步长和追踪配置，避免把同一视频同时用于
阈值选择和最终测试。

最终报告至少应记录：

- 每个行为类别的 TP、FP、FN、Precision、Recall 和 F1；
- 事件起始和结束时间误差；
- actor/target 正确率与不确定角色比例；
- chase/attack 二分类混淆矩阵，以及 nose_head/nose_tail 接触类型的混淆矩阵；
- 不同采样步长对召回率、误报率和耗时的影响；
- 普通接触、扭打和遮挡片段的单独结果。

## 11. 验证状态

仓库质量门覆盖 Ruff 格式与 lint、稳定边界的增量 mypy、仓库结构/隐私/大文件检查、完整 pytest 覆盖率运行以及 sdist/wheel 构建检查。单元、集成、回归和 CLI 冒烟测试均由同一个配置驱动，本地和 GitHub Actions 不维护两套命令。

真实视频上的 Precision、Recall、F1、actor/target accuracy 尚未作为仓库级科学验收冻结。完整旧管线仍依赖当前仓库外未提供的模块，因此正式可复现入口是轻量缓存分析；仓库测试通过不能替代代表性人工标注集的科学验证。

验证命令：

```powershell
python scripts/validate_repository.py
python scripts/run_quality.py
# 与 GitHub Actions 相同的完整门禁（含 coverage 和 package build）
python scripts/run_quality.py --ci
```

## 12. CPU、GPU 与渲染说明

轻量离线分析本身主要使用 CPU：缓存读取、轻量轨迹匹配、Pair 特征、接触事件分段和 FSM 均在 CPU 上完成。兼容性的四类视频裁剪只有显式开启时才在 CPU 上执行。GPU 只会出现在生成 YOLO Pose 缓存的上游阶段；如果缓存已经存在，轻量入口不会重新加载 YOLO 模型，也不会重复执行重型检测。

当前阶段不渲染。若未来需要渲染，应单独调用 `--render-only` 并显式指定唯一的 `--render-output` MP4；渲染不应与四类原始片段输出混在一起，也不应把渲染结果提交到 Git。

## 13. 上传边界与隐私

本仓库只上传：

- Python 源码、PowerShell 脚本和 YAML 配置；
- 单元测试、回归工具和工程说明；
- 经过版权和发布许可确认后单独发布的 Pose Release 资产（不进入源码仓库）；
- 依赖清单、README 和 Git 忽略规则。

本仓库不上传：

- 原始/渲染视频和四类视频片段；
- `yolo_precompute`、pickle、numpy、SQLite 等缓存；
- 本地分析 CSV、JSON、日志和输出目录；
- 私钥、环境变量文件、个人凭据；
- 依赖用户本地环境但未包含在本目录的完整管线模块。

上传前应检查：

```powershell
git status --short
git diff --stat
git ls-files
python scripts/validate_repository.py
python scripts/run_quality.py --ci
```

仓库公开前和每次推送前，都应再次检查代码中的实验路径、机构名称、视频文件名和任何本地账号信息；视频、缓存、权重和结果仍必须留在仓库之外。

## 14. 许可证和使用责任

当前仓库的 `LICENSE` 明确记录尚未选择开源许可证；在版权持有人补充正式许可证前，不应默认将代码当作可再分发的开源软件。行为识别结果应经过人工抽检和代表性标注集验证后再用于科研结论或自动化决策。
