# Mouse Behavior Lightweight Inference

`mouse-behavior-lightweight-inference` 是一个面向多鼠行为视频的轻量推理与离线行为分析项目。当前默认路径针对已经完成的 YOLO Pose 预推理缓存，执行轻量的个体匹配、关键点几何特征、鼻头/鼻尾接触识别以及追逐/攻击行为判定。

这个仓库保存源代码、配置、测试和运行文档，不保存实验视频、模型权重、YOLO 缓存或大体积分析结果。

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

主行为事件写入 `lightweight_behavior_events.csv`；接触事件写入独立的 `lightweight_contact_events.csv`。若一帧同时满足鼻头和鼻尾距离门，会写为 `nose_head_and_nose_tail`，并在 `contact_type_components` 中保留两个组成类型。四分类裁剪函数仍作为旧命令的显式兼容入口保留，但当前轻量默认配置不会再调用它。

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
├─ lightweight_behavior_inference.py       # 轻量单视频分析主入口
├─ lightweight_cache_behavior_analysis.py  # 旧模块名兼容层
├─ standard_behavior_engine.py             # 标准追逐/攻击行为引擎
├─ mouse_chase_attack_config.yaml           # 行为阈值和运行开关
├─ mouse_chase_attack_high_recall.py       # 完整管线兼容入口/集成入口
├─ mouse_chase_attack_extractor_base.py    # 原有提取器基础代码
├─ mask_trigger_controller.py              # 原有 Mask 触发控制模块
├─ nvenc_video_writer.py                    # NVENC 写视频辅助模块
├─ calibrate_standard_behavior.py          # 离线阈值校准工具
├─ sweep_standard_behavior.py              # 缓存特征上的阈值扫描工具
├─ run_lightweight_behavior_inference.ps1  # Windows 轻量分析脚本
├─ run_stage1_stage2.ps1                   # 完整流程兼容脚本
├─ weights/                                 # 下载 Release 后放置的本地权重目录
│  └─ README.md
├─ tests/                                   # 单元、回归和性能测试
├─ historical_v1.40_v1.41/                 # 历史工程资料
├─ historical_v1.42.1/                     # v1.42.1 资料
├─ original/                                # 原始版本备份
├─ requirements.txt                         # 运行依赖
└─ requirements-dev.txt                     # 测试依赖
```

`historical_*` 和 `original/` 用于审计与回归对照，不应作为当前运行入口。视频、缓存和生成结果由 `.gitignore` 排除；模型权重作为本仓库 Release 附件发布。

## 4.1 模型权重

仓库 Release 随附当前上游检测流程使用的两份权重：

| 文件 | 用途 |
|---|---|
| `pose_best.pt` | 下载后放到 `weights/pose/best.pt`，YOLO Pose 关键点模型 |
| `obb_best.pt` | 下载后放到 `weights/obb/best.pt`，OBB 小鼠检测模型 |

两份权重作为本仓库公开 Release 的二进制附件发布（每份约 53 MB），不会把大二进制塞进源码树。下载 Release 中的 `pose_best.pt` 和 `obb_best.pt` 后，分别重命名并放到 `weights/pose/best.pt`、`weights/obb/best.pt`。轻量入口只分析已有 `yolo_precompute` 缓存，因此不会因为读取缓存而重复加载模型；上游生成缓存或运行完整流程时，再把模型路径指向上述文件。

## 5. 环境安装

推荐使用已经安装 PyTorch/YOLO 依赖的 Conda 环境；轻量行为分析本身只需要 Python 科学计算和 OpenCV 依赖。

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
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
python .\lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --config .\mouse_chase_attack_config.yaml `
  --output-dir .\outputs\part_001 `
  --fps 29.329 `
  --expected-mice 20 `
  --sample-stride 3
```

默认会生成行为事件 CSV、Pair 汇总、证据摘要和元数据 JSON。该命令不会生成渲染视频。

需要逐帧分析、优先提高召回率时，把采样步长改成 1：

```powershell
python .\lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --config .\mouse_chase_attack_config.yaml `
  --output-dir .\outputs\part_001_stride1 `
  --fps 29.329 `
  --expected-mice 20 `
  --sample-stride 1
```

`sample_stride=3` 是当前速度与召回率的折中默认值；它不是经过所有行为类别人工标注数据最终校准的科学结论。阈值冻结前仍应使用代表性人工标注视频计算 Precision、Recall、F1 和 actor/target accuracy。

### 6.2 使用 PowerShell 包装脚本

包装脚本已经去除本机专属的输入路径，运行时必须显式传入视频和缓存：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_lightweight_behavior_inference.ps1 `
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
python .\lightweight_behavior_inference.py `
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
├─ lightweight_behavior_events.csv       # 只含 chase / attack
├─ lightweight_contact_events.csv        # nose_head / nose_tail 接触
├─ lightweight_pair_summary.csv
└─ lightweight_top_evidence.csv
```

只有显式使用 `--extract-four-class-clips` 或 `-ExtractFourClassClips` 时，才会额外生成旧四类原始片段目录。元数据会记录视频、缓存、FPS、采样步长、鼠只数量、分析帧数、运行耗时、启用的行为路径、接触事件统计、是否渲染以及轻量路径的限制。

## 9. 配置开关

`mouse_chase_attack_config.yaml` 中的当前默认配置为：

```yaml
lightweight_behavior_inference:
  enabled: true
  expected_mice: 20
  sample_stride: 3
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

`calibrate_standard_behavior.py` 用于离线校准和生成结构化结果；`sweep_standard_behavior.py` 用于在已缓存的 Pair 特征上扫描选定阈值。使用时应固定视频划分、模型、FPS、采样步长和追踪配置，避免把同一视频同时用于阈值选择和最终测试。

最终报告至少应记录：

- 每个行为类别的 TP、FP、FN、Precision、Recall 和 F1；
- 事件起始和结束时间误差；
- actor/target 正确率与不确定角色比例；
- chase/attack 二分类混淆矩阵，以及 nose_head/nose_tail 接触类型的混淆矩阵；
- 不同采样步长对召回率、误报率和耗时的影响；
- 普通接触、扭打和遮挡片段的单独结果。

## 11. 验证状态

在本地整理前的最近一次代码验证记录为：

- `pytest`：15 个测试通过；
- `compileall` 和关键模块语法检查：通过；
- YAML/FSM 不变量检查：通过；
- Identity Cascade fuzz：50/50 通过；
- 标准行为引擎的持续追逐、攻击因果链、低质量观测保护、接触与行为分离和骨架几何性质测试：通过；
- 真实视频上的 Precision、Recall、F1、actor/target accuracy：尚未作为仓库级科学验收冻结；
- 完整旧管线的实际运行：当前源码目录缺少 `disk_sequence_guard.py`、`pose_quality_recovery.py`、`mask_cluster_reid.py` 和 `adaptive_arena_boundary.py` 等外部/未随本目录提供的模块，因此优先使用轻量缓存入口。

验证命令：

```powershell
python -m pytest -q
python -m py_compile lightweight_behavior_inference.py standard_behavior_engine.py
```

## 12. CPU、GPU 与渲染说明

轻量离线分析本身主要使用 CPU：缓存读取、轻量轨迹匹配、Pair 特征、接触事件分段和 FSM 均在 CPU 上完成。兼容性的四类视频裁剪只有显式开启时才在 CPU 上执行。GPU 只会出现在生成 YOLO Pose 缓存的上游阶段；如果缓存已经存在，轻量入口不会重新加载 YOLO 模型，也不会重复执行重型检测。

当前阶段不渲染。若未来需要渲染，应单独调用 `--render-only` 并显式指定唯一的 `--render-output` MP4；渲染不应与四类原始片段输出混在一起，也不应把渲染结果提交到 Git。

## 13. 上传边界与隐私

本仓库只上传：

- Python 源码、PowerShell 脚本和 YAML 配置；
- 单元测试、回归工具和工程说明；
- GitHub Release 中明确列出的两份模型权重；
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
```

如果需要公开仓库，建议再次检查代码中的实验路径、机构名称、视频文件名和任何本地账号信息；当前仓库按 Private 方式发布。

## 14. 许可证和使用责任

当前仓库暂未声明开源许可证，因此不应默认将代码当作可再分发的开源软件。行为识别结果应经过人工抽检和代表性标注集验证后再用于科研结论或自动化决策。
