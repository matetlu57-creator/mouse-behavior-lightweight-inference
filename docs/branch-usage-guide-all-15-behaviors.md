# 15 类小鼠行为识别与模块化裁剪分支拉取及使用指南

本文档面向协作者与下游使用者，说明如何从 GitHub 拉取并配置分支 **`release/all-15-behaviors-clipping-20261125`**，以及如何运行 **全部 15 类行为识别模块**、**模块化分段裁剪（Segment Clipping）模块** 和解析输出 JSON / 视频四件套。

- **GitHub 仓库**：`https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference`
- **推荐拉取分支**：`release/all-15-behaviors-clipping-20261125`
- **模块架构参考**：[`docs/heuristic-and-clipping-architecture.md`](heuristic-and-clipping-architecture.md)
- **数据契约参考**：[`docs/index.md`](index.md)

---

## 一、在 GitHub 上拉取分支与安装环境

### 1.1 全新克隆指定分支

如果本地尚未克隆仓库，可直接只拉取该分支：

```bash
git clone -b release/all-15-behaviors-clipping-20261125 --single-branch https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference.git
cd mouse-behavior-lightweight-inference
```

如果需要保留仓库全部分支并切到当前分支：

```bash
git clone https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference.git
cd mouse-behavior-lightweight-inference
git checkout -b release/all-15-behaviors-clipping-20261125 origin/release/all-15-behaviors-clipping-20261125
```

### 1.2 已有本地仓库拉取并切换

```bash
git fetch origin release/all-15-behaviors-clipping-20261125
git checkout -b release/all-15-behaviors-clipping-20261125 origin/release/all-15-behaviors-clipping-20261125
```

后续更新该分支：

```bash
git pull --ff-only origin release/all-15-behaviors-clipping-20261125
```

### 1.3 拉取 Git LFS 模型权重（必须）

7 关键点 YOLO Pose 权重 [`weights/pose/best.pt`](../weights/pose/best.pt) 使用 Git LFS 托管。若克隆时未自动下载权重实体（文件只有百余字节的 LFS 指针），需执行：

```bash
git lfs install
git lfs pull
```

> 请确认 `weights/pose/best.pt` 已被下载为完整二进制权重文件，而非文本指针；如遇权限或网络问题导致 LFS 文件仍为指针，`build_lightweight_pose_cache.py` 加载模型时会失败。

### 1.4 Python 环境安装（Python >= 3.10）

推荐使用仓库根目录下 [`environment.yml`](../environment.yml) 创建独立 Conda 环境（Python 3.10）：

```bash
conda env create -f environment.yml
conda activate mouse-behavior
python -m pip install -e ".[review]"
```

如需同时使用校准相关工具（`scikit-learn>=1.5`），可安装扩展依赖：

```bash
python -m pip install -e ".[review,calibration]"
```

也可在已有的 Python >= 3.10 虚拟环境中直接安装核心与可选依赖（关键固定版本：`ultralytics==8.4.115`、`imageio-ffmpeg>=0.6,<0.7`，详见 [`pyproject.toml`](../pyproject.toml)）：

```bash
python -m pip install -e ".[review]"
```

### 1.5 环境与模块自检

安装完成后建议执行一次快速导入检查和裁剪回归测试，确认全部 15 类行为模块和时长配置可用：

```bash
python -c "import mouse_behavior.heuristics as h; print('exports:', len(h.__all__), 'behaviors:', len(h.MIN_CLIP_DURATION_S_BY_BEHAVIOR))"
python -m pytest -q tests/unit/test_segment_clipping_upgrade.py tests/unit/test_long_video_social_clipping_and_site.py
```

---

## 二、15 类行为总览、时长门控与标准角色命名

当前分支支持全部 **15 个保留规范行为**（个体层 4 类 + 社交/双鼠层 6 类 + 群体层 5 类）。所有默认选择器与管线**取消了默认输出数量上限**（`max_per_behavior=None` / `max_events_per_behavior=None`），同时对每个行为强制执行最短/最长有效时长门控：

| 层级 | 中文名 | 展示名 (`behavior`) | 规范名 (`canonical_behavior`) | 最少参与鼠数 | 最短时长门控 | 最长时长上限 | 有向角色 (`initiator_role -> target_role`) | 默认数量限制 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 个体 | 静止 | `Static` | `stationary` | 1 | `>= 1.0 s` | 无上限 | 无 (N/A) | 无上限 (`None`) |
| 个体 | 移动 | `Locomotion` | `locomotion` | 1 | `>= 0.5 s` | 无上限 | 无 (N/A) | 无上限 (`None`) |
| 个体 | 理毛 | `Grooming` | `grooming` | 1 | `>= 0.3 s` | 无上限 | 无 (N/A) | 无上限 (`None`) |
| 个体 | 站立 | `Rearing` | `rearing` | 1 | `>= 0.5 s` | 无上限 | 无 (N/A) | 无上限 (`None`) |
| 社交 | 接近 | `Approach` | `approach` | 2 | `>= 1.0 s` | 无上限 | `Approaching -> Approached` | 无上限 (`None`) |
| 社交 | 攻击 | `Attack` | `attack` | 2 | `> 0.5 s` | 无上限 | `Attacker -> Attacked` | 无上限 (`None`) |
| 社交 | 追逐 | `Chasing` | `chase` | 2 | `> 0.7 s` | 无上限 | `Chasing -> Chased` | 无上限 (`None`) |
| 社交 | 鼻头接触 | `Snout-head_contact` | `nose_head_contact` | 2 | `> 0.4 s` | 无上限 | `Snout_contacting -> Head_contacted` | 无上限 (`None`) |
| 社交 | 鼻尾接触 | `Snout-rear_contact` | `nose_tail_contact` | 2 | `> 0.4 s` | 无上限 | `Snout_contacting -> Rear_contacted` | 无上限 (`None`) |
| 社交 | 一起/共处 | `Together` | `together` | 2 | `>= 1.0 s` | 无上限 | 无方向 (`roles: null`, `actor_id=-1`) | 无上限 (`None`) |
| 群体 | 扎堆 | `Huddling` | `huddle` | 3 | `>= 5.0 s` | `<= 15.0 s` | 无 (N/A) | 无上限 (`None`) |
| 群体 | 社会聚集 | `Social clustering` | `social_clustering` | 3 | `> 3.0 s`（含稳定聚集） | 聚集后稳定保持最多 `3.0 s` | 无 (N/A) | 无上限 (`None`) |
| 群体 | 散开 | `Dispersal` | `dispersal` | 3 | `> 5.0 s` | `<= 15.0 s` | 无 (N/A) | 无上限 (`None`) |
| 群体 | 群体同步运动 | `Group locomotion` | `group_locomotion` | 3 | `> 3.0 s` | `<= 9.0 s` | 无 (N/A) | 无上限 (`None`) |
| 群体 | 孤立 | `Isolation` | `isolation` | 1（需 `>= 2` 在场对照帧） | `> 10.0 s` | `<= 20.0 s` | 目标个体 | 无上限 (`None`) |

> **工程契约边界**：
> - Ultralytics `YOLO.track()` 负责短时视觉跟踪 `visual_track_id`；外部 mTrack/RFID CSV 映射是稳定生物身份 `animal_id` 的唯一来源，系统绝不通过位置、毛色、亮度、姿态或运动轨迹反推缺失身份。
> - `Together` 允许存在接触且与接触类互斥判定，两者不再做同层双标签互认。

---

## 三、如何使用行为识别模块

### 3.1 方式 A：命令行端到端长视频管线（推荐）

完整流程包含「Pose 缓存构建 -> 生成来源绑定凭据 -> 15 行为识别 + 精修裁剪 + 片段导出」三步：

#### 第 1 步：生成 7 关键点 YOLO Pose 追踪缓存

使用 [`scripts/build_lightweight_pose_cache.py`](../scripts/build_lightweight_pose_cache.py)：

```powershell
python scripts/build_lightweight_pose_cache.py `
  --video D:\experiment\video.mp4 `
  --model weights\pose\best.pt `
  --output D:\experiment\yolo_precompute `
  --tracker bytetrack.yaml
```

该步骤会在 `D:\experiment\yolo_precompute` 中生成分片 `yolo_results_*.pkl.gz` 以及 `yolo_results_status.json`（含 `status=complete` 与 `source_video_sha256`）。

#### 第 2 步：生成 `source_receipt.json`

正典视频管线强制校验源视频、Pose 缓存和模型权重的 SHA-256 绑定。可运行以下单行命令生成绑定凭据（由 [`write_pose_cache_source_receipt`](../src/mouse_behavior/evaluation/group_cropping_validation.py) 提供）：

```powershell
python -c "from pathlib import Path; from mouse_behavior.evaluation.group_cropping_validation import write_pose_cache_source_receipt; print(write_pose_cache_source_receipt(Path(r'D:\experiment\video.mp4'), Path(r'D:\experiment\yolo_precompute'), Path(r'weights\pose\best.pt')))"
```

#### 第 3 步：运行全部 15 类行为识别与自动精修裁剪导出

使用 [`scripts/run_canonical_behavior_video.py`](../scripts/run_canonical_behavior_video.py) 并传入 `--all-behaviors`：

```powershell
python scripts/run_canonical_behavior_video.py `
  --video D:\experiment\video.mp4 `
  --pose-cache D:\experiment\yolo_precompute `
  --source-receipt D:\experiment\yolo_precompute\source_receipt.json `
  --config configs\default.yaml `
  --all-behaviors `
  --mode all `
  --workers 4 `
  --clip-workers 4 `
  --target-overlay `
  --output D:\experiment\run_all15_out
```

常用参数说明（详见 [`scripts/run_canonical_behavior_video.py`](../scripts/run_canonical_behavior_video.py)）：
- `--all-behaviors`：自动等价于启用全部 15 类保留行为（含标定后的 `grooming`/`rearing`）、开启模块化精修裁剪（`enable_fine_cropping=True`），并保持无数量上限导出。
- `--mode`：
  - `infer`：只输出行为事件 `events.json` 与轨迹 SQLite，不渲染视频。
  - `full`：在推理基础上输出整段带叠加标注视频 `full/clip.mp4`。
  - `clips`：裁剪导出每个行为的独立片段；若搭配 `--original-video` 或 `--target-overlay`，直接从源视频切分。
  - `all`：完整执行推理、整视频渲染与全部片段导出。
- `--events-from <已完成的旧输出目录>`：复用已生成的 `events.json` 与 `tracks.sqlite`，直接重新裁剪或渲染视频，无需重复跑识别。
- `--max-events-per-behavior`：默认 `None`（不限制数量）；只有调试抽样时才需显式传入正整数。
- `--clip-context-seconds 0`：前后额外上下文秒数（默认 `0`；注意上下文只用于审看扩展，不参与时长门控达标判定）。
- `--clip-codec libx264|h264_nvenc`：选择 CPU 软编或 NVIDIA NVENC 硬编。

#### 附：接入 mTrack/RFID 真实动物身份

若需把视觉追踪 ID 绑定为真实动物 RFID 标签，可通过安装后的 CLI `mouse-behavior-lightweight` 并传入归一化映射 CSV（格式见 [`README.md`](../README.md#mtrackrfid-交换格式)）：

```powershell
mouse-behavior-lightweight `
  --video D:\experiment\video.mp4 `
  --yolo-cache D:\experiment\yolo_precompute `
  --identity-map D:\experiment\mtrack_identity_map.csv `
  --config configs\default.yaml `
  --output-dir D:\experiment\rfid_behavior_output `
  --fps 30 `
  --expected-mice 20
```

---

### 3.2 方式 B：Python 代码调用行为识别模块

#### 1) 整视频 15 类行为一键推理 / 一键全流程

在 [`src/mouse_behavior/canonical_video_pipeline.py`](../src/mouse_behavior/canonical_video_pipeline.py) 中提供了两个专用入口：
- `infer_all_behavior_events(...)`：输入已绑定的 `TrackStore`、`binding`、`scale` 与 `params`，直接执行全部 15 类行为推理 + 模块化精修裁剪并返回 `(events, audit)`（默认 `enable_fine_cropping=True`, `max_events_per_behavior=None`）。
- `run_all_behaviors_pipeline(...)`：等价于命令行 `--all-behaviors` 的 Python 函数封装，自动落盘全部绑定校验与视频/JSON 产物。

示例：

```python
from pathlib import Path
from mouse_behavior.canonical_video_pipeline import run_all_behaviors_pipeline

receipt = run_all_behaviors_pipeline(
    video=Path("D:/experiment/video.mp4"),
    cache=Path("D:/experiment/yolo_precompute"),
    output=Path("D:/experiment/run_all15_py"),
    config=Path("configs/default.yaml"),
    source_receipt=Path("D:/experiment/yolo_precompute/source_receipt.json"),
    mode="all",
    workers=4,
    clip_workers=4,
    target_overlay=True,
)
print(receipt["event_count"], receipt["clip_count"])
```

#### 2) 通过统一门面 `mouse_behavior.heuristics` 分层调用底层识别规则

项目在 [`src/mouse_behavior/heuristics/__init__.py`](../src/mouse_behavior/heuristics/__init__.py) 提供了无副作用的惰性统一导出门面 `mouse_behavior.heuristics`：

```python
from mouse_behavior.heuristics import (
    DEFAULT_GROUP_HEURISTIC_CONFIG,
    DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG,
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    HeuristicParameters,
    group_rule_candidates,
    individual_rule_masks,
    isolation_candidates,
    pair_rule_masks,
    predict_features,
)
```

各层核心函数签名与职责：
- **个体层（4 类）**：`individual_rule_masks(centers_cm, valid, fps, keypoints_cm=..., boxes_cm=..., reference_body_cm=7.918, config=DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG)` 返回包含 `locomotion`、`static`、`grooming`、`rearing` 布尔掩码与诊断的结果对象。
- **社交/双鼠层（6 类）**：`pair_rule_masks(features, params, social_config=DEFAULT_SOCIAL_HEURISTIC_CONFIG, roles=role_dict)` 返回 `approach`、`attack`、`chase`、`nose_head_contact`、`nose_tail_contact`、`together` 的帧边掩码，并向 `roles` 写入有向 `*_actor` / `*_target` 数组。
- **群体层（5 类）**：
  - `group_rule_candidates(features, params, proposals=(), config=DEFAULT_GROUP_HEURISTIC_CONFIG)` 返回 `huddle`、`social_clustering`、`group_locomotion`、`dispersal` 候选过程。
  - `isolation_candidates(features, params, config=DEFAULT_GROUP_HEURISTIC_CONFIG)` 返回 `isolation` 候选过程。
- **特征预测聚合**：`predict_features(...)` 将个体、双鼠、群体三层证据汇总为统一评测输出。

---

## 四、如何使用裁剪模块（Segment Clipping）

### 4.1 模块拆分结构

裁剪模块以 [`mouse_behavior.behavior.segment_clipping`](../src/mouse_behavior/behavior/segment_clipping.py) 和统一门面 [`mouse_behavior.heuristics`](../src/mouse_behavior/heuristics/__init__.py) 对外暴露，内部按职责分为：

- 公共基础设施与策略表：[`segment_clipping_common.py`](../src/mouse_behavior/behavior/segment_clipping_common.py)（`InMemoryTrackStore`、`MIN_CLIP_DURATION_S_BY_BEHAVIOR`、`MAX_DURATION_S_BY_BEHAVIOR`、`qualify_clip_duration`、`endpoints_fully_observed`）
- 个体行为精修裁剪：[`segment_clipping_individual.py`](../src/mouse_behavior/behavior/segment_clipping_individual.py)（`refine_static_segments`、`refine_locomotion_segments`、`refine_grooming_segments`、`refine_rearing_segments`）
- 社交行为精修裁剪（三子模块 + 兼容门面 [`segment_clipping_social.py`](../src/mouse_behavior/behavior/segment_clipping_social.py)）：
  - [`segment_clipping_social_approach.py`](../src/mouse_behavior/behavior/segment_clipping_social_approach.py)：`refine_approach_segments`（远距离接近至包框初触补全、群聚剔除、方向性终验、角色归因）
  - [`segment_clipping_social_attack_chasing.py`](../src/mouse_behavior/behavior/segment_clipping_social_attack_chasing.py)：`refine_attack_segments`、`discover_fight_ball_attack_episodes`、`refine_chasing_segments`、`discover_fast_chasing_candidates`、`discover_id_switch_chasing_candidates`（滚球打斗/高速直线及折返追逐/ID 跳变追逐恢复）
  - [`segment_clipping_social_contact.py`](../src/mouse_behavior/behavior/segment_clipping_social_contact.py)：`refine_contact_and_together_segments`、`attach_social_roles`、`build_social_roles`（头/尾接触上下文补全、Together 门控及统一中英文标准角色写入）
- 群体行为精修裁剪（三子模块 + 兼容门面 [`segment_clipping_group.py`](../src/mouse_behavior/behavior/segment_clipping_group.py)）：
  - [`segment_clipping_group_clustering.py`](../src/mouse_behavior/behavior/segment_clipping_group_clustering.py)：`refine_social_clustering_segments`
  - [`segment_clipping_group_dispersal.py`](../src/mouse_behavior/behavior/segment_clipping_group_dispersal.py)：`refine_dispersal_segments`
  - [`segment_clipping_group_motion.py`](../src/mouse_behavior/behavior/segment_clipping_group_motion.py)：`refine_huddle_segments`、`refine_group_locomotion_segments`、`refine_isolation_segments`

### 4.2 Python 代码单独使用裁剪与导出模块

若你已经有一批粗识别事件 `raw_events` 和逐帧轨迹列表 `tracks`（每帧形如 `{"frame": f, "source_frame": f, "time": f / fps, "detections": [...]}`），可以直接脱离推理管线单独调用裁剪、去重过滤与四件套渲染导出：

```python
from pathlib import Path
from mouse_behavior.heuristics import (
    HeuristicParameters,
    InMemoryTrackStore,
    refine_behavior_segments,
    select_behavior_segments,
)
from mouse_behavior.visualization.behavior_packages import export_packages, write_manifests

# 1. 构造内存轨迹库与绑定信息（如未绑定 SQLite，可传入规则与视频元信息）
store = InMemoryTrackStore(tracks)
binding = {
    "source_video": {"fps": 30.0, "width": 1920, "height": 1080, "frame_count": 18000},
    "clipping_duration_policy": "calibrated_clip_v1",  # 允许启用 >0.4s / >0.7s / <=9s / <=20s 标定裁剪时长表
}
scale = {"cm_per_pixel": 0.08, "reference_body_cm": 7.918}
params = HeuristicParameters()

# 2. 对全部分段执行精细边界裁剪（自动按行为最低/最高时长及最少参与鼠数校验）
refined_events, refine_audit = refine_behavior_segments(
    raw_events,
    store,
    binding,
    scale=scale,
    parameters=params,
    enable_huddle_sliding=True,
    enable_fine_cropping=True,
)

# 3. 质量筛选与同鼠防重叠去重（max_per_behavior=None 表示不限制每类输出数量）
selected_events, select_summary = select_behavior_segments(
    refined_events,
    store=store,
    binding=binding,
    max_per_behavior=None,
    require_closed_boundaries=True,
    require_endpoint_visibility=True,
    strict_non_overlapping_shared_mouse=True,
)

# 4. 如已走 canonical_video_pipeline 生成完整 rule_binding（含 SHA-256），可导出四件套短视频包
# rows = export_packages(
#     Path("D:/experiment/clips_out"),
#     Path("D:/experiment/video.mp4"),
#     store,
#     selected_events,
#     full_verified_binding,
#     scale,
#     workers=4,
#     target_overlay=True,
# )
# write_manifests(Path("D:/experiment/clips_out"), rows)
```

#### 按需单独运行社交层 / 群体层全流程精修（含候选重发现）

- **社交专项全套裁剪**（包含 `refine_behavior_segments` 之外的滚球打斗、高速追逐及 ID 跳变追逐主动发现）：

  ```python
  from mouse_behavior.evaluation.social_cropping_validation import refine_all_social_events, select_social_review_clips

  post_events, refine_meta = refine_all_social_events(pre_events, store, binding, scale, params)
  review_clips = select_social_review_clips(post_events, store, binding)  # 默认 max_per_behavior=None
  ```

- **群体 + 全行为裁剪汇总与抽检**：

  ```python
  from mouse_behavior.evaluation.group_cropping_validation import refine_all_canonical_and_group_events, select_group_review_clips

  post_events, refine_meta = refine_all_canonical_and_group_events(pre_events, store, binding, scale, params)
  review_clips = select_group_review_clips(post_events, store, binding)  # 默认 max_per_behavior=None
  ```

### 4.3 长期验证脚本：直接对长视频跑社交 / 群体专项裁剪

仓库还内置两个长视频专项验证脚本（默认 `--max-clips-per-behavior 0` 即不限制数量）：

1. **社交 6 行为长视频裁剪与候选导出**（[`scripts/validate_long_video_social_clipping.py`](../scripts/validate_long_video_social_clipping.py)）：

   ```powershell
   python scripts/validate_long_video_social_clipping.py `
     --video-key my_video `
     --title "My Long Video" `
     --video D:\experiment\video.mp4 `
     --inference-dir D:\experiment\run_all15_out `
     --output-dir D:\experiment\social_clipping_val `
     --clip-workers 4 `
     --max-clips-per-behavior 0
   ```

2. **群体 5 行为与全量长视频裁剪与候选导出**（[`scripts/validate_long_video_group_clipping.py`](../scripts/validate_long_video_group_clipping.py)）：

   ```powershell
   python scripts/validate_long_video_group_clipping.py `
     --video-key my_video `
     --title "My Long Video" `
     --video D:\experiment\video.mp4 `
     --pose-cache-dir D:\experiment\yolo_precompute `
     --output-dir D:\experiment\group_clipping_val `
     --workers 4 `
     --clip-workers 4 `
     --max-clips-per-behavior 0
   ```

---

## 五、输出目录结构与 JSON 字段说明

### 5.1 标准管线输出目录

运行 `run_canonical_behavior_video.py` 或 `run_all_behaviors_pipeline()` 后，会在 `--output` 目录生成：

```text
<output>/
├── input_binding.json      # 源视频、Pose缓存与模型哈希绑定及规则源码摘要
├── scale_receipt.json      # 像素到厘米尺度标定凭据（含 reference_body_cm=7.918）
├── tracks.sqlite           # 标准化轨迹数据库
├── events.json             # 识别与裁剪后的全部行为事件列表
├── manifest.json           # 导出片段总清单（含 counts_by_behavior、actor_id、target_id、participants）
├── manifest.csv            # 同清单表格形式
├── run_receipt.json        # 运行配置、状态与产物 SHA-256 账本
├── full/                   # 整段视频渲染产物（clip.mp4 / annotation.json / metadata.json / tracks.json）
└── clips/
    └── <event_id>/
        ├── clip.mp4        # 裁剪校验后的短视频片段
        ├── annotation.json # 单条事件标注、boundary、roles 与 participants
        ├── metadata.json   # 片段尺度和帧率等元信息
        └── tracks.json     # 片段区间内逐帧小鼠轨迹
```

### 5.2 有向社交行为 vs Together 的输出 JSON 示例

**有向社交行为（以 `Chasing` 追逐为例，`annotation.json` / `events.json` / `social_events.json` / `review_candidates.json`）**：

```json
{
  "event_id": "chase_disc_1_2_00000100_00000145",
  "behavior": "Chasing",
  "canonical_behavior": "chase",
  "mouse_ids": [1, 2],
  "actor_id": 1,
  "target_id": 2,
  "initiator_id": 1,
  "receiver_id": 2,
  "role_status": "measured_directed_role",
  "role_support_frame_intervals": [[0, 45]],
  "roles": {
    "initiator_role": "Chasing",
    "target_role": "Chased",
    "initiator_label_zh": "追逐者",
    "target_label_zh": "被追逐者",
    "initiator_id": 1,
    "target_id": 2,
    "summary_zh": "V1(追逐者) → V2(被追逐者)"
  },
  "participants": [
    {"role_name": "Chasing", "role_key": "chasing", "track_ids": [1]},
    {"role_name": "Chased", "role_key": "chased", "track_ids": [2]}
  ]
}
```

> 说明：在独立片段的 `annotation.json` 中，`role_support_frame_intervals` 与 `frame_range` 会自动平移为相对于该片段起点的帧号；而全局事件文件 `events.json` / `social_events.json` 中保留原视频半开帧 `[start_frame, end_frame)`。

**无方向社交行为（`Together` 一起/共处）**：

```json
{
  "behavior": "Together",
  "canonical_behavior": "together",
  "mouse_ids": [1, 2],
  "actor_id": -1,
  "target_id": -1,
  "initiator_id": null,
  "receiver_id": null,
  "role_status": "undirected_not_applicable",
  "roles": null,
  "participants": []
}
```

---

## 六、常见问题与质量门禁验证

1. **报错 `FileExistsError: refusing to overwrite run`**：
   - `run_canonical_behavior_video.py` 拒绝覆盖已有 `--output` 目录；请每次指定一个全新目录名（或手动清理旧目录后重跑）。
2. **报错 `ValueError: duration rejected`（尤其使用 `> 0.4 s` / `> 0.7 s` / `<= 9.0 s` / `<= 20.0 s` 标定裁剪事件导出时）**：
   - 使用 [`behavior_packages.py`](../src/mouse_behavior/visualization/behavior_packages.py) 导出经裁剪模块精修后的片段时，`binding` 需要包含来自裁剪/标定管线的受信任标记 `"clipping_duration_policy": "calibrated_clip_v1"`（由规范管线或上文示例自动携带），且事件区间须满足表二中该行为的最短与最长时长门控。
3. **为何某个识别片段在裁剪后被过滤掉？**
   - 检查 `refine_audit["rejected_below_min_duration"]`、`refine_audit["rejected_insufficient_participants"]` 与 `select_summary["rejected_counts"]`：常见原因包括端点帧小鼠缺失观测（未通过 `endpoints_fully_observed`）、参与鼠数不足、不满足行为专属最短/最长时长，或与同鼠重叠的高优先片段被去重保留。
4. **本地开发自测与 CI 质量门禁**：
   - 提交任何改动前可运行全量质量门禁：

     ```bash
     python scripts/run_quality.py --ci
     ```
