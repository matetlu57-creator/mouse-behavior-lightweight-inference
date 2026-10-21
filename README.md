# Mouse behavior analysis

本项目现在只负责两件事：把外部产生的、带身份语义的小鼠轨迹规范化，以及从这些轨迹计算行为事件。检测、姿态估计和短时视觉追踪由 Ultralytics `YOLO.track()` 完成；跨遮挡和跨视觉 ID 的真实个体身份由 mTrack/RFID 映射提供。本项目不再实现第二套匈牙利匹配、亮斑识别或基于位置的身份猜测。

## 数据流

```text
video -> Ultralytics Pose track -> visual_track_id + keypoints
                                      |
mTrack/RFID -> normalized identity CSV+-> animal_id
                                      |
                                      v
                       trajectory normalization -> behavior analysis
```

`visual_track_id` 只是追踪器生成的临时轨迹编号。未提供 RFID 映射时，结果中的身份标签为 `visual:<id>`，只能作为临时视觉身份使用。提供映射后，行为事件和导出轨迹使用真实 `animal_id`；无法解析的检测保持未知并被排除，不会按位置、毛色、亮度或运动自动补配。

## 安装

建议使用 Python 3.10 的独立环境：

```powershell
conda env create -f environment.yml
conda activate mouse-behavior
python -m pip install -e .
```

Ultralytics 固定为已用当前 `Pose26` 权重验证过的 8.4.115。Pose 权重默认放在 `weights/pose/best.pt`，该路径使用 Git LFS；视频、缓存和实验输出不得提交到普通 Git。mTrack 可以保留自己的独立环境和依赖版本，只需输出下述 CSV，因此不需要为了兼容 RFID 端而降低行为分析环境的 Ultralytics 版本。

## 快速使用

先生成带 Ultralytics 视觉轨迹 ID 的 Pose 缓存：

```powershell
python scripts/build_lightweight_pose_cache.py `
  --video D:\experiment\video.mp4 `
  --model weights\pose\best.pt `
  --output D:\experiment\yolo_precompute `
  --tracker bytetrack.yaml
```

然后执行行为分析。纯视觉模式适合检查流程，但身份是临时的：

```powershell
mouse-behavior-lightweight `
  --video D:\experiment\video.mp4 `
  --yolo-cache D:\experiment\yolo_precompute `
  --config configs\default.yaml `
  --output-dir D:\experiment\behavior_output `
  --fps 30 `
  --expected-mice 20
```

RFID 模式额外传入 mTrack 归一化映射：

```powershell
mouse-behavior-lightweight `
  --video D:\experiment\video.mp4 `
  --yolo-cache D:\experiment\yolo_precompute `
  --identity-map D:\experiment\mtrack_identity_map.csv `
  --config configs\default.yaml `
  --output-dir D:\experiment\behavior_output `
  --fps 30 `
  --expected-mice 20
```

## mTrack/RFID 交换格式

适配器接受 UTF-8 CSV。逐帧格式为：

```csv
frame,visual_track_id,animal_id,identity_confidence
0,17,RFID-A,0.99
0,23,RFID-B,0.98
1,17,RFID-A,0.99
```

也可使用闭区间格式：

```csv
start_frame,end_frame,visual_track_id,animal_id,identity_confidence
0,249,17,RFID-A,0.99
250,600,31,RFID-A,0.97
```

兼容的视觉 ID 列别名为 `track_id`、`cv_track_id`；动物 ID 列别名为 `tag_id`、`rfid_id`、`epc`。同一视觉轨迹在重叠帧区间映射到不同动物时会直接报错。

## 配置与架构

默认行为参数在 `configs/default.yaml`，实验差异通过 `configs/profiles/*.yaml` 的 `extends` 覆盖。默认配置只保留笼界、尺度、特征、接触、追逐、攻击和扩展行为 FSM，不包含视觉追踪或身份恢复参数。

历史扩展接口保留跟随、群体同步运动、社会聚集和群体分散的定义。当前正典视频管线的默认活跃集合为 13 类，Following/Avoiding 停用，Grooming/Rearing 仅通过显式审核模式产生候选；不要把历史接口标签清单当作当前启用集合。角色、人数和时空定义见 [行为定义](docs/behavior-definitions.md)，当前实现、合并记录与限制见 [本轮记录](docs/posture-v3-duration-review-2026-10-21.md)。群体成员和角色不会用于反推或修复身份。

更完整的数据契约、输出语义和模块边界见 `docs/index.md` 与 `docs/adr/0001-external-tracking-and-rfid-identity.md`。

## 独立姿态候选与完整视频导出

理毛 Grooming、站立 Rearing（包括扶墙站立）使用独立规则与事件提取，默认保留停用策略；显式审核模式才输出待审核姿态候选。当前关键点证据的识别精确率较低，不能把候选当作人工确认结果，也不会自动修改 A1/A2。

已有视频和轨迹样本可直接生成完整渲染及分类四件套：

```powershell
python scripts/review_posture_sample.py --sample <样本目录> --track-id <视觉ID> --output <全新输出目录>
```

已有绑定来源的 pose cache 可通过完整视频管线运行：

```powershell
python scripts/run_canonical_behavior_video.py --video <源视频> --pose-cache <缓存目录> --source-receipt <来源绑定JSON> --posture-mode v3_review --output <全新输出目录>
```

每个分类包包含 `clip.mp4`、`annotation.json`、`metadata.json`、`tracks.json`。参与鼠以绿色粗框和原始视觉 ID 标注；缺失目标不伪造位置。独立事件在角色拆分、分段边界和编码前后按实际有效配置检查时长，保留严格 `>` 与 `>=` 的区别。

详细证据和边界见 [姿态接入与时长修复](docs/posture-v3-duration-review-2026-10-21.md) 和 [架构决策](docs/adr/2026-10-21-independent-posture-and-standalone-duration.md)。需要视频编码时安装 `python -m pip install -e .[review]`。

## 测试和质量门

```powershell
python -m pytest -q
python scripts/run_quality.py --ci
```

质量门包含 Ruff、字节码编译、测试、仓库边界/敏感信息检查和包构建。GitHub CI 分别使用 Python 3.10、3.11，检查名称为 `quality (3.10)`、`quality (3.11)`，与主分支保护一致。真实视频与 RFID 硬件联调不在单元测试内，发布前仍需使用受控实验数据做端到端验证。

## 许可证与支持

本仓库尚未选择开源许可证。在许可证明确前，不应假定可以复制、再分发或公开发布仓库代码及第三方模型/数据。问题排查请附上最小可复现配置、缓存 schema 版本和脱敏后的错误信息。
