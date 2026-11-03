# V31 行为资格与审核片段裁剪

这次更新将行为成立条件与播放上下文分开。上下文不能使原本不成立的行为通过时长、移动距离或速度门槛。

| 行为 | V31 资格变化 |
|---|---|
| 接近 | 真实事件严格 >1 秒；明确行动鼠的实际累计路径 ≥10 cm；路径只累加相邻共见观测，不计算跨缺失跳跃。角色切换后的每个子事件独立复核。 |
| 攻击 | 真实攻击过程 ≥0.7 秒；原加速度、接触支持和有界遮挡条件保留。 |
| 追逐 | 双方各自的有效相邻观测路径／相邻步时间均 ≥30 cm/s，原方向、路径与净位移条件保留。角色子事件及窗口合并后的事件再次验收，避免借用父事件平均速度或重复计算快速重叠区间。 |
| 一起 | 保留近距离、双方低速和无接触相伴的定义。额外输出鼻接触几何证据状态，不把几何候选写成人工确认的互动。 |
| 孤立 | 保留严格 >20 cm 与有效比较时间 >10 秒；按用户更正，不允许先靠近、后截取远离片段作为补充规则。完整视频最终还按同一视觉 ID 的全源有效比较最小距离复核，窗口边界不能隐藏较早的近距离帧。缺失不是远离证据，不合并不同视觉 ID。 |

Together 的 `interaction_status` 为 `contact_geometry_present`、`no_contact_geometry_observed` 或 `contact_geometry_unknown`。任一测得鼻头／鼻尾起始距离进入现有接触几何门槛即可提供阳性几何证据；所有四个方向端点均可测且均为阴性才标为未观测到接触几何，缺项保持未知。状态及解释进入 annotation 和 manifest；不改变 Together 的标签优先级。

## 裁剪与格式

用户选择真实事件前后各加 0.25 秒，不设置五秒等统一最低播放长度。按视频帧率向外取整：30 fps 时每侧 8 帧，实际约 0.2667 秒；源视频边缘截断，不重复或补造画面。

- 视频和 tracks 覆盖整个裁剪范围，局部帧号／时间从 0 开始。
- annotation 的 `frame_range` 和 `time_range` 指向裁剪视频内部的真实行为范围，不能将上下文标成行为。
- metadata 的 source 指向原视频裁剪范围；provenance 分别记录 `crop_half_open`、`source_event_half_open` 和 `event_offset_half_open`。
- 支持证据区间相对裁剪起点重定位。时长验收在编码前以及编码后实际 FPS 上各执行一次，始终使用真实事件帧数。
- 每段仍仅有 `clip.mp4`、`annotation.json`、`metadata.json`、`tracks.json`。raw 导出 tracks 的核心字段为 frame、time、detections，包含原有可见检测，不插值或重分 ID。
- `original_video=True` 导出原画面及可用音频（AAC 重新编码）。H.264 准确裁剪会重新编码，不承诺二进制无损复制。

## 调用与兼容性

Python API `export_packages` / `run_pipeline` 新增：

```python
clip_context_seconds=0.0  # 旧调用保持原范围
original_video=False     # 旧调用保持渲染画面
clip_codec="libx264"     # 也可显式选择 h264_nvenc，无自动回退
```

CLI 新增 `--clip-context-seconds`（默认 0.25）、`--original-video`、`--clip-codec`。raw 的 `--mode clips` 无需先渲染完整叠加视频；完整渲染模式仍可单独使用。

`scripts/infer_from_delivered_yolo_json.py` 接受 `--input-dir`、新 `--output`、`--workers` 及 `--posture-mode`，从已交付 YOLO JSON 创建独立派生缓存后运行规则。它核验三文件与已有模型的哈希、帧号、时间、关键点及视觉 ID；明确标记未重新运行 YOLO/SAM，不把历史缓存哈希冒充为新缓存哈希。

所有输出使用新目录，不覆盖原视频、原推理 JSON、原数据集或人工审核结果。历史 A1/A2 不在本次裁剪更新中重算。原视频没有行为真值时，输出数量只是候选数，不能用作识别准确率。

## 验证入口

新增回归见 `test_requested_social_v31.py`、`test_behavior_clip_context.py`、`test_delivered_yolo_json_import.py`，包含严格时长边界、相邻观测路径、角色拆分、合并追逐平均速度、全源孤立否决、真实编码帧对齐、音频对齐及上下文不能补足行为资格等反例。历史算子测试需要旧门槛时显式传入历史配置，不能反向降低当前默认门槛。
