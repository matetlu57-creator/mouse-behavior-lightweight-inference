# v1.42.1 FINAL CODE MERGE 工程审查

## 审查结论

代码级合并通过当前可执行的静态、单元、数值等价和回归工具自检；可以作为 RTX3060 实机 A/B 的候选 FINAL 包。由于上传内容是 backend overlay，缺少完整项目侧 `mask_cluster_reid.py`、模型权重和代表性视频，本环境不能宣称真实长视频已经达到 ID/行为 ≥99% 或整体 2–4 倍。最终科研发布门槛必须以目标机器实测 A/B 为准。

## 1. 语法检查

执行：`python -m compileall -q .`

结果：PASS。主程序、base、新 Mask Trigger、新 NVENC writer 和 tests 均可编译。

## 2. 单元/回归检查

执行：`python -m pytest -q`

最终结果：5 passed。

额外执行 `python tests/regression_performance_test.py`：PASS。该套件比较上传包内旧实现与合并实现的 detection/appearance、派生几何、cluster context、pair feature、frame records、SQLite store、checkpoint，并验证新 Identity Cascade sparse matrix 与 dense NumPy matrix 逐元素完全一致。

另外执行 50 组随机空间布局/体型/预测偏移 fuzz：`cascade_sparse_numpy` 与 dense NumPy 代价矩阵 50/50 逐元素完全一致；测试候选密度范围约 0.085–0.44。

本机微基准仅表示热点函数，不代表整段视频：

- PairFeature：约 2.90×。
- Pair CSV store：约 51.3×。
- 20×20、仅 20/400 候选的 Identity NumPy matrix：约 1.10×，且 sparse/dense exact equal。
- 派生 geometry 属性重复读取的微循环加速很大，但不应外推成总程序倍率。

## 3. Identity 关键路径检查

发现并修复过一个接线缺陷：初版修改虽然 YAML 和 `KeypointMotionIdentityAssigner` 都有 `identity_cascade`，但主程序构造每视频 `identity_runtime_cfg["performance"]` 时未复制该字段。最终版本已显式注入，并增加 regression assertion。

门控语义检查：

- legacy Stable 路径仅复用原 `_distance_and_gate()`。
- active KeypointMotion 路径仅复用原 `_cost()` 开头已有的 hard size ratio 和 predicted-center hard gate。
- 不新增新阈值来拒绝旧逻辑本可接受的 pair。
- sparse candidate path 与 dense NumPy 在 20×20 稀疏测试上逐元素完全相等。
- C++ backend 没有被降级；可用时仍优先。

风险：NumPy sparse path 在候选不够稀疏时未必更快，因此设置 `min_cells` 与 `sparse_density_threshold`，密集帧保持 dense。

## 4. Pair Behavior 检查

没有修改 chase/attack/grapple/wall-jump 判定阈值或事件逻辑。v1.41.1 已经具备每帧缓存与 vectorized pair 计算，本次保留。回归套件对 PairFeature 输出做旧/新逐值比较并通过。

## 5. Mask Trigger 检查

上传包没有 `mask_cluster_reid.py`，所以不能安全直接编辑其源码。最终采用独立 controller + 主流程接入，不遮蔽用户完整项目模块。

当前 bundled 配置中 mask 对 Identity/ReID 仍有非零权重。若普通帧直接 skip，理论上可能改变身份匹配，因此默认 `result_preserving_only: true` 时继续运行 mask。也就是说：默认 FINAL 模式不把“Mask 90% 帧跳过”当作已经证明安全的优化。

可选 aggressive 模式具备：recovery、occlusion cluster、bbox overlap、identity uncertainty、periodic refresh 触发。必须先在用户代表性数据上 A/B，再决定是否用于科研正式分析。

## 6. NVENC / 输出检查

本容器没有可用 NVIDIA NVENC runtime。测试结果：`ffmpeg_nvenc_available()` 返回 False，factory 自动选择 OpenCV fallback，并成功写出 3 帧 MP4。

代码先检查 `ffmpeg -encoders`，再做 1 帧真实 NVENC probe，因此“FFmpeg 编译了 h264_nvenc 但驱动/GPU 不可用”的情况会在正式 writer 开启前回退。

不能在本环境验证：RTX3060 上真实 h264_nvenc 吞吐、Windows FFmpeg/驱动组合、长视频 pipe 稳定性。NVENC 中途硬件故障会抛异常，而不是静默生成损坏视频。

## 7. 输入输出检查

- bundled YAML 可由 PyYAML 正常解析。
- `model.half == true`。
- `performance.yolo_first_pass.batch_size == 4`。
- Identity Cascade、Mask Trigger、video_encoding 配置存在且类型合法。
- `validate_config` 增加 cascade density/min_cells 与 mask trigger 参数检查。
- 主分析 CSV/JSON schema 没有被主动修改；metadata 只新增性能诊断字段。
- NVENC 只改变渲染 MP4 编码路径，不参与科学分析状态。

`tests/regression_v1421.py` 已用合成的 baseline/optimized 目录自比：Tracking 1.0、Behavior 1.0、Schema PASS、RESULT PASS。

## 8. 异常处理检查

- FFmpeg 不存在、无 `h264_nvenc`、NVENC 硬件 probe 失败：回退 OpenCV。
- NVENC pipe BrokenPipe：读取 ffmpeg stderr 并抛 RuntimeError。
- writer release 非零退出：抛 RuntimeError。
- Mask Trigger controller disabled：走 legacy always-run mask。
- Identity cascade disabled/候选密集：走原 dense 路径。
- 安装脚本覆盖前自动备份；Python 编译失败会终止。

本 Linux 环境没有 PowerShell，因此 `install_optimized.ps1` 只能做文本/逻辑审查，不能执行 PowerShell 语法运行测试。

## 9. 用户约束核对

符合：修改顺序以 Identity 为先；不改 chase/attack 规则；保留当前 ID 主路径；Pair 不做重复 API 重构；Mask 默认不牺牲 ID；NVENC 最后接入；RTX3060 bundled 配置 batch=4；新增 ≥99% 真实 A/B 比较器。

需要明确的两点：

1. `batch_size 8 -> 4` 可能改变 GPU kernel 调度与吞吐，不能从代码静态证明“检测逐 bit 完全一致”或一定更快；必须实机 A/B。
2. 用户给出的“整体 0.3T–0.5T / 2–4×，复杂遮挡 5×”目前是目标，不是本包已测结论。Mask 严格模式还会保留每帧 mask，因此真实总倍率取决于 YOLO、mask 占比、C++ identity backend、视频编码占比和目标 GPU。

## 10. 最小正式验收建议

至少选择 3 类片段：普通分离 20 鼠、密集接触/遮挡、快速运动/ID 易交换。对同一输入各跑 v1.41.1 与 v1.42.1，执行 `tests/regression_v1421.py`；只有 Tracking/Behavior ≥0.99 且 schema PASS，再把该配置标记为科研正式版本。若打开 aggressive Mask Trigger，应重新独立做同一套 A/B。
