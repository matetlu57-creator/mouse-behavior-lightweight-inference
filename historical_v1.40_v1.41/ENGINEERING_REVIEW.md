# mouse_behavior v1.40.1 工程审查与性能优化报告

## 1. 审查结论

本次发布为 **performance-preserving refactor（结果保持型性能重构）**。目标不是关闭算法模块或调低分析标准，而是在以下科研结果约束下减少重复计算和 I/O 开销：

- YOLO/Pose 推理参数、输入分辨率、置信度阈值、NMS、最大检测数不变；
- 每一帧仍进入原有流程，不跳帧；
- 外观描述、伪实例掩码、异常姿态恢复、遮挡簇、聚集后 ReID、身份分配全部保留；
- 身份代价权重、行为阈值、时序窗口、接触定义不变；
- 20 只小鼠仍计算全部无序鼠对及两个方向的行为特征；
- 输出记录字段、字段顺序和数值定义不变；
- `mouse_chase_attack_config.yaml` 与用户上传版本 **SHA-256 完全一致**，未修改任何算法或资源参数。

发布版本：

- 主程序：`1.40.1-performance-preserving`
- 底层模块：`1.40.1-performance-preserving`

## 2. Review 发现与处理

### P1：Stage 2 按鼠对重复扫描完整 DataFrame

原 `PairDataFrameStore.read_pair()` 每读取一个 `pair_key` 都会重新执行整列字符串转换和全表布尔筛选。20 鼠共有 190 个无序鼠对，复杂度接近 `O(鼠对数 × 总行数)`，长视频中会成为显著瓶颈。

**处理：** 构造 `PairDataFrameStore` 时一次性建立 `pair_key -> 行索引`，后续用 `iloc` 定位对应行，再保持原有稳定帧排序。输出 DataFrame 经精确比较一致。

### P1：鼠对特征重复重建轨迹历史

原流程对每个方向都执行：

1. `history.get()` 复制 deque；
2. 构造 `{frame: observation}`；
3. 求公共帧；
4. 重算轨迹相关、路径长度、目标转角和距离收缩。

同一只小鼠在同一帧会与多个伙伴重复执行相同工作。

**处理：** 新增一帧生命周期缓存：

- 每个逻辑 ID 的历史窗口只转换一次；
- A→B 和 B→A 共享公共帧、路径长度和历史窗口；
- 目标转角按目标 ID 缓存；
- 距离收缩按无序鼠对缓存；
- 每进入下一帧自动清空，历史新增观测时相应 ID 的读取缓存立即失效。

为保持严格数值一致，相关系数的反向值仍按历史调用顺序重新执行 `safe_corr(B, A)`。测试过程中曾检测到直接复用正向相关会造成约 `1e-16` 的浮点末位差异，因此未采用数学上等价但字节级不完全等价的简化。

### P1：Detection 派生几何在热路径重复计算

`center_px` 和 `body_length_px` 会被候选去重、遮挡簇、mask、身份代价和轨迹更新反复访问。原 property 每次访问都重新进行数组转换、有效点筛选、加权平均或鼻尾距离计算。

**处理：** 在 Pose 恢复、候选融合和去重完成后显式刷新一帧内缓存。缓存具有以下安全边界：

- 检测几何修改后可显式 `invalidate_derived_geometry_cache()`；
- 缓存使用普通运行时属性，不进入 dataclass 字段、比较或 `asdict()` 输出；
- 深拷贝、pickle、旧缓存恢复时不保留派生缓存；
- 旧版本序列化对象缺少运行时缓存属性时可向后兼容加载；
- 缓存只保存派生值，不改变权威关键点、置信度或检测框。

### P2：每只检测重复创建 CLAHE 对象

原外观描述为每只检测调用 `cv2.createCLAHE()`。CLAHE 参数固定，对象创建与内部缓冲区分配属于纯工程开销。

**处理：** 每个工作线程复用一个参数完全相同的 CLAHE 对象。外观向量、亮度分数、白鼠预处理分支和可靠性判定逐项精确一致。

### P2：遮挡簇证据同帧重复计算

同一个两鼠簇在“建立图边”和“生成簇上下文”阶段会重复执行检测可见性、IoU 和匈牙利匹配。

**处理：** 在一次 `build_context()` 调用内按成员集合缓存 `_member_detection_evidence()`；同时预建活动状态中的两两成员集合，替代逐鼠对扫描所有历史簇。缓存不跨帧、不跨身份分配边界。

### P2：Stage 2 记录模板与 SQLite 调用边界重复

每帧最多 190 个鼠对记录都从头创建同一份帧级默认字典；多进程返回的记录又逐条调用 `raw_store.add()`。

**处理：**

- 每帧只创建一次默认记录模板，各鼠对复制后更新；
- 新增 `PairSQLiteStore.add_many()`，保持原有行顺序和相同 flush 阈值；
- debug CSV 使用 `writerows()` 减少 Python 调用边界。

## 3. 明确未做的修改

以下做法可能提高速度，但会改变证据链或输出，故本次没有采用：

- 关闭或降低 mask、appearance、pose recovery、cluster ReID；
- 降低视频帧率、跳帧、缩小模型输入；
- 减少鼠对数量或只计算近邻；
- 调整行为阈值、身份代价权重或时间窗口；
- 将 `float64` 全局替换为 `float32`；
- 修改 YAML 中线程数、worker 数、batch、队列或输出开关。

## 4. 验证结果

### 静态验证

- `python -m py_compile`：通过；
- YAML SHA-256：优化包与原文件一致；
- 可重复补丁脚本：`apply_optimizations.py` 从 `original/` 重新生成优化版并对每个替换点做唯一性断言。

### 回归测试

`tests/regression_performance_test.py` 和 `pytest` 套件覆盖：

- 检测中心、体长、缓存失效、pickle 恢复；
- CLAHE 外观描述、归一化姿态、anchor、heading、brightness、white score；
- 主程序及底层 `PairFeatureComputer` 的全部 dataclass 字段；
- A→B/B→A 方向顺序与浮点值；
- 下一帧到来后的缓存失效；
- Stage 2 帧记录字段、字段顺序和值；
- SQLite 逐条写入与 `add_many()` 的逻辑表一致性；
- `PairDataFrameStore` 的全部鼠对读取及缺失键行为；
- 断点序列化不携带派生读取缓存。

测试状态：`1 passed`。

### 隔离微基准

本容器最近一次结果见 `TEST_RESULTS.txt`。典型结果为：

- 派生几何重复读取：约 **300×以上**；
- 20 鼠双向轨迹/行为特征热循环：约 **2.5–3×**；
- 190 个鼠对从 19 万行缓存中逐对读取：约 **40×以上**。

这些是单模块微基准，**不能相乘，也不等于整段视频总加速比**。整流程速度仍取决于 GPU 推理、GrabCut/mask、视频解码、磁盘和渲染占比。

## 5. 未能在当前环境完成的验证

当前会话只提供了主程序、底层模块和 YAML，没有以下运行依赖：

- 原始 MP4；
- `weights/best.pt`；
- `pose_quality_recovery.py`；
- `mask_cluster_reid.py`；
- `adaptive_arena_boundary.py`；
- `disk_sequence_guard.py` 及可选 pybind11 后端。

因此未声明整视频 FPS、总耗时或 GPU 利用率提升。交付包提供 `tests/compare_outputs.py`，用于在用户工作站上对原版和优化版的全部 CSV 做流式逐单元格比较。

## 6. 仍存在的工程债务

本次为低风险补丁，没有进行大规模架构重写。后续可单独立项：

1. 主程序约 1.1 万行、底层模块约 1.1 万行，并采用多代类覆盖，长期维护成本高；
2. Stage 2 多进程仍以 Python 字典列表进行 IPC，可评估列式共享内存或 worker 本地 SQLite，但必须先建立整视频 golden test；
3. 分段视频最终合并会重新解码和编码，可研究在编码参数完全一致时使用无重编码 concat；
4. mask/候选 CPU 模块的进一步优化需要补充对应源文件和真实 profiler 报告，不能凭假设改写。

## 7. 发布建议

先在同一台机器、同一驱动和同一依赖环境下，对一段 300–1000 帧代表性视频进行 A/B：

1. 原版、优化版分别输出到独立目录；
2. 运行 `tests/compare_outputs.py`；
3. 要求 `failures=0`；
4. 再比较 `运行元数据.json` 中 profiler 的阶段耗时；
5. 通过后再跑完整正式视频。
