# v1.43 STANDARD BEHAVIOR ENGINE 工程审查

日期：2026-08-10

## 1. 审查范围

本次从用户提供的 v1.42.1 FINAL CODE MERGE 升级行为学部分。目标是建立完整的标准时序行为框架，同时不改 YOLO Pose、Identity 主语义、Mask Trigger 和 NVENC。

v1.43 改动文件：

- `mouse_chase_attack_high_recall.py`
- `mouse_chase_attack_config.yaml`
- `standard_behavior_engine.py`（新增）
- `install_optimized.ps1`
- 行为单元/回归测试与文档

与 v1.42.1 基线逐文件比较确认：

- `mouse_chase_attack_extractor_base.py`：本次未改
- `mask_trigger_controller.py`：本次未改
- `nvenc_video_writer.py`：本次未改

因此 Identity Cascade、Mask Trigger、NVENC 的 v1.42.1 实现没有被行为学重构覆盖。

## 2. 算法结构审查

### 2.1 分层职责

已实现：

`Observation Quality -> Pair Feature -> Continuous Evidence -> Role -> FSM -> Ethogram -> Four-class`

旧 gate 在 standard 模式仅作为 evidence provider，不再直接 OR 到最终行为。

### 2.2 Chase

检查结果：

- A→B / B→A 分别评分；
- 连续 membership 代替阈值附近硬跳变；
- 使用 distance、speed、direction、pursuit、escape、behind、trajectory；
- 新增 radial closing speed 和 body-length normalized distance；
- `chase_role_confidence = |C_ab-C_ba|`；
- FSM 具有 IDLE/APPROACH/CHASE/RECOVERY；
- enter > exit；
- 进入需要持续确认；
- 低质量帧不能新开状态；
- wall jump/invalid/far pair 作为 veto。

### 2.3 Attack

检查结果：

- `lunge_attack`：必须遵循 initiation -> contact -> reaction；
- `grapple_fight`：持续接触 + repeated contact + angular/head motion + pose deformation；
- `occlusion_fight`：cluster overlap/deficit/motion 必须带最近 contact/initiation context；
- Attack 使用独立 role confidence，不复用 Chase role；
- dynamic lunge 要求 role confidence；
- grapple/occlusion 允许事件成立但 actor/target=-1，明确表达不确定性；
- FSM：NONE/PREPARE/CONTACT/ATTACK/RECOVERY。

### 2.4 Pose deformation

体坐标处理：

1. center 去平移；
2. heading 建立局部坐标；
3. body length 去尺度；
4. 对相邻测量帧共同有效关键点计算 RMS 内部变化。

额外修正：

- 非有限/零 body length 直接返回无效，避免 NaN 扩散；
- 修复布尔切片赋值写法，确保无效关键点真实写为 NaN；
- 单元测试证明刚体平移/旋转不产生 deformation，内部关键点变化可检测。

## 3. 角色与事件后处理审查

发现并修复两个重要问题：

1. 旧 `behavior_reconciliation` 会在事件形成后覆盖 FSM actor/target。v1.43 standard 模式默认 `preserve_standard_role_direction: true`；同一 pair 内不再覆盖标准行为角色，包括 `-1` 不确定角色。
2. 事件 actor/target 原先分别取众数，极端并列情况下可能形成不一致组合。已改成对 `(actor,target)` 配对整体取众数。

`allow_cross_pair_reassignment: false` 保持，避免事件跨屏转移到另一对鼠。

## 4. Four-class 审查

发现旧视频级分类仍会在 chase+attack 同时存在时，用 legacy `strict_chase/累计路径` 再否决 Chase FSM。

已修复：

- standard authoritative 模式：四分类是纯组合 `has_chase + 2*has_attack`；
- shadow/legacy 模式：保留旧 co-occurrence safeguard，保证回退语义。

## 5. Independent Ethogram

新增 `标准行为事件_时序引擎.csv`：

- chase 与 attack 独立聚合；
- start/peak/end；
- actor/target 与 role_ambiguous；
- attack subtype；
- mean/peak score；
- behavior/role confidence；
- 全视频范围内重新编号，避免不同 pair/chunk 出现重复 event ID。

## 6. Interaction Graph

已实现 17 cm candidate radius + occlusion pair force-keep。

科研默认：`prune_pair_computation: false`。

原因：第一次升级先保持原 pair/negative coverage。待真实视频 A/B 后再开启实际裁剪。

## 7. 输入/输出审查

### 输入

- 仍消费现有 `MouseObservation`、history、cluster context；
- 不读取文件名进行行为推断；
- 不改变 YOLO/Identity 输入。

### 输出

旧四分类/事件 CSV 保持；新增长列为 additive schema。新增：

- standard frame evidence/state/role 字段；
- `标准行为事件_时序引擎.csv`；
- metadata 中行为版本改为 `1.43.0-standard-behavior-engine`。

## 8. 异常与回退

- `decision_mode=standard/shadow/legacy` 非法值会明确抛出 ValueError；
- low quality 不允许新开 behavior；
- invalid pair/far pair/wall jump veto；
- occlusion 只有达到阈值且存在最近上下文才能越过独立中心失效；
- 安装器默认不覆盖 YAML；
- 安装器已补 `standard_behavior_engine.py`，避免漏文件导致 import error。

## 9. 实际测试结果

最终：

- `pytest -q`：14 passed
- `tests/regression_performance_test.py`：PASS
- `tests/identity_cascade_fuzz.py`：50/50 exact matrices PASS
- `python -m compileall -q .`：PASS
- YAML parse / chase weight sum / FSM hysteresis / strong≥weak / graph radius：PASS

重点单测包括：

- 持续方向性追逐进入 FSM；
- 低 pose quality 不能新开 chase；
- attack 必须满足 causal prepare/contact/reaction；
- 普通持续接触不误判 attack；
- independent chase/attack ethogram；
- pose deformation 刚体运动不变性；
- pose deformation 对内部形变敏感；
- ambiguous attack role 不被 reconciliation 强制覆盖；
- standard four-class 纯组合行为。

## 10. 未能在当前环境验证的部分

当前包不包含用户完整生产项目中的全部外部模块、实际 YOLO 权重和代表性 20 鼠长视频；当前环境也不是用户 RTX3060 Windows 实机。因此以下结论**没有被声称为已证明**：

- v1.43 对真实 Chase/Attack 的 Precision/Recall/F1；
- actor/target 实际正确率；
- bundled 初始阈值是否是用户范式的最优值；
- 20 鼠长视频总体速度；
- 开启 Interaction Graph pruning 后的真实加速与召回；
- NVENC 在用户驱动/FFmpeg 环境下的实际吞吐。

## 11. 科研正式启用建议

首轮建议用人工真值视频对比：

- v1.42.1 baseline；
- v1.43 `shadow`；
- v1.43 `standard`。

Tracking 要求保持 ≥99% 一致；Behavior 应对人工真值计算 Precision/Recall/F1，而不是要求与旧启发式 ≥99% 相同。

阈值校准完成后，再考虑打开 `interaction_graph.prune_pair_computation=true`。
