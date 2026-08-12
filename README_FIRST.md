# v1.43 STANDARD BEHAVIOR ENGINE — 先看这里

本包从 v1.42.1 FINAL CODE MERGE 继续升级。v1.43 **不修改 YOLO Pose、Identity、Observation 的核心执行语义**，主要重构追逐/攻击行为学部分，使行为判定从“多个独立 gate 最终 OR”升级为统一的标准时序框架：

`Observation Quality -> Pair Kinematics -> Contact Events + Continuous Evidence -> Behavior-specific Role -> FSM/Hysteresis -> Independent chase/attack Ethogram`

## v1.43 核心变化

- 新增 `standard_behavior_engine.py`：标准行为引擎独立模块。
- Pair 特征新增：体长归一化距离、径向 closing speed、连续 behind score、头部相对运动、nose-head/nose-tail/nose-body 接触类型、姿态形变能量。
- 轻量路径新增独立的 `lightweight_contact_events.csv`，分别记录 `nose_head` 和 `nose_tail`；接触不是攻击，普通接触不会单独打开攻击 FSM。
- Chase：A→B 与 B→A 分别计算连续证据，使用角色置信度、进入/退出不同阈值和持续时间确认。
- Attack：拆为 `lunge_attack`、`grapple_fight`、`occlusion_fight`；动态扑咬必须经过 `initiation -> contact -> reaction` 因果顺序。
- Observation Quality：低质量帧可短时维持已存在事件，但不能独立开启新行为。
- 旧 `strict/window/near/close-follow/impulse/grapple/occlusion` 规则保留为 evidence provider，不再在 standard 模式下直接 OR 成最终行为。
- Chase/Attack 角色分开推断；grapple/occlusion 可以“攻击事件成立但发起者未知”，以 `actor_id=-1` 明确表示不确定。
- 新增独立 Ethogram：`标准行为事件_时序引擎.csv`。追逐和攻击分别聚合，不因旧四分类兼容标签切换而切断持续行为。
- Interaction Graph 已实现；默认 `prune_pair_computation: false`，先保持原鼠对覆盖，实机验证后再开启裁剪。
- `decision_mode` 支持 `standard / shadow / legacy`，方便科研 A/B 与快速回退。

## 推荐安装

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_optimized.ps1
```

默认不覆盖你项目现有 YAML。若要同时安装本包的 v1.43 配置：

```powershell
.\install_optimized.ps1 -InstallBundledConfig
```

脚本会备份被覆盖文件到：

`backup_before_v1.43_YYYYMMDD_HHMMSS`

## 第一次实机运行建议

如果你希望先观察新算法而暂时不改变最终标签，把 YAML 改成：

```yaml
standard_behavior_engine:
  enabled: true
  decision_mode: shadow
```

此时所有 `*_standard_*` 分数、角色和 FSM 状态都会计算，但 `final_chase/final_attack` 仍由旧算法决定。

确认人工标注结果后再切回：

```yaml
standard_behavior_engine:
  decision_mode: standard
```

本包默认就是 `standard`。

## 科研验收原则

v1.43 是行为算法升级，所以**不能要求行为事件与 v1.42.1 ≥99% 完全相同**；那会把“新算法必须和旧算法一样”错误当成通过标准。

正确验收应拆开：

1. Tracking/Identity：应与 v1.42.1 保持等价，建议轨迹一致率 ≥99%。
2. Behavior：与人工标注比较 Precision / Recall / F1、事件起止误差、actor/target 正确率。
3. v1.42.1 vs v1.43 的行为差异只作为 A/B 报告，不作为必须相同的门槛。

A/B 工具：

```powershell
python tests\regression_v143_behavior.py "v1.42.1输出目录" "v1.43输出目录"
```

详细算法见 `V1.43_STANDARD_BEHAVIOR_ENGINE.md`，工程审查见 `ENGINEERING_REVIEW_v1.43.md`，运行步骤见 `RUNBOOK.md`。
