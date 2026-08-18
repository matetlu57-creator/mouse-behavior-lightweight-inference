# v1.43 Standard Behavior Engine 运行与回归手册

## 1. 安装

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_optimized.ps1
```

默认安装：

- `mouse_chase_attack_high_recall.py`
- `mouse_chase_attack_extractor_base.py`
- `mask_trigger_controller.py`
- `nvenc_video_writer.py`
- `standard_behavior_engine.py`

若明确需要本包 YAML：

```powershell
.\install_optimized.ps1 -InstallBundledConfig
```

## 2. 行为引擎模式

### standard

```yaml
standard_behavior_engine:
  enabled: true
  decision_mode: standard
```

新 FSM 直接决定 `weak/strong final_chase/final_attack`。

### shadow

```yaml
standard_behavior_engine:
  enabled: true
  decision_mode: shadow
```

新引擎完整运行并写出分数、状态和角色，但最终标签保留 legacy 输出。适合首轮人工 A/B。

### legacy

只保留旧 final decision，用于紧急回退；v1.43 新字段仍可用于审计（若 enabled=true）。

## 3. 新增重点输出

`成对行为标签.csv` 新增：

- `*_standard_chase_score`
- `*_standard_attack_score`
- `*_standard_approach_score`
- `*_standard_contact_score`
- `*_standard_initiation_score`
- `*_standard_reaction_score`
- `*_standard_grapple_score`
- `*_standard_occlusion_score`
- `*_standard_chase_role_confidence`
- `*_standard_attack_role_confidence`
- `*_standard_chase_state`
- `*_standard_attack_state`
- `*_standard_attack_subtype`
- `*_standard_final_chase`
- `*_standard_final_attack`

独立科学 Ethogram：

`标准行为事件_时序引擎.csv`

它分别记录 chase / attack 的 start / peak / end、actor / target、subtype、score、confidence；不会被四分类标签切换拆段。

## 4. v1.42.1 -> v1.43 Tracking 回归

同一视频、同一模型、同一 tracking 配置分别跑 v1.42.1 和 v1.43：

```powershell
python tests\regression_v143_behavior.py "D:\data\v1421_results" "D:\data\v143_results"
```

默认：

- 正式 ID 数必须一致；
- raw→logical switch 数必须一致；
- 轨迹长度一致率 ≥0.99；
- v1.43 不能删除旧输出字段；
- 新 Ethogram 结构必须有效。

行为事件与 v1.42.1 的一致率只报告，不默认作为 PASS 条件。

## 5. 行为学人工标注验收

至少准备三类视频：

1. 普通近距离社交/嗅探但无攻击；
2. 明确追逐，包含方向转换、短暂停顿；
3. 明确攻击，分别覆盖扑咬/冲撞、贴身扭打、遮挡打斗。

建议统计：

- chase event Precision / Recall / F1；
- attack event Precision / Recall / F1；
- onset / offset 绝对误差；
- actor / target 正确率；
- ambiguous role 比例；
- false attack from social contact；
- false chase from co-moving / wall jump。

不要直接根据当前 bundled 阈值宣称科研准确率。阈值是结构化初始值，必须用你的人工真值校准。

## 6. Interaction Graph

默认：

```yaml
interaction_graph:
  enabled: true
  radius_cm: 17.0
  prune_pair_computation: false
```

此时只计算 `standard_interaction_candidate`，不改变原 pair 覆盖。验证后可实验 `true`，使 20 鼠只对空间邻接 edge 计算完整行为特征；occlusion pair 会强制保留。

## 7. NVENC 与 Identity Cascade

v1.42.1 的 NVENC、Identity Cascade、Mask Trigger 保留。它们与 v1.43 行为引擎职责分离；行为 A/B 时不要同时无记录地改变 tracking/model/encoding 参数。
