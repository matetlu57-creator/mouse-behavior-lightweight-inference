# 修改记录：回退到已确认的导出数据校准版本

日期：2026-09-22

## 记录目的

在恢复用户粘贴文本中所确认的 `outputs/exported_dataset_calibration` 版本前，保存本工作树中后续实验性修改的范围、输出和验证状态。该记录不覆盖或删除工作树中用户已有的未提交修改。

## 本次回退前的实验性修改

- `src/mouse_behavior/evaluation/calibrated_rule_ranker.py`：增加观测到的单鼠/双鼠候选保留、时间特征、群体邻域/核心/剔除成员候选和社会聚集候选；为控制运行时间增加群体候选上限。
- `src/mouse_behavior/evaluation/exported_dataset.py`：实验性增加鼻-头/鼻-尾相对优势阈值，并允许群体事件在短时间缺帧时桥接；这些改动不属于本次恢复的已确认版本。
- `src/mouse_behavior/evaluation/induced_rule_tree.py` 与 `scripts/train_induced_rule_tree.py`：实验性增加会话留出门控，只有满足训练集和会话留出条件才开放验证。
- `tests/`：为上述候选、群体核心、时间特征、接触判定和缺帧桥接增加回归测试。

## 实验输出

- `outputs/existing_split_ranker_v12_listwise_full/`
- `outputs/existing_split_ranker_v17_temporal_margin/`
- `outputs/existing_split_ranker_v18_exhaustive_observed/`
- `outputs/existing_split_rules_v19_empirical_events/`
- `outputs/existing_split_rules_v20_group_cores/`（训练在候选规模过大时中断，不能作为完成结果）
- `outputs/induced_rule_tree_v1_depth12/`

上述目录均保留为实验记录，不作为本次验证结果。v20 训练未完成，不能据此声称达到训练或验证目标。

回退前关键实验文件 SHA-256：

- `src/mouse_behavior/evaluation/exported_dataset.py`: `A3C074F7F08F3328B9E3F66C2A07A73E5456C071DFFA99DC542ADC9307108B4E`
- `src/mouse_behavior/evaluation/calibrated_rule_ranker.py`: `4B9B55707D9A8A59E2AC467FC3025D7A1E460AB3AD723F61666F14DFFA9197AD`
- `src/mouse_behavior/evaluation/induced_rule_tree.py`: `4A695FC631DCF36AE0BDA12C707445D720D98CD7058D0670FF7A360964A0FE23`

## 本次恢复目标

恢复到用户粘贴文本所对应的既有结果目录和口径：

- 脚本：`scripts/calibrate_exported_dataset.py`
- 结果目录：`outputs/exported_dataset_calibration/`
- 数据划分：`outputs/exported_dataset_calibration/dataset_split/train.csv` 与 `val.csv`
- 目标 ID 级验证准确率：`220/274 = 80.29%`
- 宏平均目标 ID 准确率：`80.58%`
- 严格 Top-1：`160/274 = 58.39%`
- 宏平均 F1：`52.68%`
- 目标 ID 可用率：`100%`

恢复后只使用该目录中已经冻结的 `tuned_parameters.json`、`validation_predictions.csv` 和 `validation_metrics.json` 作为验证结果来源；后续实验文件不参与该结果计算。

## 恢复与渲染完成记录

- 已恢复 `exported_dataset.py` 到该校准路径使用的旧版接触判定和群体事件时序逻辑；恢复后文件 SHA-256 为 `AC89B6756DC49AFBCF0D93AD792868EBCA4209CCD0C32A02CC031666F6D2A20E`。
- 新增 `scripts/render_validation_result.py` 作为导出数据验证片段专用渲染入口；它读取冻结的预测 CSV 和片段轨迹，不重新调参。
- 已生成 `outputs/exported_dataset_calibration/rendered_validation/render_manifest.json` 以及每个保留行为 1 个视频，共 10 个 MP4，合计约 72.1 MB。
- 渲染入口文件 SHA-256：`17A5C2352D64F3CE63760581553AD7BDD645636D1B62554516269F035FDEC581`。
- 定向校准测试：`10 passed`。
- 完整质量门：`python scripts/run_quality.py --ci`，Ruff、compile、63 tests、repository validation、sdist/wheel build 全部 PASS。

## 识别错误视频

- `scripts/render_validation_result.py` 增加 `--errors-only`，只选择 `validation_predictions.csv` 中 `correct=False` 的样本。
- 已生成 `outputs/exported_dataset_calibration/rendered_validation_errors/render_manifest.json`。
- 共渲染 **54** 个错误样本视频，按行为分布：Approach 5、Attack 7、Huddling 12、Social clustering 22、Together 2、Walking 6；视频总大小约 226 MB。
- 错误视频目录与原有正确/示例视频目录分离，未覆盖已有 MP4。当前渲染脚本 SHA-256 为 `06FBF96BC1276D935564A2092CCEC4E285763BAC1D2B8279D6CF9A5F10EE2C8D`。
- 54 个视频均已完成文件存在性和 OpenCV 可打开检查。

## 行为标签渲染修订

- 复核确认：上一版 `render_validation_result.py` 只绘制了导出轨迹框和 ID，没有调用项目共享行为叠加逻辑，因此行为主要出现在顶部摘要，未按既有渲染风格显示在框和右侧侧栏中。
- 已修订脚本，复用 `src/mouse_behavior/visualization/overlay.py` 的 `build_mouse_overlays`、`draw_text_overlay`、`draw_behavior_sidebar`、角色颜色和行为优先级。
- 新视频左上角明确显示“标签行为 / 识别行为 / 结果 / target_hit / strict_top1”；框上和右侧侧栏显示算法识别到的行为及角色。
- 对识别错误样本，`error_reason` 明确区分“未识别标签行为”“预测行为与标签不一致”和“标签行为已检出但被其他行为占据 Top-1”。
- 修订后正确示例目录：`outputs/exported_dataset_calibration/rendered_validation_with_behavior/`，10 个视频。
- 修订后错误视频目录：`outputs/exported_dataset_calibration/rendered_validation_errors_with_behavior/`，54 个视频。
- 最终渲染脚本 SHA-256：`197150408554675AC3D3450FB47AF1D38B7016CC07336D61C1C36B99EF528D84`。
- 由于导出验证集提供的是片段级 `tracks.json`，而既有 `render_behavior_video()` 主入口要求生产轨迹 cache，本次通过该模块共用的 `visualization.overlay` 行为标签、角色颜色、框标签和侧栏组件完成渲染；没有把导出轨迹伪装成生产 cache。
- 修订后完整质量门：`python scripts/run_quality.py --ci`，63 tests、Ruff、compile、repository validation 和构建全部 PASS。

## 孤立行为框标签修正

- 原因：`Isolation` 在行为叠加层属于群体行为，但导出验证渲染入口曾把单个目标 ID 写成 `actor_id`；群体叠加只读取 `member_ids`，因此视频摘要能显示“孤立”，目标框和侧栏却退回“仅追踪”。
- 修正：`scripts/render_validation_result.py` 现在将 `Isolation` 与其他群体行为统一写入 `event_scope=group` 和 `member_ids`，单成员孤立事件也能绑定到目标框。
- 回归测试：`tests/unit/test_render_validation_result.py` 验证目标 ID 5 显示“群体：孤立”，其他 ID 仍显示“仅追踪”。
- 已重新渲染：`outputs/exported_dataset_calibration/rendered_validation_with_behavior/isolation_example_clip.mp4`。
- 当前完整质量门：`python scripts/run_quality.py --ci`，64 tests、Ruff、compile、repository validation 和构建全部 PASS。

## 正确/错误视频筛选与未识别标注修正

- 原渲染入口把 `correct=True`（仅目标 ID 命中）误当成“正确行为”，导致 10 个示例中混入行为 Top-1 错误；现在新增 `--correct-only`，只接受 `strict_top1_correct=True`。
- `--errors-only` 改为筛选 `strict_top1_correct=False`，覆盖行为误判、目标 ID 未命中和无行为预测的全部严格错误样本。
- 无预测时不再让目标框退回“仅追踪”：标签目标显示 `未识别`，其他非目标 ID 仍显示“仅追踪”；顶部摘要仍显示标签行为、识别行为和错误原因。
- 已重新生成 10 个严格正确视频、114 个严格错误视频，并重建浏览器数据：124 个条目（正确 10，错误 114）。
- 正确目录中旧版的 4 个误选视频已移入 `outputs/exported_dataset_calibration/_trash_old_renderings_2026-09-22/correct_examples_superseded_2026-09-22/`，未直接删除，且不再被清单引用。
- 空预测渲染的侧栏摘要同步改为“当前帧无行为预测，目标框显示‘未识别’”；新增 `--manifest-only` 用于从已验证可打开的 MP4 恢复完整清单，避免筛选性重渲染覆盖清单。

## 旧渲染结果清理

- 用户要求删除不满意的旧视频；旧目录已从活动结果目录移出：`rendered_validation`、`rendered_validation_errors`、`render_test_v2`。
- 为避免不可逆误删，旧目录暂存于：`outputs/exported_dataset_calibration/_trash_old_renderings_2026-09-22/`。
- 保留的正式渲染目录为 `rendered_validation_with_behavior` 和 `rendered_validation_errors_with_behavior`。

## 验证说明

本记录建立时，群体候选实验的定向测试已通过，但完整质量门是在最新群体实验修改前通过的；因此恢复后重新执行与恢复路径直接相关的测试，并单独检查验证指标和渲染产物。

## Git 状态

当前分支仍为 `main`。本次不执行 `reset --hard`、删除、commit、push 或 PR；工作树中的其他用户修改保持原样。

## 2026-09-22 群体行为语义调参与验证

- 渲染层只负责显示推理结果；本轮修正的是 `exported_dataset.py` 的行为规则，不把渲染结果当作分类器。
- Huddling 新增独立的最短休息时长与短暂掉框容忍，不再复用 Isolation 的群体确认时长。
- Isolation 新增独立的持续时长门槛；本轮训练集候选将空间阈值提高到 40 cm、持续时长设为 2 s，以减少 Walking/Stationary 被误判为 Isolation。
- Social clustering 保留至少 5 s 和最近邻距离下降判据，并新增 0.25 s 的短暂掉框容忍。
- 为区分“形成聚集”与“休息聚集”，新增 `clustering_min_mean_speed_cm_s` 规则参数；训练集局部搜索显示当前导出轨迹在 0–3 cm/s 范围内结果不变，因此冻结为 0.0，不用未经证据支持的运动阈值强行覆盖 Huddling。
- 训练集局部搜索还比较了 Social clustering 形成分数倍率 1.00–1.25；倍率 1.00 的训练严格 Top-1 最高，故恢复为 1.0，避免用 Social clustering 覆盖 Huddling。
- 训练集冻结候选参数保存于 `outputs/exported_dataset_calibration_v2/tuned_parameters.json`；验证集只在参数冻结后读取。
- 定向单元测试：`27 passed`。
- 本轮冻结候选训练结果：目标 ID 命中率 86.60%，严格 Top-1 61.56%。
- 本轮冻结候选验证结果：目标 ID 命中率 87.96%，严格 Top-1 64.96%；验证集按行为的严格 Top-1 仍未全部达到 90%，因此不能宣称达到 90% 目标。
- 该候选对应的验证集行为严格 Top-1：Approach 72.97%、Attack 46.43%、Huddling 64.00%、Isolation 84.00%、Snout-head contact 76.67%、Snout-rear contact 68.00%、Social clustering 50.00%、Static 74.07%、Together 57.69%、Walking 52.94%。

## 2026-09-22 缺帧与停走语义实验 v3

- Walking 增加内部短暂停顿桥接；Attack 增加短接触后近距离重新出现规则；Huddling 将缺帧桥接窗口扩展到 5 秒。
- 实验输出单独保存于 `outputs/exported_dataset_calibration_v3/`，未覆盖 v2 冻结结果。
- v3 验证集未识别由 13 条降至 3 条，目标 ID 命中率由 87.96% 提高到 92.34%。
- v3 严格 Top-1 为 62.77%，低于 v2 的 64.96%，因此不能作为当前严格 Top-1 主结果。
- 已用 v3 结果单独重渲染距离超过 17 cm 的 Approach 片段：`outputs/exported_dataset_calibration_v3/rendered_approach_gt17/`。
