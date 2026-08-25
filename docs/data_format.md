# 数据与输出格式

一次运行的生成物位于被 .gitignore 排除的 outputs/<run-id>/ 目录。视频、缓存、
模型权重和私人标注不应进入 Git。

## 主要文件

- lightweight_behavior_events.csv：追逐、攻击和扩展行为事件；
- lightweight_contact_events.csv：鼻头、鼻尾和组合接触事件；
- lightweight_pair_summary.csv：候选鼠对、距离、朝向、速度和 FSM 诊断；
- lightweight_top_evidence.csv：高价值证据帧摘要；
- lightweight_analysis_metadata.json：输入、配置、帧数、耗时和运行模式；
- annotation_website_export_report.json：网站导出数量和跳过原因；
- annotation_website_import/：标注网站兼容的完整视频导入包；
- 轻量行为推理_渲染.mp4：显式请求渲染时生成的可视化结果。

## 短事件字段

lightweight_behavior_events.csv 中的时间字段分为核心证据和公开上下文两组：

- analysis_start_frame、analysis_peak_frame、analysis_end_frame：分析帧索引，
  表示几何和运动证据范围；
- core_start_frame、core_end_frame：换算到原始视频后的核心帧范围；
- core_duration_s：核心证据持续时间；
- start_frame、end_frame：渲染、切片和网站导出的公开范围；
- start_time_s、end_time_s、duration_s：公开范围对应的时间；
- temporal_padding_frames：公开范围相对于核心范围增加的帧数；
- event_recovery：事件来源，普通事件为 none，短攻击和短回避恢复分别记录
  short_high_evidence 或 short_escape_evidence。

公开范围可能比核心范围长，但不会把前后文当作行为证据。统计行为持续时间时，
优先使用 core_duration_s；制作复核视频或片段时使用 start_frame 和 end_frame。

## 行为和参与者字段

- behavior：内部稳定行为名，例如 attack、avoidance、huddle；
- behavior_name_zh：中文显示名；
- actor_id、target_id：鼠对行为的主动方和被动方；
- pair_key：鼠对或事件区域标识；
- member_ids：群体事件在核心时间段内出现的实际成员；
- member_ids_at_peak：行为峰值帧的实际成员；
- candidate_level、mean_score、peak_score：候选级别和事件分数；
- analysis_mode、behavior_engine：生成路径和引擎来源。

群体事件没有可靠成员证据时，不会为了填满视频画面而虚构参与者。旧 CSV 没有
member_ids 时，标注网站导出适配器才会使用兼容的几何回退逻辑。

## 标注网站导入包

annotation_website_import/<video-name>/ 遵循 schema version 1.0，包含：

- video.mp4 或 video.mov：原始完整视频，优先使用硬链接；
- annotations.json：网站行为标注；
- tracks.jsonl：从零开始连续编号的逐帧轨迹，包含空检测帧；
- metadata.json：视频 ID、帧数、FPS、尺寸、关键点名和骨架连接。

网站文件使用排序去重后的 mouse_ids。网站导入格式不使用 ID 顺序表达主动方和
被动方，主动方和被动方仍保留在内部事件 CSV 中。

## 元数据和耗时

元数据记录原始视频、缓存、解析后的配置、分析模式和总耗时。可用的
stage_timings_s 阶段包括视频探测、笼界准备、缓存读取、运动学、候选鼠对筛选、
Pair 特征、事件汇总、网站导出和 CSV 写入。

parallel_behavior_fsm 元数据记录实际生效的 enabled、mode 和
execution_semantics。这些字段是运行溯源，不是未经验证的配置文本复制。

## 验证结果的解释

北医样例汇总中的 video_coverage=1.0 是视频级覆盖指标。没有逐帧行为起止时间和
参与者真值时，不能据此宣称算法已经达到某个 Precision、Recall、F1 或角色准确率。
