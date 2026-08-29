# 快速开始

轻量路径读取已经完成的 YOLO Pose 缓存，并在缓存基础上完成行为分析。视频、缓存
和模型权重都应放在仓库之外。

## 1. 安装项目

~~~powershell
python -m pip install -e .
python -m pip install -r requirements-dev.txt
~~~

## 2. 运行轻量分析

~~~powershell
python .\scripts\run_lightweight_behavior_inference.py --video "D:\data\part_001.mp4" --yolo-cache "D:\cache\part_001\yolo_precompute" --config .\configs\profiles\balanced.yaml --output-dir .\outputs\part_001 --fps 29.329 --expected-mice 20 --sample-stride 1
~~~

普通分析会输出行为事件、接触事件、鼠对汇总、元数据和网站导出包，不会自动
渲染视频。

## 3. 渲染行为视频

~~~powershell
python .\scripts\run_lightweight_behavior_inference.py --video "D:\data\part_001.mp4" --yolo-cache "D:\cache\part_001\yolo_precompute" --output-dir ".\outputs\part_001" --render-only --events ".\outputs\part_001\lightweight_behavior_events.csv" --render-output ".\outputs\part_001\轻量行为推理_渲染.mp4"
~~~

渲染视频包含小鼠框、ID、骨架、当前行为和行为面板。行为标签按小鼠参与者显示，
同一只小鼠按“群体行为 > 社交行为 > 个体行为”显示；群体行为只覆盖真实参与
群体事件的小鼠，不会把没有参与群体事件的小鼠统一标记为群体行为。

## 4. 生成 Pose 缓存

没有缓存时，先使用七关键点 Pose 权重：

~~~powershell
python .\scripts\build_lightweight_pose_cache.py --video "D:\data\part_001.mp4" --output "D:\cache\part_001\yolo_precompute" --model ".\weights\pose\best.pt" --device 0
~~~

## 5. 查看结果

重点文件如下：

- lightweight_behavior_events.csv：追逐、攻击和扩展行为事件；
- lightweight_contact_events.csv：鼻头、鼻尾接触事件；
- lightweight_pair_summary.csv：候选鼠对和特征摘要；
- lightweight_analysis_metadata.json：配置、缓存、帧数和耗时；
- annotation_website_import/：标注网站兼容的完整视频导入包。

短事件需要同时查看 core_duration_s 和 duration_s。前者是核心证据时长，后者
可能包含有限的显示上下文；event_recovery 记录短事件恢复来源。

更多说明见 [算法说明](algorithms.md)、[输出格式](data_format.md) 和
[故障排查](../README.md#故障排查)。
