# 配置说明

配置采用 YAML 继承。默认入口是 configs/default.yaml，运行 profile 位于
configs/profiles/，实验覆盖位于 configs/experiments/。

## 配置关系

~~~text
configs/default.yaml
        ↓
configs/profiles/fast.yaml
configs/profiles/balanced.yaml
configs/profiles/high_accuracy.yaml
        ↓
configs/experiments/<experiment>.yaml
~~~

每个 profile 只覆盖与当前运行模式有关的参数。实验配置应记录数据来源、假设、
采样步长、阈值变化和随机种子，不要直接改动默认配置。

## 代码中的读取方式

需要解析继承关系的代码使用 mouse_behavior.config.load_config()。不要直接调用
yaml.safe_load() 读取 profile，否则 extends 关系不会展开。

运行时解析后的配置会写入适用的输出目录，用于复现实际生效的参数。变更配置后，
应至少运行相关单元测试、仓库验证和一组代表性样例。

## 运行建议

- fast.yaml：快速检查输入、缓存和输出路径；
- balanced.yaml：常规行为分析；
- high_accuracy.yaml：使用更小采样步长，优先保留短暂行为；
- mouse_chase_attack_config.yaml：完整管线和旧调用方的兼容配置。

轻量命令优先使用 configs/profiles/balanced.yaml 或 configs/default.yaml。根目录
兼容配置不应被新模块直接复制或分叉。
