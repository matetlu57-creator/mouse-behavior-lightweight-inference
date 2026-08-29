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
configs/profiles/beiyi.yaml
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
- beiyi.yaml：北医短视频不使用自适应笼界，也不启用固定 polygon 笼界。北医 RFID-CV
  示例按 10 只小鼠处理，北医验证和渲染脚本默认使用 `expected_mice=10`；通用 20
  只小鼠视频应显式传入 `expected_mice=20`；
- beiyi.yaml：同时覆盖规范中可验证的行为持续时间；规范未规定固定时长的接近、
  回避、攻击和鼻头接触统一按至少 1 秒；孤立按当前项目规则按至少 3 秒。不覆盖
  个体和社交行为中尚未完成尺度校准的速度数值；短于对应规则的样例会被标记为未达到
  持续时间门；
- 群体扎堆：通用默认值使用 `huddle_distance_cm=5.0`；北医 profile 根据四个
  huddle 正向样例的稳定三鼠核心距离标定为 `11.0`。11 cm 是单条局部邻接边
  的阈值，不要求大型群体的所有两两中心距离都小于 11 cm；`huddle_density_mode`
  默认使用 `local`，所以远端对角成员不会单独使扎堆失效。北医关闭体长二次上限，
  避免白鼠和黑鼠混合时由中位体长把固定空间阈值错误缩小。无论使用哪套 profile，
  都还要求至少 3 只小鼠、局部 k-core、每只扎堆成员至少有两个同组邻居，并且事件
  至少持续 1 秒；需要保留旧的全局 all-pairs 密度时可设置 `huddle_density_mode=global`；
  这样可以拒绝两只小鼠的社交接近、三只小鼠的稀疏串联以及只有一帧的偶然聚拢；
- 轻量路径的厘米换算使用同一视频共享的体长尺度，不按白鼠或黑鼠分别缩放中心坐标，
  以避免同一画面出现不一致的坐标系。由于当前没有笼子实际尺寸或静态透视标定，
  `11.0` 是基于体长归一化的近似空间阈值；需要绝对物理厘米时，应提供固定相机
  标定参数。固定标定的配置形式如下，所有小鼠共用同一个 `cm_per_pixel`，不能按
  白鼠或黑鼠分别填写不同的比例：

  ```yaml
  scale:
    mode: fixed
    cm_per_pixel: 0.12
  ```

  未提供固定标定时，轻量路径会先对每个逻辑 ID 取身体长度中位数，再对这些 ID 的
  中位数取场景中位数，减少某一只小鼠检测帧更多造成的尺度偏移；这只能降低统计偏差，
  不能替代真实相机标定。
- 北医攻击 profile 另外配置 `bbox_occlusion_max_gap_frames=12`。该设置只生成短时
  框级遮挡桥接，供攻击接触/快速框运动证据使用，不会把预测框计入关键点轨迹或
  群体成员数；`bridge_avoidance=true` 只允许回避在短身份缺口后合并相邻片段；
- mouse_chase_attack_config.yaml：完整管线和旧调用方的兼容配置。

笼界模式由 `adaptive_arena.mode` 控制。`learned` 会从当前视频的运动轨迹
学习边界，适合有足够运动样本的长视频；`configured` 不读取短视频热力图，
直接使用 `detector_first.arena_mask.polygon` 中的固定物理笼界；`disabled`
则完全不启用笼界门控。北医 profile 使用 `disabled`，不会添加或应用固定
polygon。该选择来自显式 profile，不根据视频名称或目录名称推断。

轻量命令优先使用 configs/profiles/balanced.yaml 或 configs/default.yaml。根目录
兼容配置不应被新模块直接复制或分叉。
