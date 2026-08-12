# 模型权重

本目录是下载 GitHub Release 权重后的本地放置目录。当前项目上游检测/姿态缓存流程使用两份公开权重：

- `pose/best.pt`：YOLO Pose 小鼠关键点模型；
- `obb/best.pt`：OBB 小鼠检测模型，供完整 GUI/上游检测流程使用。

文件校验信息（SHA-256）：

- `pose/best.pt`：`E40B4A999953BEE482C960FF3DAB8EBC3A688753692853A936F985CD2A2E9EB8`；
- `obb/best.pt`：`70EBE4FEFCE3BC4CB1188EFDB769B8EF848633ED2531E3551254594E6D330AA5`。

轻量行为分析入口读取已经生成的 `yolo_precompute` 缓存，不会在缓存分析阶段重新加载这两份权重。生成缓存或运行完整流程时，可以把配置中的模型路径分别指向这两个文件。

由于权重是大二进制文件，它们作为同一 GitHub 仓库 Release 的附件上传。下载 `pose_best.pt` 与 `obb_best.pt` 后，放置为：

```text
weights/pose/best.pt
weights/obb/best.pt
```

不需要安装 Git LFS。正常克隆源码后，再从仓库的 Releases 页面下载两个附件即可。

权重文件不包含视频、缓存、账号信息或个人凭据。
