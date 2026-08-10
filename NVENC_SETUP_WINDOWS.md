# RTX3060 Laptop NVENC配置

推荐：
ffmpeg -encoders | findstr nvenc

确认存在：
h264_nvenc

建议参数：
-c:v h264_nvenc
-preset p4
-rc vbr
-cq 23

不要使用过高preset，移动版RTX3060优先稳定吞吐。
