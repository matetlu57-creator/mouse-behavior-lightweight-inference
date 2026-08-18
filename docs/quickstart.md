# Quick start

The normal lightweight path consumes a completed YOLO Pose cache:

```powershell
python .\scripts\run_lightweight_behavior_inference.py `
  --video "D:\data\part_001.mp4" `
  --yolo-cache "D:\cache\part_001\yolo_precompute" `
  --config .\configs\profiles\balanced.yaml `
  --output-dir .\outputs\part_001 `
  --fps 29.329 `
  --expected-mice 20 `
  --sample-stride 1
```

The output directory contains behavior/contact CSV files, pair summaries,
metadata and the website-compatible export. Videos, caches and model weights
remain outside Git.
