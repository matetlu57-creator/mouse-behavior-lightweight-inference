"""Prepare a local, unlabeled review batch from Phase A pose candidates.

Candidate frame intervals are clip-local and end-exclusive. Rendering never
calls the behavior predictor or changes its thresholds. Existing batches are
refused so human decisions and previous evidence cannot be overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.preprocessing.constants import SKELETON_EDGES


DECISION_FIELDS = ("decision", "observed_behavior", "reason")


def review_batch_hash(rows: list[dict], candidates_hash: str, context_s: float) -> str:
    # Review IDs are positional: a changed selection must not inherit labels
    # from a different batch produced from the same candidate CSV.
    payload = {"candidates_sha256": candidates_hash, "context_s": context_s,
               "windows": [candidate_key(row) for row in rows]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def candidate_key(row: dict) -> tuple:
    return (row["clip_path"], int(row["track_id"]), int(row["start_frame"]),
            int(row["end_frame"]), row["candidate"])


def select_batch(rows: list[dict], per_behavior: int = 15) -> list[dict]:
    """Include high-hind-speed controls, then long episodes with track diversity."""
    if per_behavior < 1:
        raise ValueError("per_behavior must be positive")
    unique = {}
    for row in rows:
        if row["candidate"] not in {"grooming", "rearing"}:
            raise ValueError(f"Unknown candidate type: {row['candidate']}")
        if int(row["start_frame"]) < 0 or int(row["end_frame"]) <= int(row["start_frame"]):
            raise ValueError("Candidate frame interval must be nonempty and nonnegative")
        for name in ("duration_s", "mean_hind_speed_cm_s", "mean_foreshorten"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Non-finite candidate metric: {name}")
        unique.setdefault(candidate_key(row), dict(row))
    selected = []
    for behavior in ("rearing", "grooming"):
        pool = sorted((r for r in unique.values() if r["candidate"] == behavior),
                      key=lambda r: (-float(r["duration_s"]), float(r["mean_foreshorten"]),
                                     candidate_key(r)))
        controls = [r for r in pool if behavior == "grooming"
                    and float(r["mean_hind_speed_cm_s"]) > 5.0]
        if len(controls) > per_behavior:
            raise ValueError("Increase per_behavior to include all high-hind-speed controls")
        chosen = list(controls)
        keys = {candidate_key(r) for r in chosen}
        tracks = {(r["clip_path"], r["track_id"]) for r in chosen}
        # First pass avoids spending the batch on repeated windows of one track.
        for diverse in (True, False):
            for row in pool:
                track = (row["clip_path"], row["track_id"])
                if len(chosen) >= per_behavior:
                    break
                if candidate_key(row) in keys or (diverse and track in tracks):
                    continue
                chosen.append(row)
                keys.add(candidate_key(row))
                tracks.add(track)
        for row in chosen:
            row["selection_reason"] = ("high_hind_speed_control" if row in controls
                                       else "long_episode_track_diversity")
            selected.append(row)
    for index, row in enumerate(selected, 1):
        row["review_id"] = f"{index:03d}"
        row.update(dict.fromkeys(DECISION_FIELDS, ""))
        row["decision"] = "pending"
    return selected


def _panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def render_candidate(row: dict, output_dir: Path, context_s: float = 1.0) -> dict:
    source_dir = Path(row["clip_path"])
    video = source_dir / "clip.mp4"
    tracks = json.loads((source_dir / "tracks.json").read_text(encoding="utf-8"))
    frames = {int(f.get("frame", i)): f.get("detections", []) for i, f in enumerate(tracks)}
    start, end, target = (int(row[k]) for k in ("start_frame", "end_frame", "track_id"))
    meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    fps = float(meta["clip"]["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid metadata FPS: {source_dir}")
    writer = None
    stem = f"{row['review_id']}_{row['candidate']}_track{target}"
    temporary = output_dir / f"{stem}.avi"
    destination = output_dir / f"{stem}.mp4"
    for path in (temporary, destination, output_dir / f"{stem}.jpg"):
        if path.exists():
            raise FileExistsError(path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(k)) for k in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if end > total or total != int(meta["clip"]["frame_count"]):
            raise ValueError(f"Video/metadata/candidate frame mismatch: {source_dir}")
        left, right = max(0, start - round(context_s * fps)), min(total, end + round(context_s * fps))
        boxes = [d["box"] for fi in range(left, right) for d in frames.get(fi, [])
                 if int(d.get("track_id", -1)) == target]
        if not boxes:
            raise ValueError(f"Target track {target} missing throughout review window")
        bounds = np.asarray(boxes, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("Invalid target bounding boxes")
        x1, y1 = np.floor(np.min(bounds[:, :2], axis=0) - 60).astype(int)
        x2, y2 = np.ceil(np.max(bounds[:, 2:], axis=0) + 60).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Target crop is outside the video")
        size = (1632, 512)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("Cannot create review intermediate")
        capture.set(cv2.CAP_PROP_POS_FRAMES, left)
        if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - left) > 0.5:
            raise RuntimeError("Video seek did not reach requested clip-local frame")
        missing = 0
        for fi in range(left, right):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode failed at frame {fi}: {video}")
            overlay = raw.copy()
            detections = [d for d in frames.get(fi, []) if int(d.get("track_id", -1)) == target]
            missing += not bool(detections)
            for detection in detections:
                points = np.asarray(detection.get("keypoints", []), dtype=float)
                if points.shape == (7, 3):
                    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
                    for a, b in SKELETON_EDGES:
                        if valid[a] and valid[b]:
                            cv2.line(overlay, tuple(np.rint(points[a, :2]).astype(int)),
                                     tuple(np.rint(points[b, :2]).astype(int)), (0, 255, 255), 2)
                    for i in np.flatnonzero(valid):
                        cv2.circle(overlay, tuple(np.rint(points[i, :2]).astype(int)),
                                   4, (0, 0, 255) if i == 0 else (0, 255, 0), -1)
                box = tuple(np.rint(detection["box"]).astype(int))
                cv2.rectangle(overlay, box[:2], box[2:], (0, 255, 255), 2)
            canvas = np.zeros((512, 1632, 3), dtype=np.uint8)
            canvas[80:, :768] = _panel(overlay, 768, 432)
            canvas[80:, 768:1200] = _panel(raw[y1:y2, x1:x2], 432, 432)
            canvas[80:, 1200:] = _panel(overlay[y1:y2, x1:x2], 432, 432)
            state = "CANDIDATE" if start <= fi < end else "CONTEXT"
            text = f"#{row['review_id']} {row['candidate']} probe | visual ID {target} | frame {fi} | {state}"
            if not detections:
                text += " | TARGET MISSING"
            cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255) if state == "CANDIDATE" else (200, 200, 200), 1)
            cv2.putText(canvas, "Scene + target skeleton                       Raw crop                  Skeleton crop",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            writer.write(canvas)
            if fi == (start + end - 1) // 2:
                if not cv2.imwrite(str(output_dir / f"{stem}.jpg"), canvas):
                    raise RuntimeError("Cannot save review poster")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-i", str(temporary),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-threads", "2",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True)
    check = cv2.VideoCapture(str(destination))
    try:
        if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != right - left:
            raise RuntimeError(f"Rendered video frame count mismatch: {destination}")
    finally:
        check.release()
    temporary.unlink()
    stat = video.stat()
    return {**row, "fps": fps, "context_start_frame": left, "context_end_frame": right,
            "rendered_frames": right - left, "target_missing_frames": missing,
            "candidate_start_s_in_review": (start - left) / fps,
            "candidate_end_s_in_review": (end - left) / fps,
            "video_file": destination.name, "poster_file": f"{stem}.jpg",
            "source_video_size": stat.st_size, "source_video_mtime_ns": stat.st_mtime_ns}


def write_review_page(rows: list[dict], output_dir: Path, batch_hash: str) -> None:
    cards = []
    for row in rows:
        def esc(key: str) -> str:
            return html.escape(str(row[key]), quote=True)
        cards.append(f'''<article data-id="{esc('review_id')}">
<h2>#{esc('review_id')} · {esc('candidate')} 候选 · visual track {esc('track_id')}</h2>
<p>{esc('clip')}<br>候选帧 [{esc('start_frame')}, {esc('end_frame')}) · {esc('duration_s')} s；
后肢均速 {esc('mean_hind_speed_cm_s')} cm/s · 缩短比 {esc('mean_foreshorten')}<br>
入选：{esc('selection_reason')}；目标缺失 {esc('target_missing_frames')} 帧</p>
<video controls preload="none" poster="{esc('poster_file')}" src="{esc('video_file')}"></video>
<div><button class="seek" data-time="{esc('candidate_start_s_in_review')}">跳到候选起点</button>
<label>播放速度 <select class="speed"><option>1</option><option selected>0.5</option><option>0.25</option></select></label></div>
<label>审计结论 <select class="decision"><option value="pending">待确认</option><option value="confirmed">确认候选</option><option value="rejected">否决候选</option><option value="uncertain">无法判定</option></select></label>
<label>实际行为 <select class="observed_behavior"><option value="">未填写</option><option value="grooming">趴着理毛</option><option value="rearing">站立 / 攀爬（含站立理毛）</option><option value="locomotion">移动</option><option value="static">静止</option><option value="other">其他</option><option value="unobservable">遮挡 / 无法观察</option></select></label>
<label>理由 <input class="reason" placeholder="如：头部抖动、遮挡、关键点塌缩；确认或否决均请记录依据"></label></article>''')
    # JSON is embedded as data, with HTML-significant characters escaped.
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>理毛 / 站立攀爬候选人工审计</title><style>
body{margin:24px auto;max-width:1400px;padding:0 20px;background:#f4f6f8;color:#18212b;font:16px/1.6 system-ui}
article{background:white;padding:20px;margin:24px 0;border-radius:12px;border:1px solid #ccd5df}h2{font-size:20px}
video{width:100%;background:#111}label{display:inline-block;margin:10px}input{width:min(650px,70vw)}button,select,input{font:inherit;padding:6px}header{position:sticky;top:0;background:#f4f6f8;padding:10px;z-index:1}p{overflow-wrap:anywhere}
</style><header><h1>30 窗口人工审计</h1><p id="status"></p><button id="export">导出审计 CSV</button></header>
<p>这些是无监督探针候选，不是已确认行为。左：场景及目标骨架；中：原始放大；右：骨架放大。黄色标题标识候选区间，CONTEXT 为前后上下文。帧号为 clip-local，终点不包含。红点为鼻头。</p>
<p>请区分趴着理毛、站立/攀爬、移动、静止，以及遮挡或姿态关键点塌缩。站立理毛归站立。允许“无法判定”，不要用片段来源文件夹当行为证据。这一批是正候选与疑似假阳审计，不能单独估计召回率。</p>
<p id="storage">审计结论暂存于当前浏览器；完成后请导出 CSV 留存。页面无需 API。</p>
CARDS<script id="rows" type="application/json">ROWS</script><script>
const rows=JSON.parse(document.getElementById('rows').textContent),key='pose-review-HASH';
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){document.getElementById('storage').textContent='浏览器暂存不可用；请导出 CSV 保存结论。'}
const fields=['decision','observed_behavior','reason'];
function status(){const done=rows.filter(r=>r.decision!=='pending').length;document.getElementById('status').textContent=`已审计 ${done}/${rows.length}；检测门仍关闭。`}
document.querySelectorAll('article').forEach(card=>{const row=rows.find(r=>r.review_id===card.dataset.id);const prior=saved[row.review_id]||{};
fields.forEach(f=>{const el=card.querySelector('.'+f);if(typeof prior[f]==='string')row[f]=prior[f];el.value=row[f];el.addEventListener('input',()=>{row[f]=el.value;saved[row.review_id]=Object.fromEntries(fields.map(k=>[k,row[k]]));try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){document.getElementById('storage').textContent='浏览器暂存不可用；请导出 CSV 保存结论。'}status()})});
const video=card.querySelector('video');video.defaultPlaybackRate=0.5;video.playbackRate=0.5;card.querySelector('.speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));card.querySelector('.seek').addEventListener('click',e=>{video.currentTime=Number(e.target.dataset.time)});
});status();document.getElementById('export').addEventListener('click',()=>{const keys=Object.keys(rows[0]);const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv=[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\\r\\n');const url=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='review_decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
</script></html>'''
    page = template.replace("HASH", batch_hash).replace("30 窗口人工审计", f"{len(rows)} 窗口人工审计")
    page = page.replace("CARDS", "\n".join(cards)).replace("ROWS", data)
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-behavior", type=int, default=15)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.context_seconds) or args.context_seconds < 0:
        parser.error("context-seconds must be finite and nonnegative")
    with args.candidates.open(encoding="utf-8-sig", newline="") as handle:
        selected = select_batch(list(csv.DictReader(handle)), args.per_behavior)
    if not selected:
        parser.error("No candidates available")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rendered = []
    for row in selected:
        rendered.append(render_candidate(row, args.output_dir, args.context_seconds))
        print(f"rendered {row['review_id']}/{len(selected)} {row['candidate']} track {row['track_id']}", flush=True)
    candidates_hash = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    batch_hash = review_batch_hash(rendered, candidates_hash, args.context_seconds)
    manifest = {"schema_version": 1, "status": "pending_human_review", "detectors_enabled": False,
                "candidates_sha256": candidates_hash, "batch_sha256": batch_hash,
                "context_seconds": args.context_seconds,
                "selection": f"{args.per_behavior} per behavior by duration, track diversity, all grooming hind>5 controls",
                "frame_interval": "clip-local [start_frame, end_frame)", "rows": rendered}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "review_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rendered[0]))
        writer.writeheader()
        writer.writerows(rendered)
    write_review_page(rendered, args.output_dir, batch_hash)
    print(f"Review ready: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
