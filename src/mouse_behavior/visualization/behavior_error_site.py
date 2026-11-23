"""Read-only primary-evaluation review with raw visual-ID video overlays."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from .comparison_site import ComparisonServerHandler

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = ROOT / "outputs/all15_latest_20261116_codex/report.json"
DEFAULT_DIAGNOSTICS = ROOT / "outputs/behavior_error_review_20261116/bound_huddle_diagnostics.json"
ASSETS = Path(__file__).with_name("error_review_assets")
LABELS = {
    "Approach": "接近", "Attack": "攻击", "Chasing": "追逐",
    "Snout-head_contact": "嗅闻头部", "Snout-rear_contact": "嗅闻尾部",
    "Together": "共处", "Static": "静止", "Locomotion": "移动",
    "Grooming": "理毛", "Rearing": "站立", "Isolation": "孤立",
    "Group locomotion": "群体同步运动", "Dispersal": "散开",
    "Social clustering": "社会聚集", "Huddling": "扎堆",
}


def primary_reason(row: dict[str, Any]) -> dict[str, str]:
    """Classify only the selected primary method, never a global rival's label."""
    behavior = row["behavior"]
    label = LABELS.get(behavior, behavior)
    primary = row.get("primary_predicted_behavior")
    prediction = LABELS.get(primary, primary) if primary else "专项检测未输出通过结果"
    note = "原因依据本次专项评测记录；候选证据不等于人工复核结论。"
    if row["primary_correct"]:
        code, title, summary = "passed", "通过", f"{label}通过当前专项评测。"
        prediction = prediction if primary else f"检测到{label}"
        pf_pred = row.get("pf_target_predicted_behavior")
        if row.get("pf_target_correct") is False and pf_pred and pf_pred != behavior:
            pf_label = LABELS.get(pf_pred, pf_pred)
            summary = f"{label}专项通过，但15类竞争模型预测为{pf_label}。"
            prediction = f"专项通过 · 15类竞争预测：{pf_label}"
    elif row.get("benchmark_status", "ok") != "ok":
        code, title, summary = "unavailable", "评测异常", "本片段评测未正常完成，不能归为行为漏检。"
    elif behavior == "Huddling":
        diagnostic = row.get("huddle_diagnostics") or {}
        if not diagnostic.get("a1_any_positive"):
            code, title, summary = "not_detected", "未检测到", "整段没有生成扎堆候选。"
        elif not diagnostic.get("a2_any_positive"):
            code, title = "other_ids", "目标 ID 不符"
            ids = ", ".join(map(str, diagnostic.get("best_huddle_target_ids", [])))
            summary = f"检测到了扎堆，但没有候选匹配完整标注目标；最佳扎堆候选 ID：{ids}。"
            prediction = f"扎堆候选存在，成员 ID 为 {ids}"
        else:
            code, title = "short_coverage", "覆盖时间不足"
            coverage = float(row.get("true_target_coverage", 0))
            summary = f"找到了完整目标集合的扎堆候选，但覆盖只有 {coverage:.1%}，未达到 50%。"
            prediction = "正确成员的扎堆候选未达到覆盖门槛"
        note = "此处按扎堆候选、完整目标 ID 和时间覆盖判断；其他鼠的其他行为不作为错判依据。"
    elif row.get("failure_category") == "misclassified":
        code, title = "other_behavior", "其他行为胜出或触发"
        rival = primary or row.get("pf_pair_predicted_behavior") or row.get("pf_target_predicted_behavior")
        rival_label = LABELS.get(rival, rival) if rival else "其他行为"
        if primary:
            summary = f"标注为{label}，专项输出为{rival_label}。"
        else:
            summary = f"{label}专项未通过；同目标上触发了{rival_label}候选。"
            prediction = f"{label}未通过；同目标候选：{rival_label}"
        note = "仅采用同目标的专项输出或规则触发证据；候选名称与最终专项输出有区别，详见原始原因。"
    elif behavior in {"Grooming", "Rearing"} and float(row.get("true_target_coverage", 0)) > 0:
        code, title = "short_coverage", "覆盖时间不足"
        summary = f"目标上有{label}证据，但覆盖 {float(row['true_target_coverage']):.1%}，未达到 50%。"
    elif float(row.get("true_target_coverage", 0)) > 0:
        code, title = "incomplete_evidence", "证据未达专项门槛"
        summary = f"目标上存在{label}证据，但未通过该行为的专项判定；具体门槛见逐段原因。"
    else:
        code, title, summary = "not_detected", "未检测到", f"目标上的{label}规则未达到触发门槛。"
    return dict(reason_code=code, reason_label=title, summary=summary,
                prediction_text=prediction, classification_note=note)


class ReviewStore:
    def __init__(self, report_path: Path, diagnostics_path: Path | None = None):
        self.report_path = Path(report_path)
        self.diagnostics_path = diagnostics_path
        self.signature: tuple[int, int] | None = None
        self.records: dict[str, dict[str, Any]] = {}
        self.catalogue: dict[str, Any] = {}
        self.gates: dict[str, dict[str, Any]] = {}

    def load(self) -> dict[str, Any]:
        stat = self.report_path.stat()
        signature = stat.st_mtime_ns, stat.st_size
        if signature == self.signature:
            return self.catalogue
        raw = self.report_path.read_bytes()
        report = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(report, dict) or report.get("schema_version") != "reconciled_behavior_evaluation_v1":
            raise ValueError("评测报告格式不匹配，未载入替代数据。")
        rows = report.get("clips")
        if not isinstance(rows, list) or not rows:
            raise ValueError("评测报告没有片段记录。")
        required = {"record_key", "dataset", "clip_id", "clip_dir", "behavior", "target_mouse_ids",
                    "primary_correct", "frame_count", "fps"}
        for row in rows:
            if (not isinstance(row, dict) or not required.issubset(row)
                    or row["dataset"] not in {"A", "B", "C"} or row["behavior"] not in LABELS
                    or not isinstance(row["primary_correct"], bool)
                    or not isinstance(row["target_mouse_ids"], list)
                    or not all(isinstance(value, int) for value in row["target_mouse_ids"])
                    or not isinstance(row["fps"], (int, float)) or not math.isfinite(row["fps"])
                    or row["fps"] <= 0 or not isinstance(row["frame_count"], int) or row["frame_count"] <= 0):
                raise ValueError("评测片段字段缺失或类型不正确。")
        records = {row["record_key"]: row for row in rows}
        if len(records) != len(rows):
            raise ValueError("评测记录键重复，无法可靠绑定视频。")
        digest = hashlib.sha256(raw).hexdigest()
        clips = []
        for row in rows:
            reason = primary_reason(row)
            clips.append(dict(id=row["record_key"], dataset=row["dataset"], behavior=row["behavior"],
                label=LABELS[row["behavior"]], clip_id=row["clip_id"], target_ids=row["target_mouse_ids"],
                correct=row["primary_correct"], in_union=bool(row.get("in_union_b_priority") or row["dataset"] in {"B", "C"}),
                pf_target_correct=row.get("pf_target_correct"),
                pf_target_predicted_behavior=row.get("pf_target_predicted_behavior"),
                duration_s=row["frame_count"] / row["fps"], fps=row["fps"], frame_count=row["frame_count"],
                source_video=row.get("source_video", ""), **reason))
        has_c = any(r["dataset"] == "C" for r in clips)
        has_a = any(r["dataset"] == "A" for r in clips)
        full_stats = report.get("full_cohort_stats") if isinstance(report.get("full_cohort_stats"), dict) else {}
        if has_c and not has_a:
            cohort_defs = (("union", "B ∪ C · 新旧清洗集合并"), ("B", "B · 605 清洗集"), ("C", "C · 764 新清洗集"))
            default_cohort = "union"
            default_status = "all"
        else:
            cohort_defs = (("B", "605 清洗集"), ("A", "429 补全集"), ("union", "去重并集 · B 优先"))
            default_cohort = "B"
            default_status = "failed"
        cohorts = {}
        for name, label in cohort_defs:
            if has_c and not has_a and name == "union":
                selected = [r for r in clips if r["dataset"] in {"B", "C"}]
            elif name == "union":
                selected = [r for r in clips if r["in_union"]]
            else:
                selected = [r for r in clips if r["dataset"] == name]
            f_cohort = full_stats.get(name, {}) if isinstance(full_stats, dict) else {}
            f_beh_map = f_cohort.get("per_behavior", {}) if isinstance(f_cohort.get("per_behavior"), dict) else {}
            behaviors = []
            for behavior, zh in LABELS.items():
                subset = [r for r in selected if r["behavior"] == behavior]
                correct = sum(r["correct"] for r in subset)
                fb = f_beh_map.get(behavior, {}) if isinstance(f_beh_map.get(behavior), dict) else {}
                b_entry = dict(behavior=behavior, label=zh, total=len(subset), correct=correct,
                    failed=len(subset) - correct, accuracy=correct / len(subset) if subset else None,
                    error_counts=dict(Counter(r["reason_code"] for r in subset if not r["correct"])))
                if fb:
                    b_entry["full_total"] = fb.get("support")
                    b_entry["full_accuracy"] = fb.get("primary_accuracy")
                    b_entry["full_precision"] = fb.get("pf_target_precision")
                    b_entry["full_recall"] = fb.get("pf_target_recall")
                behaviors.append(b_entry)
            correct = sum(r["correct"] for r in selected)
            c_entry = dict(label=label, total=len(selected), correct=correct, failed=len(selected)-correct,
                           accuracy=correct / len(selected) if selected else None, behaviors=behaviors,
                           is_bc_union=bool(has_c and not has_a and name == "union"))
            f_sum = f_cohort.get("summary", {}) if isinstance(f_cohort.get("summary"), dict) else {}
            if f_sum:
                c_entry["full_total"] = f_sum.get("total_clips")
                c_entry["full_accuracy"] = f_sum.get("primary_accuracy")
                c_entry["full_precision"] = f_sum.get("pf_target_micro_precision")
                c_entry["full_recall"] = f_sum.get("pf_target_micro_recall")
            cohorts[name] = c_entry
        hashes = report.get("provenance", {}).get("source_sha256", {})
        changed = []
        for relative, expected in hashes.items():
            source = (ROOT / relative).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                changed.append(relative)
        gates = {}
        if self.diagnostics_path is not None and self.diagnostics_path.is_file():
            diagnostics = json.loads(self.diagnostics_path.read_text(encoding="utf-8-sig"))
            if diagnostics.get("source_report_sha256") == digest:
                gates = {row["clip_id"]: row for row in diagnostics["records"]}
        self.records, self.gates = records, gates
        self.catalogue = dict(schema="behavior_error_catalogue_v1", generated_at=report.get("generated_at_epoch_s"),
            source_hash=digest, source_code_matches=bool(hashes) and not changed, changed_sources=changed,
            default_cohort=default_cohort, default_status=default_status,
            cohorts=cohorts, clips=clips)
        self.signature = signature
        self._detail.cache_clear()
        return self.catalogue

    def media_path(self, key: str) -> Path:
        self.load()
        return Path(self.records[key]["clip_dir"]) / "clip.mp4"

    def detail(self, key: str) -> dict[str, Any]:
        self.load()
        return self._detail(key)

    @lru_cache(maxsize=8)
    def _detail(self, key: str) -> dict[str, Any]:
        row = self.records[key]
        directory = Path(row["clip_dir"])
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
        media = metadata.get("clip", {})
        width, height = int(media.get("width", 0)), int(media.get("height", 0))
        fps = float(media.get("fps", row["fps"]))
        count = int(media.get("frame_count", row["frame_count"]))
        if width <= 0 or height <= 0 or count != row["frame_count"] or fps <= 0 or not math.isfinite(fps):
            raise ValueError("视频尺寸、帧数或帧率与评测记录不匹配。")
        raw = (directory / "tracks.json").read_bytes()
        if row.get("tracks_sha256") and hashlib.sha256(raw).hexdigest() != row["tracks_sha256"]:
            raise ValueError("轨迹文件已变化，请重新评测后查看高亮。")
        tracks = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(tracks, list):
            raise ValueError("轨迹不是逐帧列表。")
        targets = set(row["target_mouse_ids"])
        diagnostic = row.get("huddle_diagnostics") or {}
        candidate_ids = set(diagnostic.get("best_huddle_target_ids", []))
        frames = []
        seen: dict[int, set[int]] = {target: set() for target in targets}
        for frame in tracks:
            index = frame.get("frame")
            if not isinstance(index, int) or not 0 <= index < count or not isinstance(frame.get("detections"), list):
                raise ValueError("轨迹帧序号与原片段不匹配。")
            detections = []
            for detection in frame["detections"]:
                identity, box = detection.get("track_id"), detection.get("box")
                if (not isinstance(identity, int) or not isinstance(box, list) or len(box) != 4
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
                    raise ValueError("轨迹检测框字段不正确。")
                detections.append(dict(id=identity, box=box, target=identity in targets, candidate=identity in candidate_ids))
                if identity in seen:
                    seen[identity].add(index)
            frames.append(dict(frame=index, time=index / fps, detections=detections))
        observations = [dict(id=identity, observed_frames=len(indices), observed_fraction=len(indices)/count,
                             first_frame=min(indices) if indices else None, last_frame=max(indices) if indices else None)
                        for identity, indices in sorted(seen.items())]
        clip = next(c for c in self.catalogue["clips"] if c["id"] == key)
        return dict(schema="behavior_error_clip_v1", clip=clip,
            reason_detail=row.get("failure_reason_zh", "") if not row["primary_correct"] else "该片段通过了所选专项评测。",
            classification_note=clip["classification_note"], true_target_coverage=row.get("true_target_coverage", 0),
            best_group_ids=sorted(candidate_ids), best_group_coverage=diagnostic.get("coverage_best_any_target"),
            candidate_count=diagnostic.get("huddle_candidate_count"),
            huddle_gate=self.gates.get(row["clip_id"]) if row["dataset"] == "B" else None,
            observations=observations, video=dict(url="/api/video/" + quote(key, safe=""),
                width=width, height=height, fps=fps, frame_count=count), frames=frames,
            timeline=row.get("timeline_series") or {}, target_candidate_scores=row.get("target_candidate_scores") or {})


def create_error_review_server(report_path: Path = DEFAULT_REPORT, *, host: str = "127.0.0.1", port: int = 8776,
                               diagnostics_path: Path | None = DEFAULT_DIAGNOSTICS) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("此审核站仅绑定本机回环地址。")
    store = ReviewStore(report_path, diagnostics_path)

    class Handler(BaseHTTPRequestHandler):
        _stream_file = ComparisonServerHandler._stream_file

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, data: Any) -> None:
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path in {"/", "/index.html", "/app.css", "/app.js"}:
                name = "index.html" if path == "/" else path[1:]
                self._stream_file(ASSETS / name, head_only=self.command == "HEAD")
                return
            try:
                if path == "/api/health":
                    self._send_json(200, dict(status="ok", report_exists=store.report_path.is_file()))
                elif path == "/api/catalogue":
                    self._send_json(200, store.load())
                elif path.startswith("/api/clip/"):
                    self._send_json(200, store.detail(path[len("/api/clip/"):]))
                elif path.startswith("/api/video/"):
                    self._stream_file(store.media_path(path[len("/api/video/"):]), head_only=self.command == "HEAD")
                else:
                    self._send_json(404, {"error": "接口或片段不存在。"})
            except KeyError:
                self._send_json(404, {"error": "未找到这个评测片段。"})
            except (OSError, ValueError, TypeError) as exc:
                self._send_json(503, {"error": f"资料暂不可用：{exc}。请重试或检查报告与视频文件。"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
