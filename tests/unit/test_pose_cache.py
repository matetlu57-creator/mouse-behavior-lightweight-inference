from __future__ import annotations

import gzip
import json
import pickle

import numpy as np

from mouse_behavior.pose_cache import _payloads, build_cache


class _Tensor:
    def __init__(self, value):
        self._value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Result:
    def __init__(self, visual_id: int):
        self.keypoints = type(
            "Keypoints",
            (),
            {
                "xy": _Tensor(np.zeros((1, 7, 2), dtype=np.float32)),
                "conf": _Tensor(np.full((1, 7), 0.9, dtype=np.float32)),
            },
        )()
        self.boxes = type(
            "Boxes",
            (),
            {
                "xyxy": _Tensor(np.asarray([[1, 2, 10, 12]], dtype=np.float32)),
                "conf": _Tensor(np.asarray([0.95], dtype=np.float32)),
                "id": _Tensor(np.asarray([visual_id], dtype=np.int64)),
            },
        )()


class _EmptyResult:
    def __init__(self):
        self.keypoints = type(
            "Keypoints",
            (),
            {
                "xy": _Tensor(np.empty((0, 7, 2), dtype=np.float32)),
                "conf": _Tensor(np.empty((0, 7), dtype=np.float32)),
            },
        )()
        self.boxes = type(
            "Boxes",
            (),
            {
                "xyxy": _Tensor(np.empty((0, 4), dtype=np.float32)),
                "conf": _Tensor(np.empty((0,), dtype=np.float32)),
                "id": None,
            },
        )()


class _UntrackedResult(_Result):
    def __init__(self):
        super().__init__(visual_id=0)
        self.boxes.id = None


class _Model:
    def __init__(self):
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append((np.asarray(frame).copy(), kwargs))
        return [_Result(100 + len(self.calls))]


class _Capture:
    def __init__(self):
        self.frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(2)]
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, prop):
        return 2

    def release(self):
        return None


def test_build_cache_uses_ultralytics_track_and_persists_visual_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("mouse_behavior.pose_cache.cv2.VideoCapture", lambda _path: _Capture())
    model = _Model()

    output = build_cache(
        tmp_path / "video.mp4",
        tmp_path / "cache",
        tmp_path / "best.pt",
        model=model,
        tracker="bytetrack.yaml",
    )

    assert len(model.calls) == 2
    assert all(call[1]["persist"] is True for call in model.calls)
    assert all(call[1]["tracker"] == "bytetrack.yaml" for call in model.calls)
    with gzip.open(output / "yolo_results.000000.000000.pkl.gz", "rb") as handle:
        records = pickle.load(handle)
    assert [row["pose_detections"][0]["visual_track_id"] for row in records] == [101, 102]


def test_empty_ultralytics_track_frame_does_not_require_ids():
    assert _payloads(_EmptyResult()) == []


def test_untracked_detection_without_ultralytics_id_is_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr("mouse_behavior.pose_cache.cv2.VideoCapture", lambda _path: _Capture())

    class ModelWithOneUntrackedFrame(_Model):
        def track(self, frame, **kwargs):
            self.calls.append((np.asarray(frame).copy(), kwargs))
            return [_UntrackedResult() if len(self.calls) == 1 else _Result(102)]

    output = build_cache(
        tmp_path / "video.mp4",
        tmp_path / "cache",
        tmp_path / "best.pt",
        model=ModelWithOneUntrackedFrame(),
    )

    with gzip.open(output / "yolo_results.000000.000000.pkl.gz", "rb") as handle:
        records = pickle.load(handle)
    assert records[0]["pose_detections"] == []
    assert records[1]["pose_detections"][0]["visual_track_id"] == 102
    status = json.loads((output / "yolo_results_status.json").read_text(encoding="utf-8"))
    assert status["untracked_detection_count"] == 1
    assert status["untracked_detection_frame_count"] == 1
