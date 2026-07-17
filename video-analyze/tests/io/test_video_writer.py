import queue
from pathlib import Path

import numpy as np

from video_analyze.io.video_writer import MultiStreamVideoWriter

_FRAME = np.zeros((2, 2, 3), dtype=np.uint8)


def test_stream_worker_closes_each_segment_only_once_when_second_open_fails(tmp_path):
    # 第二支片段 _open_writer 失敗時，第一支片段的 writer 只應被 _close 一次；
    # 修正前 current 未歸零，收尾時會對同一片段重複呼叫 _close。
    writer = MultiStreamVideoWriter(output_root=Path(tmp_path))
    close_calls = []
    writer._close = lambda segment: close_calls.append(segment.relpath)

    open_calls = {"count": 0}

    class _FakeWriter:
        def write(self, frame):
            pass

    def fake_open_writer(segment_relpath, frame, fps):
        open_calls["count"] += 1
        if open_calls["count"] == 1:
            return _FakeWriter()
        raise ValueError("模擬第二支片段開檔失敗")

    writer._open_writer = fake_open_writer

    q: queue.Queue = queue.Queue()
    q.put(("seg1.mkv", _FRAME, 30.0))
    q.put(("seg2.mkv", _FRAME, 30.0))
    q.put(None)  # 收尾訊號

    writer._stream_worker(q)

    assert close_calls == ["seg1.mkv"]
