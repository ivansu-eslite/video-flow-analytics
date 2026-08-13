"""`_collect_batch` 的取用順序測試：釘住九路輪替，不再退化成依序處理。

`for stream_id in range(self.num_streams)` 固定從 0 起跑，只要 stream 0 供得上，內層
`while` 會取到滿批才換手，其餘八路永遠輪不到（issue #100）。這裡直接呼叫
`_collect_batch`，不拉起子進程，沿用 `test_inference_loop.py` 的 stub 手法
（`queue.Queue` 取代 `mp.Queue`）。
"""

import datetime
import queue as pyqueue
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from video_analyze.services.inference import InferencePipeline
from video_analyze.services.video_reader import FrameShape

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 8, 1, 11, 0, tzinfo=_TAIPEI)
_SEGMENT = "cam/2026/08/01/030000.000Z.mkv"


class _RingStub:
    """環形緩衝的替身：slot 內容與這裡的測試無關，回傳固定大小的空影格即可。"""

    def read_slot(self, slot: int) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _make_pipeline(num_streams: int, tmp_path: Path) -> InferencePipeline:
    names = [f"cam{i:03d}" for i in range(num_streams)]
    return InferencePipeline(
        stream_names=names,
        detector=None,
        tracker=None,
        output_root=tmp_path / "outputs",
        results_path=tmp_path / "outputs" / "tracking_results.parquet",
        frame_shapes=[FrameShape(height=4, width=4)] * num_streams,
    )


def _fill(q: pyqueue.Queue, n: int) -> None:
    """塞 n 格假資料，`frame_index` 從 0 起連續遞增（供釘住不亂序用）。"""
    for i in range(n):
        q.put((i % 16, _SEGMENT, i, _BASE, 15.0))


def test_batches_rotate_across_streams_when_all_have_data(tmp_path):
    """九路都有資料時，連續九批的來源涵蓋九路——改回原本固定起點會失敗。"""
    num_streams = 9
    pipeline = _make_pipeline(num_streams, tmp_path)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    free_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    for q in data_queues:
        _fill(q, 200)

    sources_per_batch = []
    for _ in range(num_streams):
        packets, stream_ids, _finished = pipeline._collect_batch(
            data_queues, free_queues, rings
        )
        assert len(packets) == pipeline._target_batch
        sources_per_batch.append(set(stream_ids))

    # 每批仍只來自單一路：不引入混解析度批次
    assert all(len(sources) == 1 for sources in sources_per_batch)
    assert set.union(*sources_per_batch) == set(range(num_streams))


def test_full_batch_still_fills_from_a_single_remaining_stream(tmp_path):
    """只有一路還有資料（其餘已讀完）時，輪替不會讓批次縮水。"""
    num_streams = 9
    pipeline = _make_pipeline(num_streams, tmp_path)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    free_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    pipeline.finished_streams = set(range(num_streams)) - {5}
    _fill(data_queues[5], 200)

    for _ in range(3):
        packets, stream_ids, _finished = pipeline._collect_batch(
            data_queues, free_queues, rings
        )
        assert len(packets) == pipeline._target_batch
        assert set(stream_ids) == {5}


def test_frame_order_within_a_stream_is_preserved_across_rotated_batches(tmp_path):
    """單路內的影格順序不變：`track_id` 延續與 `timestamp` 推導都依賴這件事。"""
    num_streams = 3
    pipeline = _make_pipeline(num_streams, tmp_path)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    free_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    for q in data_queues:
        _fill(q, 200)

    indices_by_stream: dict[int, list[int]] = {i: [] for i in range(num_streams)}
    for _ in range(6):
        packets, stream_ids, _finished = pipeline._collect_batch(
            data_queues, free_queues, rings
        )
        for packet, stream_id in zip(packets, stream_ids, strict=True):
            indices_by_stream[stream_id].append(packet.frame_index)

    for indices in indices_by_stream.values():
        assert indices == sorted(indices)
