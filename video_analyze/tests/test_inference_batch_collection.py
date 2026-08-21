"""`_collect_batch` 的取用順序測試：釘住九路輪替，不再退化成依序處理。

`for stream_id in range(self.num_streams)` 固定從 0 起跑，只要 stream 0 供得上，內層
`while` 會取到滿批才換手，其餘八路永遠輪不到（issue #100）。這裡直接呼叫
`_collect_batch`，不拉起子進程，沿用 `test_inference_loop.py` 的 stub 手法
（`queue.Queue` 取代 `mp.Queue`）。
"""

import datetime
import queue as pyqueue
from zoneinfo import ZoneInfo

import numpy as np

from video_analyze.services.inference import InferencePipeline

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 8, 1, 11, 0, tzinfo=_TAIPEI)


class _RingStub:
    """環形緩衝的替身：slot 內容與這裡的測試無關，回傳固定大小的空影格即可。"""

    def view_slot(self, slot: int) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _make_pipeline(num_streams: int) -> InferencePipeline:
    names = [f"cam{i:03d}" for i in range(num_streams)]
    return InferencePipeline(
        stream_names=names,
        detector=None,
        track_queue=pyqueue.Queue(),
    )


def _fill(q: pyqueue.Queue, n: int) -> None:
    """塞 n 格假資料，`frame_index` 從 0 起連續遞增（供釘住不亂序用）。"""
    for i in range(n):
        q.put((i % 16, i, _BASE))


def test_batches_rotate_across_streams_when_all_have_data():
    """九路都有資料時，連續九批的來源涵蓋九路——改回原本固定起點會失敗。"""
    num_streams = 9
    pipeline = _make_pipeline(num_streams)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    for q in data_queues:
        _fill(q, 200)

    sources_per_batch = []
    for _ in range(num_streams):
        packets, stream_ids, _held = pipeline._collect_batch(data_queues, rings)
        assert len(packets) == pipeline._target_batch
        sources_per_batch.append(set(stream_ids))

    # 每路都預先塞滿，起點那路一次就取到滿批，因此每批只有單一來源——這是本 fixture
    # 的產物，不是實作保證（供不上時仍會混來源，見
    # `test_batch_mixes_sources_when_the_starting_stream_runs_short`）
    assert all(len(sources) == 1 for sources in sources_per_batch)
    assert set.union(*sources_per_batch) == set(range(num_streams))


def test_batch_mixes_sources_when_the_starting_stream_runs_short():
    """起點那路供不上滿批時，同一批會由下一路補齊——混來源批次仍可能發生。

    正式執行時各路解碼速度不同（4K 那路補滿一批要數百 ms），輪替過後回到同一路時
    queue 未必已補滿。混來源批次的影格尺寸可能不同，ultralytics 會走
    `same_shapes=False` 的 letterbox 分支，前處理隨批次組成變動——這是 CLAUDE.md 記載
    的「`tracking_results.parquet` 不可重現」來源之一，本次改動沒有消除它。
    """
    num_streams = 3
    pipeline = _make_pipeline(num_streams)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    short = pipeline._target_batch // 4
    _fill(data_queues[0], short)  # 起點那路只剩零星幾格
    for q in data_queues[1:]:
        _fill(q, 200)

    packets, stream_ids, _held = pipeline._collect_batch(data_queues, rings)

    assert len(packets) == pipeline._target_batch
    assert stream_ids[:short] == [0] * short
    assert set(stream_ids) == {0, 1}


def test_full_batch_still_fills_from_a_single_remaining_stream():
    """只有一路還有資料（其餘已讀完）時，輪替不會讓批次縮水。"""
    num_streams = 9
    pipeline = _make_pipeline(num_streams)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    pipeline.finished_streams = set(range(num_streams)) - {5}
    _fill(data_queues[5], 200)

    for _ in range(3):
        packets, stream_ids, _held = pipeline._collect_batch(data_queues, rings)
        assert len(packets) == pipeline._target_batch
        assert set(stream_ids) == {5}


def test_frame_order_within_a_stream_is_preserved_across_rotated_batches():
    """單路內的影格順序不變：`track_id` 延續與 `timestamp` 推導都依賴這件事。"""
    num_streams = 3
    pipeline = _make_pipeline(num_streams)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    for q in data_queues:
        _fill(q, 200)

    indices_by_stream: dict[int, list[int]] = {i: [] for i in range(num_streams)}
    for _ in range(6):
        packets, stream_ids, _held = pipeline._collect_batch(data_queues, rings)
        for packet, stream_id in zip(packets, stream_ids, strict=True):
            indices_by_stream[stream_id].append(packet.frame_index)

    for indices in indices_by_stream.values():
        # 連續且不重複：只驗非遞減的話，重送（`[0, 0, 1]`）與漏格（`[0, 1, 5]`）都會過，
        # 而這兩者正是會打壞 `track_id` 延續與 `timestamp` 推導的情況
        assert indices == list(range(len(indices)))


def test_collect_batch_reports_every_slot_it_takes():
    """取用的 slot 一格不漏地交給呼叫端，由呼叫端在推論後歸還。

    影格是共享記憶體的 view，收集期間歸還等於允許 reader 在推論進行中覆寫仍在用的
    畫面（覆寫進去的是同一路幾格之後的正常畫面，偵測數不會崩、輸出也完全正常，
    只有座標靜默偏移）。「不歸還」本身已由簽章保證——`_collect_batch` 拿不到
    `free_queues`；這裡驗的是另一半：漏報一格，該 slot 就永遠回不到 reader 手上。
    """
    num_streams = 3
    pipeline = _make_pipeline(num_streams)
    data_queues = [pyqueue.Queue() for _ in range(num_streams)]
    rings = [_RingStub() for _ in range(num_streams)]
    for q in data_queues:
        _fill(q, 200)

    packets, stream_ids, held = pipeline._collect_batch(data_queues, rings)

    assert len(packets) == pipeline._target_batch
    # `_fill` 塞的 slot 是 `frame_index % 16`，故 held_slots 該逐項對上本批的來源與格
    assert held == [
        (stream_id, packet.frame_index % 16)
        for stream_id, packet in zip(stream_ids, packets, strict=True)
    ]
