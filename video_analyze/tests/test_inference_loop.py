"""推理主迴圈的接線契約測試：哪一份資料交給誰、在哪一步做。

追蹤與落盤搬進獨立進程後（issue #109），這個迴圈只剩「湊批 → 推論 → 送 payload」，
因此本檔的斷言對象是**送進 `track_queue` 的 payload**，而不再是寫出的 parquet。
軌跡與落腳點那一側的接線由 test_track_worker.py 釘住。

三件事都是「主迴圈接錯線，但被接的函式自己的測試全綠、parquet 也完全正常」：

1. **payload 帶的是該格的全部偵測框（含 head）**，拆分留給追蹤進程。這裡只送 fbody
   的話，落腳點推算永遠配不到頭、每列靜默退回框底邊中點（ADR-009 的偏移就這樣回來）。
2. **`frame_index` 與 `timestamp` 逐格對得上**。payload 是 parquet 那兩欄的唯一來源，
   錯開後每一列的時間戳都配到別格，而列數與 `track_id` 全部正常。
3. **歸還 slot 晚於推論、且早於送 payload**。影格是共享記憶體的 view，任何早於推論
   完成的歸還都會讓 reader 覆寫正在推論的畫面（ADR-010）。

因此本檔的偵測資料一律以**推論尺度**（640×384）表示——那就是縮放前移之後 YOLO 實際
輸出的座標系，也是 payload 內座標的尺度（反算在追蹤進程做）。

主迴圈在多進程 pipeline 裡，但它與子進程之間只隔著佇列與環形緩衝的介面，因此這裡
用 stub 取代佇列、環形緩衝與偵測器直接跑 `start_loop`，不拉起任何子進程。
"""

import datetime
import queue
from itertools import accumulate
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes, Results

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services.batching import RING_SLOTS, TARGET_BATCH
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.track_worker import TRACK_DONE, TRACK_FAILED
from video_analyze.services.video_reader import READER_DONE, READER_FAILED

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
# 偵測結果的座標系：影格在讀取端就縮好了，ultralytics 的 `orig_shape` 也是這個尺寸
_INFER_SHAPE = (INFER_HEIGHT, INFER_WIDTH)
_CLASS_NAMES = {HEAD_CLASS_ID: "head", FBODY_CLASS_ID: "fbody"}


def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> Boxes:
    """`rows` 為 `(x1, y1, x2, y2, conf, cls)`，與 ultralytics 的 data 佈局一致。"""
    return Boxes(torch.tensor(rows, dtype=torch.float32), _INFER_SHAPE)


# 兩格偵測：各兩個 fbody，配上罩得住的 head，第二格另有一顆配不到人的孤兒 head。
# 座標都在 640×384 之內：縮放前移之後，YOLO 收到的與吐出的都是推論尺度。
_FRAME_0_DETECTIONS = _boxes(
    [
        (30.0, 45.0, 44.0, 62.0, 0.9, HEAD_CLASS_ID),
        (27.0, 42.0, 67.0, 145.0, 0.8, FBODY_CLASS_ID),
        (205.0, 40.0, 220.0, 58.0, 0.7, HEAD_CLASS_ID),
        (200.0, 36.0, 233.0, 166.0, 0.6, FBODY_CLASS_ID),
    ]
)
_FRAME_1_DETECTIONS = _boxes(
    [
        (31.0, 46.0, 45.0, 63.0, 0.9, HEAD_CLASS_ID),
        (28.0, 43.0, 68.0, 146.0, 0.8, FBODY_CLASS_ID),
        (203.0, 36.0, 234.0, 166.0, 0.6, FBODY_CLASS_ID),
        (500.0, 300.0, 514.0, 317.0, 0.5, HEAD_CLASS_ID),
    ]
)
# 空格：`torch.tensor([])` 是 1D、`Boxes` 會判成 0 欄，要明確給 (0, 6)
_EMPTY_DETECTIONS = Boxes(torch.zeros((0, 6), dtype=torch.float32), _INFER_SHAPE)


class _ScriptedDetector:
    """依序吐出預先寫好的每格偵測結果，不載入模型。

    回傳**真正的 `Results`**（而非只帶 `boxes` 的替身）：主迴圈在歸還 slot 前會把
    `result.orig_img` 清成 `None`，那道保護只有對真實物件才驗得到——替身是普通
    dataclass，`orig_img` 賦值必定成功，日後 ultralytics 把該欄位改成唯讀或加上
    `__slots__` 時，production 每批都拋錯而測試全綠。

    另記下每次 predict 當下 free queue 的長度與該批格數：影格是共享記憶體的 view，
    任何早於推論完成的歸還都會讓 reader 覆寫正在推論的畫面（見 ADR-010），而那件事
    只在「呼叫的當下」看得出來，事後從輸出完全看不出來。
    """

    # 引擎綁的最大批次。正式的 `YOLODetector` 從引擎 metadata 讀，這裡照正式設定的
    # 單次批次給，讓 `start_loop` 開頭的上限檢查在真實條件下跑過
    max_batch = TARGET_BATCH

    def __init__(self, per_frame: list[Boxes], free_queue: queue.Queue):
        self._remaining = iter(per_frame)
        self._free_queue = free_queue
        # 逐次 predict 的 (呼叫當下已歸還的 slot 數, 本批格數)
        self.predict_log: list[tuple[int, int]] = []
        # 交出去的 Results，供呼叫端檢查歸還前是否已切斷對共享記憶體的參照
        self.returned_results: list[Results] = []

    def predict(self, batch_frames: list[np.ndarray]) -> list[Results]:
        self.predict_log.append((self._free_queue.qsize(), len(batch_frames)))
        results = [
            Results(
                orig_img=frame,
                path="scripted.mkv",
                names=_CLASS_NAMES,
                boxes=next(self._remaining).data,
            )
            for frame in batch_frames
        ]
        self.returned_results.extend(results)
        return results


class _FrameRingStub:
    """環形緩衝的替身：slot 內容與本測試無關，回傳固定大小的空影格即可。

    `num_slots` 沿用正式執行的推導值，讓 `start_loop` 開頭的格數不變量檢查在測試裡
    也跑在真實條件下。
    """

    num_slots = RING_SLOTS

    def view_slot(self, slot: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _run_loop(
    per_frame: list[Boxes], last_signal: str = READER_DONE
) -> tuple[list, queue.Queue, _ScriptedDetector]:
    """跑一次單路的推理主迴圈。

    回傳送進 `track_queue` 的全部元素、歸還時序要看的 free queue，以及偵測器。
    `last_signal` 換成 `READER_FAILED` 就能驗上游崩潰時的失效路徑。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector(per_frame, free_queue)
    track_queue: queue.Queue = queue.Queue()
    pipeline = InferencePipeline(
        stream_names=["loc_cam001"],
        detector=detector,
        track_queue=track_queue,
    )

    data_queue: queue.Queue = queue.Queue()
    for frame_index in range(len(per_frame)):
        data_queue.put(
            (
                frame_index,  # slot
                frame_index,
                _BASE + datetime.timedelta(seconds=frame_index),
            )
        )
    data_queue.put(last_signal)

    if last_signal == READER_FAILED:
        # 只有這條路徑預期拋錯，訊號本身由呼叫端斷言。無條件吞掉 RuntimeError 會讓其餘
        # 幾支測試在主迴圈真的壞掉時仍然全綠
        with pytest.raises(RuntimeError):
            pipeline.start_loop([data_queue], [free_queue], [_FrameRingStub()])
    else:
        pipeline.start_loop([data_queue], [free_queue], [_FrameRingStub()])
    dispatched = []
    while not track_queue.empty():
        dispatched.append(track_queue.get_nowait())
    return dispatched, free_queue, detector


def test_payload_carries_every_detected_class_including_heads():
    """送出的是該格**全部**類別的框：head 留在裡面，拆分是追蹤進程的事。

    只送 fbody 的話 `FootPointEstimator` 永遠配不到頭，每列靜默退回框底邊中點——
    列數、`track_id`、bbox 全部正常，只有落腳點偏掉（ADR-009 要修掉的偏移）。
    """
    dispatched, _free_queue, _detector = _run_loop(
        [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    payloads = [item for item in dispatched if item != TRACK_DONE]
    assert len(payloads) == 2
    for payload, source in zip(
        payloads, [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS], strict=True
    ):
        _stream_id, box_data, _frame_index, _timestamp = payload
        np.testing.assert_allclose(box_data, source.data.numpy())
        assert sorted(int(c) for c in box_data[:, -1]) == sorted(
            int(c) for c in source.cls.tolist()
        )


def test_payload_keeps_frame_index_and_timestamp_aligned_per_frame():
    """payload 是 parquet 的 `frame_id`／`timestamp` 唯一來源，逐格不可錯開。"""
    dispatched, _free_queue, _detector = _run_loop(
        [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    payloads = [item for item in dispatched if item != TRACK_DONE]
    assert [(p[0], p[2], p[3]) for p in payloads] == [
        (0, 0, _BASE),
        (0, 1, _BASE + datetime.timedelta(seconds=1)),
    ]


def test_empty_detections_are_still_dispatched():
    """整格沒有偵測也要送 payload（`box_data` 為 None），不可整格跳過。

    追蹤進程靠每格一個 payload 推進 `BYTETracker` 的 frame_id 與軌跡老化；空格被
    吞掉的話已離開畫面的人不會在 `track_buffer` 到期時被移除，而輸出檔完全正常。
    """
    dispatched, _free_queue, _detector = _run_loop([_EMPTY_DETECTIONS])

    payloads = [item for item in dispatched if item != TRACK_DONE]
    assert len(payloads) == 1
    assert payloads[0][1] is None


def test_track_done_is_the_last_thing_sent_on_the_normal_path():
    """正常跑完要送 `TRACK_DONE`：那是追蹤進程唯一的正常結束途徑，不送就會卡住。"""
    dispatched, _free_queue, _detector = _run_loop([_FRAME_0_DETECTIONS])

    assert dispatched[-1] == TRACK_DONE
    assert TRACK_FAILED not in dispatched


def test_track_failed_is_sent_when_the_loop_aborts():
    """上游崩潰時送 `TRACK_FAILED` 而非 `TRACK_DONE`。

    收不到任何訊號的話追蹤進程會等到被父進程 terminate，而 terminate 不走 Python 的
    `except`／`finally`，`.tmp` 暫存檔會被留在輸出目錄；送成 `TRACK_DONE` 更糟——
    截斷的結果會被 rename 成正式檔名。
    """
    dispatched, _free_queue, _detector = _run_loop(
        [_FRAME_0_DETECTIONS], last_signal=READER_FAILED
    )

    assert dispatched[-1] == TRACK_FAILED
    assert TRACK_DONE not in dispatched


def test_slots_are_returned_after_inference_and_before_dispatch():
    """歸還晚於推論、且整批推論完就歸還——免複製消費唯一會出錯的地方。

    每次 predict 當下已歸還的 slot 數，必須恰好等於**先前各批**的格數總和：多一格
    就代表本批的 slot 在推論中被放行，reader 可以覆寫正在推論的畫面。餵超過一批的
    量，讓跨批的歸還順序也被涵蓋（只驗單批的話，第二批之後歸還早一步也看不出來）。
    末尾再驗全數歸還——漏還的話 reader 會卡在 `free_queue.get()`，整條 pipeline 停住。
    """
    num_frames = TARGET_BATCH + 4  # > 一批，確保跑到第二批
    _dispatched, free_queue, detector = _run_loop([_FRAME_0_DETECTIONS] * num_frames)

    assert len(detector.predict_log) >= 2, "沒跑到第二批，跨批歸還沒被涵蓋"
    returned_before = [entry[0] for entry in detector.predict_log]
    batch_sizes = [entry[1] for entry in detector.predict_log]
    assert returned_before == list(accumulate([0, *batch_sizes[:-1]]))

    returned = []
    while not free_queue.empty():
        returned.append(free_queue.get_nowait())

    # `_run_loop` 每格用一個 slot（slot 索引即 frame_index）
    assert sorted(returned) == list(range(num_frames))
    # 歸還前先切斷對共享記憶體的參照：`orig_img` 是該 slot 的活別名（ADR-010）
    assert all(result.orig_img is None for result in detector.returned_results)


def test_a_batch_larger_than_the_engine_ceiling_aborts_with_track_failed():
    """實際批次超過引擎綁的最大批次要**在跑之前**擋下，並送得出 `TRACK_FAILED`。

    引擎的 batch 維度上限是建置時綁死的。超過的話 ultralytics 的 TensorRT backend 會在
    `forward` 的 assert 失敗，訊息只講「input size 不等於 model size」——看不出是
    `[model].batch` 與引擎對不上，而這兩者是分別維護的（一個在 `config.toml`，一個在
    建引擎時的 `--batch`）。

    檢查放在 `start_loop` 的 try 內而非之前：追蹤進程此時已經等在 queue 上，
    走 `TRACK_FAILED` 才不必等父進程 SIGTERM（見 `services/pipeline.py`）。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector([_FRAME_0_DETECTIONS], free_queue)
    detector.max_batch = TARGET_BATCH - 1  # 剛好差一格
    track_queue: queue.Queue = queue.Queue()
    pipeline = InferencePipeline(
        stream_names=["loc_cam001"],
        detector=detector,
        track_queue=track_queue,
    )

    with pytest.raises(ValueError, match="最大批次"):
        pipeline.start_loop([queue.Queue()], [free_queue], [_FrameRingStub()])

    assert track_queue.get_nowait() == TRACK_FAILED
