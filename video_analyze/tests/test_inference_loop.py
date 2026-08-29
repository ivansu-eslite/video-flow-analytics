"""推理主迴圈的接線契約測試：哪一份資料交給誰、在哪一步做。

追蹤與落盤搬進獨立進程後（issue #109），這個迴圈只剩「湊批 → 推論 → 送 payload」，
因此本檔的斷言對象是**送進 `track_queue` 的 payload**，而不再是寫出的 parquet。
軌跡與落腳點那一側的接線由 test_track_worker.py 釘住。

偵測改成兩批深度的流水之後（ADR-014），這個迴圈多了一件只有它做得到的事：**記住送出去
但還沒回來的那幾批分別是哪些影格**。錯配的後果與下面第 2 點同型，但更難察覺——每一格的
框都是真的，只是配到隔壁批的時間戳。

五件事都是「主迴圈接錯線，但被接的函式自己的測試全綠、parquet 也完全正常」：

1. **payload 帶的是該格的全部偵測框（含 head）**，拆分留給追蹤進程。這裡只送 fbody
   的話，落腳點推算永遠配不到頭、每列靜默退回框底邊中點（ADR-009 的偏移就這樣回來）。
2. **`frame_index` 與 `timestamp` 逐格對得上**。payload 是 parquet 那兩欄的唯一來源，
   錯開後每一列的時間戳都配到別格，而列數與 `track_id` 全部正常。
3. **歸還 slot 晚於前處理、且早於 forward**。影格是共享記憶體的 view，任何早於
   `preprocess` 回來的歸還都會讓 reader 覆寫正在被讀的畫面；而前處理一回來像素就已經
   複製走了，晚還只是讓 reader 多空等一整段 forward（ADR-010、ADR-013）。
4. **payload 進到路由指定的那一片、結束與失敗訊號送到每一片**。送錯片沒有任何直接
   症狀（那片也有全部路的 tracker），唯一的訊號在追蹤進程入口；訊號漏送則會讓那一片
   一直等在 `get()` 上、缺一支 part 檔。
5. **在途批與 `_pending` 同進退**：序號對不上要當場拋錯，所有路讀完後要把剩下的在途批
   收乾淨（漏收就是 parquet 少幾百格，而檔案完全正常）。

因此本檔的偵測資料一律以**推論尺度**（640×384）表示——那就是縮放前移之後 YOLO 實際
輸出的座標系，也是 payload 內座標的尺度（反算在追蹤進程做）。

主迴圈在多進程 pipeline 裡，但它與子進程之間只隔著佇列與環形緩衝的介面，因此這裡
用 stub 取代佇列、環形緩衝與偵測器直接跑 `start_loop`，不拉起任何子進程。
"""

import datetime
import queue
from collections import deque
from itertools import accumulate
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services.batching import (
    RING_SLOTS,
    TARGET_BATCH,
    TRACK_QUEUE_SLOTS,
)
from video_analyze.services.detector import PIPELINE_DEPTH
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.track_worker import TRACK_DONE, TRACK_FAILED
from video_analyze.services.video_reader import READER_DONE, READER_FAILED

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> np.ndarray:
    """`rows` 為 `(x1, y1, x2, y2, conf, cls)`，與 `YOLODetector.infer` 的每格佈局一致。"""
    return np.array(rows, dtype=np.float32)


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
# 空格：欄數仍是 6，只是一列都沒有
_EMPTY_DETECTIONS = np.zeros((0, 6), dtype=np.float32)


class _ScriptedDetector:
    """依序吐出預先寫好的每格偵測結果，不載入模型；介面是流水的 `submit`／`collect`。

    結果**刻意延後到 `collect` 才取用**（`submit` 只記下這批有幾格），所以「主迴圈把
    第 k 批的框配到第 k+1 批的影格」在這個替身上真的做得出來，不是被替身的同步行為
    蓋掉。

    兩處都記下當下 free queue 的長度與該批格數：影格是共享記憶體的 view，早於 `submit`
    回來的歸還會讓 reader 覆寫正在被讀的畫面（ADR-010），而那件事只在「呼叫的當下」
    看得出來，事後從輸出完全看不出來。

    `submit` 回傳的序號故意不是「第幾次呼叫」以外的東西——主迴圈唯一該做的就是把它跟
    `collect` 回傳的那個比對。
    """

    # 引擎綁的最大批次。正式的 `YOLODetector` 從引擎的 optimization profile 讀，這裡
    # 照正式設定的單次批次給，讓 `start_loop` 開頭的上限檢查在真實條件下跑過
    max_batch = TARGET_BATCH

    def __init__(self, per_frame: list[np.ndarray], free_queue: queue.Queue):
        self._remaining = iter(per_frame)
        self._free_queue = free_queue
        self._queued: deque[tuple[int, int]] = deque()  # (序號, 該批格數)
        self._next_seq = 0
        # 逐批的 (呼叫當下已歸還的 slot 數, 本批格數)
        self.submit_log: list[tuple[int, int]] = []
        self.collect_log: list[tuple[int, int]] = []

    @property
    def in_flight(self) -> int:
        return len(self._queued)

    def submit(self, batch_frames: list[np.ndarray]) -> int:
        assert len(self._queued) < PIPELINE_DEPTH, "主迴圈沒有先 collect 就又 submit"
        self.submit_log.append((self._free_queue.qsize(), len(batch_frames)))
        seq = self._next_seq
        self._next_seq += 1
        self._queued.append((seq, len(batch_frames)))
        return seq

    def collect(self) -> tuple[int, list[np.ndarray]]:
        seq, num_frames = self._queued.popleft()
        self.collect_log.append((self._free_queue.qsize(), num_frames))
        return seq, [next(self._remaining) for _ in range(num_frames)]


class _FrameRingStub:
    """環形緩衝的替身：slot 內容與本測試無關，回傳固定大小的空影格即可。

    `num_slots` 沿用正式執行的推導值，讓 `start_loop` 開頭的格數不變量檢查在測試裡
    也跑在真實條件下。
    """

    num_slots = RING_SLOTS

    def view_slot(self, slot: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _run_loop(
    per_frame: list[np.ndarray], last_signal: str = READER_DONE
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
        track_queues=[track_queue],
        route=[0],
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
        np.testing.assert_allclose(box_data, source)
        assert sorted(int(c) for c in box_data[:, -1]) == sorted(
            int(c) for c in source[:, -1]
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


def test_slots_are_returned_after_submit_and_before_the_result_is_collected():
    """歸還晚於送出、早於取回結果——歸還點前移之後，免複製消費唯一會出錯的地方。

    `submit` 當下已歸還的 slot 數必須恰好等於**先前各批**的格數總和（本批一格都還沒
    還：`np.stack` 還在讀共享記憶體，早還會讓 reader 邊寫邊被讀）；`collect` 當下則必須
    已經含那一批（像素在 `submit` 回來時就複製進 pinned buffer 了，晚還只是讓 reader
    多空等一整段 forward）。餵超過兩批的量，讓跨批的順序與「流水滿了才收」都被涵蓋。
    末尾再驗全數歸還——漏還的話 reader 會卡在 `free_queue.get()`，整條 pipeline 停住。

    改動前這支驗的是「歸還晚於 `predict`」並另外斷言 `result.orig_img is None`；
    `Results` 不在正式路徑上之後，推論輸出不再攜帶任何影格參照，那條斷言失去對象
    （ADR-010 Decision 4 的修訂）。
    """
    num_frames = TARGET_BATCH * PIPELINE_DEPTH + 4  # 確保流水填滿後還有一批
    _dispatched, free_queue, detector = _run_loop([_FRAME_0_DETECTIONS] * num_frames)

    assert len(detector.submit_log) > PIPELINE_DEPTH, "流水沒被填滿，跨批歸還沒被涵蓋"
    batch_sizes = [entry[1] for entry in detector.submit_log]
    assert [entry[0] for entry in detector.submit_log] == list(
        accumulate([0, *batch_sizes[:-1]])
    )
    # `collect` 回收的是最舊那一批，所以當下已歸還的量至少含它自己那一批
    collected_through = list(accumulate(batch_sizes))
    for index, (returned_count, _size) in enumerate(detector.collect_log):
        assert returned_count >= collected_through[index]

    returned = []
    while not free_queue.empty():
        returned.append(free_queue.get_nowait())

    # `_run_loop` 每格用一個 slot（slot 索引即 frame_index）
    assert sorted(returned) == list(range(num_frames))


def test_every_frame_gets_its_own_payload_across_batches_and_the_final_drain():
    """跨批與收尾 drain 都不可掉格或錯位：逐格的 `frame_index` 要完整且遞增。

    流水化之後最後那幾批是在所有路都讀完之後才收的（主迴圈的 drain 段）。漏掉 drain
    的症狀是 parquet 少了最後幾百格——列數少一點，而檔案的欄位、時間範圍全部正常。
    """
    num_frames = TARGET_BATCH * PIPELINE_DEPTH + 4
    dispatched, _free_queue, _detector = _run_loop([_FRAME_0_DETECTIONS] * num_frames)

    payloads = [item for item in dispatched if item != TRACK_DONE]
    assert [p[2] for p in payloads] == list(range(num_frames))
    assert [p[3] for p in payloads] == [
        _BASE + datetime.timedelta(seconds=index) for index in range(num_frames)
    ]


class _SeqShufflingDetector(_ScriptedDetector):
    """回傳的序號比實際的多 1，模擬兩邊的在途佇列錯開一批。"""

    def collect(self) -> tuple[int, list[np.ndarray]]:
        seq, boxes = super().collect()
        return seq + 1, boxes


def test_a_batch_whose_sequence_number_does_not_match_aborts_the_loop():
    """序號對不上要當場拋錯，不能照樣把框送出去。

    錯配的輸出檔**完全正常**：每一格的框都是真的，只是配到隔壁批的 `frame_index` 與
    時間戳。逐值比對也只表現為配對率下降，看不出是錯配還是偵測變了。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _SeqShufflingDetector([_FRAME_0_DETECTIONS] * 2, free_queue)
    track_queue: queue.Queue = queue.Queue()
    pipeline = InferencePipeline(
        stream_names=["loc_cam001"],
        detector=detector,
        track_queues=[track_queue],
        route=[0],
    )
    data_queue: queue.Queue = queue.Queue()
    for frame_index in range(2):
        data_queue.put((frame_index, frame_index, _BASE))
    data_queue.put(READER_DONE)

    with pytest.raises(RuntimeError, match="批"):
        pipeline.start_loop([data_queue], [free_queue], [_FrameRingStub()])

    assert _drain(track_queue) == [TRACK_FAILED]


class _StateDrivenQueue:
    """先送完 `items`，之後依偵測器的在途狀態決定給 Empty 還是收尾訊號。

    用「在途批數」而不是「第幾次呼叫」當條件，是因為湊批的等待（`_FILL_MAX_WAIT`）是
    計時的——用呼叫次數寫死會在慢一點的機器上換一條路徑跑，而那正是這兩支要區分的
    兩條路徑。

    等待仍設上限：等的那件事沒發生時要收尾讓斷言失敗，不是把測試掛住。
    """

    _MAX_EMPTY = 200

    def __init__(
        self,
        items: list,
        detector,
        final_signal: str,
        *,
        wait_for_collect: bool = False,
        probe=None,
    ):
        self._items = list(items)
        self._detector = detector
        self._final_signal = final_signal
        self._wait_for_collect = wait_for_collect
        self._probe = probe
        self._empties = 0
        self.probed = None

    def _empty(self):
        self._empties += 1
        raise queue.Empty

    def get_nowait(self):
        if self._items:
            return self._items.pop(0)
        if self._empties < self._MAX_EMPTY:
            if not self._detector.submit_log:
                # 還沒送出任何一批：讓湊批的等待走完，把手上那一格送進流水
                self._empty()
            if self._wait_for_collect and self._detector.in_flight:
                # 沒有新影格、但有一批在 GPU 上：主迴圈應該趁這時把它收回來
                self._empty()
        if self._probe is not None:
            self.probed = self._probe()
        return self._final_signal


def test_an_in_flight_batch_is_collected_while_no_new_frames_arrive():
    """沒有新影格時就把在途批收回來，不必等湊出下一批。

    等下一批才收的話，供料斷續時 payload 會一路壓在偵測器裡——輸出檔完全正常，只是
    追蹤進程整段時間沒事做，而端到端的吞吐看起來就只是「比較慢」。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector([_FRAME_0_DETECTIONS], free_queue)
    track_queue: queue.Queue = queue.Queue()
    pipeline = InferencePipeline(
        stream_names=["loc_cam001"],
        detector=detector,
        track_queues=[track_queue],
        route=[0],
    )
    data_queue = _StateDrivenQueue(
        [(0, 0, _BASE)],
        detector,
        READER_DONE,
        wait_for_collect=True,
        probe=track_queue.qsize,
    )

    pipeline.start_loop([data_queue], [free_queue], [_FrameRingStub()])

    # 該路讀完的當下，那一格的 payload 已經送出去了——不是留到迴圈結束的 drain 才送
    assert data_queue.probed == 1


def test_track_failed_reaches_every_shard_when_a_batch_is_still_in_flight():
    """上游在有批次在途時崩潰，失敗訊號仍要送到每一片。

    在途批的結果就此丟掉是對的（結果不完整，寧可不寫），但訊號不能跟著丟——收不到的
    那片會等到父進程 SIGTERM，把「上游崩潰」與「被 terminate」混成同一種結束方式。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector([_FRAME_0_DETECTIONS], free_queue)
    track_queues: list[queue.Queue] = [queue.Queue(), queue.Queue()]
    pipeline = InferencePipeline(
        stream_names=["loc_cam001", "loc_cam002"],
        detector=detector,
        track_queues=track_queues,
        route=[0, 1],
    )
    live_queue: queue.Queue = queue.Queue()
    live_queue.put(READER_DONE)
    failing_queue = _StateDrivenQueue([(0, 0, _BASE)], detector, READER_FAILED)

    with pytest.raises(RuntimeError, match="中途例外"):
        pipeline.start_loop(
            [failing_queue, live_queue],
            [free_queue, free_queue],
            [_FrameRingStub(), _FrameRingStub()],
        )

    assert [_drain(q)[-1] for q in track_queues] == [TRACK_FAILED, TRACK_FAILED]


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
        track_queues=[track_queue],
        route=[0],
    )

    with pytest.raises(ValueError, match="最大批次"):
        pipeline.start_loop([queue.Queue()], [free_queue], [_FrameRingStub()])

    assert track_queue.get_nowait() == TRACK_FAILED


def _run_two_stream_loop(
    route: list[int], num_shards: int, last_signal: str = READER_DONE
) -> list[queue.Queue]:
    """兩路各送一格，跑完回傳各片 queue 收到的東西。

    偵測內容與這幾支無關（要驗的是「哪一格進了哪一條 queue」），故兩格都給同一份。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector([_FRAME_0_DETECTIONS] * 2, free_queue)
    track_queues: list[queue.Queue] = [
        queue.Queue(maxsize=TRACK_QUEUE_SLOTS) for _ in range(num_shards)
    ]
    pipeline = InferencePipeline(
        stream_names=["loc_cam001", "loc_cam002"],
        detector=detector,
        track_queues=track_queues,
        route=route,
    )
    data_queues: list[queue.Queue] = [queue.Queue(), queue.Queue()]
    for stream_id, data_queue in enumerate(data_queues):
        data_queue.put((stream_id, 0, _BASE))
        data_queue.put(last_signal)

    rings = [_FrameRingStub(), _FrameRingStub()]
    if last_signal == READER_FAILED:
        with pytest.raises(RuntimeError):
            pipeline.start_loop(data_queues, [free_queue, free_queue], rings)
    else:
        pipeline.start_loop(data_queues, [free_queue, free_queue], rings)
    return track_queues


def _drain(track_queue: queue.Queue) -> list:
    items = []
    while not track_queue.empty():
        items.append(track_queue.get_nowait())
    return items


def test_each_payload_goes_to_the_shard_that_owns_its_stream():
    """payload 只進它歸屬的那一條 queue。

    送錯片不會有任何直接症狀：那片也有全部路的 tracker，會照樣追蹤、照樣寫進自己的
    part，只是該路的軌跡被切成兩段而 `track_id` 分裂，合併後的檔案完全正常。唯一的
    訊號在追蹤進程入口的歸屬檢查，而那要等到跑起來才會炸。
    """
    shard0, shard1 = _run_two_stream_loop(route=[1, 0], num_shards=2)

    assert [item[0] for item in _drain(shard0) if item != TRACK_DONE] == [1]
    assert [item[0] for item in _drain(shard1) if item != TRACK_DONE] == [0]


def test_track_done_reaches_every_shard():
    """結束訊號要送到每一片：漏掉哪一片，那一片不會 `save()`，合併就缺一支 part。"""
    track_queues = _run_two_stream_loop(route=[0, 1], num_shards=2)

    assert [_drain(q)[-1] for q in track_queues] == [TRACK_DONE, TRACK_DONE]


def test_track_failed_reaches_every_shard():
    """失敗訊號同理：漏掉的那片會一直等在 `get()` 上，直到父進程 terminate。"""
    track_queues = _run_two_stream_loop(
        route=[0, 1], num_shards=2, last_signal=READER_FAILED
    )

    assert [_drain(q)[-1] for q in track_queues] == [TRACK_FAILED, TRACK_FAILED]


def test_track_failed_still_reaches_the_later_shards_when_one_queue_is_full():
    """前一片死了、它的 queue 又滿了的時候，後面的片仍要收得到 `TRACK_FAILED`。

    fan-out 是序列的 `put`：不帶 timeout 的話會永久阻塞在那條滿的 queue 上，**後面的
    片一個都收不到**，只能等父進程的 SIGTERM——那條路徑清得掉 part 檔，但會把「上游
    崩潰」與「被 terminate」混成同一種結束方式，正是這個 in-band 訊號要分開的東西。
    """
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector([], free_queue)
    dead_shard: queue.Queue = queue.Queue(maxsize=1)
    dead_shard.put("塞住這條 queue 的舊 payload")  # 該片已經死了，沒有人在消化
    live_shard: queue.Queue = queue.Queue(maxsize=TRACK_QUEUE_SLOTS)
    pipeline = InferencePipeline(
        stream_names=["loc_cam001", "loc_cam002"],
        detector=detector,
        track_queues=[dead_shard, live_shard],
        route=[0, 1],
    )
    data_queues: list[queue.Queue] = [queue.Queue(), queue.Queue()]
    data_queues[0].put(READER_FAILED)
    data_queues[1].put(READER_DONE)

    with pytest.raises(RuntimeError):
        pipeline.start_loop(
            data_queues, [free_queue, free_queue], [_FrameRingStub(), _FrameRingStub()]
        )

    assert _drain(live_shard) == [TRACK_FAILED]


def test_a_route_that_does_not_cover_every_stream_is_rejected():
    """路由長度與路數對不上要在組裝時擋下，不能等跑起來才 `IndexError`。"""
    with pytest.raises(ValueError, match="與路數"):
        InferencePipeline(
            stream_names=["loc_cam001", "loc_cam002"],
            detector=None,
            track_queues=[queue.Queue()],
            route=[0],
        )


def test_a_route_pointing_at_a_missing_shard_is_rejected():
    """指到不存在的片同理——那會在每格都走到的路徑上炸。"""
    with pytest.raises(ValueError, match="不存在的片"):
        InferencePipeline(
            stream_names=["loc_cam001"],
            detector=None,
            track_queues=[queue.Queue()],
            route=[1],
        )
