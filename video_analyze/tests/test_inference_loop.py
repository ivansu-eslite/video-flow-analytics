"""推理主迴圈的接線契約測試：交給 tracker 的只有 fbody。

`split_detections` 拆得對不對由 test_inference_split.py 釘住，這裡釘的是另一件事
——主迴圈實際把拆出來的**哪一份**交給 `tracker.update`。把引數改回未拆分的偵測
結果（`results[idx].boxes.cpu()`），拆分函式的測試依然全綠，但同一個人會多出一條
頭部軌跡，`track_id` 的語義從「一個人」變成「一個偵測目標」，下游的不重複訪客與
進出人數直接翻倍，而 `tracking_results.parquet` 本身完全正常。

主迴圈在多進程 pipeline 裡，但它與子進程之間只隔著佇列與環形緩衝的介面，因此這裡
用 stub 取代佇列、環形緩衝、偵測器與追蹤器直接跑 `start_loop`，不拉起任何子進程。
"""

import datetime
import queue
from itertools import accumulate
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import torch
from ultralytics.engine.results import Boxes, Results

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services.frame_ring import RING_SLOTS
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.video_reader import READER_DONE, FrameShape

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
_ORIG_SHAPE = (1080, 1920)
_SHAPE = FrameShape(height=1080, width=1920)
_CLASS_NAMES = {HEAD_CLASS_ID: "head", FBODY_CLASS_ID: "fbody"}


def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> Boxes:
    """`rows` 為 `(x1, y1, x2, y2, conf, cls)`，與 ultralytics 的 data 佈局一致。"""
    return Boxes(torch.tensor(rows, dtype=torch.float32), _ORIG_SHAPE)


# 兩格偵測：各兩個 fbody，配上罩得住的 head，第二格另有一顆配不到人的孤兒 head。
_FRAME_0_DETECTIONS = _boxes(
    [
        (100.0, 100.0, 140.0, 150.0, 0.9, HEAD_CLASS_ID),
        (80.0, 90.0, 200.0, 400.0, 0.8, FBODY_CLASS_ID),
        (620.0, 110.0, 660.0, 160.0, 0.7, HEAD_CLASS_ID),
        (600.0, 100.0, 700.0, 500.0, 0.6, FBODY_CLASS_ID),
    ]
)
_FRAME_1_DETECTIONS = _boxes(
    [
        (105.0, 105.0, 145.0, 155.0, 0.9, HEAD_CLASS_ID),
        (85.0, 95.0, 205.0, 405.0, 0.8, FBODY_CLASS_ID),
        (610.0, 100.0, 700.0, 500.0, 0.6, FBODY_CLASS_ID),
        (1500.0, 900.0, 1540.0, 950.0, 0.5, HEAD_CLASS_ID),
    ]
)


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


class _RecordingTracker:
    """記下每次 `update` 收到的類別，並為**每個輸入框**回一條軌跡。

    「一個偵測目標一條軌跡」正是 ByteTrack 的行為，也是 head 不可進 tracker 的原因；
    照這樣回傳，head 混進來時多出的軌跡就會一路寫進 parquet。
    """

    def __init__(self):
        self.received_cls: list[list[int]] = []

    def update(self, stream_id: int, yolo_boxes: Boxes) -> np.ndarray:
        cls = [int(c) for c in yolo_boxes.cls.tolist()]
        self.received_cls.append(cls)
        xyxy = yolo_boxes.xyxy.numpy()
        if len(xyxy) == 0:
            return np.array([])
        # 列格式同 MultiStreamByteTracker.update：[x1,y1,x2,y2,track_id,score,cls,idx]
        return np.array(
            [
                [*box, i + 1, 0.9, c, i]
                for i, (box, c) in enumerate(zip(xyxy, cls, strict=True))
            ]
        )


class _FrameRingStub:
    """環形緩衝的替身：slot 內容與本測試無關，回傳固定大小的空影格即可。

    `num_slots` 沿用正式執行的推導值，讓 `start_loop` 開頭的格數不變量檢查在測試裡
    也跑在真實條件下。
    """

    num_slots = RING_SLOTS

    def view_slot(self, slot: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _run_loop(
    tmp_path, per_frame: list[Boxes]
) -> tuple[_RecordingTracker, pl.DataFrame, queue.Queue, _ScriptedDetector]:
    """跑一次單路的推理主迴圈。

    回傳追蹤器的收件紀錄、寫出的 parquet，以及歸還時序要看的 free queue 與偵測器。
    """
    tracker = _RecordingTracker()
    free_queue: queue.Queue = queue.Queue()
    detector = _ScriptedDetector(per_frame, free_queue)
    results_path = Path(tmp_path) / "tracking_results.parquet"
    pipeline = InferencePipeline(
        stream_names=["loc_cam001"],
        detector=detector,
        tracker=tracker,
        results_path=results_path,
        frame_shapes=[_SHAPE],
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
    data_queue.put(READER_DONE)

    pipeline.start_loop([data_queue], [free_queue], [_FrameRingStub()])
    return tracker, pl.read_parquet(results_path), free_queue, detector


def test_main_loop_hands_only_fbody_to_the_tracker(tmp_path):
    tracker, _df, _free_queue, _detector = _run_loop(
        tmp_path, [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    assert tracker.received_cls == [
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
    ]


def test_heads_do_not_become_extra_tracks_in_the_output(tmp_path):
    """同一件事在下游的樣子：每格兩個人就是兩條軌跡，不會被 head 撐成四條。"""
    _tracker, df, _free_queue, _detector = _run_loop(
        tmp_path, [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    assert df.height == 4  # 兩格 × 每格兩條軌跡
    assert sorted(df["track_id"].unique().to_list()) == [1, 2]


def test_slots_are_returned_after_inference_not_before(tmp_path):
    """歸還晚於推論、且整批推論完就歸還——免複製消費唯一會出錯的地方。

    每次 predict 當下已歸還的 slot 數，必須恰好等於**先前各批**的格數總和：多一格
    就代表本批的 slot 在推論中被放行，reader 可以覆寫正在推論的畫面。餵超過一批的
    量，讓跨批的歸還順序也被涵蓋（只驗單批的話，第二批之後歸還早一步也看不出來）。
    末尾再驗全數歸還——漏還的話 reader 會卡在 `free_queue.get()`，整條 pipeline 停住。
    """
    num_frames = 20  # > _target_batch（batch=8 → 16），確保跑到第二批
    _tracker, _df, free_queue, detector = _run_loop(
        tmp_path, [_FRAME_0_DETECTIONS] * num_frames
    )

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
