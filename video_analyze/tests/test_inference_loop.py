"""推理主迴圈的接線契約測試：哪一份資料交給誰、在哪一步做。

三件事都是「主迴圈接錯線，但被接的函式自己的測試全綠、parquet 也完全正常」：

1. **交給 tracker 的只有 fbody**。`split_detections` 拆得對不對由
   test_inference_split.py 釘住，這裡釘的是主迴圈實際把拆出來的**哪一份**交給
   `tracker.update`。把引數改回未拆分的偵測結果（`results[idx].boxes.cpu()`），同一個
   人會多出一條頭部軌跡，`track_id` 的語義從「一個人」變成「一個偵測目標」，下游的
   不重複訪客與進出人數直接翻倍。
2. **座標反算在落腳點推算之後**。影格在讀取端就縮成推論尺寸（issue #108），框與落腳點
   要在寫出前映射回原始解析度；反算若提早到 `estimate` 之前，`heads` 與 `tracks` 會落在
   不同尺度上而配不到頭，每列靜默退回框底邊中點。
3. **`frame_shapes` 是原始解析度，不是推論尺寸**。傳成推論尺寸會讓反算退化成恆等、
   parquet 的 `frame_width` 一起寫成 640。

因此本檔的偵測資料一律以**推論尺度**（640×384）表示——那就是縮放前移之後 YOLO 實際
輸出的座標系。

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
import pytest
import torch
from ultralytics.engine.results import Boxes, Results

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services.frame_ring import RING_SLOTS
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    letterbox_params,
)
from video_analyze.services.video_reader import READER_DONE, FrameShape

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
# 偵測結果的座標系：影格在讀取端就縮好了，ultralytics 的 `orig_shape` 也是這個尺寸
_INFER_SHAPE = (INFER_HEIGHT, INFER_WIDTH)
# 該路的原始解析度，也就是反算的目標尺度
_SHAPE = FrameShape(height=1080, width=1920)
_SCALE, _PAD_X, _PAD_Y = letterbox_params(_SHAPE.height, _SHAPE.width)
_CLASS_NAMES = {HEAD_CLASS_ID: "head", FBODY_CLASS_ID: "fbody"}


def _to_source(x: float, y: float) -> tuple[float, float]:
    """把推論尺度的一個點換算成原始解析度，供期望值使用。"""
    return (x - _PAD_X) / _SCALE, (y - _PAD_Y) / _SCALE


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


def test_boxes_and_foot_points_are_mapped_back_to_the_source_resolution(tmp_path):
    """**反算必須在落腳點推算之後**：框與落腳點都要回到原始解析度，且互相自洽。

    第一格第一條軌跡：fbody `(27, 42, 67, 145)`、配到的 head `(30, 45, 44, 62)`，
    在推論尺度上 `foot = 2 × C_fbody − H = (57, 142)`（`H` 為 head 框頂邊中點），
    換算回 1080p 是 `(171, 390)`；框底邊中點則是 `(141, 399)`。

    把反算移到 `estimate` 之前，`heads` 仍在推論尺度、`tracks` 已回原始解析度，
    `_match_head` 全數回 `None`，落腳點正好退回那個框底邊中點——列數、`track_id`、
    bbox 全部正常，只有落腳點靜默偏掉（ADR-009 要修掉的偏移就這樣回來）。
    """
    _tracker, df, _free_queue, _detector = _run_loop(tmp_path, [_FRAME_0_DETECTIONS])
    row = df.filter(pl.col("track_id") == 1).row(0, named=True)

    expected_box = [*_to_source(27.0, 42.0), *_to_source(67.0, 145.0)]
    assert [row["x1"], row["y1"], row["x2"], row["y2"]] == pytest.approx(
        expected_box, abs=0.5
    )
    expected_foot = _to_source(57.0, 142.0)
    assert (row["foot_x"], row["foot_y"]) == pytest.approx(expected_foot, abs=0.5)
    # 反算提早做的話會落在這裡：換算過的框底邊中點
    bottom_center = ((row["x1"] + row["x2"]) / 2, row["y2"])
    assert (row["foot_x"], row["foot_y"]) != pytest.approx(bottom_center, abs=0.5)


def test_frame_size_columns_stay_at_the_source_resolution(tmp_path):
    """尺寸欄位寫的是原始解析度，不是推論尺寸。

    這兩欄是 `line_counting`／`zone_mapping` 換算 1080p 基準像素的唯一來源，兩包都只
    檢查欄位存在、不檢查值合理性（ADR-004／ADR-006）：寫成 640 的話下游幾何全部縮到
    約 1/3 而不報錯。
    """
    _tracker, df, _free_queue, _detector = _run_loop(tmp_path, [_FRAME_0_DETECTIONS])

    assert df["frame_width"].unique().to_list() == [_SHAPE.width]
    assert df["frame_height"].unique().to_list() == [_SHAPE.height]


def test_frame_shapes_of_inference_size_are_rejected(tmp_path):
    """`frame_shapes` 傳成推論尺寸要當場拋錯，這是上一支測試那條路徑的來源。

    緩衝改吃推論尺寸之後，順手把 `frame_shapes` 也改成推論尺寸是很自然的下一步，而
    後果全是靜默的：反算參數退化成恆等（scale=1、pad=0），座標停在 640×384，parquet
    的 `frame_width` 同時寫成 640。正式來源 `probe_frame_shape` 是 16:9，不會等於 5:3
    的推論尺寸，所以這道檢查不會擋到合法輸入。
    """
    with pytest.raises(ValueError, match="推論尺寸"):
        InferencePipeline(
            stream_names=["loc_cam001"],
            detector=_ScriptedDetector([], queue.Queue()),
            tracker=_RecordingTracker(),
            results_path=Path(tmp_path) / "tracking_results.parquet",
            frame_shapes=[FrameShape(height=INFER_HEIGHT, width=INFER_WIDTH)],
        )
