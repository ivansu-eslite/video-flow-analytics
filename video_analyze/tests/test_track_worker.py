"""追蹤進程的接線契約測試：七步的順序、payload 的往返、以及兩條結束訊號。

追蹤、落腳點推算、座標反算與 parquet 落盤在 issue #109 從推論進程搬到這裡。搬移過程
最容易打亂的是**順序**，而打亂的後果全是靜默的——parquet 的列數、`track_id` 與 bbox
都完全正常：

1. **交給 tracker 的只有 fbody**。`split_detections` 拆得對不對由 test_inference_split.py
   釘住，這裡釘的是本模組實際把拆出來的**哪一份**交給 `tracker.update`。把引數改回
   未拆分的偵測結果，同一個人會多出一條頭部軌跡，`track_id` 的語義從「一個人」變成
   「一個偵測目標」，下游的不重複訪客與進出人數直接翻倍。
2. **座標反算在落腳點推算之後**。`heads` 是唯一沒有反算的陣列，反算提早到 `estimate`
   之前會讓兩者落在不同尺度上而配不到頭，每列靜默退回框底邊中點（ADR-009 要修掉的
   偏移）。這條在 issue #108 就用迴歸測試鎖住了，整段搬進子進程時鎖也跟著搬。
3. **`frame_shapes` 是原始解析度，不是推論尺寸**。傳成推論尺寸會讓反算退化成恆等、
   parquet 的 `frame_width` 一起寫成 640。

另外兩條是本次新增的：**空格也要走完整條路徑**（`BYTETracker` 的軌跡老化靠每格呼叫
推進），以及**`TRACK_FAILED` 要清掉暫存檔**（推論進程的 `collector.discard()` 隨
collector 一起搬走了，那條路徑改由訊號覆蓋）。

issue #113 再補一條收尾路徑：推論進程被 SIGKILL 時 `TRACK_FAILED` 送不出來，本進程等到的
是父進程 terminate 的 **SIGTERM**，要攔下來走既有的清理。認領輸出位置那一條已經不在本
進程（分片之後由主進程認領 `parts/.lock`），那批測試在 `test_output_parts.py`。

分片之後多一條：**收到不屬於自己的 stream_id 要拋錯**。那是路由送錯片時唯一的訊號——
每片都建滿了全部路的 tracker，送錯的 payload 會被正常追蹤、正常寫進這一片的 part，
只是該路的軌跡被切成兩段而 `track_id` 分裂，合併後的檔案完全正常。

因此本檔的偵測資料一律以**推論尺度**（640×384）表示——那就是 payload 內座標的尺度。
真正的 `MultiStreamByteTracker` 會做 Kalman 平滑而算不出可斷言的期望值，故以每個輸入
框回一條軌跡的替身取代（那正是 ByteTrack「一個偵測目標一條軌跡」的行為）。
"""

import datetime
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import torch
from ultralytics.engine.results import Boxes

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services import track_worker as tw
from video_analyze.services import tracking_results as tr
from video_analyze.services.batching import TARGET_BATCH, TRACK_QUEUE_SLOTS
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    letterbox_params,
)
from video_analyze.services.track_worker import (
    TRACK_DONE,
    TRACK_FAILED,
    run_track_worker,
    to_payload,
)
from video_analyze.services.video_reader import FrameShape

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
# payload 內座標的座標系：影格在讀取端就縮好了，反算是本模組的責任
_INFER_SHAPE = (INFER_HEIGHT, INFER_WIDTH)
# 該路的原始解析度，也就是反算的目標尺度
_SHAPE = FrameShape(height=1080, width=1920)
_SCALE, _PAD_X, _PAD_Y = letterbox_params(_SHAPE.height, _SHAPE.width)


def _to_source(x: float, y: float) -> tuple[float, float]:
    """把推論尺度的一個點換算成原始解析度，供期望值使用。"""
    return (x - _PAD_X) / _SCALE, (y - _PAD_Y) / _SCALE


def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> Boxes:
    """`rows` 為 `(x1, y1, x2, y2, conf, cls)`，與 ultralytics 的 data 佈局一致。"""
    return Boxes(torch.tensor(rows, dtype=torch.float32), _INFER_SHAPE)


# 兩格偵測：各兩個 fbody，配上罩得住的 head，第二格另有一顆配不到人的孤兒 head。
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


class _RecordingTracker:
    """記下每次 `update` 收到的類別，並為**每個輸入框**回一條軌跡。

    「一個偵測目標一條軌跡」正是 ByteTrack 的行為，也是 head 不可進 tracker 的原因；
    照這樣回傳，head 混進來時多出的軌跡就會一路寫進 parquet。
    """

    def __init__(self, num_streams: int):
        self.num_streams = num_streams
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


def _install_recording_tracker(monkeypatch) -> list[_RecordingTracker]:
    """把 `MultiStreamByteTracker` 換成替身，回傳「已建立的實例」清單。

    追蹤器是 `run_track_worker` 在進程內自己建的（它持有跨格狀態，不從外面傳入），
    所以要從這裡把實例撈出來才驗得到它收到什麼。
    """
    created: list[_RecordingTracker] = []

    def factory(num_streams: int) -> _RecordingTracker:
        tracker = _RecordingTracker(num_streams)
        created.append(tracker)
        return tracker

    monkeypatch.setattr(tw, "MultiStreamByteTracker", factory)
    return created


def _part_path(tmp_path) -> Path:
    """這一片的 part 檔。分片之後 worker 寫的是它，正式檔名由主進程合併產生。"""
    return Path(tmp_path) / "tracking_results.parts" / "shard0.parquet"


def _run_worker(
    tmp_path,
    monkeypatch,
    per_frame: list[Boxes],
    last_signal: str = TRACK_DONE,
    frame_shapes: list[FrameShape] | None = None,
) -> tuple[list[_RecordingTracker], Path]:
    """把 `per_frame` 逐格轉成 payload 餵給追蹤進程主體，跑完回傳追蹤器與 part 檔路徑。

    `track_queue` 用 `queue.Queue` 取代 `mp.Queue`（介面相同），不拉起任何子進程。
    `last_signal` 換成 `TRACK_FAILED` 就能驗推論進程崩潰時的失效路徑。

    餵進去的 `Boxes` 先 `clone()` 一份：`to_payload` 對已在 CPU 的 tensor 回傳的是**共享
    記憶體的 numpy view**，而正式執行時 `mp.Queue` 會 pickle（等於複製），這裡用
    `queue.Queue` 就沒有那道複製。不 clone 的話 `_track_one` 的 `clip_to_content_inplace`
    會就地改掉模組層級的偵測資料，污染同檔後續測試——目前所有框都在內容區以內、裁切是
    no-op 所以打不到，但只要有人加一個落在填充帶的框（正是 clip 要測的情境）就會踩到。
    """
    trackers = _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    track_queue: queue.Queue = queue.Queue()
    for frame_index, boxes in enumerate(per_frame):
        track_queue.put(
            to_payload(
                0,
                Boxes(boxes.data.clone(), boxes.orig_shape),
                frame_index,
                _BASE + datetime.timedelta(seconds=frame_index),
            )
        )
    track_queue.put(last_signal)

    run_track_worker(
        track_queue=track_queue,
        stream_names=["loc_cam001"],
        frame_shapes=frame_shapes or [_SHAPE],
        part_path=part_path,
        shard_id=0,
        owned_stream_ids=frozenset({0}),
    )
    return trackers, part_path


def test_only_fbody_is_handed_to_the_tracker(tmp_path, monkeypatch):
    trackers, _path = _run_worker(
        tmp_path, monkeypatch, [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    assert trackers[0].received_cls == [
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
    ]


def test_heads_do_not_become_extra_tracks_in_the_output(tmp_path, monkeypatch):
    """同一件事在下游的樣子：每格兩個人就是兩條軌跡，不會被 head 撐成四條。"""
    _trackers, path = _run_worker(
        tmp_path, monkeypatch, [_FRAME_0_DETECTIONS, _FRAME_1_DETECTIONS]
    )

    df = pl.read_parquet(path)
    assert df.height == 4  # 兩格 × 每格兩條軌跡
    assert sorted(df["track_id"].unique().to_list()) == [1, 2]


def test_boxes_and_foot_points_are_mapped_back_to_the_source_resolution(
    tmp_path, monkeypatch
):
    """**反算必須在落腳點推算之後**：框與落腳點都要回到原始解析度，且互相自洽。

    第一格第一條軌跡：fbody `(27, 42, 67, 145)`、配到的 head `(30, 45, 44, 62)`，
    在推論尺度上 `foot = 2 × C_fbody − H = (57, 142)`（`H` 為 head 框頂邊中點），
    換算回 1080p 是 `(171, 390)`；框底邊中點則是 `(141, 399)`。

    把反算移到 `estimate` 之前，`heads` 仍在推論尺度、`tracks` 已回原始解析度，
    `_match_head` 全數回 `None`，落腳點正好退回那個框底邊中點——列數、`track_id`、
    bbox 全部正常，只有落腳點靜默偏掉（ADR-009 要修掉的偏移就這樣回來）。
    """
    _trackers, path = _run_worker(tmp_path, monkeypatch, [_FRAME_0_DETECTIONS])
    row = pl.read_parquet(path).filter(pl.col("track_id") == 1).row(0, named=True)

    expected_box = [*_to_source(27.0, 42.0), *_to_source(67.0, 145.0)]
    assert [row["x1"], row["y1"], row["x2"], row["y2"]] == pytest.approx(
        expected_box, abs=0.5
    )
    expected_foot = _to_source(57.0, 142.0)
    assert (row["foot_x"], row["foot_y"]) == pytest.approx(expected_foot, abs=0.5)
    # 反算提早做的話會落在這裡：換算過的框底邊中點
    bottom_center = ((row["x1"] + row["x2"]) / 2, row["y2"])
    assert (row["foot_x"], row["foot_y"]) != pytest.approx(bottom_center, abs=0.5)


def test_frame_size_columns_stay_at_the_source_resolution(tmp_path, monkeypatch):
    """尺寸欄位寫的是原始解析度，不是推論尺寸。

    這兩欄是 `line_counting`／`zone_mapping` 換算 1080p 基準像素的唯一來源，兩包都只
    檢查欄位存在、不檢查值合理性（ADR-004／ADR-006）：寫成 640 的話下游幾何全部縮到
    約 1/3 而不報錯。
    """
    _trackers, path = _run_worker(tmp_path, monkeypatch, [_FRAME_0_DETECTIONS])

    df = pl.read_parquet(path)
    assert df["frame_width"].unique().to_list() == [_SHAPE.width]
    assert df["frame_height"].unique().to_list() == [_SHAPE.height]


def test_frame_shapes_of_inference_size_are_rejected(tmp_path, monkeypatch):
    """`frame_shapes` 傳成推論尺寸要當場拋錯，這是上一支測試那條路徑的來源。

    緩衝改吃推論尺寸之後，順手把 `frame_shapes` 也改成推論尺寸是很自然的下一步，而
    後果全是靜默的：反算參數退化成恆等（scale=1、pad=0），座標停在 640×384，parquet
    的 `frame_width` 同時寫成 640。代價是真的以推論尺寸為來源解析度的攝影機也會被擋
    （那種輸入其實跑得動），但推論尺寸是 5:3、不是常見的攝影機規格，訊息也指到這件事。
    """
    with pytest.raises(ValueError, match="推論尺寸"):
        _run_worker(
            tmp_path,
            monkeypatch,
            [],
            frame_shapes=[FrameShape(height=INFER_HEIGHT, width=INFER_WIDTH)],
        )


def test_frames_without_detections_still_reach_the_tracker(tmp_path, monkeypatch):
    """空格也要呼叫 `tracker.update`，不可因為 payload 是 None 就整格跳過。

    `BYTETracker` 的 `frame_id` 與軌跡老化（`track_buffer` 到期即移除）都靠每格呼叫
    推進：跳過空格會讓已離開畫面的人一直留在 lost 狀態、之後被錯誤地接回，而輸出檔
    的列數、`track_id` 與 bbox 完全正常。`FootPointEstimator` 的偏移量 TTL 同理——它
    按「有軌跡的幀」計 tick，那個早退要由 `estimate` 自己做，不能由呼叫端代勞。
    """
    empty = Boxes(torch.zeros((0, 6), dtype=torch.float32), _INFER_SHAPE)
    trackers, path = _run_worker(
        tmp_path, monkeypatch, [_FRAME_0_DETECTIONS, empty, _FRAME_1_DETECTIONS]
    )

    # 三格都進 tracker，中間那格是空輸入而不是被吞掉
    assert trackers[0].received_cls == [
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
        [],
        [FBODY_CLASS_ID, FBODY_CLASS_ID],
    ]
    # 空格不產生列（沒有軌跡），但前後兩格照樣寫出
    assert pl.read_parquet(path)["frame_id"].to_list() == [0, 0, 2, 2]


def test_track_done_saves_the_parquet(tmp_path, monkeypatch):
    """收到 `TRACK_DONE` 才 `save()`：正式檔名出現、暫存檔不留。"""
    _trackers, path = _run_worker(tmp_path, monkeypatch, [_FRAME_0_DETECTIONS])

    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_track_failed_discards_the_partial_output(tmp_path, monkeypatch):
    """收到 `TRACK_FAILED` 要拋錯並清掉暫存檔，正式檔名不得出現。

    推論進程原本的 `try/except BaseException: collector.discard()` 隨 collector 一起
    搬走了，那道保護改由這條訊號覆蓋。把 flush 門檻降到 1 讓 `.tmp` 真的被建出來——
    否則幾格資料還在記憶體緩衝裡，測試會在什麼都沒發生的情況下通過。
    """
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")

    with pytest.raises(RuntimeError, match="推論進程中途例外"):
        _run_worker(
            tmp_path, monkeypatch, [_FRAME_0_DETECTIONS], last_signal=TRACK_FAILED
        )

    assert not part_path.exists()
    assert not tmp_file.exists()


def test_a_payload_for_another_shard_is_rejected(tmp_path, monkeypatch):
    """收到不屬於自己的 stream_id 要當場拋錯，這是路由送錯片唯一的訊號。

    每片都建滿了全部路的 tracker（讓 stream_id 維持全域編號、payload 不必轉換），
    所以 `MultiStreamByteTracker.update` 對未知 stream_id 回空陣列那條靜默路徑在這裡
    根本走不到：送錯片的 payload 會被正常追蹤、正常寫進這一片的 part，只是該路的軌跡
    被切成兩段而 `track_id` 分裂，而合併後的檔案完全正常、列數也不變。
    """
    _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    track_queue: queue.Queue = queue.Queue()
    track_queue.put(
        to_payload(1, Boxes(_FRAME_0_DETECTIONS.data.clone(), _INFER_SHAPE), 0, _BASE)
    )

    with pytest.raises(RuntimeError, match="不屬於自己的 stream_id"):
        run_track_worker(
            track_queue=track_queue,
            stream_names=["loc_cam001", "loc_cam002"],
            frame_shapes=[_SHAPE, _SHAPE],
            part_path=part_path,
            shard_id=0,
            owned_stream_ids=frozenset({0}),  # cam002 是另一片的
        )

    assert not part_path.exists()


def test_a_shard_only_writes_its_own_part(tmp_path, monkeypatch):
    """`TRACK_DONE` 只收尾自己那一份：別片的 part 檔一個 byte 都不能動。

    各片是獨立進程、寫的是同一個目錄底下的不同檔案，收尾若依目錄（而不是依傳進來的
    那條路徑）動作，就會把還在寫的鄰居收掉，而兩邊的 log 都寫著「已寫入」。
    """
    part_path = _part_path(tmp_path)
    neighbour = part_path.with_name("shard1.parquet")
    neighbour.parent.mkdir(parents=True, exist_ok=True)
    neighbour.write_bytes(b"y" * 4096)

    _trackers, written = _run_worker(tmp_path, monkeypatch, [_FRAME_0_DETECTIONS])

    assert written == part_path
    assert part_path.exists()
    assert neighbour.read_bytes() == b"y" * 4096


def test_a_failing_shard_only_discards_its_own_part(tmp_path, monkeypatch):
    """例外時同樣只清自己那一份，鄰居正在寫的暫存檔不能被波及。"""
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    part_path = _part_path(tmp_path)
    neighbour_tmp = part_path.with_name("shard1.parquet.tmp")
    neighbour_tmp.parent.mkdir(parents=True, exist_ok=True)
    neighbour_tmp.write_bytes(b"y" * 4096)

    with pytest.raises(RuntimeError, match="推論進程中途例外"):
        _run_worker(
            tmp_path, monkeypatch, [_FRAME_0_DETECTIONS], last_signal=TRACK_FAILED
        )

    assert not part_path.exists()
    assert not part_path.with_name(part_path.name + ".tmp").exists()
    assert neighbour_tmp.read_bytes() == b"y" * 4096


def test_to_payload_returns_none_for_frames_without_detections():
    """空框送 None，省下每格一次空陣列的 pickle。"""
    empty = Boxes(torch.zeros((0, 6), dtype=torch.float32), _INFER_SHAPE)

    assert to_payload(3, empty, 7, _BASE) == (3, None, 7, _BASE)
    assert to_payload(3, None, 7, _BASE) == (3, None, 7, _BASE)


def test_to_payload_hands_over_plain_cpu_numpy():
    """轉成 CPU numpy 而不是直接傳 `Boxes`：後者內含 tensor、可能還在 GPU 上，
    pickle 過去會連帶把 CUDA 狀態拖進來。"""
    stream_id, box_data, frame_index, timestamp = to_payload(
        3, _FRAME_0_DETECTIONS, 7, _BASE
    )

    assert (stream_id, frame_index, timestamp) == (3, 7, _BASE)
    assert isinstance(box_data, np.ndarray)
    assert box_data.shape == (4, 6)


def test_payload_round_trip_preserves_confidence_and_class():
    """往返後重新包成 `Boxes`，衍生屬性要與原始一致。

    `orig_shape` 給推論尺寸——此時座標確實在那個尺度上，給錯（例如給原始解析度）會讓
    `Boxes` 的衍生屬性算錯，而 `.data` 本身看起來完全正常。
    """
    _stream_id, box_data, _frame_index, _timestamp = to_payload(
        0, _FRAME_0_DETECTIONS, 0, _BASE
    )
    restored = Boxes(torch.from_numpy(box_data), _INFER_SHAPE)

    np.testing.assert_allclose(
        restored.conf.numpy(), _FRAME_0_DETECTIONS.conf.numpy()
    )
    np.testing.assert_allclose(restored.cls.numpy(), _FRAME_0_DETECTIONS.cls.numpy())
    np.testing.assert_allclose(
        restored.xyxy.numpy(), _FRAME_0_DETECTIONS.xyxy.numpy()
    )
    assert restored.orig_shape == _INFER_SHAPE


def test_track_queue_capacity_is_a_few_batches_of_slack():
    """佇列上限是背壓，不是調校參數：沒有它，追蹤一落後 payload 就無上限堆在推論進程，
    而 `TRACK_FAILED` 也會晚到父進程 terminate 之後（見 `TRACK_QUEUE_SLOTS` 的說明）。

    釘住「以單次推論批次為單位的鬆弛」這個口徑：只調 `[model].batch` 而沒同步調這裡，
    鬆弛會不足一批而讓兩個進程互等。
    """
    assert TRACK_QUEUE_SLOTS % TARGET_BATCH == 0
    slack_batches = TRACK_QUEUE_SLOTS // TARGET_BATCH
    assert 2 <= slack_batches <= 8, (
        f"鬆弛 {slack_batches} 批：太少會讓正常抖動變成互等，太多等於把 backlog 換個"
        "地方堆，還會讓 TRACK_FAILED 的送達延遲超過父進程的 terminate 時限"
    )


def test_payload_shares_memory_with_the_source_boxes():
    """`to_payload` 回的是共享記憶體的 view，不是副本——正式路徑靠 `mp.Queue` 的 pickle
    複製，測試用 `queue.Queue` 時就得自己 clone（見 `_run_worker`）。

    這支測試的用途是讓「哪一邊負責複製」這件事有訊號：日後若 `to_payload` 改成回副本
    （例如加上 `.copy()`），這裡會失敗，提醒同時檢查 `_run_worker` 的 clone 還需不需要。
    """
    _stream_id, box_data, _frame_index, _timestamp = to_payload(
        0, _FRAME_0_DETECTIONS, 0, _BASE
    )

    assert np.shares_memory(box_data, _FRAME_0_DETECTIONS.data.numpy())


class _SigtermAfterGets(queue.Queue):
    """第 N 次 `get()` 回傳之後對自己送 SIGTERM，模擬父進程的 `_terminate_all`。

    正式路徑上訊號是打斷阻塞中的 `mp.Queue.get()`（handler 拋出的例外從 `get()` 內部
    往外傳）；這裡用同步送達換取確定性，驗的是「handler 有裝上、且拋出的例外會走到
    `collector.discard()`」這段接線。
    """

    def __init__(self, kill_after: int, on_kill):
        super().__init__()
        self._remaining = kill_after
        self._on_kill = on_kill

    def get(self, *args, **kwargs):
        item = super().get(*args, **kwargs)
        self._remaining -= 1
        if self._remaining == 0:
            self._on_kill()
            os.kill(os.getpid(), signal.SIGTERM)
        return item


def test_sigterm_discards_the_partial_output(tmp_path, monkeypatch):
    """收到 SIGTERM 要清掉暫存檔：那是推論進程被 SIGKILL 之後唯一還走得到的路徑。

    上游被 SIGKILL 時 `TRACK_FAILED` 送不出來（`track_queue` 的 pipe 寫入端 fd 被父進程
    與九個讀取進程一起繼承，`get()` 收不到 EOF），本進程等到的是父進程 terminate 的
    SIGTERM。不攔的話預設處置直接結束進程，不走 `except`／`finally`，`.tmp` 就留在
    輸出目錄了。

    把 flush 門檻降到 1，讓 `.tmp` 在訊號抵達前真的被建出來——`saw_tmp` 釘住這件事，
    否則這支測試會在什麼都沒發生的情況下通過。
    """
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")
    saw_tmp: list[bool] = []

    def _fallback(signum, frame):
        # 保險絲：handler 沒被裝上時 SIGTERM 會當場殺掉整個 pytest 進程，那看起來像
        # crash 而不是測試失敗。run_track_worker 會換掉它，並在離開時還原
        raise AssertionError("run_track_worker 沒有裝上 SIGTERM handler")

    track_queue = _SigtermAfterGets(
        kill_after=2, on_kill=lambda: saw_tmp.append(tmp_file.exists())
    )
    for frame_index in range(2):
        track_queue.put(
            to_payload(
                0,
                Boxes(_FRAME_0_DETECTIONS.data.clone(), _INFER_SHAPE),
                frame_index,
                _BASE + datetime.timedelta(seconds=frame_index),
            )
        )

    previous = signal.signal(signal.SIGTERM, _fallback)
    try:
        with pytest.raises(SystemExit) as excinfo:
            run_track_worker(
                track_queue=track_queue,
                stream_names=["loc_cam001"],
                frame_shapes=[_SHAPE],
                part_path=part_path,
                shard_id=0,
                owned_stream_ids=frozenset({0}),
            )
        assert excinfo.value.code == 143  # 128 + SIGTERM
        # 離開時要還原成呼叫前的 handler：本函式在測試中是 in-process 呼叫的
        assert signal.getsignal(signal.SIGTERM) is _fallback
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert saw_tmp == [True], "訊號抵達時還沒有暫存檔，這支測試沒驗到東西"
    assert not tmp_file.exists()
    assert not part_path.exists()


def test_a_failure_before_the_loop_leaves_no_residue(tmp_path, monkeypatch):
    """建 collector 之後、進主迴圈之前的建構步驟若拋錯，也不能留下暫存檔。

    `[foot_point].method` 填錯就是這條路徑的真實觸發方式，留下的殘檔要等同一個 bucket
    的同一天再跑一次才清得掉——正是 issue #113 要消除的東西。

    同一個窗口的另一半：SIGTERM 的 handler 必須在 collector 建起來**之前**就註冊好，
    否則這段期間收到父進程的 terminate 走的仍是預設處置。`handler_ready` 釘住這個順序。
    """
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")
    handler_ready: list[bool] = []

    def _exploding_estimator(method: str):
        handler_ready.append(signal.getsignal(signal.SIGTERM) is tw._raise_on_sigterm)
        raise ValueError("人工注入：建構落腳點推算器失敗")

    _install_recording_tracker(monkeypatch)
    monkeypatch.setattr(tw, "FootPointEstimator", _exploding_estimator)
    track_queue: queue.Queue = queue.Queue()
    track_queue.put(TRACK_DONE)

    with pytest.raises(ValueError, match="人工注入"):
        run_track_worker(
            track_queue=track_queue,
            stream_names=["loc_cam001"],
            frame_shapes=[_SHAPE],
            part_path=part_path,
            shard_id=0,
            owned_stream_ids=frozenset({0}),
        )

    assert handler_ready == [True], "SIGTERM handler 比 collector 還晚註冊"
    assert not tmp_file.exists()
    assert not part_path.exists()


def test_sigterm_interrupts_a_blocking_queue_get(tmp_path, monkeypatch):
    """例外要從**阻塞中的** `mp.Queue.get()` 內部往外傳，不能只驗到接線。

    整個修法的支點就是這個前提。上面那支用的是同步送達的假 queue，驗得到「handler 有裝
    上、拋出的例外會走到 `discard()`」，但沒有任何東西擋住「日後把 `get()` 改成帶 timeout
    的重試迴圈、或把讀取搬到背景執行緒（handler 只在主執行緒跑）」——那樣這條路徑會靜默
    失效而那支測試照樣全綠。

    這裡用真的 `mp.Queue`：主執行緒處理完唯一那一格之後就卡在 `get()` 的 pipe 讀取上，
    由背景執行緒送 SIGTERM（訊號的 Python handler 一律在主執行緒執行）。末尾比對
    traceback 有沒有經過 `multiprocessing/queues.py`，把「從 get() 內部傳出來」這件事釘死。

    不拉子進程是刻意的：從載了 torch、又有多條執行緒的 pytest 進程 fork 出去，子進程會
    卡在繼承來的鎖上（`os.fork()` 自己就會發 DeprecationWarning 警告這件事），驗到的會是
    測試環境的問題而不是本模組的行為。
    """
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")

    def _fallback(signum, frame):
        raise AssertionError("run_track_worker 沒有裝上 SIGTERM handler")

    def _signal_once_it_blocks() -> None:
        # 等暫存檔出現＝那一格處理完了，主執行緒此刻正卡在 get() 上
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not tmp_file.exists():
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGTERM)

    track_queue: mp.Queue = mp.Queue()
    track_queue.put(
        to_payload(0, Boxes(_FRAME_0_DETECTIONS.data.clone(), _INFER_SHAPE), 0, _BASE)
    )
    previous = signal.signal(signal.SIGTERM, _fallback)
    killer = threading.Thread(target=_signal_once_it_blocks, daemon=True)
    try:
        killer.start()
        with pytest.raises(SystemExit) as excinfo:
            run_track_worker(
                track_queue=track_queue,
                stream_names=["loc_cam001"],
                frame_shapes=[_SHAPE],
                part_path=part_path,
                shard_id=0,
                owned_stream_ids=frozenset({0}),
            )
    finally:
        signal.signal(signal.SIGTERM, previous)
        killer.join(timeout=5)
        track_queue.close()
        track_queue.join_thread()

    assert excinfo.value.code == 143
    assert any("queues.py" in str(entry.path) for entry in excinfo.traceback), (
        "例外不是從阻塞中的 mp.Queue.get() 內部傳出來的"
    )
    assert not tmp_file.exists()
    assert not part_path.exists()


def test_sigterm_during_cleanup_does_not_interrupt_it(tmp_path, monkeypatch):
    """清理途中再收到 SIGTERM 不能把清理打斷。

    Ctrl+C 走的是 process group 的 SIGINT：本進程收到後進 `except` 開始 `discard()`，
    父進程同時在 `_terminate_all` 送 SIGTERM。handler 若還開著，就會在 `discard()` 內部
    再拋一次例外——落在 `_writer.close()` 之後、`unlink()` 之前的話，整天的 `.tmp` 就留
    下了，而外層那個 `except` 已經進來過、不會再跑一次。
    """
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")
    real_discard = tr.TrackingResultCollector.discard

    def _discard_interrupted_by_sigterm(self) -> None:
        # 模擬「清理已經開始，這時父進程的 SIGTERM 到了」
        os.kill(os.getpid(), signal.SIGTERM)
        real_discard(self)

    monkeypatch.setattr(
        tr.TrackingResultCollector, "discard", _discard_interrupted_by_sigterm
    )
    track_queue: queue.Queue = queue.Queue()
    track_queue.put(
        to_payload(0, Boxes(_FRAME_0_DETECTIONS.data.clone(), _INFER_SHAPE), 0, _BASE)
    )
    track_queue.put(TRACK_FAILED)  # 先以另一條路徑進入清理

    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with pytest.raises(RuntimeError, match="推論進程中途例外"):
            run_track_worker(
                track_queue=track_queue,
                stream_names=["loc_cam001"],
                frame_shapes=[_SHAPE],
                part_path=part_path,
                shard_id=0,
                owned_stream_ids=frozenset({0}),
            )
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert not tmp_file.exists(), "清理被 SIGTERM 打斷，暫存檔留下來了"
    assert not part_path.exists()


def test_a_second_ctrl_c_during_cleanup_does_not_interrupt_it(tmp_path, monkeypatch):
    """連按兩次 Ctrl+C 也不能把清理打斷。

    第一次 Ctrl+C 讓本進程以 `KeyboardInterrupt` 進到清理，第二次又是一個 SIGINT；只擋
    SIGTERM 的話它會在 `discard()` 內部打斷，落在 `_writer.close()` 之後、`unlink()`
    之前就留下整天的暫存檔。
    """
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 1)
    _install_recording_tracker(monkeypatch)
    part_path = _part_path(tmp_path)
    tmp_file = part_path.with_name(part_path.name + ".tmp")
    real_discard = tr.TrackingResultCollector.discard

    def _discard_interrupted_by_sigint(self) -> None:
        os.kill(os.getpid(), signal.SIGINT)  # 使用者又按了一次 Ctrl+C
        real_discard(self)

    monkeypatch.setattr(
        tr.TrackingResultCollector, "discard", _discard_interrupted_by_sigint
    )
    track_queue: queue.Queue = queue.Queue()
    track_queue.put(
        to_payload(0, Boxes(_FRAME_0_DETECTIONS.data.clone(), _INFER_SHAPE), 0, _BASE)
    )
    track_queue.put(TRACK_FAILED)

    try:
        with pytest.raises(RuntimeError, match="推論進程中途例外"):
            run_track_worker(
                track_queue=track_queue,
                stream_names=["loc_cam001"],
                frame_shapes=[_SHAPE],
                part_path=part_path,
                shard_id=0,
                owned_stream_ids=frozenset({0}),
            )
    except KeyboardInterrupt:
        # 沒擋 SIGINT 的話它會一路逸出到測試外，而 pytest 把 KeyboardInterrupt 當成
        # 「使用者中止」、整個 session 停掉——那看起來像中斷而不是測試失敗
        pytest.fail("第二次 Ctrl+C 打斷了清理：KeyboardInterrupt 逸出到 run_track_worker 之外")

    # 離開時 SIGINT 的處置要還原，否則之後整個進程都不再理 Ctrl+C
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN
    assert not tmp_file.exists(), "清理被第二次 Ctrl+C 打斷，暫存檔留下來了"
    assert not part_path.exists()
