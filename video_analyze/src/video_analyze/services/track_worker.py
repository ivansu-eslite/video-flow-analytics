"""把 ByteTrack 比對、落腳點推算與結果落盤搬出推論進程，改由獨立子進程做。

推論進程是整條 pipeline 的序列瓶頸：解碼有九個進程並行，但湊批 → 推論 → 追蹤 → 寫檔
全部擠在同一個進程裡一件一件做，時間直接相加。追蹤實測每格 1.81 ms、佔該進程 8.4%，
而它與 GPU 推論之間沒有資料相依（下一批的推論不需要上一批的軌跡），可以重疊。

**跨進程傳的是偵測框而不是影格**：每格幾十個框、幾 KB，pickle 成本遠低於影格。影格在
推論完成後就沒有用途（環形緩衝的 slot 在推論完成當下就歸還了，見 ADR-010），追蹤只需
要框。座標一路都停在推論尺度上（影格在讀取端就縮好了），本模組在寫出前才映回原始解析
度——所以每路的反算參數與內容區也跟著搬到這裡。

推論進程中途例外時送 `TRACK_FAILED`（fan-out 到每一片），收到就清掉自己的 `.tmp`
暫存檔再拋錯；與
`video_reader.py` 的 `READER_DONE`／`READER_FAILED` 同一套設計——明確的訊號而非裸
`None`，讓「正常結束」與「上游崩潰」不可能被混為一談。

推論進程被 SIGKILL 或整機掛掉時收不到那個訊號：`track_queue` 的 pipe 寫入端 fd 被父進程
與九個讀取進程一起繼承，寫入端永遠不會全部關閉，`get()` 收不到 EOF、只會一直等
（issue #113）。暫存檔因此靠另外兩道，都不需要動 queue 的內部狀態：

- **本進程攔下 SIGTERM**。等下去的結局是父進程 `_terminate_all` 的 SIGTERM，而它是
  攔得到的——handler 拋出的例外會從 `track_queue.get()` 內部往外傳，走既有的
  `except BaseException: collector.discard()`。
- **主進程認領輸出位置**（`output_parts.claim_parts_dir`）。整機重啟這種「所有進程都
  沒機會執行清理」的情境沒有任何 in-process 的機制擋得住，只能由下一次寫同一天的執行
  順手清掉。認領從本進程移到主進程是因為追蹤進程有 N 個（見下段），而認領、跑、合併
  三段要被同一把鎖蓋住。

**本進程有 N 個**（`[tracker].shards`），各自負責一組固定的攝影機：各路的
`BYTETracker` 與 `FootPointEstimator` 狀態按 stream_id 獨立，路與路之間沒有共享狀態，
所以追蹤天然可切。切不開的是輸出，故每片寫自己的 part 檔、由主進程合併（ADR-012）。
`stream_id` 仍是**全域**編號（`stream_names`／`frame_shapes` 各片都收完整清單），
`owned_stream_ids` 是唯一擋得下路由送錯片的東西——送錯的 payload 會被正常追蹤、正常
寫進那片的 part，只是該路的軌跡被切成兩段而 `track_id` 分裂，合併後的檔案完全正常。
"""

import multiprocessing as mp
import signal
import time
from datetime import datetime
from pathlib import Path
from queue import Full
from types import FrameType

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from vfa_observability import StructuredLogger

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.models.config import settings
from video_analyze.services.foot_point import FootPointEstimator
from video_analyze.services.fps_meter import FpsMeter
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    clip_to_content_inplace,
    content_box,
    letterbox_params,
    unscale_boxes_inplace,
    unscale_points_inplace,
)
from video_analyze.services.tracker import MultiStreamByteTracker
from video_analyze.services.tracking_results import TrackingResultCollector
from video_analyze.services.video_reader import FrameShape

logger = StructuredLogger(component="track_worker")

# 推論進程送完所有影格後放入 queue 的結束訊號。與 READER_DONE 同一套設計：明確的訊號
# 而非裸 None，讓「正常結束」與「上游崩潰導致 queue 永遠不再有東西」能分開處理。
TRACK_DONE = "__TRACK_DONE__"

# 推論進程中途例外時放入 queue 的錯誤訊號。收到就清掉暫存檔並拋錯，避免把上游崩潰
# 當成正常結束而把截斷的結果 rename 成正式檔名。
TRACK_FAILED = "__TRACK_FAILED__"

# fan-out `TRACK_FAILED` 時每一片最多等這麼久。1 秒對活著的片綽綽有餘（預設 batch 下
# 64 格的 backlog 約 70 ms 就消化完），對死掉的片則放棄得夠快。
TRACK_SIGNAL_PUT_TIMEOUT = 1.0

# `to_payload` 對空框送 None（省下每格一次空陣列的 pickle），本模組收到後補回一個空的
# 偵測陣列——欄數與 ultralytics 的 `Boxes.data` 一致（xyxy + conf + cls）。
_EMPTY_DETECTION_COLUMNS = 6


def fanout_track_signal(
    track_queues: list[mp.Queue], signal_value: str, timeout: float | None = None
) -> None:
    """把結束或失敗訊號送到**每一片**追蹤進程。

    payload 只送它歸屬的那一片，但結束與失敗是整條 pipeline 的事：漏掉哪一片，那一片
    就會一直等在 `track_queue.get()` 上（寫入端 fd 被父進程與各讀取進程一起繼承，上游
    死掉不會讓它收到 EOF，見模組 docstring），只能等父進程的 SIGTERM。

    Args:
        track_queues: 各片的 queue，索引即片編號。
        signal_value: `TRACK_DONE` 或 `TRACK_FAILED`。
        timeout: 每一片的 `put` 上限秒數；`None` 為阻塞到送出為止。
            **`TRACK_FAILED` 一定要給**（用 `TRACK_SIGNAL_PUT_TIMEOUT`）：fan-out 是
            序列的 `put`，而每條 queue 都有 `TRACK_QUEUE_SLOTS` 的上限。前一片若已經
            死了、它的 queue 又是滿的，無 timeout 的 `put` 會永久阻塞，**後面的片一個
            都收不到訊號**——那時只剩父進程的 SIGTERM 收尾，part 檔仍清得掉，但「上游
            崩潰」與「被 terminate」會混成同一種結束方式，正是這個 in-band 訊號當初
            要分開的東西。
            **`TRACK_DONE` 反過來不給**：走到那裡各片都在正常消化、不會卡住，而這個
            訊號寧可等也不能掉——掉了那一片不會 `save()`，合併就缺一支 part。
    """
    for shard_id, track_queue in enumerate(track_queues):
        try:
            track_queue.put(signal_value, timeout=timeout)
        except Full:
            logger.warning(
                "訊號送不進這一片的 queue（該片可能已經死了），續送下一片",
                shard_id=shard_id,
                signal=signal_value,
            )


def split_detections(boxes: Boxes) -> tuple[Boxes, np.ndarray]:
    """把一格的偵測結果拆成「要餵給 tracker 的 fbody 子集」與「head 框座標」。

    head 只用來推算落腳點，**不可進 tracker**：送進去的話同一個人會多出一條頭部
    軌跡，`track_id` 的語義從「一個人」變成「一個偵測目標」，下游的不重複訪客與
    進出人數會直接翻倍。

    Args:
        boxes: 單格的偵測結果，需已在 CPU（`Boxes.cpu()`）。

    Returns:
        `(fbody 子集, head 框的 [M, 4] xyxy)`；沒有 head 時第二項為 `(0, 4)` 空陣列。
    """
    cls = boxes.cls
    return boxes[cls == FBODY_CLASS_ID], boxes.xyxy[cls == HEAD_CLASS_ID].numpy()


def to_payload(
    stream_id: int, boxes: Boxes | None, frame_index: int, timestamp: datetime
) -> tuple[int, np.ndarray | None, int, datetime]:
    """把推論結果轉成可跨進程傳遞的輕量元組。

    在推論端做這個轉換而不是直接把 `Boxes` 丟進 queue：`Boxes` 內含 tensor、可能還在
    GPU 上，pickle 過去會連帶把 CUDA 狀態拖進來。轉成 CPU numpy 是幾 KB 的純資料。

    送的是**全部類別**的框（含 head），拆分留到 `_track_one` 做：`split_detections` 是
    純函式，而本模組本來就要把 numpy 包回 `Boxes` 才餵得了 tracker，拆在這一側等於零成本。

    Args:
        stream_id: 該格所屬的攝影機編號。
        boxes: 該格的 `results[idx].boxes`，座標位於推論尺度。
        frame_index: 該影格在所屬片段內的序號。
        timestamp: 該影格的時間戳（台北在地時間）。

    Returns:
        `(stream_id, box_data, frame_index, timestamp)`；`box_data` 是 `Boxes.data` 的
        CPU numpy（N×6：x1、y1、x2、y2、conf、cls），該格沒有任何偵測時為 `None`。
    """
    if boxes is None or len(boxes) == 0:
        return stream_id, None, frame_index, timestamp
    return stream_id, boxes.data.cpu().numpy(), frame_index, timestamp


def _reject_inference_sized_shapes(frame_shapes: list[FrameShape]) -> None:
    """`frame_shapes` 任一路等於推論尺寸就拋錯——那代表傳進來的是縮放後的尺寸。

    影格在讀取端就縮成推論尺寸了，`frame_shapes` 卻**必須維持原始解析度**：傳成推論
    尺寸的話反算參數會退化成恆等（scale=1、pad=0），框與落腳點靜默停在 640×384 尺度，
    而 parquet 的 frame_width 也一起寫成 640——下游 zone／line 只檢查這兩欄存在、不檢查
    值是否合理（ADR-004／ADR-006），幾何全部縮到約 1/3 而不報錯。

    代價是「來源本身真的就是 640×384」會被一起擋掉（那種輸入其實跑得動，letterbox 會
    原樣放行、反算是恆等）；推論尺寸是 5:3、不是任何常見的攝影機規格，拿誤判換掉一條
    靜默路徑划算，訊息裡也寫明了這個情況。

    這道檢查原本在 `InferencePipeline.__init__`，隨著 `frame_shapes` 的兩個消費端
    （反算參數、parquet 的尺寸欄位）一起搬到本模組——留在推論端會變成擋一個它已經不再
    持有的值。

    Raises:
        ValueError: 任一路的 `FrameShape` 等於推論尺寸。
    """
    infer_shaped = [
        index
        for index, shape in enumerate(frame_shapes)
        if (shape.height, shape.width) == (INFER_HEIGHT, INFER_WIDTH)
    ]
    if infer_shaped:
        raise ValueError(
            f"frame_shapes 有 {len(infer_shaped)} 路（stream_id={infer_shaped}）"
            f"等於推論尺寸 {INFER_WIDTH}×{INFER_HEIGHT}，需傳入各路的原始解析度："
            "它要寫進 parquet 供下游換算 1080p 基準像素，也是座標反算的依據，"
            "傳成推論尺寸會讓兩者一起靜默出錯。若這幾路的來源**真的**是 "
            f"{INFER_WIDTH}×{INFER_HEIGHT}（不是傳錯值），請改這道檢查——"
            "那種輸入本身跑得動。"
        )


def _track_one(
    tracker: MultiStreamByteTracker,
    foot_estimator: FootPointEstimator,
    stream_id: int,
    box_data: np.ndarray | None,
    content: tuple[int, int, int, int],
    unscale: tuple[float, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """把一格的偵測框走完裁切 → 拆分 → 追蹤 → 落腳點 → 反算，回傳 `(tracks, foot_points)`。

    這幾步的**順序不可調動**，其中兩處錯了都不會有任何錯誤訊息：

    - 裁切在拆分與 `tracker.update` 之前。改動前 ultralytics 在反算座標時就把框裁進畫面
      了，少了這一步 4K 的框會在反算時外擴最多 8 px（每個推論像素放大 6 倍），而 head
      配對與 tracker 也會看到與改動前不同的框（見 `letterbox.py`）。
    - **反算在 `foot_estimator.estimate` 之後**。`heads` 是這裡唯一沒有反算的陣列，先把
      `tracks` 換算回原始解析度會讓兩者尺度不一致，`_match_head` 全數回 `None`、每列退回
      框底邊中點，而列數、`track_id`、bbox 全部正常（ADR-009 要修掉的偏移就這樣靜默回
      來）。在推論尺度上配對是安全的：三個判準（中心落在框內、面積比、主軸傾角）在等比
      縮放＋等量平移下都不變。

    `BYTETracker` 吃的是 ultralytics `Boxes` 而不是裸陣列（它要 `.conf`／`.cls`／
    `.xywh`），所以這裡把跨進程傳來的 numpy 重新包成 `Boxes`。`orig_shape` 給**推論
    尺寸**——此時座標確實在那個尺度上（影格在讀取端就縮好了），給錯會讓 `Boxes` 的
    衍生屬性算錯。

    Args:
        tracker: 多路 ByteTrack 狀態管理器（跨格重用，維持軌跡延續）。
        foot_estimator: 落腳點推算器（跨格重用，記著每條軌跡上次的偏移量）。
        stream_id: 該格所屬的攝影機編號。
        box_data: `to_payload` 給的 N×6 偵測陣列；該格沒有偵測時為 `None`。
        content: 該路在推論尺度上的內容區（`letterbox.content_box`）。
        unscale: 該路的 `(scale, pad_x, pad_y)`（`letterbox.letterbox_params`）。

    Returns:
        `(tracks, foot_points)`，兩者都已映回原始解析度且逐列對應。
    """
    if box_data is None:
        # 空框也要走完整條路徑、照樣呼叫 `tracker.update`：BYTETracker 的 frame_id 與
        # 軌跡老化（`track_buffer` 到期就移除）都靠每格呼叫推進，跳過空格會讓已經離開
        # 畫面的人一直留在 lost 狀態、之後被錯誤地接回，而輸出檔本身完全正常。
        # `FootPointEstimator` 的 TTL 也是同理：它按「有軌跡的幀」計 tick，空格由
        # `estimate` 自己早退，不能由這裡代為跳過而改變 tick 的推進節奏。
        box_data = np.zeros((0, _EMPTY_DETECTION_COLUMNS), dtype=np.float32)
    detections = Boxes(torch.from_numpy(box_data), (INFER_HEIGHT, INFER_WIDTH))
    clip_to_content_inplace(detections.data, content)
    fbody_boxes, heads = split_detections(detections)
    tracks = tracker.update(stream_id, fbody_boxes)
    # 配對用 tracker 輸出的 Kalman 平滑框，不用 detection 框：落腳點才與同列寫進
    # parquet 的 bbox 自洽（tracker 回傳的 idx 也不是傳入陣列的索引，無法拿來回填，
    # 見 tracker.py 的 Returns 說明）
    foot_points = foot_estimator.estimate(stream_id, tracks, heads)
    unscale_boxes_inplace(tracks, *unscale)
    unscale_points_inplace(foot_points, *unscale)
    return tracks, foot_points


def _raise_on_sigterm(signum: int, _frame: FrameType | None) -> None:
    """SIGTERM 的 handler：拋 `SystemExit` 讓既有的清理路徑走得到。

    本進程等不到 `TRACK_FAILED` 的情況（推論進程被 SIGKILL、整機 OOM）最後都收斂成
    同一件事：父進程的 `_terminate_all` 送 SIGTERM。預設處置是直接結束進程，不走
    Python 的 `except`／`finally`，`.tmp` 就留在輸出目錄了。

    攔下來拋例外，`run_track_worker` 既有的 `except BaseException: collector.discard()`
    就會執行——**這裡不新增任何清理路徑，只是讓既有那條被走到**。阻塞中的
    `track_queue.get()` 也擋不住：signal handler 在主執行緒跑，拋出的例外從 `get()`
    內部往外傳（實測見 PR）。

    用 `SystemExit` 而不是自訂例外：`multiprocessing` 會把它的 code 直接當 exitcode，
    不印 traceback——關機路徑上那份 traceback 只是雜訊。128 + 訊號編號是 shell 對
    「被訊號終止」的慣例（SIGTERM → 143）。

    ⚠ 只涵蓋第一個 SIGTERM。清理途中再被送一次（或被 SIGKILL）仍會留下暫存檔，那條
    由下次執行認領 parts 目錄時清掉（`output_parts.claim_parts_dir`）。
    """
    raise SystemExit(128 + signum)


def run_track_worker(
    track_queue: mp.Queue,
    stream_names: list[str],
    frame_shapes: list[FrameShape],
    part_path: Path,
    shard_id: int,
    owned_stream_ids: frozenset[int],
) -> None:
    """追蹤子進程進入點：吃偵測框 → 追蹤與落腳點 → 映回原解析度 → 寫這一片的 part 檔。

    以 `mp.Process(target=run_track_worker, ...)` 於子進程執行，故參數需為可 pickle
    的型別。`TrackingResultCollector` 在本進程內建立而非由父進程傳入：它持有 parquet
    writer 的檔案 handle，跨進程傳遞行不通，而 `part_path` 是純量、傳過來就夠了。

    成功收到 `TRACK_DONE` 才 `collector.save()`（原子性 rename 成 part 檔名）；任何
    例外——含收到 `TRACK_FAILED`——都會先 `collector.discard()` 清掉暫存檔再重新拋出。
    part 檔缺一支，主進程的合併就 fail loud，正式檔名下不會出現不完整的結果。

    **本進程不認領輸出位置**：那把鎖（`parts/.lock`）由主進程在 `fork` 之前取得，本
    進程只是繼承了那個 fd（見 `services/output_parts.py`）。

    Args:
        track_queue: 推論進程送來的佇列，元素為 `to_payload` 的四元組，或
            `TRACK_DONE`／`TRACK_FAILED` 訊號。
        stream_names: **全部**攝影機的 `stream_dirname`，索引即 stream_id，同時作為
            `TrackingResultCollector` 記錄的 camera_id。各片都收完整清單而不是自己
            那幾路，是為了讓 stream_id 維持全域編號、payload 與索引都不必轉換；代價
            只是每片多建了幾個用不到的 `BYTETracker`（初始化只是幾個空 list）。
        frame_shapes: 各路的**原始** `FrameShape`，索引即 stream_id；逐列寫進追蹤結果
            parquet 供下游做解析度相關的參數換算，同時是把框與落腳點映射回原始解析度
            的依據（見 `services/letterbox.py`）。同樣是完整清單。
        part_path: 這一片的 part 檔路徑（`tracking_results.parts/shard<k>.parquet`）。
        shard_id: 這一片的編號，只進 log——N 行「追蹤進程結束」要分得出誰是誰。
        owned_stream_ids: 這一片負責的 stream_id 集合。收到不屬於自己的就拋錯：那是
            路由寫錯時**唯一**的訊號。`MultiStreamByteTracker.update` 對未知 stream_id
            回空陣列那條靜默路徑在這裡走不到（每片都建滿了全部路的 tracker），送錯片
            的 payload 會被正常追蹤、正常寫進這一片的 part，只是該路的軌跡被切成兩段
            而 `track_id` 分裂，而合併後的檔案完全正常。

    Raises:
        ValueError: 任一路的 `FrameShape` 等於推論尺寸（見 `_reject_inference_sized_shapes`）。
        RuntimeError: 推論進程回報 `TRACK_FAILED`，或收到不屬於本片的 stream_id。
        BaseException: 其他子系統拋出的例外，會原樣重新拋出。
    """
    _reject_inference_sized_shapes(frame_shapes)
    # 註冊在建 collector 之前：第一次 flush 一旦把 `.tmp` 建出來，這之後的每一步（含
    # 建構 tracker、落腳點推算器）被 SIGTERM 打斷都得走得到清理，否則那個窗口內收到
    # 訊號就會留下殘檔
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    # SIGINT 不改處置（Ctrl+C 照常以 KeyboardInterrupt 進到清理），只記下來，清理期間
    # 要連它一起擋掉
    previous_sigint = signal.getsignal(signal.SIGINT)
    # 在 try 內才建立，故先給 None：`except` 這時不會去 discard 一個還不存在的東西
    collector: TrackingResultCollector | None = None
    try:
        collector = TrackingResultCollector(part_path)
        tracker = MultiStreamByteTracker(num_streams=len(stream_names))
        # 跨格重用：它記著每條軌跡上一次成功推算的落腳點偏移量
        foot_estimator = FootPointEstimator(settings.foot_point.method)
        # 每路一組 (scale, pad_x, pad_y) 與一個內容區，都跟著 stream_id 走
        unscale_params = [
            letterbox_params(shape.height, shape.width) for shape in frame_shapes
        ]
        content_boxes = [
            content_box(shape.height, shape.width) for shape in frame_shapes
        ]
        fps_meter = FpsMeter()
        logger.info(
            "追蹤進程啟動",
            shard_id=shard_id,
            num_streams=len(stream_names),
            owned_stream_ids=sorted(owned_stream_ids),
        )
        # 計時從**第一個 payload 抵達**才起算，不含等推論進程載完 YOLO 權重的那段空窗：
        # 推論端的 `start` 也在載完模型之後（`start_loop` 內），兩邊口徑一致，餘裕才比得
        # 出意義。把空窗算進來會壓低 `overall_fps`，讓印出的餘裕系統性偏大——而那個數字
        # 是容量決策的依據，偏大的方向正好是會誤事的方向
        first_payload_at: float | None = None
        while True:
            item = track_queue.get()
            if item == TRACK_DONE:  # 推論進程正常送完整天影格
                break
            if item == TRACK_FAILED:
                # 推論進程中途例外，寧可什麼都不留也不寫出截斷的結果
                raise RuntimeError("推論進程中途例外結束，中止追蹤。")
            stream_id, box_data, frame_index, timestamp = item
            if stream_id not in owned_stream_ids:
                raise RuntimeError(
                    f"第 {shard_id} 片收到不屬於自己的 stream_id={stream_id}"
                    f"（負責的是 {sorted(owned_stream_ids)}），路由與實際送法對不上，"
                    "本次執行中止。照樣追蹤下去的話該路的軌跡會被切成兩段、track_id "
                    "分裂，而合併後的檔案完全正常，沒有任何其他訊號。"
                )
            track_start = time.perf_counter()
            if first_payload_at is None:
                first_payload_at = track_start
            tracks, foot_points = _track_one(
                tracker,
                foot_estimator,
                stream_id,
                box_data,
                content_boxes[stream_id],
                unscale_params[stream_id],
            )
            shape = frame_shapes[stream_id]
            collector.add(
                camera_id=stream_names[stream_id],
                frame_index=frame_index,
                timestamp=timestamp,
                tracks=tracks,
                foot_points=foot_points,
                frame_width=shape.width,
                frame_height=shape.height,
            )
            # 量的是**本進程每格的全部工作**（裁切、拆分、追蹤、落腳點、反算、累積結果），
            # 不只 `tracker.update`：這個進程接手成為新的候選瓶頸，決定它餘裕的是整段
            # 序列工作，只量其中一段會高估餘裕
            fps_meter.add_tracking_time(time.perf_counter() - track_start)
            fps_meter.record(stream_names[stream_id])  # 一格真的處理完了
        # 在 save 之前印，數字只反映純處理，且即使 save 失敗仍看得到
        elapsed = (
            0.0 if first_payload_at is None else time.perf_counter() - first_payload_at
        )
        _log_fps_summary(fps_meter, elapsed, shard_id)
        # 收尾的最後一步同樣不受理終止訊號：`save()` 關掉 writer 之後、rename 之前被打斷
        # 的話，暫存檔此刻已經是一份**完整**的 parquet，而 `except` 的 `discard()` 會把它
        # 刪掉——本來只要人工把 `.tmp` 改個名就救得回來的東西就沒了。`finally` 會還原
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        collector.save()  # 僅推論進程正常送完才原子性改名成正式檔名
    except BaseException:
        # 清理期間不再受理終止訊號。Ctrl+C 走的是 process group 的 SIGINT，本進程進到
        # 這裡開始 discard() 的同時，父進程也在 `_terminate_all` 送 SIGTERM，而使用者
        # 連按兩次 Ctrl+C 又會再來一個 SIGINT——任何一個在 `_writer.close()` 之後、
        # `unlink()` 之前打進來，整天的 `.tmp` 就留下了，而外層這個 except 已經進來過、
        # 不會再跑一次。清理很短（關 writer 再刪檔），真的卡住仍有父進程的 SIGKILL 收尾。
        # ⚠ 這是把窗口**縮到**「例外拋出到下面兩行之間」，不是關掉——用訊號就到這裡為止
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if collector is not None:
            collector.discard()  # fail-loud：不留下不完整結果
        raise
    finally:
        # handler 的存續期到「暫存檔不再會被寫」為止。還原是為了在測試中 in-process
        # 呼叫本函式時不留下副作用。**鎖不在這裡放**：本進程只是繼承了主進程的 fd，
        # 進程結束時 kernel 自然關掉它——提早關會讓「孤兒子進程仍守著鎖」那道保護少
        # 一個持有者
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def _log_fps_summary(
    fps_meter: FpsMeter, elapsed_seconds: float, shard_id: int
) -> None:
    """把這一片的吞吐與純處理速度印成一行。

    追蹤移出推論進程後，這個進程就是下一個候選瓶頸——實測改完之後它的餘裕只剩約 1.68
    倍，人流再多約 2.6 倍就換它當瓶頸。`tracking_fps`（純處理速度）除以 `overall_fps`
    （含等上游的實際吞吐）就是那個餘裕，所以兩個數字要一起印，只印其中一個看不出來。
    分片之後每片各印一行，**餘裕取各片的最小值**：整條 pipeline 的瓶頸是最慢的那片，
    平均會把它蓋掉。

    不印 `detection_fps`：本進程不做偵測，那個欄位必為 0，是誤導而不是缺值。
    """
    summary = fps_meter.summary(elapsed_seconds)
    logger.info(
        "追蹤進程結束",
        shard_id=shard_id,
        frames=summary.total_frames,
        elapsed_seconds=round(summary.elapsed_seconds, 1),
        overall_fps=round(summary.overall_fps, 2),
        tracking_fps=round(summary.tracking_fps, 2),
    )
