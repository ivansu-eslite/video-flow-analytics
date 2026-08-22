"""把 ByteTrack 比對、落腳點推算與結果落盤搬出推論進程，改由獨立子進程做。

推論進程是整條 pipeline 的序列瓶頸：解碼有九個進程並行，但湊批 → 推論 → 追蹤 → 寫檔
全部擠在同一個進程裡一件一件做，時間直接相加。追蹤實測每格 1.81 ms、佔該進程 8.4%，
而它與 GPU 推論之間沒有資料相依（下一批的推論不需要上一批的軌跡），可以重疊。

**跨進程傳的是偵測框而不是影格**：每格幾十個框、幾 KB，pickle 成本遠低於影格。影格在
推論完成後就沒有用途（環形緩衝的 slot 在推論完成當下就歸還了，見 ADR-010），追蹤只需
要框。座標一路都停在推論尺度上（影格在讀取端就縮好了），本模組在寫出前才映回原始解析
度——所以每路的反算參數與內容區也跟著搬到這裡。

推論進程中途例外時送 `TRACK_FAILED`，本進程收到就清掉 `.tmp` 暫存檔再拋錯；與
`video_reader.py` 的 `READER_DONE`／`READER_FAILED` 同一套設計——明確的訊號而非裸
`None`，讓「正常結束」與「上游崩潰」不可能被混為一談。

推論進程被 SIGKILL 或整機掛掉時收不到那個訊號：`track_queue` 的 pipe 寫入端 fd 被父進程
與九個讀取進程一起繼承，寫入端永遠不會全部關閉，`get()` 收不到 EOF、只會一直等
（issue #113）。暫存檔因此靠另外兩道，都不需要動 queue 的內部狀態：

- **本進程攔下 SIGTERM**。等下去的結局是父進程 `_terminate_all` 的 SIGTERM，而它是
  攔得到的——handler 拋出的例外會從 `track_queue.get()` 內部往外傳，走既有的
  `except BaseException: collector.discard()`。
- **啟動時認領輸出位置**（`tracking_results.claim_tmp_slot`）。整機重啟這種「兩個進程
  都沒機會執行清理」的情境沒有任何 in-process 的機制擋得住，只能由下一次寫同一條路徑
  的執行順手清掉。
"""

import multiprocessing as mp
import os
import signal
import time
from datetime import datetime
from pathlib import Path
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
from video_analyze.services.tracking_results import (
    TrackingResultCollector,
    claim_tmp_slot,
)
from video_analyze.services.video_reader import FrameShape

logger = StructuredLogger(component="track_worker")

# 推論進程送完所有影格後放入 queue 的結束訊號。與 READER_DONE 同一套設計：明確的訊號
# 而非裸 None，讓「正常結束」與「上游崩潰導致 queue 永遠不再有東西」能分開處理。
TRACK_DONE = "__TRACK_DONE__"

# 推論進程中途例外時放入 queue 的錯誤訊號。收到就清掉暫存檔並拋錯，避免把上游崩潰
# 當成正常結束而把截斷的結果 rename 成正式檔名。
TRACK_FAILED = "__TRACK_FAILED__"

# `to_payload` 對空框送 None（省下每格一次空陣列的 pickle），本模組收到後補回一個空的
# 偵測陣列——欄數與 ultralytics 的 `Boxes.data` 一致（xyxy + conf + cls）。
_EMPTY_DETECTION_COLUMNS = 6

# `track_queue` 的容量上限（payload 個數）。**這個上限是背壓，不是調校參數**——它擋的
# 兩件事都沒有其他機制在擋：
#
# - **backlog 無上限成長**。影格側的背壓是「reader 拿不到空 slot 就阻塞」，而 slot 在
#   predict 完成當下就歸還（ADR-010），所以那條保護只覆蓋到推論為止。追蹤搬出去之後，
#   追蹤只要比推論慢，payload 就會以 Python 物件的形式堆在推論進程裡（OS pipe 只緩衝
#   約 64 KB，其餘都在 feeder thread 的緩衝），整天數百萬格可以堆到 GB 級而全程沒有訊號。
#   給上限之後 `put` 會阻塞推論迴圈 → 推論不再消費 slot → reader 跟著阻塞，整條 pipeline
#   收斂到最慢的階段，與追蹤還在推論進程內時的行為一致。
# - **`TRACK_FAILED` 送不到**。它是排在同一條 FIFO 尾端的 in-band 訊號，送達延遲與
#   backlog 成正比；而推論進程一死，父進程約 0.5 秒內就 `_terminate_all`。backlog 大到
#   消化不完那幾秒的量時，本進程會在還沒讀到訊號時就被 SIGTERM 收掉。**暫存檔仍然清得
#   掉**（issue #113 之後 SIGTERM 有 handler，走的是同一個 `collector.discard()`），但
#   那條 in-band 的失效路徑等於沒作用，而且會把「上游崩潰」與「被 terminate」混成同一種
#   結束方式。有了上限，backlog 至多這麼多格——預設 batch（8，即上限 64 格）下約 70 ms
#   的工作量，遠小於父進程的偵測延遲，訊號趕得上。
#   ⚠ **這是「backlog 有界」的推論，不是時序保證**：上限隨 `MODEL__BATCH` 線性成長，而
#   父進程的偵測延遲不隨它成長，所以把 batch 調得很大時這個邊際會變薄（例如
#   `MODEL__BATCH=64` → 上限 512 格）。調大 batch 時要一併重新評估這條路徑。
#
# 取「單次推論批次的 4 倍」：一批推論完會連續 put 一整批，留幾批的鬆弛才不會讓正常抖動
# 變成兩個進程互等。`settings.model.batch` 的 ×2 與 `frame_ring.RING_SLOTS` 的第一個 ×2
# 同源——ultralytics 對 in-memory list source 一次 forward 整個 list，故單次批次是該值的
# 2 倍，即 `InferencePipeline._target_batch`；只調 batch 而沒同步調這裡，鬆弛會不足一批。
#
# 代價：追蹤進程先死時，推論進程的 `put` 會阻塞而不再是立即返回。這與 reader 卡在
# `free_queue.get()` 是同一種收斂方式——父進程的 `_raise_if_abnormal` 偵測到非零 exitcode
# 後 `_terminate_all` 收掉，不會 hang；而那種情況下暫存檔已由本進程自己的 `discard()` 清掉。
TRACK_QUEUE_SLOTS = settings.model.batch * 2 * 4


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
    由下次啟動的 `claim_tmp_slot` 收掉。
    """
    raise SystemExit(128 + signum)


def run_track_worker(
    track_queue: mp.Queue,
    stream_names: list[str],
    frame_shapes: list[FrameShape],
    results_path: Path,
) -> None:
    """追蹤子進程進入點：吃偵測框 → 追蹤與落腳點 → 映回原解析度 → 寫 parquet。

    以 `mp.Process(target=run_track_worker, ...)` 於子進程執行，故參數需為可 pickle
    的型別。`TrackingResultCollector` 在本進程內建立而非由父進程傳入：它持有 parquet
    writer 的檔案 handle，跨進程傳遞行不通，而 `results_path` 是純量、傳過來就夠了。

    成功收到 `TRACK_DONE` 才 `collector.save()`（原子性 rename 成正式 parquet）；任何
    例外——含收到 `TRACK_FAILED`——都會先 `collector.discard()` 清掉暫存檔再重新拋出。

    Args:
        track_queue: 推論進程送來的佇列，元素為 `to_payload` 的四元組，或
            `TRACK_DONE`／`TRACK_FAILED` 訊號。
        stream_names: 各路攝影機的 `stream_dirname`，索引即 stream_id，同時作為
            `TrackingResultCollector` 記錄的 camera_id。
        frame_shapes: 各路的**原始** `FrameShape`，索引即 stream_id；逐列寫進追蹤結果
            parquet 供下游做解析度相關的參數換算，同時是把框與落腳點映射回原始解析度
            的依據（見 `services/letterbox.py`）。
        results_path: 追蹤結果 parquet 的目標路徑。

    Raises:
        ValueError: 任一路的 `FrameShape` 等於推論尺寸（見 `_reject_inference_sized_shapes`）。
        RuntimeError: 推論進程回報 `TRACK_FAILED`，或這條輸出路徑的暫存檔正被另一個
            執行中的進程持有（見 `tracking_results.claim_tmp_slot`）。
        BaseException: 其他子系統拋出的例外，會原樣重新拋出。
    """
    _reject_inference_sized_shapes(frame_shapes)
    # 註冊在認領之前：`claim_tmp_slot` 一旦把 `.tmp` 建出來，這之後的每一步（含建構
    # tracker、落腳點推算器）被 SIGTERM 打斷都得走得到清理，否則那個窗口內收到訊號
    # 就會留下殘檔。⚠ 有一段仍蓋不到：`claim_tmp_slot` 內部從 `os.open` 到回傳之間，
    # `lock_fd` 與 `collector` 都還是 `None`，`except` 兩件事都不做，訊號落在那裡會留下
    # 一個 0 byte 的 `.tmp`。窗口不到毫秒、殘檔無害，且下次認領同一條路徑時會清掉
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    # SIGINT 不改處置（Ctrl+C 照常以 KeyboardInterrupt 進到清理），只記下來，清理期間
    # 要連它一起擋掉
    previous_sigint = signal.getsignal(signal.SIGINT)
    # 兩者都在 try 內才建立，故先給 None：認領失敗代表那個 `.tmp` 屬於另一個**執行中**
    # 的進程，此時 `collector` 還是 None，`except` 不會去 `discard()` 掉別人正在寫的檔
    lock_fd: int | None = None
    collector: TrackingResultCollector | None = None
    try:
        lock_fd = claim_tmp_slot(results_path)
        collector = TrackingResultCollector(results_path)
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
        logger.info("追蹤進程啟動", num_streams=len(stream_names))
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
        _log_fps_summary(fps_meter, elapsed)
        collector.save()  # 僅推論進程正常送完才原子性改名成正式檔名
    except BaseException:
        # 清理期間不再受理終止訊號。Ctrl+C 走的是 process group 的 SIGINT，本進程進到
        # 這裡開始 discard() 的同時，父進程也在 `_terminate_all` 送 SIGTERM，而使用者
        # 連按兩次 Ctrl+C 又會再來一個 SIGINT——任何一個在 `_writer.close()` 之後、
        # `unlink()` 之前打進來，整天的 `.tmp` 就留下了，而外層這個 except 已經進來過、
        # 不會再跑一次。清理很短（關 writer 再刪檔），真的卡住仍有父進程的 SIGKILL 收尾
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if collector is not None:
            collector.discard()  # fail-loud：不留下不完整結果
        raise
    finally:
        # handler 與鎖的存續期都到「暫存檔不再會被寫」為止。還原 handler 是為了在
        # 測試中 in-process 呼叫本函式時不留下副作用；關 fd 則是釋放 flock——正式執行
        # 時進程接著就結束了，kernel 也會做同一件事
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        if lock_fd is not None:
            os.close(lock_fd)


def _log_fps_summary(fps_meter: FpsMeter, elapsed_seconds: float) -> None:
    """把追蹤進程的吞吐與純處理速度印成一行。

    追蹤移出推論進程後，這個進程就是下一個候選瓶頸——實測改完之後它的餘裕只剩約 1.68
    倍，人流再多約 2.6 倍就換它當瓶頸。`tracking_fps`（純處理速度）除以 `overall_fps`
    （含等上游的實際吞吐）就是那個餘裕，所以兩個數字要一起印，只印其中一個看不出來。

    不印 `detection_fps`：本進程不做偵測，那個欄位必為 0，是誤導而不是缺值。
    """
    summary = fps_meter.summary(elapsed_seconds)
    logger.info(
        "追蹤進程結束",
        frames=summary.total_frames,
        elapsed_seconds=round(summary.elapsed_seconds, 1),
        overall_fps=round(summary.overall_fps, 2),
        tracking_fps=round(summary.tracking_fps, 2),
    )
