import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

import numpy as np
from ultralytics.engine.results import Boxes
from vfa_observability import StructuredLogger

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.models.config import settings
from video_analyze.services.detector import YOLODetector
from video_analyze.services.foot_point import FootPointEstimator
from video_analyze.services.fps_meter import FpsMeter
from video_analyze.services.frame_ring import FrameRing
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    clip_to_content_inplace,
    letterbox_params,
    unscale_boxes_inplace,
    unscale_points_inplace,
)
from video_analyze.services.tracker import MultiStreamByteTracker
from video_analyze.services.tracking_results import TrackingResultCollector
from video_analyze.services.video_reader import (
    READER_DONE,
    READER_FAILED,
    FramePacket,
    FrameShape,
)

logger = StructuredLogger(component="inference")

# 影格不足目標批次時最多再等這麼久湊批（實測 batch 4→8 可讓每格推理 3.8ms→2.4ms）
_FILL_MAX_WAIT = 0.004
_FILL_POLL = 0.0005


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


class InferencePipeline:
    """推理進程主迴圈：非阻塞湊批 → YOLO 偵測 → 多路 ByteTrack → 收集結果。"""

    def __init__(
        self,
        stream_names: list[str],
        detector: YOLODetector,
        tracker: MultiStreamByteTracker,
        results_path: Path,
        frame_shapes: list[FrameShape],
    ):
        """組裝推理迴圈所需的各個子系統（偵測、追蹤、收集結果）。

        Args:
            stream_names: 各路攝影機的 `stream_dirname`，索引即 stream_id，
                同時作為 `TrackingResultCollector` 記錄的 camera_id。
            detector: 已載入模型的 YOLO 偵測器（跨批次重用）。
            tracker: 多路 ByteTrack 狀態管理器（跨批次重用，維持軌跡延續）。
            results_path: 追蹤結果 parquet 的目標路徑。
            frame_shapes: 各路的**原始** `FrameShape`，索引即 stream_id；逐列寫進
                追蹤結果 parquet 供下游做解析度相關的參數換算，同時是本迴圈把框與
                落腳點映射回原始解析度的依據（見 `services/letterbox.py`）。

        Raises:
            ValueError: 任一路的 `FrameShape` 剛好等於推論尺寸——那代表呼叫端傳進來
                的是縮放後的尺寸而非原始解析度，見下方註解。
        """
        # 影格在讀取端就縮成推論尺寸了，`frame_shapes` 卻**必須維持原始解析度**：傳成
        # 推論尺寸的話反算參數會退化成恆等（scale=1、pad=0），框與落腳點靜默停在
        # 640×384 尺度，而 parquet 的 frame_width 也一起寫成 640——下游 zone／line 只
        # 檢查這兩欄存在、不檢查值是否合理（ADR-004／ADR-006），幾何全部縮到約 1/3 而
        # 不報錯。正式來源 `probe_frame_shape` 是 16:9，永遠不會等於 5:3 的推論尺寸，
        # 故這道檢查不會擋到合法輸入
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
                "傳成推論尺寸會讓兩者一起靜默出錯。"
            )
        self.stream_names = stream_names
        self.frame_shapes = frame_shapes
        # 每路一組 (scale, pad_x, pad_y)，跟著 stream_id 走
        self._unscale_params = [
            letterbox_params(shape.height, shape.width) for shape in frame_shapes
        ]
        self.num_streams = len(stream_names)
        self.finished_streams = set()
        # 記住上一次湊批的起點，下一批從下一路開始繞一圈，避免固定從 0 起跑讓單一路
        # 供得上時就一路取到滿批、其餘路永遠輪不到（issue #100）
        self._next_stream_start = 0
        self.detector = detector
        self.tracker = tracker
        # 跨批次重用：它記著每條軌跡上一次成功推算的落腳點偏移量
        self.foot_estimator = FootPointEstimator(settings.foot_point.method)
        self.collector = TrackingResultCollector(results_path)
        self.fps_meter = FpsMeter()
        # ultralytics 對 in-memory list source 一次 forward 整個 list（batch=
        # 只對檔案來源的 LoadImagesAndVideos 有效），故單次 forward 實際批次為
        # settings.model.batch 的 2 倍；此處湊批目標維持現狀（見 detector.py
        # 移除 no-op 的 batch= kwarg 說明），未量測前不改行為
        self._target_batch = settings.model.batch * 2

    def _collect_batch(
        self,
        data_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> tuple[list[FramePacket], list[int], list[tuple[int, int]]]:
        # 影格以 view 免複製取用，slot 因此不在這裡歸還，改由呼叫端在 predict 完成後
        # 統一歸還（見 start_loop 的歸還迴圈）；在途影格數受環形緩衝 slot 數上限，
        # 不爆記憶體。刻意不收 free_queues：拿不到就不可能提早歸還，比只靠測試擋強。
        batch_packets: list[FramePacket] = []
        batch_stream_ids: list[int] = []
        # 本批取用、待推論完成後歸還的 (stream_id, slot)。與 packet 同進退，故空批時
        # 必然為空，呼叫端的空批分支不需要額外歸還
        held_slots: list[tuple[int, int]] = []
        fill_deadline: float | None = None
        order = [
            (self._next_stream_start + offset) % self.num_streams
            for offset in range(self.num_streams)
        ]
        self._next_stream_start = (self._next_stream_start + 1) % self.num_streams
        while len(batch_packets) < self._target_batch:
            progressed = False
            for stream_id in order:
                if stream_id in self.finished_streams:
                    continue
                data_queue = data_queues[stream_id]
                while len(batch_packets) < self._target_batch:
                    try:
                        item = data_queue.get_nowait()
                    except Empty:
                        break
                    progressed = True
                    if item == READER_DONE:  # 該路正常讀完
                        self.finished_streams.add(stream_id)
                        break
                    if item == READER_FAILED:
                        # 讀取進程中途例外，寧可中止整個推理迴圈也不寫出截斷的結果
                        raise RuntimeError(
                            f"讀取進程（stream_id={stream_id}）中途例外結束，中止推理。"
                        )
                    slot, frame_index, timestamp = item
                    frame = rings[stream_id].view_slot(slot)
                    held_slots.append((stream_id, slot))
                    batch_packets.append(
                        FramePacket(
                            frame=frame,
                            frame_index=frame_index,
                            timestamp=timestamp,
                        )
                    )
                    batch_stream_ids.append(stream_id)
            if len(batch_packets) >= self._target_batch:
                break
            if not progressed:
                if not batch_packets:
                    break  # 當下完全沒有資料，交回呼叫端短暫休眠
                # 已有部分影格：短暫等待，嘗試湊到較滿的批次再送 GPU
                now = time.perf_counter()
                if fill_deadline is None:
                    fill_deadline = now + _FILL_MAX_WAIT
                if now >= fill_deadline:
                    break
                time.sleep(_FILL_POLL)
        return batch_packets, batch_stream_ids, held_slots

    def start_loop(
        self,
        data_queues: list[mp.Queue],
        free_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> None:
        """執行推理主迴圈直到所有路都讀完，並負責結果的落盤/清理。

        成功跑完會 `collector.save()`（原子性 rename 成正式 parquet）；任何
        例外都會先 `collector.discard()` 清理不完整輸出，再重新拋出（fail-loud）。

        Args:
            data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
            free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
            rings: 各路的共享記憶體環形緩衝，索引為 stream_id。

        Raises:
            ValueError: 任一路的環形緩衝格數不足單次批次的 2 倍。
            RuntimeError: 任一路讀取進程回報 `READER_FAILED`。
            BaseException: 其他子系統拋出的例外，會原樣重新拋出。
        """
        # `frame_ring.RING_SLOTS` 的推導式已保證這條成立，所以這裡防的不是「有人調
        # batch」，而是日後兩處推導公式漂移：一批會同時扣住同一路最多 _target_batch
        # 個 slot 直到推論結束，總數不足其 2 倍時 reader 在整個推論期間都拿不到空位、
        # 完全停擺，而且不會有任何錯誤訊息
        if any(ring.num_slots < 2 * self._target_batch for ring in rings):
            raise ValueError(
                f"環形緩衝格數需至少為單次批次（{self._target_batch}）的 2 倍，"
                f"實得 {[ring.num_slots for ring in rings]}；"
                "推論完才歸還 slot 的設計下，格數不足會讓讀取進程停擺。"
            )
        logger.info("模組化推理流程啟動...")
        start = time.perf_counter()
        try:
            while len(self.finished_streams) < self.num_streams:
                batch_packets, batch_stream_ids, held_slots = self._collect_batch(
                    data_queues, rings
                )
                if not batch_packets:
                    # 所有 queue 當下都沒有資料，短暫休眠避免忙等待耗盡 CPU
                    time.sleep(0.001)
                    continue
                detect_start = time.perf_counter()
                results = self.detector.predict([p.frame for p in batch_packets])
                # predict 回傳時 boxes 已具體化到 CPU（隱含同步），此 wall time 已含
                # GPU 實際耗時，不需額外 cuda.synchronize()
                self.fps_meter.add_detection_time(time.perf_counter() - detect_start)
                # predict 的前處理已把影格 letterbox 成新陣列並上傳 GPU，共享記憶體
                # 可以放行了。歸還點卡在這裡的三個方向：
                # - 不能更早：predict 的前處理還在讀共享記憶體，早還會讓 reader 邊寫
                #   邊被讀。
                # - 不能更晚：下面的逐格 tracking 迴圈在 T4 上一批 16 格要數十 ms，
                #   拖過去只是白讓 reader 空等。
                # - `results[i].orig_img` 就是這些 slot 的 view（ultralytics
                #   `engine/results.py` 是 `self.orig_img = orig_img`，沒有 `.copy()`），
                #   歸還之後其內容不再可信。本迴圈不使用該欄位，但這是**本版消費路徑
                #   的性質，不是 ultralytics 的保證**：前處理已把像素複製兩次、後處理
                #   只取 `shape`、BYTETracker 收到的 `img` 是 None（`img` 只有 BOTSORT
                #   的 gmc 分支會用）、結果收集只用 `frame_index` 與 `timestamp`。日後
                #   開 `verbose=True`、呼叫 `plot()` 或升級 ultralytics 都可能讓
                #   `orig_img` 重新被讀到，屆時要改回取副本，見 ADR-010；歸還後把該
                #   欄位一併清成 None，就是為了讓那種情況當場炸出來。
                # - **ultralytics 內部還留著同一批 view 的別名**：`predictor.dataset.im0`
                #   與 `predictor.batch[1]` 存的就是我們傳進去的 list（`_single_check`
                #   對 numpy 原樣回傳），predictor 掛在 model 上、要到下一次 predict 才
                #   被替換。那兩處我們不清——去動別人的內部狀態，風險大於收益。因此下面
                #   這道保護擋的是「我們自己的程式碼日後誤用」，不是所有路徑。
                #
                # 刻意不包 try/finally：predict 或 READER_FAILED 拋出時 held_slots 不
                # 歸還，該路 reader 會卡在 free_queue.get()，但不會 hang——推理進程死亡
                # 後 pipeline.py 的 _raise_if_abnormal 偵測到非零 exitcode，
                # _terminate_all 會殺掉所有 reader。包起來得讓 held_slots 的作用域橫跨
                # _collect_batch 與本函式兩層，不值得。
                # 先切斷**我們自己持有**的兩個 slot 參照，再放行記憶體——順序不能對調：中間那段
                # 「slot 已歸還、參照還在」正是這道保護要消滅的狀態，而 zip 的 strict
                # 也可能在此拋錯。清成 None 讓「日後有人在歸還之後讀影格」從靜默讀到
                # 同一路幾格之後的畫面（內容正常、只是錯格，比對輸出也看不出來）變成
                # 當場拋錯。`orig_img` 清得掉是因為 Results 建構時已把 `orig_shape`
                # 另存一份，本迴圈之後只用 `boxes`（其座標系也綁在 `orig_shape` 上）；
                # strict 順帶釘住 predict 逐格回傳一個 result
                for packet, result in zip(batch_packets, results, strict=True):
                    packet.frame = None
                    result.orig_img = None
                for held_stream_id, held_slot in held_slots:
                    free_queues[held_stream_id].put(held_slot)
                for idx, stream_id in enumerate(batch_stream_ids):
                    packet = batch_packets[idx]
                    detections = results[idx].boxes.cpu()
                    # 先裁進填充帶以內，再拆分：改動前 ultralytics 在反算座標時就把框
                    # 裁進畫面了，少了這一步，4K 的框會在反算時外擴最多 8 px（每個推論
                    # 像素放大 6 倍），而 head 配對與 tracker 也會看到與改動前不同的框
                    _scale, pad_x, pad_y = self._unscale_params[stream_id]
                    clip_to_content_inplace(detections.data, pad_x, pad_y)
                    fbody_boxes, heads = split_detections(detections)
                    track_start = time.perf_counter()
                    tracks = self.tracker.update(stream_id, fbody_boxes)
                    self.fps_meter.add_tracking_time(time.perf_counter() - track_start)
                    # 配對用 tracker 輸出的 Kalman 平滑框，不用 detection 框：落腳點
                    # 才與同列寫進 parquet 的 bbox 自洽（tracker 回傳的 idx 也不是
                    # 傳入陣列的索引，無法拿來回填，見 tracker.py 的 Returns 說明）
                    foot_points = self.foot_estimator.estimate(stream_id, tracks, heads)
                    # 反算的位置只能在這裡：影格在讀取端就縮過，YOLO 與 tracker 全程在
                    # 推論尺度上工作，寫出去的座標卻必須是原始解析度（下游拿它跟攝影機
                    # 幾何比對）。**不可以移到 `estimate` 之前**——`heads` 是這裡唯一沒
                    # 反算的陣列，先把 `tracks` 換算回原始解析度會讓兩者尺度不一致，
                    # `_match_head` 全數回 None、每列退回框底邊中點，而列數、track_id、
                    # bbox 全部正常（ADR-009 要修掉的偏移就這樣靜默回來）。在推論尺度上
                    # 配對是安全的：三個判準（中心落在框內、面積比、主軸傾角）在等比縮放
                    # ＋等量平移下都不變
                    unscale_boxes_inplace(tracks, *self._unscale_params[stream_id])
                    unscale_points_inplace(
                        foot_points, *self._unscale_params[stream_id]
                    )
                    shape = self.frame_shapes[stream_id]
                    self.collector.add(
                        camera_id=self.stream_names[stream_id],
                        packet=packet,
                        tracks=tracks,
                        foot_points=foot_points,
                        frame_width=shape.width,
                        frame_height=shape.height,
                    )
                    self.fps_meter.record(self.stream_names[stream_id])
            # 在 save 之前印，數字只反映純處理，且即使 save 失敗仍看得到
            self._log_fps_summary(time.perf_counter() - start)
            self.collector.save()  # 僅全部串流跑完才原子性改名成正式檔名
        except BaseException:
            self.collector.discard()  # fail-loud：不留下不完整結果
            raise

    def _log_fps_summary(self, elapsed_seconds: float) -> None:
        """把處理 FPS 統計逐路、整體、階段各印一行。"""
        summary = self.fps_meter.summary(elapsed_seconds)
        for camera_id, fps in summary.per_camera_fps.items():
            logger.info(
                "FPS 逐路",
                camera_id=camera_id,
                frames=summary.per_camera_frames[camera_id],
                fps=round(fps, 2),
            )
        logger.info(
            "FPS 整體",
            total_frames=summary.total_frames,
            elapsed_seconds=round(summary.elapsed_seconds, 1),
            overall_fps=round(summary.overall_fps, 2),
        )
        logger.info(
            "FPS 階段",
            detection_fps=round(summary.detection_fps, 2),
            tracking_fps=round(summary.tracking_fps, 2),
        )
