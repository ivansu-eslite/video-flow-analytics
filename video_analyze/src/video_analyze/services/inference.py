import multiprocessing as mp
import time
from queue import Empty

from vfa_observability import StructuredLogger

from video_analyze.services.batching import TARGET_BATCH
from video_analyze.services.detector import YOLODetector
from video_analyze.services.fps_meter import FpsMeter
from video_analyze.services.frame_ring import FrameRing
from video_analyze.services.track_worker import (
    TRACK_DONE,
    TRACK_FAILED,
    to_payload,
)
from video_analyze.services.video_reader import (
    READER_DONE,
    READER_FAILED,
    FramePacket,
)

logger = StructuredLogger(component="inference")

# 影格不足目標批次時最多再等這麼久湊批（實測 batch 4→8 可讓每格推理 3.8ms→2.4ms）
_FILL_MAX_WAIT = 0.004
_FILL_POLL = 0.0005


class InferencePipeline:
    """推理進程主迴圈：非阻塞湊批 → YOLO 偵測 → 把偵測框送往追蹤進程。

    追蹤、落腳點推算、座標反算與 parquet 落盤都不在這裡——它們搬到獨立進程了
    （見 `services/track_worker.py`），本類只負責「湊批 → 推論 → 把框丟出去」。
    """

    def __init__(
        self,
        stream_names: list[str],
        detector: YOLODetector,
        track_queue: mp.Queue,
    ):
        """組裝推理迴圈所需的各個子系統（偵測、湊批、送往追蹤進程）。

        Args:
            stream_names: 各路攝影機的 `stream_dirname`，索引即 stream_id。
            detector: 已載入模型的 YOLO 偵測器（跨批次重用）。
            track_queue: 送往追蹤進程的佇列，元素格式見 `track_worker.to_payload`。
        """
        self.stream_names = stream_names
        self.num_streams = len(stream_names)
        self.finished_streams = set()
        # 記住上一次湊批的起點，下一批從下一路開始繞一圈，避免固定從 0 起跑讓單一路
        # 供得上時就一路取到滿批、其餘路永遠輪不到（issue #100）
        self._next_stream_start = 0
        self.detector = detector
        self.track_queue = track_queue
        self.fps_meter = FpsMeter()
        self._target_batch = TARGET_BATCH

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
        """執行推理主迴圈直到所有路都讀完，並在結束或例外時通知追蹤進程。

        落盤不在這裡：正常跑完送 `TRACK_DONE`，追蹤進程收到才 `save()`；中途例外送
        `TRACK_FAILED` 讓它清掉不完整的暫存檔，再把例外重新拋出（fail-loud）。
        **本函式內的例外都送得到訊號**；在此之前就失敗的路徑（例如
        `run_inference_pipeline` 建 `YOLODetector` 時拋錯）不由本函式覆蓋，改由該函式
        自己的 `except` 送出（見 `services/pipeline.py`）。

        Args:
            data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
            free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
            rings: 各路的共享記憶體環形緩衝，索引為 stream_id。

        Raises:
            ValueError: 單次批次超過引擎綁的最大批次。
            RuntimeError: 任一路讀取進程回報 `READER_FAILED`。
            BaseException: 其他子系統拋出的例外，會原樣重新拋出。
        """
        logger.info("模組化推理流程啟動...")
        start = time.perf_counter()
        try:
            # 引擎的 batch 維度上限是建置時綁死的（dynamic 引擎的 optimization
            # profile 取 max batch = 匯出時的 `batch`）。超過的話 ultralytics 的
            # TensorRT backend 會在 `forward` 的 assert 失敗——那條訊息只講「input
            # size 不等於 model size」，看不出是批次設定與引擎對不上。在這裡先擋，
            # 訊息才指得出要改哪一邊。放在 try 內而非之前：追蹤進程此時已經起來、
            # 等在 queue 上，要送得出 TRACK_FAILED，不必等父進程 SIGTERM。
            #
            # 檢查放在推理端而不是 detector 內：湊到幾格是本模組決定的
            # （`_collect_batch`），detector 只負責把拿到的 list 送進引擎
            if (
                self.detector.max_batch is not None
                and self._target_batch > self.detector.max_batch
            ):
                raise ValueError(
                    f"單次推理批次 {self._target_batch}（[model].batch）超過引擎綁的"
                    f"最大批次 {self.detector.max_batch}。請把 [model].batch 調到 "
                    f"{self.detector.max_batch} 以下，或用 "
                    "`tools/build_engine.py --batch` 重建一顆容得下的引擎。"
                )
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
                # - 不能更晚：追蹤搬進獨立進程後這裡只剩「把框丟進 queue」（每格幾十
                #   微秒），拖過去的代價比從前小，但方向不變——歸還越晚，reader 空等
                #   越久。
                # - `results[i].orig_img` 就是這些 slot 的 view（ultralytics
                #   `engine/results.py` 是 `self.orig_img = orig_img`，沒有 `.copy()`），
                #   歸還之後其內容不再可信。本迴圈不使用該欄位，但這是**本版消費路徑
                #   的性質，不是 ultralytics 的保證**：前處理已把像素複製兩次、後處理
                #   只取 `shape`、送往追蹤進程的 payload 只取 `boxes.data`（見
                #   `track_worker.to_payload`，影格不跨進程）。日後開 `verbose=True`、
                #   呼叫 `plot()` 或升級 ultralytics 都可能讓 `orig_img` 重新被讀到，
                #   屆時要改回取副本，見 ADR-010；歸還後把該欄位一併清成 None，就是
                #   為了讓那種情況當場炸出來。
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
                # 送 payload 排在歸還之後：`to_payload` 取的是 `results[idx].boxes`
                # （推論輸出，不是影格），與 slot 無關，歸還之後仍可取
                for idx, stream_id in enumerate(batch_stream_ids):
                    packet = batch_packets[idx]
                    # 追蹤、落腳點推算、座標反算與寫 parquet 都在追蹤進程做（實測追蹤
                    # 每格 1.81 ms、佔本進程 8.4%，而它與下一批的 GPU 推論之間沒有資料
                    # 相依）。這裡只把該格的全部偵測框丟出去，含 head——拆分也在那邊做。
                    self.track_queue.put(
                        to_payload(
                            stream_id,
                            results[idx].boxes,
                            packet.frame_index,
                            packet.timestamp,
                        )
                    )
                    # 本進程的口徑是「該路的一格已推論完並送出」，不是「一格已完整處理
                    # 完」（那件事只有追蹤進程知道，它自己也記一份）。這個計數同時是
                    # detection_fps 與 overall_fps 的分子，拿掉會讓那兩個數字變成 0
                    self.fps_meter.record(self.stream_names[stream_id])
            self._log_fps_summary(time.perf_counter() - start)
            # 落盤改由追蹤進程負責（parquet 的內容都在那邊產生）。這個訊號是它唯一的
            # 正常結束途徑，不送的話那邊會一直等在 queue 上
            self.track_queue.put(TRACK_DONE)
        except BaseException:
            # 本進程已經沒有 collector 可以 discard 了，改用訊號請追蹤進程代為清理：
            # 不送的話它會一直等在 queue.get() 上，直到被父進程 terminate 才收尾——那條
            # 路徑也會清掉 `.tmp`（追蹤進程攔了 SIGTERM，見 `services/track_worker.py`），
            # 但要多等父進程的偵測延遲，中間再被 SIGKILL 就得留到下一次執行才清
            self.track_queue.put(TRACK_FAILED)
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
        # 不印 tracking_fps：追蹤已搬到獨立進程，本進程量不到它，印出來永遠是 0，
        # 那是誤導而不是缺值。該進程結束時會自己印一行（見 track_worker.py）
        logger.info(
            "FPS 階段",
            detection_fps=round(summary.detection_fps, 2),
        )
