import multiprocessing as mp
import time
from collections import deque
from queue import Empty

from vfa_observability import StructuredLogger

from video_analyze.services.batching import TARGET_BATCH
from video_analyze.services.detector import PIPELINE_DEPTH, YOLODetector
from video_analyze.services.fps_meter import FpsMeter
from video_analyze.services.frame_ring import FrameRing
from video_analyze.services.track_worker import (
    TRACK_DONE,
    TRACK_FAILED,
    TRACK_SIGNAL_PUT_TIMEOUT,
    fanout_track_signal,
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
    """推理進程主迴圈：非阻塞湊批 → YOLO 偵測 → 把偵測框送往它歸屬的追蹤進程。

    追蹤、落腳點推算、座標反算與 parquet 落盤都不在這裡——它們搬到獨立進程了
    （見 `services/track_worker.py`），本類只負責「湊批 → 推論 → 把框丟出去」。
    追蹤進程有 N 個、各自負責一組固定的攝影機，所以「丟出去」多了一步查表。

    偵測那一步是**兩批深度的流水**（ADR-014）：`submit` 把一批排進 GPU 就回來，主迴圈
    接著去湊下一批、歸還 slot、送上一批的 payload，`collect` 才取回結果。所以本迴圈
    自己要記「送出去但還沒回來的那幾批分別是哪些影格」——`_pending` 與偵測器的在途
    佇列同進退，序號由 `submit` 給、`collect` 回傳時核對。
    """

    def __init__(
        self,
        stream_names: list[str],
        detector: YOLODetector,
        track_queues: list[mp.Queue],
        route: list[int],
    ):
        """組裝推理迴圈所需的各個子系統（偵測、湊批、送往各片追蹤進程）。

        Args:
            stream_names: 各路攝影機的 `stream_dirname`，索引即 stream_id。
            detector: 已載入模型的 YOLO 偵測器（跨批次重用）。
            track_queues: 各片追蹤進程的佇列，索引即片編號；元素格式見
                `track_worker.to_payload`。
            route: 各路歸屬的片編號，索引即 stream_id（`output_parts.plan_routes`
                在主進程算好後傳進來，整天不變）。

        Raises:
            ValueError: `route` 的長度與路數不符，或指到不存在的片——兩者都會在跑起來
                之後才以 `IndexError` 炸在每格都會走到的路徑上。
        """
        self.stream_names = stream_names
        self.num_streams = len(stream_names)
        if len(route) != self.num_streams:
            raise ValueError(
                f"route 有 {len(route)} 筆，與路數 {self.num_streams} 不符。"
            )
        out_of_range = [k for k in route if not 0 <= k < len(track_queues)]
        if out_of_range:
            raise ValueError(
                f"route 指到不存在的片 {sorted(set(out_of_range))}，"
                f"目前只有 {len(track_queues)} 片。"
            )
        self.finished_streams = set()
        # 記住上一次湊批的起點，下一批從下一路開始繞一圈，避免固定從 0 起跑讓單一路
        # 供得上時就一路取到滿批、其餘路永遠輪不到（issue #100）
        self._next_stream_start = 0
        self.detector = detector
        self.track_queues = track_queues
        self.route = route
        self.fps_meter = FpsMeter()
        self._target_batch = TARGET_BATCH
        # 已 submit、還沒 collect 的批次：(序號, 該批影格, 該批各格的 stream_id)
        self._pending: deque[tuple[int, list[FramePacket], list[int]]] = deque()

    def _collect_batch(
        self,
        data_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> tuple[list[FramePacket], list[int], list[tuple[int, int]]]:
        # 影格以 view 免複製取用，slot 因此不在這裡歸還，改由呼叫端在**前處理**完成後
        # 統一歸還（見 start_loop 的歸還迴圈）；在途影格數受環形緩衝 slot 數上限，
        # 不爆記憶體。刻意不收 free_queues：拿不到就不可能提早歸還，比只靠測試擋強。
        batch_packets: list[FramePacket] = []
        batch_stream_ids: list[int] = []
        # 本批取用、待前處理完成後歸還的 (stream_id, slot)。與 packet 同進退，故空批時
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
            # profile 取 max batch = 匯出時的 `batch`）。超過的話會炸在
            # `TrtRunner.enqueue` 的 `set_input_shape`——那條訊息看得出形狀與上限，但
            # 看不出是 `[model].batch` 與引擎對不上，而這兩者是分別維護的。在這裡先擋，
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
                    # 當下沒有新影格。有在途批就趁這時把它收回來——不收的話那一批的
                    # payload 要等到下一批湊成才送得出去，而供料本來就是斷續的
                    if self.detector.in_flight:
                        self._collect_and_dispatch()
                    else:
                        time.sleep(0.001)  # 完全沒事做，休眠避免忙等待耗盡 CPU
                    continue
                if self.detector.in_flight >= PIPELINE_DEPTH:
                    # 流水滿了才收，收的是最舊那一批。排在湊批**之後**，讓 GPU 在 host
                    # 湊批的那段時間仍有一批在算
                    self._collect_and_dispatch()
                submit_start = time.perf_counter()
                seq = self.detector.submit([p.frame for p in batch_packets])
                # `submit` 只量到這裡：中間的歸還是幾個 queue.put，不計入 detection_time
                # ——它的口徑是「submit 與 collect 的 wall time 合計」，兩邊同一把尺
                self.fps_meter.add_detection_time(time.perf_counter() - submit_start)
                # `submit` 回來時像素已經複製進 pinned buffer（`stack_frames`），共享
                # 記憶體可以放行了。歸還點卡在這裡的兩個方向：
                # - 不能更早：`np.stack` 還在讀共享記憶體，早還會讓 reader 邊寫邊被讀。
                # - 不能更晚：歸還越晚 reader 空等越久。它從「整批推論完成」前移到這裡
                #   （ADR-013），讓 reader 早一整段 forward 的時間拿回空位；流水化之後
                #   那段等待更長，前移的價值也更大（ADR-014）。
                # 影格參照只剩我們自己持有的 `packet.frame` 要切斷：推論輸出不再攜帶
                # 影格（`Results` 已經不在正式路徑上），也就沒有 ultralytics 內部的活
                # 別名（ADR-010 Decision 4 記的擋不住的那條，見 ADR-013）。清成 None 讓
                # 「日後有人在歸還之後讀影格」從靜默讀到同一路幾格之後的畫面（內容正常、
                # 只是錯格，比對輸出也看不出來）變成當場拋錯；這一步與下面的歸還不可
                # 對調，中間那段「slot 已歸還、參照還在」正是它要消滅的狀態。
                #
                # 刻意不包 try/finally：submit／collect 或 READER_FAILED 拋出時
                # held_slots 不歸還，該路 reader 會卡在 free_queue.get()，但不會 hang
                # ——推理進程死亡後 pipeline.py 的 _raise_if_abnormal 偵測到非零
                # exitcode，_terminate_all 會殺掉所有 reader。包起來得讓 held_slots 的
                # 作用域橫跨 _collect_batch 與本函式兩層，不值得。
                for packet in batch_packets:
                    packet.frame = None
                for held_stream_id, held_slot in held_slots:
                    free_queues[held_stream_id].put(held_slot)
                self._pending.append((seq, batch_packets, batch_stream_ids))
            # 所有路都讀完了，把還在 GPU 上的那幾批收乾淨——漏掉的話那些格不會有
            # payload，而 parquet 少幾百格完全看不出來
            while self.detector.in_flight:
                self._collect_and_dispatch()
            self._log_fps_summary(time.perf_counter() - start)
            # 落盤改由追蹤進程負責（parquet 的內容都在那邊產生）。這個訊號是它唯一的
            # 正常結束途徑，不送的話那邊會一直等在 queue 上——**每一片都要送到**，漏掉
            # 哪一片就會缺一支 part 檔而讓主進程的合併 fail loud
            fanout_track_signal(self.track_queues, TRACK_DONE)
        except BaseException:
            # 本進程已經沒有 collector 可以 discard 了，改用訊號請各片代為清理：
            # 不送的話它們會一直等在 queue.get() 上，直到被父進程 terminate 才收尾——
            # 那條路徑也會清掉 `.tmp`（追蹤進程攔了 SIGTERM，見
            # `services/track_worker.py`），但要多等父進程的偵測延遲，中間再被 SIGKILL
            # 就得留到下一次執行才清。這裡帶 timeout：已經死掉的那片若 queue 是滿的，
            # 阻塞的 put 會讓**後面的片一個都收不到**
            fanout_track_signal(
                self.track_queues, TRACK_FAILED, timeout=TRACK_SIGNAL_PUT_TIMEOUT
            )
            raise

    def _collect_and_dispatch(self) -> None:
        """取回最舊那一批的偵測結果，核對序號，逐格送往它歸屬的追蹤進程。

        **序號核對是這條流水的主要保護**。`collect` 回傳的是偵測器在途佇列最前面那一
        批，`_pending` 最前面應該是同一批——兩者一旦錯開，第 k 批的框就會配到第 k+1
        批的影格，而輸出檔完全正常，只有時間戳對到隔壁批。這種錯法逐值比對也只表現為
        配對率下降，看不出是錯配還是偵測變了，所以要在當場擋。

        Raises:
            RuntimeError: 序號與 `_pending` 的最舊一批對不上。
        """
        collect_start = time.perf_counter()
        seq, detections = self.detector.collect()
        # 口徑是「submit 與 collect 的 wall time 合計」：collect 內等 GPU 的時間算在
        # 內，被 host 工作蓋掉的那段 GPU 時間不算。流水化之後 detection_fps 因此不可
        # 與改動前的數字直接相減（見 video_analyze/README.md）
        self.fps_meter.add_detection_time(time.perf_counter() - collect_start)
        expected_seq, batch_packets, batch_stream_ids = self._pending.popleft()
        if seq != expected_seq:
            raise RuntimeError(
                f"偵測器回傳第 {seq} 批的結果，但主迴圈等的是第 {expected_seq} 批。"
                "這代表兩邊的在途佇列已經錯開，框會配到別批的影格——輸出檔會完全"
                "正常，只有時間戳對到隔壁批。"
            )
        # strict 順帶釘住 collect 逐格回傳一個結果
        for packet, stream_id, boxes in zip(
            batch_packets, batch_stream_ids, detections, strict=True
        ):
            # 追蹤、落腳點推算、座標反算與寫 parquet 都在追蹤進程做（實測追蹤每格
            # 1.81 ms、佔本進程 8.4%，而它與下一批的 GPU 推論之間沒有資料相依）。這裡
            # 只把該格的全部偵測框丟出去，含 head——拆分也在那邊做。依路由送給該路歸屬
            # 的那一片。送錯片不會有任何直接症狀——那片也有全部路的 tracker，會照樣
            # 追蹤、照樣寫進自己的 part，只是該路的軌跡被切成兩段而 track_id 分裂；
            # 唯一的訊號是追蹤進程入口的歸屬檢查（見 `track_worker.run_track_worker`）
            self.track_queues[self.route[stream_id]].put(
                to_payload(
                    stream_id,
                    boxes,
                    packet.frame_index,
                    packet.timestamp,
                )
            )
            # 本進程的口徑是「該路的一格已推論完並送出」，不是「一格已完整處理完」
            # （那件事只有追蹤進程知道，它自己也記一份）。這個計數同時是 detection_fps
            # 與 overall_fps 的分子，拿掉會讓那兩個數字變成 0
            self.fps_meter.record(self.stream_names[stream_id])

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
