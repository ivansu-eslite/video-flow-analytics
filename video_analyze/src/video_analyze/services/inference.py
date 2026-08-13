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
            frame_shapes: 各路的 `FrameShape`，索引即 stream_id；逐列寫進追蹤結果
                parquet，供下游做解析度相關的參數換算。
        """
        self.stream_names = stream_names
        self.frame_shapes = frame_shapes
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
        free_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> tuple[list[FramePacket], list[int]]:
        # slot 讀出後立即歸還 free_queue；在途影格數受環形緩衝 slot 數上限，不爆記憶體。
        batch_packets: list[FramePacket] = []
        batch_stream_ids: list[int] = []
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
                    frame = rings[stream_id].read_slot(slot)
                    free_queues[stream_id].put(slot)  # 立即歸還 slot 供 reader 覆寫
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
        return batch_packets, batch_stream_ids

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
            RuntimeError: 任一路讀取進程回報 `READER_FAILED`。
            BaseException: 其他子系統拋出的例外，會原樣重新拋出。
        """
        logger.info("模組化推理流程啟動...")
        start = time.perf_counter()
        try:
            while len(self.finished_streams) < self.num_streams:
                batch_packets, batch_stream_ids = self._collect_batch(
                    data_queues, free_queues, rings
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
                for idx, stream_id in enumerate(batch_stream_ids):
                    packet = batch_packets[idx]
                    fbody_boxes, heads = split_detections(results[idx].boxes.cpu())
                    track_start = time.perf_counter()
                    tracks = self.tracker.update(stream_id, fbody_boxes)
                    self.fps_meter.add_tracking_time(time.perf_counter() - track_start)
                    # 配對用 tracker 輸出的 Kalman 平滑框，不用 detection 框：落腳點
                    # 才與同列寫進 parquet 的 bbox 自洽（tracker 回傳的 idx 也不是
                    # 傳入陣列的索引，無法拿來回填，見 tracker.py 的 Returns 說明）
                    foot_points = self.foot_estimator.estimate(stream_id, tracks, heads)
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
