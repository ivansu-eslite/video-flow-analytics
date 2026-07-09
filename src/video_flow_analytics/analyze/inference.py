import logging
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

from video_flow_analytics.analyze.detector import YOLODetector
from video_flow_analytics.analyze.tracker import MultiStreamByteTracker
from video_flow_analytics.analyze.tracking_results import TrackingResultCollector
from video_flow_analytics.core.config import settings
from video_flow_analytics.io.frame_ring import FrameRing
from video_flow_analytics.io.video_reader import READER_DONE, READER_FAILED, FramePacket
from video_flow_analytics.io.video_writer import MultiStreamVideoWriter
from video_flow_analytics.visualization.visualizer import TrackAnnotator

logger = logging.getLogger(__name__)

# 影格不足目標批次時最多再等這麼久湊批（實測 batch 4→8 可讓每格推理 3.8ms→2.4ms）
_FILL_MAX_WAIT = 0.004
_FILL_POLL = 0.0005


class InferencePipeline:
    """推理進程主迴圈：非阻塞湊批 → YOLO 偵測 → 多路 ByteTrack →
    收集結果 → 標註 → 寫檔。
    """

    def __init__(
        self,
        stream_names: list[str],
        detector: YOLODetector,
        tracker: MultiStreamByteTracker,
        output_root: Path,
        results_path: Path,
    ):
        """組裝推理迴圈所需的各個子系統（偵測、追蹤、寫檔、收集結果）。

        Args:
            stream_names: 各路攝影機的 `stream_dirname`，索引即 stream_id，
                同時作為 `TrackingResultCollector` 記錄的 camera_id。
            detector: 已載入模型的 YOLO 偵測器（跨批次重用）。
            tracker: 多路 ByteTrack 狀態管理器（跨批次重用，維持軌跡延續）。
            output_root: 標註影片輸出根目錄。
            results_path: 追蹤結果 parquet 的目標路徑。
        """
        self.stream_names = stream_names
        self.num_streams = len(stream_names)
        self.finished_streams = set()
        self.detector = detector
        self.tracker = tracker
        self.writer = MultiStreamVideoWriter(output_root=output_root)
        self.annotator = TrackAnnotator()
        self.collector = TrackingResultCollector(results_path)
        # 湊約兩倍 batch 讓 ultralytics 組成完整批
        self._target_batch = settings.model.batch * 2

    def _collect_batch(
        self,
        data_queues: list[mp.Queue],
        free_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> tuple[list[FramePacket], list[int], list[int]]:
        # slot 讀出後立即歸還 free_queue；在途影格數受環形緩衝 slot 數上限，不爆記憶體。
        # 讀到 READER_DONE 只記進 newly_finished、不在此關閉 writer：本批已收但未寫出的
        # 同路影格若此時 close_stream，會被 writer 背景緒搶先關檔、之後補寫時重開檔案
        # 而截斷。
        batch_packets: list[FramePacket] = []
        batch_stream_ids: list[int] = []
        newly_finished: list[int] = []
        fill_deadline: float | None = None
        while len(batch_packets) < self._target_batch:
            progressed = False
            for stream_id in range(self.num_streams):
                if stream_id in self.finished_streams:
                    continue
                data_queue = data_queues[stream_id]
                while len(batch_packets) < self._target_batch:
                    try:
                        item = data_queue.get_nowait()
                    except Empty:
                        break
                    progressed = True
                    if item == READER_DONE:  # 該路正常讀完，close 延後（見上方說明）
                        self.finished_streams.add(stream_id)
                        newly_finished.append(stream_id)
                        break
                    if item == READER_FAILED:
                        # 讀取進程中途例外，寧可中止整個推理迴圈也不寫出截斷的結果
                        raise RuntimeError(
                            f"讀取進程（stream_id={stream_id}）中途例外結束，中止推理。"
                        )
                    slot, relpath, frame_index, timestamp, fps = item
                    frame = rings[stream_id].read_slot(slot)
                    free_queues[stream_id].put(slot)  # 立即歸還 slot 供 reader 覆寫
                    batch_packets.append(
                        FramePacket(
                            frame=frame,
                            segment_relpath=relpath,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            fps=fps,
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
        return batch_packets, batch_stream_ids, newly_finished

    def start_loop(
        self,
        data_queues: list[mp.Queue],
        free_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> None:
        """執行推理主迴圈直到所有路都讀完，並負責結果的落盤/清理。

        成功跑完會 `writer.close_all()` 並 `collector.save()`（原子性
        rename 成正式 parquet）；任何例外都會先 `collector.discard()` 與
        `writer.abort()` 清理不完整輸出，再重新拋出（fail-loud）。

        Args:
            data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
            free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
            rings: 各路的共享記憶體環形緩衝，索引為 stream_id。

        Raises:
            RuntimeError: 任一路讀取進程回報 `READER_FAILED`。
            BaseException: writer 背景執行緒或其他子系統拋出的例外，會原樣
                重新拋出。
        """
        logger.info("模組化推理流程啟動...")
        try:
            while len(self.finished_streams) < self.num_streams:
                batch_packets, batch_stream_ids, newly_finished = self._collect_batch(
                    data_queues, free_queues, rings
                )
                if not batch_packets:
                    if newly_finished:
                        # 沒有影格但有路剛讀完（例如空批同時收到 READER_DONE），
                        # 仍要關其 writer
                        for stream_id in newly_finished:
                            self.writer.close_stream(stream_id)
                        continue
                    # 所有 queue 當下都沒有資料，短暫休眠避免忙等待耗盡 CPU
                    time.sleep(0.001)
                    continue
                results = self.detector.predict([p.frame for p in batch_packets])
                for idx, stream_id in enumerate(batch_stream_ids):
                    packet = batch_packets[idx]
                    tracks = self.tracker.update(stream_id, results[idx].boxes)
                    self.collector.add(
                        camera_id=self.stream_names[stream_id],
                        packet=packet,
                        tracks=tracks,
                    )
                    annotated_frame = self.annotator.draw_bboxes(packet.frame, tracks)
                    self.writer.write(
                        stream_id, packet.segment_relpath, annotated_frame, packet.fps
                    )
                # 本批已全部 write()，才可安全 close_stream（避免截斷，見上方說明）
                for stream_id in newly_finished:
                    self.writer.close_stream(stream_id)
            self.writer.close_all()  # 會把 writer 背景緒的中途例外重拋到這裡
            self.collector.save()  # 僅全部串流（含編碼）跑完才原子性改名成正式檔名
        except BaseException:
            self.collector.discard()  # fail-loud：不留下不完整結果
            self.writer.abort()
            raise
