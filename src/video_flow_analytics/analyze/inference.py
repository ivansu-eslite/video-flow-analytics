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
from video_flow_analytics.io.video_reader import READER_FAILED, FramePacket
from video_flow_analytics.io.video_writer import MultiStreamVideoWriter
from video_flow_analytics.visualization.visualizer import TrackAnnotator

logger = logging.getLogger(__name__)

# 湊批：當下可取的影格不足目標批次時，最多再等這麼久嘗試湊更多，讓 YOLO 吃到較滿
# 的批次（實測 batch=4 → 8 可讓每格推理從 ~3.8ms 降到 ~2.4ms）。等待期以此輪詢間隔
# 短暫休眠。離線批次處理不在意單格延遲，這點等待換來的批次效率是划算的。
_FILL_MAX_WAIT = 0.004
_FILL_POLL = 0.0005


class InferencePipeline:
    def __init__(
        self,
        stream_names: list[str],
        detector: YOLODetector,
        tracker: MultiStreamByteTracker,
        output_root: Path,
        results_path: Path,
    ):
        self.stream_names = stream_names
        self.num_streams = len(stream_names)
        self.finished_streams = set()
        self.detector = detector
        self.tracker = tracker
        self.writer = MultiStreamVideoWriter(output_root=output_root)
        self.annotator = TrackAnnotator()
        self.collector = TrackingResultCollector(results_path)
        # 每次推理湊到約兩個模型批次量再送 predict，讓 ultralytics 內部能組成完整批
        self._target_batch = settings.model.batch * 2

    def _collect_batch(
        self,
        data_queues: list[mp.Queue],
        free_queues: list[mp.Queue],
        rings: list[FrameRing],
    ) -> tuple[list[FramePacket], list[int], list[int]]:
        # 逐路非阻塞取出「slot 索引 + metadata」，從共享環形緩衝把該格複製成私有陣列
        # 後，立刻把 slot 還回 free_queue 供 reader 覆寫。累積到 _target_batch 就送出；
        # 不足時最多再等 _FILL_MAX_WAIT 嘗試湊更多，以維持 GPU 批次效率。記憶體不會
        # 因追著 reader 而爆——在途影格數受環形緩衝 slot 數硬性上限。
        #
        # 讀到某路的 None（正常讀完）時，只記進 newly_finished、不在此關閉該路 writer：
        # 本批 batch_packets 裡先前收進來的、屬於這一路的影格還沒寫出（要等 collect
        # 返回後才 writer.write）。若在這裡就 close_stream，writer 執行緒會先收尾關檔，
        # 之後那些影格再 write 會重開同一個檔、把它截斷。故延到本批影格全部寫完再關。
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
                    if item is None:
                        # None 是讀取進程放入的結束訊號，代表該路已正常讀完；writer 延到
                        # 本批影格寫完後再關（見上方說明），這裡只標記結束、停止輪詢該路
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
        logger.info("模組化推理流程啟動...")
        try:
            while len(self.finished_streams) < self.num_streams:
                batch_packets, batch_stream_ids, newly_finished = self._collect_batch(
                    data_queues, free_queues, rings
                )
                if not batch_packets:
                    if newly_finished:
                        # 沒有影格但有路剛讀完（例如空批同時收到 None），仍要關其 writer
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
                # 本批影格已全部寫入各自的 writer queue，這時才安全地關閉剛讀完的路，
                # 避免上面尚未寫出的影格在 writer 收尾後又被重開寫入、把檔案截斷
                for stream_id in newly_finished:
                    self.writer.close_stream(stream_id)
            # 編碼在背景 writer 執行緒進行，收尾要等它們寫完；close_all() 也會把
            # 任一路 writer 執行緒的中途例外重拋到這裡，讓 parquet 走 discard 而非 save
            self.writer.close_all()
            # 僅在全部串流正常跑完（含影片編碼）時才把結果原子性地改名成正式檔名
            self.collector.save()
        except BaseException:
            # 中途例外（含 Ctrl+C、writer 執行緒失敗）：清掉尚未改名成正式檔名的
            # 暫存 parquet，並停掉所有 writer 執行緒，寧可不留下不完整結果（fail-loud）
            self.collector.discard()
            self.writer.abort()
            raise
