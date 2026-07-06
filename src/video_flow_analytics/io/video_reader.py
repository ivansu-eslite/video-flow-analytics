import multiprocessing as mp
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from video_flow_analytics.io.frame_ring import FrameRing

# 讀取進程中途例外時放入 queue 的錯誤訊號，讓推理進程與「正常讀完」的 None 區分開，
# 避免把中途崩潰誤判為該路已完整讀完而寫出截斷的結果。
READER_FAILED = "__READER_FAILED__"


@dataclass
class SegmentInfo:
    """一支影片片段。start 為檔名解析出的錄影起始時間（UTC），
    relpath 為相對 bucket 根的路徑（輸出影片會鏡射它）。"""

    path: Path
    start: datetime
    relpath: Path


@dataclass
class FramePacket:
    """讀取進程送往推理進程的單格資料，timestamp 由片段起始時間 + 幀序推得。"""

    frame: np.ndarray
    # 相對 bucket 根的路徑，輸出影片鏡射此路徑；fps 供逐片段開輸出檔用
    segment_relpath: str
    frame_index: int
    timestamp: datetime
    fps: float


def _parse_segment_start(path: Path, day: date) -> datetime:
    # 檔名格式 {HHmmss}.{SSS}Z.{ext}，Z 為 RFC 3339 的 UTC 標記
    stem = path.stem
    if not stem.endswith("Z"):
        raise ValueError(f"片段檔名不符合 HHmmss.SSSZ 格式: {path}")
    t = datetime.strptime(stem.removesuffix("Z"), "%H%M%S.%f")
    return datetime.combine(day, t.time(), tzinfo=timezone.utc)


def probe_frame_shape(segment: SegmentInfo) -> tuple[int, int]:
    """讀出片段首格以取得 (height, width)，供父進程一次配置該路的環形緩衝。

    假設單一攝影機整天解析度固定，故只探測第一支片段的首格即可。
    """
    cap = cv2.VideoCapture(str(segment.path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"無法開啟影片片段: {segment.path}")
    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"片段讀不到任何影格，無法探測解析度: {segment.path}")
        height, width = frame.shape[:2]
        return height, width
    finally:
        cap.release()


def discover_segments(
    bucket_dir: Path, stream_dirname: str, day: date, file_ext: str
) -> list[SegmentInfo]:
    """列出某攝影機在指定日期的所有片段，依起始時間排序。"""
    day_dir = bucket_dir / stream_dirname / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    if not day_dir.is_dir():
        return []
    segments = [
        SegmentInfo(
            path=p,
            start=_parse_segment_start(p, day),
            relpath=p.relative_to(bucket_dir),
        )
        for p in day_dir.glob(f"*.{file_ext}")
    ]
    segments.sort(key=lambda s: s.start)
    return segments


class DailyStreamVideoReader:
    """依時間序逐段讀取單一攝影機一整天的片段。

    影格不再逐格 pickle 進 queue，而是 memcpy 進共享環形緩衝的某個 slot：先從
    free_queue 領一個空 slot（無空 slot 時阻塞，形成對推理進程的天然背壓），寫入
    後只把「slot 索引 + metadata」丟進 data_queue。推理進程讀出 slot 後會把索引還回
    free_queue 供覆寫。
    """

    def __init__(
        self,
        stream_id: int,
        segments: list[SegmentInfo],
        data_queue: mp.Queue,
        free_queue: mp.Queue,
        ring: FrameRing,
    ):
        self.stream_id = stream_id
        self.segments = segments
        self.data_queue = data_queue
        self.free_queue = free_queue
        self.ring = ring

    def _read_segment(self, segment: SegmentInfo) -> None:
        cap = cv2.VideoCapture(str(segment.path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"無法開啟影片片段: {segment.path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(f"無法讀取影片 FPS: {segment.path}")
            relpath = str(segment.relpath)
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                slot = self.free_queue.get()  # 無空 slot 時阻塞（背壓）
                self.ring.write_slot(slot, frame)
                timestamp = segment.start + timedelta(seconds=frame_index / fps)
                self.data_queue.put((slot, relpath, frame_index, timestamp, fps))
                frame_index += 1
        finally:
            cap.release()

    def run(self) -> None:
        # free_queue 由 reader 自己在起跑時填滿（避免「父進程先 put 再 fork」在
        # mp.Queue feeder 執行緒上的競態）；推理進程只負責歸還，不會早於此執行。
        for slot in range(self.ring.num_slots):
            self.free_queue.put(slot)
        failed = False
        try:
            for segment in self.segments:
                self._read_segment(segment)
        except Exception:
            failed = True
            raise
        finally:
            # None 是給推理引擎的「正常讀完」結束訊號；例外時改送 READER_FAILED，
            # 讓推理進程能區分並中止，而非把這一路當成正常結束繼續寫出結果
            self.data_queue.put(READER_FAILED if failed else None)


def run_video_reader(
    stream_id: int,
    segments: list[SegmentInfo],
    data_queue: mp.Queue,
    free_queue: mp.Queue,
    ring_buffer,
    num_slots: int,
    height: int,
    width: int,
) -> None:
    ring = FrameRing(ring_buffer, num_slots, height, width)
    reader = DailyStreamVideoReader(stream_id, segments, data_queue, free_queue, ring)
    reader.run()
