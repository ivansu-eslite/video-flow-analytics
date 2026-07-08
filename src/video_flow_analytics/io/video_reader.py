import multiprocessing as mp
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from video_flow_analytics.io.frame_ring import FrameRing

# 讀取進程中途例外時放入 queue 的錯誤訊號，讓推理進程與「正常讀完」的 None 區分開，
# 避免把中途崩潰誤判為該路已完整讀完而寫出截斷的結果。
READER_FAILED = "__READER_FAILED__"

# 攝影機錄影時鐘本身就是台北時間，並非真正的 UTC；即使檔名格式的 "Z" 尾綴
# 看起來像 RFC 3339 的 UTC 標記，實際寫入的 wall-clock 值就是台北時間。
_RECORDING_TZ = ZoneInfo("Asia/Taipei")


@dataclass
class SegmentInfo:
    """一支影片片段。

    Attributes:
        path: 片段檔案的完整路徑。
        start: 檔名解析出的錄影起始時間（標記為台北時間 UTC+8，見
            `_RECORDING_TZ`）。
        relpath: 相對 bucket 根的路徑（輸出影片會鏡射它）。
    """

    path: Path
    start: datetime
    relpath: Path


@dataclass
class FramePacket:
    """讀取進程送往推理進程的單格資料。

    Attributes:
        frame: 影格畫面（BGR）。
        segment_relpath: 相對 bucket 根的路徑，輸出影片鏡射此路徑。
        frame_index: 該影格在所屬片段內的序號（從 0 起算）。
        timestamp: 由片段起始時間 + 幀序（`frame_index / fps`）推得的時間戳。
        fps: 所屬片段的影格率，供逐片段開輸出檔用。
    """

    frame: np.ndarray
    segment_relpath: str
    frame_index: int
    timestamp: datetime
    fps: float


def _parse_segment_start(path: Path, day: date) -> datetime:
    # 檔名格式 {HHmmss}.{SSS}Z.{ext}：格式沿用 RFC 3339 的 "Z" 尾綴排版，但實際
    # 錄影時鐘是台北時間（見模組層級 _RECORDING_TZ 註解），不可標記成 timezone.utc。
    stem = path.stem
    if not stem.endswith("Z"):
        raise ValueError(f"片段檔名不符合 HHmmss.SSSZ 格式: {path}")
    t = datetime.strptime(stem.removesuffix("Z"), "%H%M%S.%f")
    return datetime.combine(day, t.time(), tzinfo=_RECORDING_TZ)


def probe_frame_shape(segment: SegmentInfo) -> tuple[int, int]:
    """讀出片段首格以取得 (height, width)，供父進程一次配置該路的環形緩衝。

    假設單一攝影機整天解析度固定，故只探測第一支片段的首格即可。

    Args:
        segment: 要探測的片段（通常是當天第一支片段）。

    Returns:
        `(height, width)`。

    Raises:
        ValueError: 片段無法開啟，或讀不到任何影格。
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
    """列出某攝影機在指定日期的所有片段，依起始時間排序。

    Args:
        bucket_dir: bucket 根目錄。
        stream_dirname: 攝影機目錄名（`<location>_<camera_id>`）。
        day: 要列出的日期。
        file_ext: 片段檔案副檔名（不含點號）。

    Returns:
        依起始時間排序的 `SegmentInfo` 清單；當天目錄不存在時回傳空清單。

    Raises:
        ValueError: 任一片段檔名不符合 `HHmmss.SSSZ` 命名格式。
    """
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

    影格 memcpy 進共享環形緩衝的 slot、queue 只傳「slot 索引 + metadata」，避免逐格
    pickle。無空 slot 時阻塞，形成對推理進程的天然背壓。
    """

    def __init__(
        self,
        stream_id: int,
        segments: list[SegmentInfo],
        data_queue: mp.Queue,
        free_queue: mp.Queue,
        ring: FrameRing,
    ):
        """綁定該路要讀取的片段清單與 IPC 通道，尚未開始實際讀取。

        Args:
            stream_id: 該路攝影機的編號。
            segments: 當天要依序讀取的片段清單（需已依起始時間排序）。
            data_queue: 送往推理進程的資料佇列（存放 slot 索引與 metadata）。
            free_queue: 供推理進程歸還已消費 slot 的佇列。
            ring: 該路專用的共享記憶體環形緩衝。
        """
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
        """依序讀完 `self.segments` 所有片段，並在結束或例外時發出結束訊號。

        正常讀完送 `None`；中途例外送 `READER_FAILED` 並重新拋出例外，讓
        推理進程能區分兩者、避免把中途崩潰誤判為正常結束繼續寫出結果。

        Raises:
            ValueError: 任一片段開檔或讀取 FPS 失敗（見 `_read_segment`）。
        """
        # free_queue 由 reader 自己起跑時填滿，避免「父進程先 put 再 fork」的競態
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
    """讀取子進程的進入點：建構 `FrameRing` 與 `DailyStreamVideoReader` 並執行。

    Args:
        stream_id: 該路攝影機的編號。
        segments: 當天要依序讀取的片段清單。
        data_queue: 送往推理進程的資料佇列。
        free_queue: 供推理進程歸還已消費 slot 的佇列。
        ring_buffer: `create_ring_buffer` 建立的共享記憶體。
        num_slots: 環形緩衝的 slot 數。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。
    """
    ring = FrameRing(ring_buffer, num_slots, height, width)
    reader = DailyStreamVideoReader(stream_id, segments, data_queue, free_queue, ring)
    reader.run()
