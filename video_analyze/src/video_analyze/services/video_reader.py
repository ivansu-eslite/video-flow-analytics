import multiprocessing as mp
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import av
import numpy as np

from video_analyze.services.frame_ring import FrameRing
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    letterbox,
)

# 讀取進程正常讀完整天片段時放入 queue 的結束訊號；與 READER_FAILED 對稱，讓推理進程
# 能明確區分「正常讀完」與「中途例外」兩種結束，而非依賴裸 None。
READER_DONE = "__READER_DONE__"

# 讀取進程中途例外時放入 queue 的錯誤訊號，讓推理進程與 READER_DONE 區分開，
# 避免把中途崩潰誤判為該路已完整讀完而寫出截斷的結果。
READER_FAILED = "__READER_FAILED__"

# 檔名的 "Z" 尾綴依 RFC 3339 為真正的 UTC；解析後於來源（ingestion）即轉換成
# 台北在地時間，讓下游 parquet / zone_counts / report 一律以台北 wall-clock 處理。
_FILENAME_TZ = timezone.utc
_LOCAL_TZ = ZoneInfo("Asia/Taipei")


class FrameShape(NamedTuple):
    """一路串流的影像尺寸。

    刻意用 NamedTuple 而非 tuple：欄位順序是 `(height, width)`（沿用 numpy 的
    `frame.shape`），寫進 parquet 時若把兩者對調不會有型別錯誤，只會讓下游的解析度
    換算靜默算錯。呼叫端一律用 `.height` / `.width` 取值，讓寫反變成 attribute 錯誤。
    仍是 tuple，既有的 `height, width = shape` 解包與 `mp.Process` 的 pickle 都不受影響。
    """

    height: int
    width: int


@dataclass
class SegmentInfo:
    """一支影片片段。

    Attributes:
        path: 片段檔案的完整路徑。
        start: 檔名解析出的錄影起始時間（檔名為 UTC，已轉換為台北時間 UTC+8，
            見 `_FILENAME_TZ` / `_LOCAL_TZ`）。
    """

    path: Path
    start: datetime


@dataclass
class FramePacket:
    """讀取進程送往推理進程的單格資料。

    Attributes:
        frame: 影格畫面（BGR，已在讀取端縮成推論尺寸）。推理進程取的是共享記憶體的
            view（見 `FrameRing.view_slot`），**歸還 slot 之後會被設成 `None`**——
            那之後該記憶體隨時會被 reader 覆寫，讓存取直接拋錯比靜默讀到別格畫面好。
        frame_index: 該影格在所屬片段內的序號（從 0 起算）。
        timestamp: 由片段起始時間 + 幀序（`frame_index / fps`）推得的時間戳。
    """

    frame: np.ndarray | None
    frame_index: int
    timestamp: datetime


def _parse_segment_start(path: Path, day: date) -> datetime:
    """解析片段檔名為台北在地時間的起始時間。

    Args:
        path: 片段檔案路徑，檔名須為 `{HHmmss}.{SSS}Z.{ext}`。
        day: 該片段所在目錄的 `{YYYY}/{MM}/{DD}` 日期，語義為 **UTC 曆日**
            （與轉換後 `start` 的台北曆日不同語義，見下方 Raises）。

    Returns:
        轉換為台北在地時間（`Asia/Taipei`）的起始時間。

    Raises:
        ValueError: 檔名不符合 `HHmmss.SSSZ` 格式；或轉換後的台北曆日與
            `day`（UTC 曆日）不同——`day` 來自目錄結構、為 UTC 曆日，`start`
            為台北時間，UTC 16:00 之後兩者會分岔到不同曆日，此時代表片段
            落在錯誤的輸出日期目錄下，須 fail-loud 而非靜默寫錯天。
    """
    # 檔名格式 {HHmmss}.{SSS}Z.{ext}：依 RFC 3339，"Z" 尾綴為真正的 UTC；先以 UTC
    # 結合當日日期，再轉成台北在地時間（見模組層級 _FILENAME_TZ / _LOCAL_TZ 註解）。
    stem = path.stem
    if not stem.endswith("Z"):
        raise ValueError(f"片段檔名不符合 HHmmss.SSSZ 格式: {path}")
    t = datetime.strptime(stem.removesuffix("Z"), "%H%M%S.%f")
    start = datetime.combine(day, t.time(), tzinfo=_FILENAME_TZ).astimezone(_LOCAL_TZ)
    if start.date() != day:
        raise ValueError(
            f"片段 {path} 的錄影時間轉換台北時區後跨到 {start.date()}，與目錄"
            f"日期 {day}（UTC 曆日）不同：目錄結構以 UTC 曆日分層，但輸出的"
            f"timestamp 為台北在地時間，兩者語義不同——UTC 16:00 之後的片段"
            f"會落在錯誤的輸出日期目錄下，須避免此片段時間範圍。"
        )
    return start


def probe_frame_shape(segment: SegmentInfo) -> FrameShape:
    """讀出片段首格以取得該路的**原始**影像尺寸。

    假設單一攝影機整天解析度固定，故只探測第一支片段的首格即可。

    影格在讀取端就被縮成推論尺寸，環形緩衝也照那個尺寸配置，所以這裡回傳的值**不再
    用來配置緩衝**，剩下的兩個用途都需要原始解析度：逐列寫進 `tracking_results.parquet`
    的 `frame_width`／`frame_height`（下游 zone／line 兩包換算 1080p 基準像素的唯一
    來源，見 ADR-004／ADR-006），以及餵 `letterbox_params` 算該路的反算參數。**不可
    改成回傳推論尺寸**——parquet 會靜默寫成 640×384，而兩個下游只檢查欄位存在。

    Args:
        segment: 要探測的片段（通常是當天第一支片段）。

    Returns:
        該路的 `FrameShape(height, width)`。

    Raises:
        ValueError: 片段無法開啟，或讀不到任何影格。
    """
    try:
        container = av.open(str(segment.path))
    except av.FFmpegError as exc:
        raise ValueError(f"無法開啟影片片段: {segment.path}") from exc
    try:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError(f"片段讀不到任何影格，無法探測解析度: {segment.path}")
        height, width = frame.to_ndarray(format="bgr24").shape[:2]
        return FrameShape(height=height, width=width)
    finally:
        container.close()


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
        ValueError: 任一片段檔名不符合 `HHmmss.SSSZ` 命名格式，或轉換台北時區後
            跨到與 `day` 不同的曆日（見 `_parse_segment_start`）。本函式在主進程
            逐攝影機呼叫，故任一路踩到都會讓當天整批中止。
    """
    day_dir = bucket_dir / stream_dirname / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    if not day_dir.is_dir():
        return []
    segments = [
        SegmentInfo(path=p, start=_parse_segment_start(p, day))
        for p in day_dir.glob(f"*.{file_ext}")
    ]
    segments.sort(key=lambda s: s.start)
    return segments


class DailyStreamVideoReader:
    """依時間序逐段讀取單一攝影機一整天的片段。

    影格先 letterbox 成推論尺寸（見 `services/letterbox.py`）再 memcpy 進共享環形緩衝的
    slot、queue 只傳「slot 索引 + metadata」，避免逐格 pickle。無空 slot 時阻塞，形成對
    推理進程的天然背壓。
    """

    def __init__(
        self,
        stream_id: int,
        segments: list[SegmentInfo],
        data_queue: mp.Queue,
        free_queue: mp.Queue,
        ring: FrameRing,
        source_shape: FrameShape,
    ):
        """綁定該路要讀取的片段清單與 IPC 通道，尚未開始實際讀取。

        Args:
            stream_id: 該路攝影機的編號。
            segments: 當天要依序讀取的片段清單（需已依起始時間排序）。
            data_queue: 送往推理進程的資料佇列（存放 slot 索引與 metadata）。
            free_queue: 供推理進程歸還已消費 slot 的佇列。
            ring: 該路專用的共享記憶體環形緩衝。
            source_shape: `probe_frame_shape` 探測到的原始解析度，逐格核對用
                （見 `_read_segment`）。
        """
        self.stream_id = stream_id
        self.segments = segments
        self.data_queue = data_queue
        self.free_queue = free_queue
        self.ring = ring
        self.source_shape = source_shape

    def _read_segment(self, segment: SegmentInfo) -> None:
        """讀完單一片段的所有影格，逐格核對解析度後縮放、寫入 slot。

        Raises:
            ValueError: 片段無法開啟、讀不到 FPS，或任一影格的解析度與
                `source_shape` 不符（見迴圈內註解）。
        """
        try:
            container = av.open(str(segment.path))
        except av.FFmpegError as exc:
            raise ValueError(f"無法開啟影片片段: {segment.path}") from exc
        try:
            stream = container.streams.video[0]
            fps = stream.average_rate
            if not fps or fps <= 0:
                raise ValueError(f"無法讀取影片 FPS: {segment.path}")
            fps = float(fps)
            frame_index = 0
            for av_frame in container.decode(video=0):
                frame = av_frame.to_ndarray(format="bgr24")
                # 「整天解析度固定」的 fail-loud 檢查要在這裡做：縮放前移之前，這件事
                # 由 `FrameRing.write_slot` 的形狀檢查順便擋下（緩衝依首格解析度配置），
                # 但 letterbox 會把任何尺寸都抹平成推論尺寸，那道網就失效了。中途換
                # 解析度而沒擋下的後果全是靜默的：反算參數仍是首格那組，座標按錯誤的
                # 比例縮放（1080p→4K 差一半），parquet 的 frame_width 每列照樣寫首格
                # 的值，下游「該攝影機 frame_width 唯一」的檢查也照樣通過。
                if frame.shape[:2] != (
                    self.source_shape.height,
                    self.source_shape.width,
                ):
                    raise ValueError(
                        f"{segment.path} 第 {frame_index} 格的解析度 "
                        f"{frame.shape[1]}×{frame.shape[0]} 與該路探測到的 "
                        f"{self.source_shape.width}×{self.source_shape.height} 不符"
                        "（假設單一攝影機整天解析度固定）：座標反算與 parquet 的尺寸"
                        "欄位都綁在探測值上，中途變動只會讓輸出靜默算錯。"
                    )
                # 縮到推論尺寸這件事本來在推論進程內由 ultralytics 做（每格 1.84 ms、
                # 佔該進程 8.5%），而那是序列瓶頸；N 個讀取進程各自做則是並行的。
                # 代價是框與落腳點的反算改由推論端負責，見 services/letterbox.py。
                frame = letterbox(frame)
                slot = self.free_queue.get()  # 無空 slot 時阻塞（背壓）
                self.ring.write_slot(slot, frame)
                timestamp = segment.start + timedelta(seconds=frame_index / fps)
                self.data_queue.put((slot, frame_index, timestamp))
                frame_index += 1
        finally:
            container.close()

    def run(self) -> None:
        """依序讀完 `self.segments` 所有片段，並在結束或例外時發出結束訊號。

        正常讀完送 `READER_DONE`；中途例外送 `READER_FAILED` 並重新拋出例外，讓
        推理進程能區分兩者、避免把中途崩潰誤判為正常結束繼續寫出結果。

        Raises:
            ValueError: 任一片段開檔／讀取 FPS 失敗，或影格解析度與探測值不符
                （見 `_read_segment`）。
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
            # READER_DONE 是給推理引擎的「正常讀完」結束訊號；例外時改送 READER_FAILED，
            # 讓推理進程能區分並中止，而非把這一路當成正常結束繼續寫出結果
            self.data_queue.put(READER_FAILED if failed else READER_DONE)


def run_video_reader(
    stream_id: int,
    segments: list[SegmentInfo],
    data_queue: mp.Queue,
    free_queue: mp.Queue,
    ring_buffer,
    num_slots: int,
    source_shape: FrameShape,
) -> None:
    """讀取子進程的進入點：建構 `FrameRing` 與 `DailyStreamVideoReader` 並執行。

    尺寸不由呼叫端傳入，直接取 `INFER_*`：`_read_segment` 寫進 slot 之前已把影格
    letterbox 成推論尺寸，緩衝的尺寸因此不再是「該路的解析度」這種 per-stream 屬性，
    而是與 `letterbox()` 綁在一起的固定約定。留成參數只會讓日後有人又傳回原始解析度。

    Args:
        stream_id: 該路攝影機的編號。
        segments: 當天要依序讀取的片段清單。
        data_queue: 送往推理進程的資料佇列。
        free_queue: 供推理進程歸還已消費 slot 的佇列。
        ring_buffer: `create_ring_buffer` 建立的共享記憶體。
        num_slots: 環形緩衝的 slot 數。
        source_shape: 該路的原始解析度（`probe_frame_shape` 的探測值）。**這不是緩衝
            的尺寸**（緩衝照推論尺寸配置），只用來逐格核對解析度沒有中途變動。
    """
    ring = FrameRing(ring_buffer, num_slots, INFER_HEIGHT, INFER_WIDTH)
    reader = DailyStreamVideoReader(
        stream_id, segments, data_queue, free_queue, ring, source_shape
    )
    reader.run()
