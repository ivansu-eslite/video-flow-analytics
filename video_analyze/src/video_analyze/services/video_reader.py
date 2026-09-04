import multiprocessing as mp
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import av
import numpy as np
from av.codec.hwaccel import HWAccel
from vfa_observability import StructuredLogger

from video_analyze.services.frame_ring import FrameRing
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    letterbox_nv12,
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

# NVDEC 硬解解出來的畫面格式（`av_hwframe_transfer_data` 在 decode 當下就下載回主
# 記憶體）。縮放在這個格式的兩個平面上做，故讀到別的格式要擋下，見 `_read_segment`。
_HW_PIXEL_FORMAT = "nv12"

logger = StructuredLogger(component="video_reader")


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
        ValueError: 片段無法開啟、不含視訊串流，或讀不到任何影格。
    """
    try:
        container = av.open(str(segment.path))
    except av.FFmpegError as exc:
        raise ValueError(f"無法開啟影片片段: {segment.path}") from exc
    try:
        if not container.streams.video:
            # ffmpeg 開得起來但沒有視訊串流（例如只有音軌）時，`decode(video=0)` 會拋出
            # 不帶檔名的 `IndexError`；cv2 這種檔案是 `isOpened()` 為否，走 fail loud。
            raise ValueError(f"片段不含視訊串流: {segment.path}")
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError(f"片段讀不到任何影格，無法探測解析度: {segment.path}")
        height, width = frame.to_ndarray(format="bgr24").shape[:2]
        return FrameShape(height=height, width=width)
    finally:
        container.close()


def _stream_fps(stream, path: Path) -> float:
    """讀出視訊串流的 fps，讀不到就帶檔名 fail loud。

    `average_rate` 是容器標的 `avg_frame_rate`，某些檔案為 0／None；cv2 的
    `CAP_PROP_FPS` 這時會退回 `r_frame_rate`，`guessed_rate` 是同一個值，不接的話
    這路整天會以「無法讀取影片 FPS」中止。

    回傳 `float` 而非 `Fraction`：兩個消費端都吃不了 `Fraction`——`_read_segment` 拿它
    算 `timedelta`（會拋 `TypeError`），路由拿它當權重。

    `_read_segment` 與 `probe_stream_fps` 共用這一份：兩邊各寫一次的話，「fps 讀不到
    要怎麼辦」會分兩處漂移，而其中一處放寬（例如預設 30）會讓時間戳靜默算錯。

    Raises:
        ValueError: fps 缺值或非正數。
    """
    fps = stream.average_rate or stream.guessed_rate
    if not fps or fps <= 0:
        raise ValueError(f"無法讀取影片 FPS: {path}")
    return float(fps)


def probe_stream_fps(segment: SegmentInfo) -> float:
    """讀出片段標的 fps，**不解碼任何影格**（每次數十毫秒）。

    主進程用它算追蹤進程的路由權重（`services/output_parts.py` 的 `plan_routes`）：
    registry 沒有 fps 欄位，而各路的工作量與 fps 成正比。假設單一攝影機整天 fps 固定，
    故只探測第一支片段。

    **刻意不與 `probe_frame_shape` 合併**：後者回傳的 `FrameShape` 被 reader、
    collector、letterbox 三處消費，為了多帶一個 fps 去改它的形狀不划算。

    這裡不掛 `hwaccel`：只讀容器標頭，沒有影格要解。

    Args:
        segment: 要探測的片段（通常是當天第一支片段）。

    Returns:
        該路的 fps。

    Raises:
        ValueError: 片段無法開啟、不含視訊串流，或讀不到 fps。
    """
    try:
        container = av.open(str(segment.path))
    except av.FFmpegError as exc:
        raise ValueError(f"無法開啟影片片段: {segment.path}") from exc
    try:
        if not container.streams.video:
            raise ValueError(f"片段不含視訊串流: {segment.path}")
        return _stream_fps(container.streams.video[0], segment.path)
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

    影格先在解碼出的 nv12 平面上 letterbox 成推論尺寸（見 `services/letterbox.py`）再
    memcpy 進共享環形緩衝的 slot、queue 只傳「slot 索引 + metadata」，避免逐格 pickle。
    無空 slot 時阻塞，形成對推理進程的天然背壓。
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
        # 失敗當下正在讀哪一支，供 `run_video_reader` 的例外紀錄取用。**這行不可省**：
        # `run()` 進 for 之前還有填 `free_queue` 的一段，`segments` 為空時 for 也一次都
        # 不跑——沒有初始值的話，入口的 except handler 自己會拋 AttributeError 把根因蓋掉。
        # 刻意不記格序：那要逐格更新，而 `_read_segment` 的兩道逐格檢查訊息本來就帶格序
        self.current_segment: SegmentInfo | None = None

    def _read_segment(self, segment: SegmentInfo) -> None:
        """讀完單一片段的所有影格，逐格核對解析度與像素格式後縮放、寫入 slot。

        Raises:
            ValueError: 片段無法開啟、不含視訊串流、讀不到 FPS，或任一影格的解析度與
                `source_shape` 不符、像素格式不是 `nv12`（見迴圈內註解）。
        """
        try:
            container = av.open(
                str(segment.path),
                # `allow_software_fallback=False`：硬解不成立要在這裡直接拋出，不准
                # 靜默退回軟解——允許 fallback 的話，量到的效能數字會是另一個組態的
                # （同 D⑫ 的教訓，見 report.md 5.2「第二階段」）。與已驗證過的
                # `outputs/vfa_perf/code/bench_pyav_ctx.py:open_container` 同一組參數。
                hwaccel=HWAccel(device_type="cuda", allow_software_fallback=False),
            )
        except (av.FFmpegError, RuntimeError) as exc:
            # `allow_software_fallback=False` 且沒有任何串流吃得到硬解時，PyAV 18.1.0
            # 是從 `InputContainer` 的初始化路徑直接拋一個裸 `RuntimeError`
            # （"Hardware accelerated decode requested but no stream is compatible"，
            # `av/container/input.py`），不是 `av.FFmpegError` 的子類別——只接
            # `FFmpegError` 會讓這個失敗漏網，跳過帶檔名的 `ValueError`，這一路的
            # 對應片段就無從查起。
            raise ValueError(f"無法開啟影片片段: {segment.path}") from exc
        try:
            if not container.streams.video:
                # 同 `probe_frame_shape`：沒有視訊串流時要帶檔名擋下，不要讓
                # `streams.video[0]` 拋出不帶檔名的 `IndexError`。
                raise ValueError(f"片段不含視訊串流: {segment.path}")
            fps = _stream_fps(container.streams.video[0], segment.path)
            frame_index = 0
            for av_frame in container.decode(video=0):
                # 「整天解析度固定」的 fail-loud 檢查要在這裡做：縮放前移之前，這件事
                # 由 `FrameRing.write_slot` 的形狀檢查順便擋下（緩衝依首格解析度配置），
                # 但 letterbox 會把任何尺寸都抹平成推論尺寸，那道網就失效了。中途換
                # 解析度而沒擋下的後果全是靜默的：反算參數仍是首格那組，座標按錯誤的
                # 比例縮放（1080p→4K 差一半），parquet 的 frame_width 每列照樣寫首格
                # 的值，下游「該攝影機 frame_width 唯一」的檢查也照樣通過。
                if (av_frame.height, av_frame.width) != (
                    self.source_shape.height,
                    self.source_shape.width,
                ):
                    raise ValueError(
                        f"{segment.path} 第 {frame_index} 格的解析度 "
                        f"{av_frame.width}×{av_frame.height} 與該路探測到的 "
                        f"{self.source_shape.width}×{self.source_shape.height} 不符"
                        "（假設單一攝影機整天解析度固定）：座標反算與 parquet 的尺寸"
                        "欄位都綁在探測值上，中途變動只會讓輸出靜默算錯。"
                    )
                # 縮放吃的是 nv12 的平面佈局，格式不符就 fail loud。yuv420p（軟解的
                # 輸出）攤成 ndarray 的形狀與 nv12 完全相同，只是後半段是分開的 U、V
                # 兩塊而非交錯，當成 nv12 縮只會得到顏色錯亂的畫面而不會拋錯；10-bit
                # 來源的 p010 則連 dtype 都不同。`allow_software_fallback=False` 擋的
                # 是「整條退回軟解」，擋不到這裡。
                if av_frame.format.name != _HW_PIXEL_FORMAT:
                    raise ValueError(
                        f"{segment.path} 第 {frame_index} 格的像素格式為 "
                        f"{av_frame.format.name}，非預期的 {_HW_PIXEL_FORMAT}："
                        "讀取端的縮放綁在 nv12 的平面佈局上，換格式會讓畫面靜默錯亂。"
                    )
                # 縮到推論尺寸這件事本來在推論進程內由 ultralytics 做（每格 1.84 ms、
                # 佔該進程 8.5%），而那是序列瓶頸；N 個讀取進程各自做則是並行的。
                # 代價是框與落腳點的反算改由推論端負責，見 services/letterbox.py。
                # 縮放在 nv12 兩個平面上做、縮完才轉 BGR：搬動量只有 BGR 的一半，
                # 也省下對整張原始解析度畫面做色彩轉換。
                frame = letterbox_nv12(
                    av_frame.to_ndarray(), av_frame.height, av_frame.width
                )
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
            ValueError: 任一片段開檔／讀取 FPS 失敗，或影格解析度與探測值不符、
                像素格式不是 `nv12`（見 `_read_segment`）。
        """
        failed = False
        try:
            # free_queue 由 reader 自己起跑時填滿，避免「父進程先 put 再 fork」的競態。
            # **這一段要包在 try 內**：它失敗（queue 已關閉、資源吃緊時 feeder thread
            # 起不來）而 finally 沒送訊號的話，推理進程會一直阻塞在 `data_queue.get()`，
            # 只能等主進程掃到非零 exitcode 再 SIGTERM——與 `run_video_reader` 的組裝段
            # 補送 `READER_FAILED` 是同一個空窗
            for slot in range(self.ring.num_slots):
                self.free_queue.put(slot)
            for segment in self.segments:
                self.current_segment = segment
                self._read_segment(segment)
            # 讀完就清掉，否則收尾階段的失敗會指向一支已經讀完的片段
            self.current_segment = None
        except Exception:
            failed = True
            raise
        finally:
            # READER_DONE 是給推理引擎的「正常讀完」結束訊號；例外時改送 READER_FAILED，
            # 讓推理進程能區分並中止，而非把這一路當成正常結束繼續寫出結果
            self.data_queue.put(READER_FAILED if failed else READER_DONE)


def run_video_reader(
    stream_id: int,
    stream_name: str,
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
    而是與 `letterbox_nv12()` 綁在一起的固定約定。留成參數只會讓日後有人又傳回原始
    解析度。

    **失敗時先落一筆結構化 ERROR 再原樣往外拋**：`_read_segment` 內多數 fail-loud 檢查
    的訊息自帶片段路徑，但初始化段、以及不屬於那幾個 `ValueError` 的例外（解碼器的
    OSError、`FrameRing.write_slot` 的形狀錯誤）只會留下 `multiprocessing` 預設輸出的
    裸 traceback——與另外十幾個進程的輸出交錯在同一份 stderr，按行切割的日誌系統上還會
    散成十幾筆 DEFAULT 等級的紀錄。攝影機名在此之前根本沒傳進讀取進程（只有整數
    `stream_id`），所以 `stream_name` 是新加的參數而非從既有值推導。

    Args:
        stream_id: 該路攝影機的編號。
        stream_name: 該路攝影機的 `stream_dirname`，例外紀錄的 `camera` 欄位用。
            與 `stream_id` 相鄰擺放，兩個一起看才對得起來。
        segments: 當天要依序讀取的片段清單。
        data_queue: 送往推理進程的資料佇列。
        free_queue: 供推理進程歸還已消費 slot 的佇列。
        ring_buffer: `create_ring_buffer` 建立的共享記憶體。
        num_slots: 環形緩衝的 slot 數。
        source_shape: 該路的原始解析度（`probe_frame_shape` 的探測值）。**這不是緩衝
            的尺寸**（緩衝照推論尺寸配置），只用來逐格核對解析度沒有中途變動。
    """
    try:
        ring = FrameRing(ring_buffer, num_slots, INFER_HEIGHT, INFER_WIDTH)
        reader = DailyStreamVideoReader(
            stream_id, segments, data_queue, free_queue, ring, source_shape
        )
    except BaseException as exc:
        # 這一段**不**放行 `KeyboardInterrupt`（與下面 `reader.run()` 那段不同）：兩個
        # 建構子之間的窗口是微秒級、reader 又是最後才 fork 的一批，Ctrl+C 落在這裡的
        # 機率可忽略，而放行就得同時決定「不送 READER_FAILED 會不會讓推理進程卡住」。
        # 界線與 `pipeline.run_inference_pipeline` 的組裝段一致（那裡也是純
        # `except BaseException`）。代價是這個窗口內的 Ctrl+C 會留一筆 ERROR
        logger.exception(
            "讀取進程在初始化階段失敗",
            error=exc,
            stream_id=stream_id,
            camera=stream_name,
            # 這個階段還沒有片段可指，且訊息本身不斷言階段
            segment=None,
        )
        # 組裝失敗時 `run()` 的 finally 還沒有機會執行，這一路的結束訊號沒人送——推理
        # 進程會一直阻塞到父進程 `_terminate_all` 的 SIGTERM（`data_queue` 無 maxsize，
        # 這個 put 不會阻塞，也不需要 `TRACK_FAILED` 那種 timeout）。同
        # `pipeline.run_inference_pipeline` 組裝段的理由，也同它「只包組裝這一段」的界線：
        # 包到 `reader.run()` 會在同一次失敗送出兩個 READER_FAILED
        data_queue.put(READER_FAILED)
        raise
    try:
        reader.run()
    except KeyboardInterrupt:
        # 終端機 Ctrl+C 送給整個 process group，每一路 reader 都會在 `_read_segment` 的
        # 阻塞點收到它，而本 repo 把 Ctrl+C 當非錯誤路徑（`pipeline` 那側是 warning、
        # `main` 是 sys.exit(130)）。不先攔就會在使用者主動中斷時噴出 N 筆 severity=ERROR
        # 與 N 份 KeyboardInterrupt stacktrace，在雲端日誌上是假警報。相對地
        # `_terminate_all` 的 SIGTERM 對 reader 是預設處置（不像 `run_track_worker` 攔
        # 起來轉 SystemExit），不會進到 Python 的例外路徑，被父進程收掉不會多噴紀錄
        raise
    except BaseException as exc:
        # 這裡用 `.exception()`（完整 stacktrace），與 `detector.py` 記引擎 metadata 時
        # 「刻意不用 .exception()」那條決定不衝突：那筆是非致命的 WARNING、訊息可有可無，
        # 寧可短以免長行超過 PIPE_BUF 4096 被切斷；這筆是致命失敗路徑上的根因來源，被
        # 切斷的 traceback 仍優於沒有 traceback。交錯的機率也不同——那筆在啟動期九路引擎
        # metadata 同時在寫，這筆在失敗當下通常只有出事的那一路在寫
        seg = reader.current_segment
        logger.exception(
            "讀取進程失敗",
            error=exc,
            stream_id=stream_id,
            camera=stream_name,
            # 傳 `str(seg.path)` 而非 `seg`：`SegmentInfo` 是 dataclass 而非 pydantic
            # model，`_normalize_log_value` 會 fallback 到 `str(value)`，欄位值變成
            # dataclass repr——可讀但不好斷言，也不是路徑
            segment=str(seg.path) if seg else None,
        )
        raise
