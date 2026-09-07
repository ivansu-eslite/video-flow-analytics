import datetime
import queue
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from video_analyze.services import video_reader
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.video_reader import FrameShape, _parse_segment_start


def test_frame_shape_unpacks_as_height_width():
    """欄位順序是 `(height, width)`（沿用 numpy 的 `frame.shape`）。

    環形緩衝改照推論尺寸配置之後，這份尺寸剩下的兩個消費端都是靜默的：寫進 parquet
    的 `frame_width`／`frame_height`，以及 `letterbox_params(height, width)` 算出的
    座標反算參數。順序若被調換，兩邊都不會有型別錯誤，只會讓下游的解析度換算與反算
    出來的座標一起算錯。
    """
    shape = FrameShape(height=1080, width=1920)

    height, width = shape
    assert (height, width) == (1080, 1920)
    assert shape.height == 1080
    assert shape.width == 1920

_TAIPEI = ZoneInfo("Asia/Taipei")


def test_parse_segment_start_converts_utc_filename_to_taipei():
    # 檔名的 "Z" 為真正的 UTC；錄影窗起點 03:00Z 應轉成台北 11:00（+08:00），
    # 而非把 03:00 直接當成台北 wall-clock（舊邏輯會得到 03:00，此測試會擋下）。
    start = _parse_segment_start(
        Path("loc_cam/2026/07/08/030000.000Z.mkv"), datetime.date(2026, 7, 8)
    )
    assert start.utcoffset() == datetime.timedelta(hours=8)
    assert start.replace(tzinfo=None) == datetime.datetime(2026, 7, 8, 11, 0)


def test_parse_segment_start_end_of_recording_window_stays_same_taipei_day():
    # 錄影窗終點 14:00Z → 台北 22:00，仍落在同一台北曆日（無跨日）。
    start = _parse_segment_start(
        Path("loc_cam/2026/07/08/140000.000Z.mkv"), datetime.date(2026, 7, 8)
    )
    assert start == datetime.datetime(2026, 7, 8, 22, 0, tzinfo=_TAIPEI)


def test_parse_segment_start_rejects_non_z_suffix():
    with pytest.raises(ValueError):
        _parse_segment_start(
            Path("loc_cam/2026/07/08/030000.000.mkv"), datetime.date(2026, 7, 8)
        )


def test_parse_segment_start_rejects_when_taipei_day_crosses_dir_day():
    # 16:00Z 之後轉台北時間會跨到目錄日期（UTC 曆日）的隔天，
    # 與輸出目錄日期分岔，須 fail-loud 而非靜默寫到錯誤的日期目錄。
    with pytest.raises(ValueError, match="跨到"):
        _parse_segment_start(
            Path("loc_cam/2026/07/08/160000.000Z.mkv"), datetime.date(2026, 7, 8)
        )


class _FakeFormat:
    """`av` 影格的像素格式替身；讀取端只用到 `name`。"""

    def __init__(self, name):
        self.name = name


class _FakeAvFrame:
    """`av` 硬解出的影格替身：nv12，`to_ndarray()` 給 `(H * 3 // 2, W)` 的平面佈局。

    替身刻意不吐 BGR：讀取端改成在 nv12 的平面上縮放之後，畫面尺寸要從 `height` /
    `width` 屬性取得（不是 `to_ndarray().shape`），格式也要逐格核對——用 BGR 替身
    的話這兩件事都測不到。
    """

    def __init__(self, height, width, format_name="nv12"):
        self.height = height
        self.width = width
        self.format = _FakeFormat(format_name)

    def to_ndarray(self):
        return np.zeros((self.height + self.height // 2, self.width), dtype=np.uint8)


class _FakeStream:
    """視訊串流的替身。速率用 `Fraction`，與 `av` 實際回傳的型別一致。"""

    def __init__(self, average_rate, guessed_rate):
        self.average_rate = average_rate
        self.guessed_rate = guessed_rate


class _FakeStreams:
    def __init__(self, video):
        self.video = video


class _FakeContainer:
    """`av.open()` 的替身，依序吐出指定的影格。"""

    def __init__(
        self,
        frames,
        average_rate=Fraction(30, 1),
        guessed_rate=Fraction(30, 1),
        has_video=True,
    ):
        self._frames = frames
        streams = [_FakeStream(average_rate, guessed_rate)] if has_video else []
        self.streams = _FakeStreams(streams)

    def decode(self, video=0):  # 參數名對齊 av 的介面
        yield from self._frames

    def close(self):
        pass


class _RingStub:
    """環形緩衝的替身，只記下每次寫入的形狀。"""

    num_slots = 4

    def __init__(self):
        self.written = []

    def write_slot(self, slot, frame):
        self.written.append(frame.shape)


def _reader_over(monkeypatch, frames, source_shape, **container_kwargs):
    """組一個讀取器，讓 `_read_segment` 讀到 `frames` 這幾格。

    `av.open` 的替身額外記下呼叫時收到的關鍵字參數（`captured_open_kwargs`），
    供斷言 `hwaccel` 有沒有被正確傳入。
    """
    captured_open_kwargs: dict = {}

    def fake_open(_path, **kwargs):
        captured_open_kwargs.update(kwargs)
        return _FakeContainer(frames, **container_kwargs)

    monkeypatch.setattr(video_reader.av, "open", fake_open)
    free_queue: queue.Queue = queue.Queue()
    for slot in range(_RingStub.num_slots):
        free_queue.put(slot)
    reader = video_reader.DailyStreamVideoReader(
        stream_id=0,
        segments=[],
        data_queue=queue.Queue(),
        free_queue=free_queue,
        ring=_RingStub(),
        source_shape=source_shape,
    )
    reader.captured_open_kwargs = captured_open_kwargs
    return reader


def _segment() -> video_reader.SegmentInfo:
    return video_reader.SegmentInfo(
        path=Path("loc_cam/2026/07/08/030000.000Z.mkv"),
        start=datetime.datetime(2026, 7, 8, 11, 0, tzinfo=_TAIPEI),
    )


def test_reader_rejects_a_resolution_change_mid_stream(monkeypatch):
    """整天解析度改變必須當場中止，不能被 letterbox 抹平後靜默算錯。

    縮放前移之前，這件事由 `FrameRing.write_slot` 的形狀檢查順便擋下（緩衝依首格
    解析度配置）；letterbox 把任何尺寸都變成推論尺寸之後那道網就失效了，而後果全是
    靜默的：反算參數仍是首格那組（1080p→4K 差一半）、`frame_width` 每列照樣寫首格的
    值，連下游「該攝影機 frame_width 唯一」的檢查都照樣通過。
    """
    frames = [
        _FakeAvFrame(1080, 1920),
        _FakeAvFrame(2160, 3840),  # 中途換成 4K
    ]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    with pytest.raises(ValueError, match="解析度"):
        reader._read_segment(_segment())


def test_reader_letterboxes_every_frame_to_the_inference_size(monkeypatch):
    """解析度一致時照常讀完，且寫進 slot 的一律是推論尺寸。"""
    frames = [_FakeAvFrame(1080, 1920) for _ in range(3)]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    reader._read_segment(_segment())

    assert reader.ring.written == [(INFER_HEIGHT, INFER_WIDTH, 3)] * 3


def test_reader_opens_with_hwaccel_and_forbids_software_fallback(monkeypatch):
    """硬解不成立要 fail loud，`allow_software_fallback` 絕不能是 `True`——

    允許 fallback 的話，硬解失敗會靜默退化成軟解，量到的效能數字是另一個組態的
    （同 D⑫ 的教訓，見 report.md 5.2「第二階段」）。`HWAccel` 沒有公開的
    `device_type` 屬性可斷言，`allow_software_fallback` 是唯一能斷言、也是
    規格唯一要求的屬性。
    """
    frames = [_FakeAvFrame(1080, 1920)]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    reader._read_segment(_segment())

    hwaccel = reader.captured_open_kwargs["hwaccel"]
    assert hwaccel.allow_software_fallback is False


def test_reader_wraps_hwaccel_incompatibility_as_value_error(monkeypatch):
    """硬解不成立時 PyAV 18.1.0 拋的是裸 `RuntimeError`（不是 `av.FFmpegError` 的
    子類別，見 `av/container/input.py`「Hardware accelerated decode requested but
    no stream is compatible」），只接 `FFmpegError` 會讓這個失敗漏網、跳過帶檔名的
    `ValueError`，這一路對應的片段就無從查起。
    """

    def fake_open(_path, **_kwargs):
        raise RuntimeError("Hardware accelerated decode requested but no stream is compatible")

    monkeypatch.setattr(video_reader.av, "open", fake_open)
    reader = video_reader.DailyStreamVideoReader(
        stream_id=0,
        segments=[],
        data_queue=queue.Queue(),
        free_queue=queue.Queue(),
        ring=_RingStub(),
        source_shape=FrameShape(height=1080, width=1920),
    )

    with pytest.raises(ValueError, match="無法開啟影片片段"):
        reader._read_segment(_segment())


def test_reader_rejects_a_frame_whose_pixel_format_is_not_nv12(monkeypatch):
    """縮放綁在 nv12 的平面佈局上，讀到別的格式必須當場中止。

    `allow_software_fallback=False` 擋的是「整條退回軟解」，擋不到這裡：yuv420p 攤成
    ndarray 的形狀與 nv12 完全相同（`(H * 3 // 2, W)`），只是後半段是分開的 U、V 兩塊
    而非交錯，當成 nv12 縮不會拋錯，只會讓每一格的顏色靜默錯亂。
    """
    frames = [_FakeAvFrame(1080, 1920, format_name="yuv420p")]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    with pytest.raises(ValueError, match="像素格式"):
        reader._read_segment(_segment())


def _timestamps_of(reader) -> list[datetime.datetime]:
    """把 `_read_segment` 放進 data_queue 的逐格時間戳取出來。"""
    stamps = []
    while not reader.data_queue.empty():
        _slot, _frame_index, timestamp = reader.data_queue.get()
        stamps.append(timestamp)
    return stamps


def test_reader_derives_frame_timestamps_from_a_fractional_frame_rate(monkeypatch):
    """`av` 的速率是 `Fraction`，必須轉成 float 才算得出 `timedelta`。

    直接把 `Fraction` 餵給 `timedelta(seconds=...)` 會拋 `TypeError`，而那是在讀取
    子進程裡、每一支真實片段都會走到的路徑；替身若用 float 就完全測不到這件事。
    """
    frames = [_FakeAvFrame(1080, 1920) for _ in range(3)]
    reader = _reader_over(
        monkeypatch,
        frames,
        FrameShape(height=1080, width=1920),
        average_rate=Fraction(15, 1),
    )

    reader._read_segment(_segment())

    start = _segment().start
    assert _timestamps_of(reader) == [
        start,
        start + datetime.timedelta(seconds=1 / 15),
        start + datetime.timedelta(seconds=2 / 15),
    ]


def test_reader_falls_back_to_guessed_rate_when_average_rate_is_missing(monkeypatch):
    """容器沒標 `avg_frame_rate` 時退回 `guessed_rate`，比照 cv2 的 `r_frame_rate`。

    不接這個 fallback 的話，該攝影機一整天會以「無法讀取影片 FPS」中止。
    """
    frames = [_FakeAvFrame(1080, 1920) for _ in range(2)]
    reader = _reader_over(
        monkeypatch,
        frames,
        FrameShape(height=1080, width=1920),
        average_rate=None,
        guessed_rate=Fraction(15, 1),
    )

    reader._read_segment(_segment())

    start = _segment().start
    assert _timestamps_of(reader) == [start, start + datetime.timedelta(seconds=1 / 15)]


def test_reader_rejects_a_segment_whose_frame_rate_is_unknown(monkeypatch):
    """兩個速率都取不到就 fail loud——猜一個 FPS 會讓整天的時間戳靜默錯位。"""
    frames = [_FakeAvFrame(1080, 1920)]
    reader = _reader_over(
        monkeypatch,
        frames,
        FrameShape(height=1080, width=1920),
        average_rate=Fraction(0, 1),
        guessed_rate=None,
    )

    with pytest.raises(ValueError, match="FPS"):
        reader._read_segment(_segment())


def test_reader_rejects_a_segment_without_a_video_stream(monkeypatch):
    """ffmpeg 開得起來但沒有視訊串流時，要帶著檔名擋下，不是拋裸的 `IndexError`。"""
    reader = _reader_over(
        monkeypatch, [], FrameShape(height=1080, width=1920), has_video=False
    )

    with pytest.raises(ValueError, match="視訊串流"):
        reader._read_segment(_segment())


def test_probe_frame_shape_rejects_a_segment_without_a_video_stream(monkeypatch):
    """探測解析度在主進程跑，訊息裡沒有檔名的話整批中止後查不出是哪一支片段。"""
    monkeypatch.setattr(
        video_reader.av, "open", lambda _path: _FakeContainer([], has_video=False)
    )

    with pytest.raises(ValueError, match="視訊串流"):
        video_reader.probe_frame_shape(_segment())


def _probe_over(monkeypatch, container: _FakeContainer) -> dict:
    """讓 `probe_stream_fps` 開到這個替身容器，並記下 `av.open` 收到的參數。"""
    captured: dict = {}

    def fake_open(_path, **kwargs):
        captured.update(kwargs)
        return container

    monkeypatch.setattr(video_reader.av, "open", fake_open)
    return captured


def test_probe_stream_fps_reads_the_container_rate(monkeypatch):
    """路由的權重就是這個值：讀不對只會讓兩片分配失衡，輸出完全正常。"""
    _probe_over(monkeypatch, _FakeContainer([], average_rate=Fraction(20, 1)))

    assert video_reader.probe_stream_fps(_segment()) == 20.0


def test_probe_stream_fps_returns_a_plain_float(monkeypatch):
    """`Fraction` 不能外流：`_read_segment` 拿它算 `timedelta` 會拋 `TypeError`。"""
    _probe_over(
        monkeypatch, _FakeContainer([], average_rate=Fraction(30000, 1001))
    )

    fps = video_reader.probe_stream_fps(_segment())

    assert isinstance(fps, float)


def test_probe_stream_fps_falls_back_to_the_guessed_rate(monkeypatch):
    """與 `_read_segment` 共用同一份讀法，fallback 不能只有其中一邊有。"""
    _probe_over(
        monkeypatch,
        _FakeContainer([], average_rate=None, guessed_rate=Fraction(15, 1)),
    )

    assert video_reader.probe_stream_fps(_segment()) == 15.0


def test_probe_stream_fps_rejects_a_segment_whose_frame_rate_is_unknown(monkeypatch):
    """讀不到就帶檔名 fail loud——猜一個值只會讓路由靜默失衡。"""
    _probe_over(
        monkeypatch,
        _FakeContainer([], average_rate=Fraction(0, 1), guessed_rate=None),
    )

    with pytest.raises(ValueError, match="FPS"):
        video_reader.probe_stream_fps(_segment())


def test_probe_stream_fps_rejects_a_segment_without_a_video_stream(monkeypatch):
    """同 `probe_frame_shape`：這一步在主進程跑，訊息沒有檔名就查不出是哪一支。"""
    _probe_over(monkeypatch, _FakeContainer([], has_video=False))

    with pytest.raises(ValueError, match="視訊串流"):
        video_reader.probe_stream_fps(_segment())


def test_probe_stream_fps_decodes_nothing(monkeypatch):
    """只讀容器標頭。解一格要數十毫秒到數百毫秒，而這一步是九路各跑一次的啟動成本。"""

    class _CountingContainer(_FakeContainer):
        def __init__(self):
            super().__init__([], average_rate=Fraction(20, 1))
            self.decode_calls = 0

        def decode(self, video=0):
            self.decode_calls += 1
            yield from ()

    container = _CountingContainer()
    _probe_over(monkeypatch, container)

    video_reader.probe_stream_fps(_segment())

    assert container.decode_calls == 0
