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


class _FakeAvFrame:
    """`av` 解碼出的影格替身，只需支援 `to_ndarray(format="bgr24")`。"""

    def __init__(self, array):
        self._array = array

    def to_ndarray(self, format="bgr24"):  # noqa: A002 — 對齊 av 的介面命名
        return self._array


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
        for frame in self._frames:
            yield _FakeAvFrame(frame)

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
    """組一個讀取器，讓 `_read_segment` 讀到 `frames` 這幾格。"""
    monkeypatch.setattr(
        video_reader.av, "open", lambda _path: _FakeContainer(frames, **container_kwargs)
    )
    free_queue: queue.Queue = queue.Queue()
    for slot in range(_RingStub.num_slots):
        free_queue.put(slot)
    return video_reader.DailyStreamVideoReader(
        stream_id=0,
        segments=[],
        data_queue=queue.Queue(),
        free_queue=free_queue,
        ring=_RingStub(),
        source_shape=source_shape,
    )


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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        np.zeros((2160, 3840, 3), dtype=np.uint8),  # 中途換成 4K
    ]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    with pytest.raises(ValueError, match="解析度"):
        reader._read_segment(_segment())


def test_reader_letterboxes_every_frame_to_the_inference_size(monkeypatch):
    """解析度一致時照常讀完，且寫進 slot 的一律是推論尺寸。"""
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    reader = _reader_over(monkeypatch, frames, FrameShape(height=1080, width=1920))

    reader._read_segment(_segment())

    assert reader.ring.written == [(INFER_HEIGHT, INFER_WIDTH, 3)] * 3


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
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
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
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(2)]
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
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
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
