import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

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
