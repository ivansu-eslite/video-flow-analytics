import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from video_analyze.io.video_reader import _parse_segment_start

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
