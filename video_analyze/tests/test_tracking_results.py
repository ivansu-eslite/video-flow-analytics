"""`TrackingResultCollector` 的寫入契約測試。

守住三件事：parquet 的欄位與值符合 `TRACKING_RESULTS_SCHEMA`（下游 zone_mapping／
line_counting 直接吃這份 schema）、跨 flush 的資料不遺漏也不重複、以及 `.tmp` 只在
`save()` 成功後才變成正式檔（中途例外不留半成品）。

`frame_width` / `frame_height` 特別要測：來源 `frame_shapes` 存的是 `(height, width)`，
順序寫反不會有型別錯誤，只會讓下游的解析度換算靜默算錯。因此這裡一律用寬高不相等的
尺寸，寬高互換就會讓斷言失敗。

`foot_x` / `foot_y` 同屬「寫錯也不會報錯」的欄位：它與 `tracks` 是兩個獨立參數，
列數對不上或順序錯開都會讓每一列的落腳點配到別條軌跡，而 parquet 本身完全正常。
"""

import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from video_analyze.config.constants import TRACKING_RESULTS_SCHEMA
from video_analyze.services import tracking_results as tr
from video_analyze.services.tracking_results import TrackingResultCollector
from video_analyze.services.video_reader import FramePacket

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)

# 寬 1920 / 高 1080：兩者不相等，寬高寫反時斷言會失敗
_WIDTH, _HEIGHT = 1920, 1080


def _packet(frame_index: int) -> FramePacket:
    return FramePacket(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_index=frame_index,
        timestamp=_BASE + datetime.timedelta(seconds=frame_index),
    )


def _tracks(*rows: tuple[float, float, float, float, int]) -> np.ndarray:
    return np.array(rows, dtype=float).reshape(len(rows), 5)


def _feet(*rows: tuple[float, float]) -> np.ndarray:
    """落腳點由 `services/foot_point.py` 算好後傳入，collector 只負責逐列寫下。

    測試一律給「不等於框底邊中點」的值：collector 若自作主張從 bbox 重算，斷言就會
    失敗（落腳點改用 head 推算後，這兩者本來就不再相等）。
    """
    return np.array(rows, dtype=float).reshape(len(rows), 2)


def test_add_writes_all_schema_columns_with_expected_values(tmp_path):
    """一格兩條軌跡：既有欄位、落腳點與兩個尺寸欄位都要逐列寫對。"""
    results_path = tmp_path / "tracking_results.parquet"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        packet=_packet(7),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1), (50.0, 60.0, 70.0, 80.0, 2)),
        foot_points=_feet((23.0, 41.0), (64.0, 77.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    collector.save()

    df = pl.read_parquet(results_path)
    assert df.columns == list(TRACKING_RESULTS_SCHEMA)
    assert df.to_dicts() == [
        {
            "camera_id": "loc_cam001",
            "frame_id": 7,
            "timestamp": _BASE + datetime.timedelta(seconds=7),
            "track_id": 1,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 30.0,
            "y2": 40.0,
            "foot_x": 23.0,
            "foot_y": 41.0,
            "frame_width": _WIDTH,
            "frame_height": _HEIGHT,
        },
        {
            "camera_id": "loc_cam001",
            "frame_id": 7,
            "timestamp": _BASE + datetime.timedelta(seconds=7),
            "track_id": 2,
            "x1": 50.0,
            "y1": 60.0,
            "x2": 70.0,
            "y2": 80.0,
            "foot_x": 64.0,
            "foot_y": 77.0,
            "frame_width": _WIDTH,
            "frame_height": _HEIGHT,
        },
    ]


def test_add_rejects_foot_points_of_mismatched_length():
    """落腳點與軌跡列數不一致就 fail loud：錯位後每一列的落腳點都配到別條軌跡，
    輸出檔看起來完全正常，下游查不出來。"""
    collector = TrackingResultCollector(Path("unused.parquet"))

    with pytest.raises(ValueError, match="逐列對應"):
        collector.add(
            camera_id="loc_cam001",
            packet=_packet(0),
            tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1), (50.0, 60.0, 70.0, 80.0, 2)),
            foot_points=_feet((23.0, 41.0)),
            frame_width=_WIDTH,
            frame_height=_HEIGHT,
        )


def test_add_keeps_each_camera_own_frame_size(tmp_path):
    """混解析度的 bucket：每台攝影機各自的尺寸不可被另一台蓋掉。"""
    results_path = tmp_path / "tracking_results.parquet"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        packet=_packet(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=1920,
        frame_height=1080,
    )
    collector.add(
        camera_id="loc_cam002",
        packet=_packet(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=3840,
        frame_height=2160,
    )
    collector.save()

    df = pl.read_parquet(results_path)
    assert df.select("camera_id", "frame_width", "frame_height").rows() == [
        ("loc_cam001", 1920, 1080),
        ("loc_cam002", 3840, 2160),
    ]


def test_add_with_empty_tracks_adds_no_rows(tmp_path):
    """該格沒有任何軌跡時不新增列（空陣列不該被當成一列 null）。"""
    results_path = tmp_path / "tracking_results.parquet"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        packet=_packet(0),
        tracks=np.empty((0, 5), dtype=float),
        foot_points=np.empty((0, 2), dtype=float),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    collector.save()

    assert pl.read_parquet(results_path).height == 0


def test_rows_survive_flush_boundary(tmp_path, monkeypatch):
    """跨 flush 門檻（多個 row group）時列數與內容都要完整——flush 後緩衝要清空，
    既不能漏寫也不能把同一批重複寫進下一個 row group。"""
    monkeypatch.setattr(tr, "_FLUSH_EVERY_ROWS", 3)
    results_path = tmp_path / "tracking_results.parquet"
    collector = TrackingResultCollector(results_path)

    for i in range(7):
        collector.add(
            camera_id="loc_cam001",
            packet=_packet(i),
            tracks=_tracks((float(i), 20.0, 30.0, 40.0, i)),
            foot_points=_feet((float(i) + 3.0, 41.0)),
            frame_width=_WIDTH,
            frame_height=_HEIGHT,
        )
    collector.save()

    df = pl.read_parquet(results_path)
    assert df.height == 7
    assert df["frame_id"].to_list() == list(range(7))
    assert df["x1"].to_list() == [float(i) for i in range(7)]
    assert df["frame_width"].unique().to_list() == [_WIDTH]


def test_save_without_any_row_still_writes_schema(tmp_path):
    """全天沒有任何追蹤結果時仍要寫出空 parquet，且欄位／型別與 schema 一致——
    下游讀到缺欄位的空檔會以為是舊版產物。"""
    results_path = tmp_path / "tracking_results.parquet"
    TrackingResultCollector(results_path).save()

    df = pl.read_parquet(results_path)
    assert df.height == 0
    assert dict(df.schema) == TRACKING_RESULTS_SCHEMA


def test_results_path_appears_only_after_save(tmp_path):
    """flush 只寫 `.tmp`，正式檔名要到 save() 才出現（原子性 rename）。"""
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path / "tracking_results.parquet.tmp"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        packet=_packet(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    collector._flush()
    assert tmp_file.exists()
    assert not results_path.exists()

    collector.save()
    assert results_path.exists()
    assert not tmp_file.exists()


def test_discard_leaves_no_partial_output(tmp_path):
    """中途例外時 discard()：暫存檔刪掉、正式檔不產生（fail-loud，不留半成品）。"""
    results_path = tmp_path / "tracking_results.parquet"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        packet=_packet(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    collector._flush()
    collector.discard()

    assert not results_path.exists()
    assert not (tmp_path / "tracking_results.parquet.tmp").exists()


def test_schema_declares_frame_size_columns():
    """下游（line_counting）靠這兩欄做解析度換算，schema 少了任一欄就是破壞契約。"""
    assert TRACKING_RESULTS_SCHEMA["frame_width"] == pl.Int64
    assert TRACKING_RESULTS_SCHEMA["frame_height"] == pl.Int64
