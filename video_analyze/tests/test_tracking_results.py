"""`TrackingResultCollector` 的寫入契約測試。

守住三件事：parquet 的欄位與值符合 `TRACKING_RESULTS_SCHEMA`（下游 zone_mapping／
line_counting 直接吃這份 schema）、跨 flush 的資料不遺漏也不重複、以及 `.tmp` 只在
`save()` 成功後才變成正式檔（中途例外不留半成品）。

`frame_width` / `frame_height` 特別要測：來源 `frame_shapes` 存的是 `(height, width)`，
順序寫反不會有型別錯誤，只會讓下游的解析度換算靜默算錯。因此這裡一律用寬高不相等的
尺寸，寬高互換就會讓斷言失敗。

`foot_x` / `foot_y` 同屬「寫錯也不會報錯」的欄位：它與 `tracks` 是兩個獨立參數，
列數對不上或順序錯開都會讓每一列的落腳點配到別條軌跡，而 parquet 本身完全正常。

第四件是 `claim_tmp_slot`（issue #113）：追蹤進程被 SIGKILL 或整機斷電時 `save()` 與
`discard()` 都走不到，`.tmp` 留在輸出目錄沒有人會回來收，只能由下一次寫同一條路徑的
執行清掉。它要清得掉殘檔、又不能誤刪另一個執行**正在寫**的暫存檔，所以這裡的斷言
分成「拿得到鎖 → 清掉」與「拿不到鎖 → 一個 byte 都不准動」兩側。
"""

import datetime
import fcntl
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from video_analyze.config.constants import TRACKING_RESULTS_SCHEMA
from video_analyze.services import tracking_results as tr
from video_analyze.services.tracking_results import (
    TrackingResultCollector,
    claim_tmp_slot,
    tmp_path_for,
)

_TAIPEI = ZoneInfo("Asia/Taipei")
_BASE = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)

# 寬 1920 / 高 1080：兩者不相等，寬高寫反時斷言會失敗
_WIDTH, _HEIGHT = 1920, 1080


def _stamp(frame_index: int) -> datetime.datetime:
    return _BASE + datetime.timedelta(seconds=frame_index)


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
        frame_index=7,
        timestamp=_stamp(7),
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
            frame_index=0,
            timestamp=_stamp(0),
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
        frame_index=0,
        timestamp=_stamp(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=1920,
        frame_height=1080,
    )
    collector.add(
        camera_id="loc_cam002",
        frame_index=0,
        timestamp=_stamp(0),
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
        frame_index=0,
        timestamp=_stamp(0),
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
            frame_index=i,
            timestamp=_stamp(i),
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


def test_save_without_any_row_takes_the_claimed_tmp_file_with_it(tmp_path):
    """零列的那條分支不經過 rename，`claim_tmp_slot` 建出來的空暫存檔要自己收掉。

    留著的話正常跑完的一天也會在輸出目錄留下一個 `.tmp`，而這個 issue 要的正是
    「輸出目錄不留無人清理的暫存檔」——留一個 0 byte 的等於讓那個判準失去意義。
    """
    results_path = tmp_path / "tracking_results.parquet"
    fd = claim_tmp_slot(results_path)
    assert tmp_path_for(results_path).exists()  # 認領當下就把檔案建出來了

    TrackingResultCollector(results_path).save()
    os.close(fd)

    assert results_path.exists()
    assert not tmp_path_for(results_path).exists()


def test_results_path_appears_only_after_save(tmp_path):
    """flush 只寫 `.tmp`，正式檔名要到 save() 才出現（原子性 rename）。"""
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path / "tracking_results.parquet.tmp"
    collector = TrackingResultCollector(results_path)

    collector.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_stamp(0),
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
        frame_index=0,
        timestamp=_stamp(0),
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


def test_claim_clears_the_residue_left_by_a_dead_run(tmp_path):
    """拿得到鎖 → 這個殘檔的持有者已經不在了 → 就地清空。

    斷言看的是**大小**而不是存在與否：留著 inode 是刻意的（鎖掛在上面），要回收的是
    那份整天追蹤明細的空間。
    """
    results_path = tmp_path / "tracking_results.parquet"
    residue = tmp_path_for(results_path)
    residue.write_bytes(b"x" * 4096)  # 上一次執行被 SIGKILL 時留下的半成品

    fd = claim_tmp_slot(results_path)
    os.close(fd)

    assert residue.exists()
    assert residue.stat().st_size == 0


def test_claim_is_refused_while_another_run_holds_the_tmp_file(tmp_path):
    """拿不到鎖 → 那是另一個執行**正在寫**的檔 → fail loud，一個 byte 都不准動。

    這條是「判斷依據不能只看檔名」的正面表述：本測試裡的殘檔與上一支測試的長得一模
    一樣，差別只在還有沒有人持有鎖。改成看檔名或 mtime 的話，這支會把別人寫到一半的
    整天明細清掉，而且不會有任何訊號。
    """
    results_path = tmp_path / "tracking_results.parquet"
    in_flight = tmp_path_for(results_path)
    in_flight.write_bytes(b"x" * 4096)
    holder = os.open(in_flight, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        with pytest.raises(RuntimeError, match="正被另一個執行中的進程持有"):
            claim_tmp_slot(results_path)
        assert in_flight.stat().st_size == 4096
    finally:
        os.close(holder)


def test_claim_only_touches_its_own_output_path(tmp_path):
    """只認領自己那一條路徑，不掃目錄——別的 bucket／別的日期的暫存檔不能被波及。

    輸出路徑是 `outputs/<bucket>/<date>/`，並行執行本來就落在不同目錄；掃目錄的做法
    在共用輸出根目錄時會跨過這條界線。
    """
    mine = tmp_path / "2026-08-01" / "tracking_results.parquet"
    other = tmp_path / "2026-08-02" / "tracking_results.parquet"
    other.parent.mkdir(parents=True)
    other_tmp = tmp_path_for(other)
    other_tmp.write_bytes(b"x" * 4096)

    fd = claim_tmp_slot(mine)
    os.close(fd)

    assert other_tmp.stat().st_size == 4096


def test_tmp_naming_has_a_single_source(tmp_path):
    """collector 與 claim 必須指到同一個檔：兩邊各寫一次命名規則的話，改了後綴會讓
    claim 靜默守著另一個檔名——鎖照樣拿得到、殘檔照樣沒清掉，也不會有錯誤訊息。"""
    results_path = tmp_path / "tracking_results.parquet"

    collector = TrackingResultCollector(results_path)
    fd = claim_tmp_slot(results_path)
    os.close(fd)

    assert collector._tmp_path == tmp_path_for(results_path)
    assert tmp_path_for(results_path).exists()


def test_claim_does_not_truncate_a_file_that_was_renamed_away(tmp_path, monkeypatch):
    """`os.open` 與 `flock` 之間有空窗，鎖到的可能已經不是這個檔名指向的東西。

    另一個執行正好在這期間 `save()`（把同一個 inode rename 成正式檔名）的話，鎖是拿得到
    的、`fstat` 也看得到非零大小，接著的 `ftruncate` 會把對方剛寫完的**整天結果**清成
    0 byte，而 log 還寫著「清掉前一次執行留下的暫存檔」。

    這裡用一個會在拿到鎖之後做 rename 的 `flock` 替身，把那個交錯變成確定會發生。
    """
    results_path = tmp_path / "tracking_results.parquet"
    in_flight = tmp_path_for(results_path)
    in_flight.write_bytes(b"x" * 4096)
    real_flock = fcntl.flock
    raced: list[int] = []

    def racing_flock(fd: int, operation: int) -> None:
        real_flock(fd, operation)
        if not raced:  # 只在第一輪：模擬「等鎖的期間對方 save() 把它 rename 走了」
            raced.append(fd)
            in_flight.replace(results_path)

    monkeypatch.setattr(tr.fcntl, "flock", racing_flock)

    fd = claim_tmp_slot(results_path)
    os.close(fd)

    assert raced, "替身沒被呼叫到，這支測試沒驗到東西"
    assert results_path.stat().st_size == 4096, "把對方剛完成的結果清空了"
    assert tmp_path_for(results_path).stat().st_size == 0  # 自己認到的是新建的空檔
