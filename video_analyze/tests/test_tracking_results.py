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
    fd, identity = claim_tmp_slot(results_path)
    assert tmp_path_for(results_path).exists()  # 認領當下就把檔案建出來了

    TrackingResultCollector(results_path, identity).save()
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

    fd, _identity = claim_tmp_slot(results_path)
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
        with pytest.raises(RuntimeError, match="正被另一個進程持有"):
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

    fd, _identity = claim_tmp_slot(mine)
    os.close(fd)

    assert other_tmp.stat().st_size == 4096


def test_tmp_naming_has_a_single_source(tmp_path):
    """collector 與 claim 必須指到同一個檔：兩邊各寫一次命名規則的話，改了後綴會讓
    claim 靜默守著另一個檔名——鎖照樣拿得到、殘檔照樣沒清掉，也不會有錯誤訊息。"""
    results_path = tmp_path / "tracking_results.parquet"

    collector = TrackingResultCollector(results_path)
    fd, _identity = claim_tmp_slot(results_path)
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

    fd, _identity = claim_tmp_slot(results_path)
    os.close(fd)

    assert raced, "替身沒被呼叫到，這支測試沒驗到東西"
    assert results_path.stat().st_size == 4096, "把對方剛完成的結果清空了"
    assert tmp_path_for(results_path).stat().st_size == 0  # 自己認到的是新建的空檔


def test_discard_leaves_a_different_file_at_the_same_path_alone(tmp_path):
    """同一個路徑此刻若已經是別份檔案（inode 不同），`discard()` 一個字節都不能動。

    `claim_tmp_slot` 的鎖擋得住「兩個執行同時認領」，但擋不住有人手動 `rm` 掉暫存檔之後
    另一個執行接手。依路徑刪的話會刪掉對方寫到一半的內容，而對方要到最後 `replace()`
    才會以 `FileNotFoundError` 爆掉——錯誤發生的位置離原因很遠。
    """
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path_for(results_path)
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
    tmp_file.unlink()  # 運維人員手動清掉
    tmp_file.write_bytes(b"y" * 128)  # 另一個執行接手，建了新的一份

    collector.discard()

    assert tmp_file.stat().st_size == 128


def test_save_refuses_to_rename_a_file_it_did_not_write(tmp_path):
    """`replace()` 依路徑動作，而這個路徑此刻可能已經是別人的檔——那時要 fail loud。

    不擋的話會把對方**寫到一半**的內容改名成正式檔名，log 還照樣印「追蹤結果已寫入
    rows=N」，下游要到 `pl.read_parquet` 才以「must end with PAR1」崩在離原因很遠的地方。
    """
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path_for(results_path)
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
    tmp_file.unlink()  # 運維人員手動清掉
    tmp_file.write_bytes(b"y" * 21)  # 另一個執行接手，正在寫的半成品

    with pytest.raises(RuntimeError, match="已不是本次執行認領的那一份"):
        collector.save()

    assert not results_path.exists()
    assert tmp_file.stat().st_size == 21  # 對方的檔還在原地，沒被改名


def test_discard_removes_the_tmp_even_if_closing_the_writer_fails(tmp_path):
    """`close()` 寫 parquet footer 時失敗（磁碟滿）正是最需要收掉半成品的時候。

    不隔開的話刪檔整段被跳過，而「不留殘檔」是這次改動立起來的不變量。
    """
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path_for(results_path)
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

    class _FailingClose:
        def close(self) -> None:
            raise OSError(28, "No space left on device")

    collector._writer = _FailingClose()

    with pytest.raises(OSError, match="No space left"):
        collector.discard()

    assert not tmp_file.exists()


def test_flush_refuses_to_write_over_a_file_it_did_not_claim(tmp_path):
    """真正造成破壞的是**寫入**那一步：`ParquetWriter` 以路徑開檔、帶 `O_TRUNC`。

    路徑此刻若已經是別人的檔（有人手動 `rm` 掉暫存檔之後另一個執行接手），第一次 flush
    就會把對方的內容截掉並蓋寫。只在 `save()` 檢查已經來不及——那時對方的資料早就沒了，
    而錯誤訊息還會把受害者講反。
    """
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path_for(results_path)
    fd, identity = claim_tmp_slot(results_path)  # 本次認領到的那一份
    collector = TrackingResultCollector(results_path, identity)
    tmp_file.unlink()  # 被手動清掉
    tmp_file.write_bytes(b"y" * 4096)  # 另一個執行接手，正在寫的內容

    collector.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_stamp(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    with pytest.raises(RuntimeError, match="不能開始寫入"):
        collector._flush()
    os.close(fd)

    assert tmp_file.read_bytes() == b"y" * 4096  # 對方的內容一個字節都沒動


def test_flush_fails_loud_when_the_claimed_tmp_was_removed(tmp_path):
    """認領到的暫存檔被外部清掉時，要在寫入之前就中止，而不是寫完才判定遺失。

    不擋的話 `ParquetWriter` 會在同一個路徑上建一個**新的** inode 並把整天的結果寫進去，
    然後 `save()` 因為 inode 對不上而拒絕改名——一份完整的結果被判成遺失、還以 `.tmp`
    留在輸出目錄，兩邊都錯。
    """
    results_path = tmp_path / "tracking_results.parquet"
    fd, identity = claim_tmp_slot(results_path)
    collector = TrackingResultCollector(results_path, identity)
    tmp_path_for(results_path).unlink()  # 例如被清理排程掃掉

    collector.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_stamp(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    with pytest.raises(RuntimeError, match="不能開始寫入"):
        collector._flush()
    os.close(fd)

    assert not results_path.exists()


def test_collector_identity_comes_from_the_claim_not_from_the_path(tmp_path):
    """身分要由認領那一步帶進來，不能在 collector 裡用 `os.stat` 重新推導一次。

    認領完到建 collector 之間這個名字仍可能被換掉（有人手動 `rm`、另一個執行接手）。
    重新推導的話 collector 會把**別人的 inode** 當成自己的，之後寫入、刪除、改名三道
    比對全部通過——正是 `_require_own_tmp` 要擋的那件事，卻擋不到。
    """
    results_path = tmp_path / "tracking_results.parquet"
    tmp_file = tmp_path_for(results_path)
    fd, identity = claim_tmp_slot(results_path)
    tmp_file.unlink()
    tmp_file.write_bytes(b"y" * 3400)  # 另一個執行接手，正在寫的內容

    collector = TrackingResultCollector(results_path, identity)
    collector.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_stamp(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    with pytest.raises(RuntimeError, match="不能開始寫入"):
        collector._flush()
    os.close(fd)

    assert tmp_file.read_bytes() == b"y" * 3400
    assert not results_path.exists()


def test_zero_row_save_never_leaves_a_half_written_official_file(tmp_path, monkeypatch):
    """零列那條分支也要先寫暫存檔再 rename，不能直接寫正式檔名。

    直接寫的話 `write_parquet` 中途失敗（磁碟滿）會在**正式檔名**下留一個截斷的檔，而
    `discard()` 只認得 `.tmp`、收不掉它，下游要到 `pl.read_parquet` 才以「must end with
    PAR1」崩在離原因很遠的地方。
    """
    results_path = tmp_path / "tracking_results.parquet"
    fd, identity = claim_tmp_slot(results_path)
    collector = TrackingResultCollector(results_path, identity)

    def _out_of_space(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1_trunc")  # 寫了一半才失敗
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _out_of_space)
    with pytest.raises(OSError, match="No space left"):
        collector.save()
    monkeypatch.undo()
    collector.discard()
    os.close(fd)

    assert not results_path.exists(), "半成品寫到正式檔名下了"
    assert not tmp_path_for(results_path).exists()


def test_discard_after_a_successful_save_says_nothing(tmp_path, capsys):
    """`save()` 之後暫存檔已經改名走了，再呼叫 `discard()` 不該有任何抱怨。

    不清掉身分的話，`_remove_tmp` 會拿它去比對此刻可能屬於別人的同名檔，印出「不是本次
    執行建立的那一份」——把運維指向一個不存在的並行執行。
    """
    results_path = tmp_path / "tracking_results.parquet"
    fd, identity = claim_tmp_slot(results_path)
    collector = TrackingResultCollector(results_path, identity)
    collector.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_stamp(0),
        tracks=_tracks((10.0, 20.0, 30.0, 40.0, 1)),
        foot_points=_feet((23.0, 41.0)),
        frame_width=_WIDTH,
        frame_height=_HEIGHT,
    )
    collector.save()
    capsys.readouterr()  # 丟掉 save() 自己那行「追蹤結果已寫入」

    collector.discard()
    os.close(fd)

    # `StructuredLogger` 是 print 到 stdout 的，不走 logging，所以看 capsys 而不是 caplog
    assert "略過刪除" not in capsys.readouterr().out
    assert results_path.exists()
