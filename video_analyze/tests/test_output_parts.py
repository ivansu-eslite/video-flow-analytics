"""parts 目錄的三件事：鎖、路由、合併。

**鎖那一節是 issue #113 的等價物搬過來的**（原本在 `test_tracking_results.py`，守著
`claim_tmp_slot`）。認領的對象從整天檔的 `.tmp` 換成 `parts/.lock`、持有者從追蹤進程
換成主進程，但它保護的東西一個字都沒變：不覆蓋別人正在寫的東西、順手清掉沒人會回來收
的殘骸、以及鎖必須鎖在「此刻真的叫這個名字」的那個 inode 上。多了一條這一版才有的
——**清殘骸不能把 `.lock` 自己刪掉**：刪掉之後鎖留在一個沒有檔名的 inode 上，另一個
執行馬上能在新建的 inode 上取得鎖，兩邊都以為自己獨佔，而兩邊的輸出檔都完全正常。

路由那一節釘的是**可重現**：分配錯了不會讓輸出變錯（每路的 tracker 狀態獨立），只會
讓兩輪量測不可比，所以沒有別的訊號。

合併那一節釘的是列數與收尾順序：先 `replace()` 成正式檔、再清 parts，清理失敗不能把
一份已經在位的完整結果判成失敗。
"""

import datetime
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from video_analyze.config.constants import (
    PARTS_LOCK_FILENAME,
    TRACKING_RESULTS_SCHEMA,
)
from video_analyze.services import output_parts as op
from video_analyze.services.output_parts import (
    claim_parts_dir,
    merge_parts,
    parts_dir_for,
    plan_routes,
    shard_part_path,
    sweep_parts_dir,
)
from video_analyze.services.tracking_results import (
    TrackingResultCollector,
    tmp_path_for,
)

_STAMP = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))


def _results_path(tmp_path: Path) -> Path:
    return Path(tmp_path) / "2026-08-01" / "tracking_results.parquet"


def _lock_path(results_path: Path) -> Path:
    return parts_dir_for(results_path) / PARTS_LOCK_FILENAME


def _write_part(path: Path, camera_id: str, rows: int) -> None:
    """寫一支欄位符合 `TRACKING_RESULTS_SCHEMA` 的 part 檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "camera_id": [camera_id] * rows,
            "frame_id": list(range(rows)),
            "timestamp": [_STAMP] * rows,
            "track_id": list(range(rows)),
            "x1": [1.0] * rows,
            "y1": [2.0] * rows,
            "x2": [3.0] * rows,
            "y2": [4.0] * rows,
            "foot_x": [2.0] * rows,
            "foot_y": [4.0] * rows,
            "frame_width": [1920] * rows,
            "frame_height": [1080] * rows,
        }
    ).with_columns(pl.col("timestamp").cast(TRACKING_RESULTS_SCHEMA["timestamp"]))
    frame.write_parquet(path)


# --- 鎖 -----------------------------------------------------------------


def test_claim_clears_the_residue_left_by_a_dead_run(tmp_path):
    """拿得到鎖 → 那些 part 的持有者已經不在了 → 清掉。

    整機重啟這種「所有進程都沒機會執行清理」的情境沒有 in-process 的機制擋得住，只能
    由下一次寫同一天的執行順手收拾。
    """
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    parts_dir.mkdir(parents=True)
    residue = shard_part_path(parts_dir, 0)
    residue.write_bytes(b"x" * 4096)
    half_written = tmp_path_for(shard_part_path(parts_dir, 1))
    half_written.write_bytes(b"y" * 128)

    fd = claim_parts_dir(results_path)
    os.close(fd)

    assert not residue.exists()
    assert not half_written.exists()


def test_claim_does_not_remove_the_lock_file_while_sweeping(tmp_path):
    """清殘骸清的是目錄**內容**，`.lock` 自己不算殘骸，連 inode 都不能換。

    `rmtree` 掉整個目錄的話，鎖就留在一個沒有檔名的 inode 上——另一個執行馬上能在新建
    的 inode 上取得鎖，兩邊都以為自己獨佔，而兩邊的輸出檔都完全正常。這是 issue #113
    的殘檔用 `ftruncate` 而非 `unlink` 清的同型情況。
    """
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    parts_dir.mkdir(parents=True)
    shard_part_path(parts_dir, 0).write_bytes(b"x" * 4096)

    fd = claim_parts_dir(results_path)
    try:
        lock_path = _lock_path(results_path)
        assert lock_path.exists()
        by_fd = os.fstat(fd)
        by_name = lock_path.stat()
        assert (by_fd.st_dev, by_fd.st_ino) == (by_name.st_dev, by_name.st_ino)
    finally:
        os.close(fd)


def test_a_second_run_on_the_same_day_is_refused(tmp_path):
    """拿不到鎖 → 那一天正在被另一個執行寫 → fail loud，殘骸一個 byte 都不准動。

    這條是「判斷依據不能只看檔名」的正面表述：目錄裡的東西與上一支測試長得一樣，差別
    只在還有沒有人持有鎖。改成看檔名或 mtime 的話，這支會把別人正在寫的 part 清掉。
    """
    results_path = _results_path(tmp_path)
    first_fd = claim_parts_dir(results_path)
    in_flight = shard_part_path(parts_dir_for(results_path), 0)
    in_flight.write_bytes(b"x" * 4096)

    try:
        with pytest.raises(RuntimeError, match="正被另一個執行持有"):
            claim_parts_dir(results_path)
        assert in_flight.stat().st_size == 4096
    finally:
        os.close(first_fd)


def test_the_lock_survives_the_death_of_the_process_that_took_it(tmp_path):
    """持有者消失（fd 全關）之後，下一個執行拿得到鎖。

    `flock` 由 kernel 在持有者死亡時釋放，正是「主進程被 SIGKILL、沒有人執行清理」
    那條路徑所依賴的性質。這裡用「關掉 fd」模擬進程結束。
    """
    results_path = _results_path(tmp_path)
    os.close(claim_parts_dir(results_path))

    fd = claim_parts_dir(results_path)
    os.close(fd)


def test_a_child_process_keeps_holding_the_lock_after_the_parent_dies(tmp_path):
    """`fork` 出去的子進程繼承同一個 open file description，鎖因此不隨主進程消失。

    這是「主進程被 SIGKILL 之後孤兒子進程仍守住鎖」那道保護的支點，也是本模組**依賴
    `fork`**（而非 spawn）的地方——改成 spawn 會靜默失去它。
    """
    results_path = _results_path(tmp_path)
    fd = claim_parts_dir(results_path)
    ready = mp.Event()
    release = mp.Event()

    def _hold_the_inherited_lock() -> None:
        ready.set()
        release.wait(timeout=10)

    holder = mp.get_context("fork").Process(target=_hold_the_inherited_lock)
    holder.start()
    try:
        ready.wait(timeout=10)
        os.close(fd)  # 主進程這一側放掉，鎖仍該在子進程手上

        with pytest.raises(RuntimeError, match="正被另一個執行持有"):
            claim_parts_dir(results_path)
    finally:
        release.set()
        holder.join(timeout=10)


def test_claim_only_touches_its_own_day(tmp_path):
    """只認領自己那一天，不掃輸出根目錄——別的日期／別的 bucket 不能被波及。"""
    mine = Path(tmp_path) / "2026-08-01" / "tracking_results.parquet"
    other = Path(tmp_path) / "2026-08-02" / "tracking_results.parquet"
    other_part = shard_part_path(parts_dir_for(other), 0)
    other_part.parent.mkdir(parents=True)
    other_part.write_bytes(b"x" * 4096)

    fd = claim_parts_dir(mine)
    os.close(fd)

    assert other_part.stat().st_size == 4096


def test_claim_retries_when_the_lock_file_is_swapped_under_it(tmp_path, monkeypatch):
    """`os.open` 與 `flock` 之間有空窗，鎖到的可能已經不是這個檔名指向的東西。

    有人手動 `rm .lock` 之後另一個執行接手時就是這個情況：鎖是拿得到的，但它掛在一個
    已經沒有檔名（或已被換掉）的 inode 上，等於沒鎖。這裡用一個會在拿到鎖之後把鎖檔
    抽換掉的 `flock` 替身，把那個交錯變成確定會發生。
    """
    results_path = _results_path(tmp_path)
    lock_path = _lock_path(results_path)
    real_flock = fcntl.flock
    raced: list[int] = []

    def racing_flock(fd: int, operation: int) -> None:
        real_flock(fd, operation)
        if not raced:  # 只在第一輪：模擬「等鎖的期間鎖檔被換成另一個 inode」
            raced.append(fd)
            lock_path.unlink()
            lock_path.write_bytes(b"")

    monkeypatch.setattr(op.fcntl, "flock", racing_flock)

    fd = claim_parts_dir(results_path)
    try:
        assert raced, "替身沒被呼叫到，這支測試沒驗到東西"
        by_fd = os.fstat(fd)
        by_name = lock_path.stat()
        assert (by_fd.st_dev, by_fd.st_ino) == (by_name.st_dev, by_name.st_ino)
    finally:
        os.close(fd)


def test_claim_fails_loud_when_the_lock_file_keeps_changing(tmp_path, monkeypatch):
    """連續數輪都撞上抽換就 fail loud，不無限重試。

    真的連續撞上代表有東西在同一條路徑上反覆建檔，那時繼續轉圈只會讓問題沒有訊號。
    """
    results_path = _results_path(tmp_path)
    lock_path = _lock_path(results_path)
    real_flock = fcntl.flock

    def always_racing_flock(fd: int, operation: int) -> None:
        real_flock(fd, operation)
        lock_path.unlink()
        lock_path.write_bytes(b"")

    monkeypatch.setattr(op.fcntl, "flock", always_racing_flock)

    with pytest.raises(RuntimeError, match="都在拿到鎖的當下發現它已經換了"):
        claim_parts_dir(results_path)


def test_claim_leaves_the_pre_sharding_tmp_file_alone(tmp_path):
    """改版前留下的整天 `.tmp` 只記 warning，不刪。

    它可能仍被舊版的孤兒追蹤進程持有著 flock，而新版已經不看那把鎖了——刪掉等於把
    對方正在寫的東西抽走。合併的暫存檔也因此改放在 parts 目錄裡，不寫這條路徑。
    """
    results_path = _results_path(tmp_path)
    legacy = tmp_path_for(results_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"x" * 4096)

    fd = claim_parts_dir(results_path)
    os.close(fd)

    assert legacy.stat().st_size == 4096


def test_sweep_clears_directory_shaped_residue(tmp_path):
    """殘骸是目錄時也要清掉：「清空這個目錄」才是這一步的語義。"""
    parts_dir = Path(tmp_path) / "tracking_results.parts"
    (parts_dir / "loc_cam001").mkdir(parents=True)
    (parts_dir / "loc_cam001" / "090000.000Z.parquet").write_bytes(b"x" * 32)
    lock_path = parts_dir / PARTS_LOCK_FILENAME
    lock_path.write_bytes(b"")

    sweep_parts_dir(parts_dir)

    assert list(parts_dir.iterdir()) == [lock_path]


# --- 路由 ---------------------------------------------------------------


def test_routes_balance_the_fps_load(tmp_path):
    """依 fps 加權的貪婪分配：兩片的 fps 總和要盡量接近。

    九路的實際組態（4K@20 ×3、1080p@30 ×1、1080p@15 ×5，合計 165）在 N=2 下是
    80／85——各路 fps 都是 5 的倍數，切不出比這更平均的組合。
    """
    stream_fps = [20.0, 20.0, 20.0, 30.0, 15.0, 15.0, 15.0, 15.0, 15.0]

    route = plan_routes(stream_fps, shards=2)

    loads = [
        sum(fps for sid, fps in enumerate(stream_fps) if route[sid] == shard)
        for shard in (0, 1)
    ]
    assert sorted(loads) == [80.0, 85.0]


def test_routes_are_reproducible_when_every_stream_has_the_same_fps(tmp_path):
    """fps 全相同時仍要有確定的分配：tie-break 用 stream_id。

    不釘住的話，同一組態的兩輪量測可能把不同的路湊在一起，兩輪的餘裕數字就不可比——
    而那正是分片要量的東西。輸出本身不會錯，所以沒有別的訊號。
    """
    route = plan_routes([15.0] * 6, shards=2)

    assert route == [0, 1, 0, 1, 0, 1]
    assert plan_routes([15.0] * 6, shards=2) == route


def test_routes_are_a_pure_function_of_the_fps_list(tmp_path):
    """同一份輸入呼叫幾次都給同一個答案（沒有殘留狀態）。"""
    stream_fps = [20.0, 15.0, 30.0]

    assert plan_routes(stream_fps, 2) == plan_routes(stream_fps, 2)


def test_more_shards_than_streams_is_clamped(tmp_path):
    """分片數多於路數時 clamp 到路數，不 fail：多起的片只是白佔一個進程。"""
    route = plan_routes([20.0, 15.0], shards=5)

    assert sorted(route) == [0, 1]


def test_every_shard_gets_at_least_one_stream(tmp_path):
    """每片都要分到東西——空片會讓主進程等一支永遠不會出現的 part 檔。"""
    route = plan_routes([20.0, 20.0, 20.0, 20.0], shards=3)

    assert set(route) == {0, 1, 2}


def test_a_single_shard_takes_every_stream(tmp_path):
    """N=1 是預設值以外的合法組態（量測用的對照組），全部路都歸片 0。"""
    assert plan_routes([20.0, 15.0, 30.0], shards=1) == [0, 0, 0]


def test_routing_rejects_a_shard_count_below_one(tmp_path):
    assert plan_routes([20.0], shards=1) == [0]
    with pytest.raises(ValueError, match="分片數必須 >= 1"):
        plan_routes([20.0], shards=0)


# --- 合併 ---------------------------------------------------------------


def test_merge_concatenates_every_part(tmp_path):
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=3)
    _write_part(shard_part_path(parts_dir, 1), "loc_cam002", rows=5)

    rows = merge_parts(
        results_path, [shard_part_path(parts_dir, 0), shard_part_path(parts_dir, 1)]
    )

    assert rows == 8
    merged = pl.read_parquet(results_path)
    assert len(merged) == 8
    assert merged.columns == list(TRACKING_RESULTS_SCHEMA)
    assert merged["camera_id"].unique().sort().to_list() == ["loc_cam001", "loc_cam002"]


def test_merge_keeps_the_downstream_schema(tmp_path):
    """下游 zone／line 直接吃這份 schema，合併不能改動任何一欄的型別。"""
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=2)

    merge_parts(results_path, [shard_part_path(parts_dir, 0)])

    merged = pl.read_parquet(results_path)
    for name, dtype in TRACKING_RESULTS_SCHEMA.items():
        assert merged.schema[name] == dtype


def test_merge_accepts_a_part_from_a_shard_that_saw_nothing(tmp_path):
    """一格都沒收到的那片仍會寫出一支零列的 part，合併不能被它擋下。

    `TrackingResultCollector` 的零列分支走的是 `pl.DataFrame.write_parquet`，有列的那條
    走 `pq.ParquetWriter`——兩條路徑產出的 arrow schema 只要有一點不同，`write_table`
    就會拋錯，而這件事只在「某片整天都沒有偵測」時才發生（各片各寫各的，單元測試分開
    看不出來）。所以這裡刻意用真的 collector 產這兩支，不用測試自製的 parquet。
    """
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    parts_dir.mkdir(parents=True)
    busy = TrackingResultCollector(shard_part_path(parts_dir, 0))
    busy.add(
        camera_id="loc_cam001",
        frame_index=0,
        timestamp=_STAMP,
        tracks=np.array([[1.0, 2.0, 3.0, 4.0, 7]]),
        foot_points=np.array([[2.0, 4.0]]),
        frame_width=1920,
        frame_height=1080,
    )
    busy.save()
    TrackingResultCollector(shard_part_path(parts_dir, 1)).save()  # 整天沒有任何偵測

    rows = merge_parts(
        results_path,
        [shard_part_path(parts_dir, 0), shard_part_path(parts_dir, 1)],
    )

    assert rows == 1
    assert pl.read_parquet(results_path).schema == pl.Schema(TRACKING_RESULTS_SCHEMA)


def test_merge_streams_row_groups_instead_of_reading_whole_parts(tmp_path, monkeypatch):
    """合併要逐 row group 搬，不能把整支 part 具體化成一個 arrow table。

    一整天的追蹤明細磁碟上約 2 GB、攤成 arrow 是 4.9 GB，而 `TRACKER__SHARDS=1`
    （量測用的對照組）就是一支 part 裝整天。`pq.read_table` 那條路徑輸出完全正確，
    只是峰值記憶體跟著天數規模走——沒有任何其他訊號。

    把批次列數壓到 3 才驗得到：正式值是 65536，而測試資料不可能有那麼多列。
    """
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    parts_dir.mkdir(parents=True)
    part_path = shard_part_path(parts_dir, 0)
    _write_part(part_path, "loc_cam001", rows=9)
    # 重寫成多個 row group，讓「有沒有串流」看得出差別
    pq.write_table(pq.read_table(part_path), part_path, row_group_size=3)
    batch_rows: list[int] = []
    real_write_batch = pq.ParquetWriter.write_batch

    def _recording_write_batch(self, batch, *args, **kwargs):
        batch_rows.append(batch.num_rows)
        return real_write_batch(self, batch, *args, **kwargs)

    monkeypatch.setattr(op.pq.ParquetWriter, "write_batch", _recording_write_batch)
    monkeypatch.setattr(op, "_MERGE_BATCH_ROWS", 3)

    rows = merge_parts(results_path, [part_path])

    assert rows == 9
    assert batch_rows, "沒有走 write_batch，整支 part 被讀進記憶體了"
    assert max(batch_rows) < 9, f"一次搬了 {max(batch_rows)} 列，等於整支讀進來"


def test_merge_removes_the_parts_directory(tmp_path):
    """正常跑完之後 parts 目錄不該留下，輸出樹要與分片前逐檔一致。"""
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=1)

    merge_parts(results_path, [shard_part_path(parts_dir, 0)])

    assert not parts_dir.exists()
    assert list(results_path.parent.iterdir()) == [results_path]


def test_merge_fails_loud_when_a_part_is_missing(tmp_path):
    """缺一支 part 就中止，正式檔名下不能出現不完整的結果。"""
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=1)

    with pytest.raises(RuntimeError, match="part 檔不存在"):
        merge_parts(
            results_path,
            [shard_part_path(parts_dir, 0), shard_part_path(parts_dir, 1)],
        )

    assert not results_path.exists()


def test_merge_writes_its_tmp_inside_the_parts_directory(tmp_path, monkeypatch):
    """合併的暫存檔要落在 parts 目錄裡，不能寫到整天檔的 `.tmp`。

    那條路徑上可能還躺著改版前留下的殘檔（認領時只警告、不刪），寫過去等於動它；而且
    合併中途崩掉留下的半成品若在 parts 目錄外，就沒有任何一步會回來收。
    """
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=1)
    seen: list[str] = []
    real_writer = pq.ParquetWriter

    def _recording_writer(where, *args, **kwargs):
        seen.append(str(where))
        return real_writer(where, *args, **kwargs)

    monkeypatch.setattr(op.pq, "ParquetWriter", _recording_writer)

    merge_parts(results_path, [shard_part_path(parts_dir, 0)])

    assert seen and all(Path(path).parent == parts_dir for path in seen)
    assert not tmp_path_for(results_path).exists()


def test_a_failed_cleanup_still_leaves_the_official_file(tmp_path, monkeypatch):
    """清 parts 失敗只記 warning：正式檔已經在位，這時中止只會把成功判成失敗。"""
    results_path = _results_path(tmp_path)
    parts_dir = parts_dir_for(results_path)
    _write_part(shard_part_path(parts_dir, 0), "loc_cam001", rows=2)

    def _refuse(path, *args, **kwargs):
        raise OSError(39, "Directory not empty")

    monkeypatch.setattr(op.shutil, "rmtree", _refuse)

    rows = merge_parts(results_path, [shard_part_path(parts_dir, 0)])

    assert rows == 2
    assert results_path.exists()
