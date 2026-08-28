"""`tracking_results.parts/` 的全部知識：認領、清殘骸、片檔命名、路由、合併。

追蹤進程從一個變成 N 個之後，切不開的是**輸出**——`pq.ParquetWriter` 是行程內的檔案
handle，N 個進程不可能共寫同一份。所以每片各寫自己的 part 檔，主進程在全部到齊後合併
成下游看的那一個檔名（見 ADR-012）。

```
outputs/<bucket>/<date>/
  tracking_results.parquet        ← 契約不變，下游只看這個
  tracking_results.parts/         ← 只在跑到一半時存在
    .lock                         ← flock 認領這一天（0 byte，主進程持有）
    shard0.parquet                ← 該片正常收尾後 rename 出來的
    shard1.parquet.tmp            ← 該片還在寫
```

**鎖的持有者是主進程**，不是任何一個追蹤進程：認領、跑、合併三段要被同一把鎖蓋住，
只有主進程橫跨全程。改動前那把鎖由追蹤進程持有、鎖的對象是整天檔的 `.tmp`
（issue #113 的 `claim_tmp_slot`），語義與清殘骸的手法原樣搬進本模組。

**保護不因此退化**：主進程持鎖後 `fork` 出的所有子進程繼承同一個 open file
description，而 flock 屬於 description 不屬於 fd，因此只要任一進程還開著它鎖就在——
主進程被 SIGKILL 之後孤兒子進程仍守著鎖，另一個執行照樣擋得下。⚠ 這道保護**依賴
`fork`**：改成 spawn 會靜默失去它（子進程不繼承 fd），連帶讓「孤兒守鎖」的錯誤訊息
變成謊話。
"""

import fcntl
import os
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
from vfa_observability import StructuredLogger

from video_analyze.config.constants import (
    PARTS_LOCK_FILENAME,
    TRACKING_RESULTS_PARTS_DIRNAME,
)
from video_analyze.services.tracking_results import is_file_at, tmp_path_for

logger = StructuredLogger(component="output_parts")

# 認領時，「拿到鎖卻發現檔名已經指向別的 inode」最多重來這麼多次。正常情況一次就過；
# 會重來代表剛好撞上另一個執行的 unlink 或 rename，那是瞬間的動作，不會連續發生。
# 給上限而不是無限重試：真的連續撞上代表有東西在同一條路徑上反覆建檔，該讓它 fail loud
_CLAIM_ATTEMPTS = 5

# 合併輸出的暫存檔名。**刻意放在 parts 目錄裡、不用整天檔的 `.tmp`**：那條路徑上可能
# 還躺著改版前留下的殘檔（見 `claim_parts_dir` 的 warning），寫過去等於動它；而且合併
# 中途崩掉留下的半成品若在 parts 目錄外，就沒有任何一步會回來收——parts 目錄裡的東西
# 則由下一次認領一併清掉。與 part 檔名不衝突（`shard<k>.parquet`），故合併讀入的清單
# 由呼叫端明確給定，本模組不掃目錄。
_MERGE_TMP_FILENAME = "merged.parquet.tmp"


def parts_dir_for(results_path: Path) -> Path:
    """回傳 `results_path` 對應的 parts 目錄（與正式輸出同一層）。"""
    return results_path.with_name(TRACKING_RESULTS_PARTS_DIRNAME)


def shard_part_path(parts_dir: Path, shard_id: int) -> Path:
    """回傳第 `shard_id` 片的 part 檔路徑。

    part 檔名**不參與任何跨執行的判定**（本版沒有續跑，parts 是單次執行的中間產物，
    跑完即刪），所以編號只要在一次執行內唯一就夠。
    """
    return parts_dir / f"shard{shard_id}.parquet"


def claim_parts_dir(results_path: Path) -> int:
    """認領這一天的輸出：對 `parts/.lock` 上鎖，並清掉前一次執行留下的殘骸。

    **判準是「還有沒有人持有這把鎖」，不是檔名、不是 mtime**：多個 bucket 或多個日期
    並行時各自寫各自的 parts 目錄（輸出路徑帶 bucket 名與日期），本來就碰不到彼此；
    真正需要判斷的是同一天被兩個執行同時寫，而檔名與 mtime 都分不出「上次留下的」與
    「另一個執行正在寫的」。`flock` 由 kernel 在持有者死亡時釋放——SIGKILL 與整機重啟
    都算——正好對上「兩個進程都沒機會執行清理」這個情境。

    **拿到鎖之後要再確認手上的 inode 還是那個檔名指向的東西**：`os.open` 與 `flock`
    之間有空窗，另一個執行可能正好在這期間把鎖檔刪掉重建（例如有人手動 `rm .lock`
    之後別的執行接手）。對不上就關掉重開——鎖在一個沒有檔名的 inode 上等於沒鎖，兩邊
    都會以為自己獨佔。

    清殘骸清的是**目錄內容，不是目錄本身，而且不碰 `.lock`**：`rmtree` 會把鎖檔一起
    帶走，鎖就留在沒有檔名的 inode 上，另一個執行馬上能在新建的 inode 上取得鎖，兩邊
    都以為自己獨佔。這與 issue #113 的殘檔用 `ftruncate` 而非 `unlink` 清是同一件事。

    `flock` 是 BSD 介面（不在 POSIX 內；POSIX 的 advisory lock 是 `fcntl(F_SETLK)`），
    只在都走這個機制的進程之間有效。本 repo 只跑 Linux；輸出目錄若日後掛到 NFS，語義
    要重新確認——Linux 的 NFS 從 2.6.37 起才用 POSIX lock 去模擬 flock，跨用戶端的行為
    與本機不同。

    Args:
        results_path: 追蹤結果 parquet 的正式輸出路徑。

    Returns:
        持有 `flock` 的 file descriptor。**必須持到合併完成為止**（fd 一關鎖就沒了），
        而且要在起子進程**之前**取得——子進程靠 `fork` 繼承它，鎖才在主進程死後仍然
        有效。

    Raises:
        RuntimeError: 這一天正被另一個執行持有，或連續數輪都在拿到鎖的當下發現鎖檔
            已經換了 inode。
        OSError: 開檔、上鎖或 stat 本身失敗（例如 NFS 回 `ENOLCK`、權限不足）。這類
            錯誤原樣往外拋，不轉成 `RuntimeError`——它們與「有人正在寫」是不同的事，
            蓋掉會讓排查方向偏掉。
    """
    parts_dir = parts_dir_for(results_path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    lock_path = parts_dir / PARTS_LOCK_FILENAME
    for _ in range(_CLAIM_ATTEMPTS):
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claimed = is_file_at(fd, lock_path)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                f"{parts_dir} 正被另一個執行持有，本次執行中止。"
                "同一個 bucket 的同一天不能有兩個執行同時跑——兩邊會各自寫進同一組 "
                "part 檔，再各自合併成正式檔名；要並行請分開 bucket 或分開日期。"
                "若確定沒有別的執行在跑，持有者可能是上一次留下的孤兒子進程"
                "（主進程被 SIGKILL 之後，繼承了這把鎖的追蹤進程會一直等在 "
                "track_queue.get() 上，讀取進程則等在 free_queue.get() 上）："
                "以 `pgrep -af video_analyze` 找出來收掉之後再重跑。"
            ) from exc
        except BaseException:
            # flock 的其他失敗（例如 NFS 上的 ENOLCK）與 is_file_at 的 stat 失敗都原樣
            # 往外拋，但 fd 不能跟著漏掉
            os.close(fd)
            raise
        if claimed:
            break
        os.close(fd)  # 等鎖的期間這個 inode 已經不叫 `.lock` 了，鎖到的是別人的東西
    else:
        raise RuntimeError(
            f"連續 {_CLAIM_ATTEMPTS} 次認領 {lock_path} 都在拿到鎖的當下發現它已經換了"
            "inode，本次執行中止。同一條輸出路徑上有其他執行正在頻繁地建立與收尾。"
        )
    sweep_parts_dir(parts_dir)
    _warn_about_legacy_tmp(results_path)
    return fd


def sweep_parts_dir(parts_dir: Path) -> None:
    """清掉 parts 目錄裡除了鎖檔以外的全部內容。

    走到這裡代表鎖已經拿到手，殘骸的上一個持有者不在了（否則拿不到鎖），這些檔沒有人
    會回來收。**鎖檔一個字節都不能動**——理由見 `claim_parts_dir`。

    目錄型的殘骸一併清掉：本版的 part 全是平鋪的檔案，但「清空這個目錄」才是這一步的
    語義，只清檔案會在日後多出子目錄時靜默留下東西。
    """
    residue = [p for p in parts_dir.iterdir() if p.name != PARTS_LOCK_FILENAME]
    if not residue:
        return
    total_bytes = 0
    for path in residue:
        if path.is_dir() and not path.is_symlink():
            total_bytes += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path)
        else:
            total_bytes += path.stat().st_size
            path.unlink()
    logger.warning(
        "清掉前一次執行留下的 part 殘骸",
        path=str(parts_dir),
        entries=len(residue),
        bytes=total_bytes,
    )


def _warn_about_legacy_tmp(results_path: Path) -> None:
    """看到改版前的整天 `.tmp` 殘檔就記一行 warning，**不自動刪**。

    改版後那條路徑不再有人寫（每片寫自己的 part、合併的暫存檔在 parts 目錄裡），也就
    不再有人清。不自動刪是因為它可能仍被舊版的孤兒追蹤進程持有著 flock，而新版已經
    不看那把鎖了——刪掉等於把對方正在寫的東西抽走。
    """
    legacy_tmp = tmp_path_for(results_path)
    if not legacy_tmp.exists():
        return
    logger.warning(
        "輸出目錄有改版前留下的整天暫存檔，本次執行不會使用也不會清理它，"
        "確認沒有舊版進程還在跑之後可手動刪除",
        path=str(legacy_tmp),
        bytes=legacy_tmp.stat().st_size,
    )


def plan_routes(stream_fps: list[float], shards: int) -> list[int]:
    """算出「哪一路歸哪一片」的靜態路由，索引為 stream_id、值為片編號。

    權重取各路的 fps：本版所有路都跑整天全部片段，工作量與 fps 成正比。依
    `(-fps, stream_id)` 排序後貪婪分配，每次丟進當前權重總和最小的那片。

    **tie-break 必須用 stream_id**，否則 fps 相同的幾路在不同次執行可能落到不同片。
    那不影響輸出正確性（每路的 tracker 狀態獨立、part 檔按路分開），但會讓兩輪量測
    不可比——而分片的整個目的就是量出餘裕。

    **不用 `stream_id % shards`**：平衡完全取決於 registry 的排列順序。

    Args:
        stream_fps: 各路的 fps，索引即 stream_id。
        shards: 設定的分片數（`[tracker].shards`）。大於路數時 clamp 到路數並記一行
            log，不 fail——多起的片不會收到任何 payload，只是白佔一個進程。

    Returns:
        `route`，長度同 `stream_fps`；片編號為 `0 ~ min(shards, 路數) - 1`，且每一片
        都至少分到一路（排序後的前幾路必然落在不同片上）。

    Raises:
        ValueError: `stream_fps` 為空，或 `shards` 小於 1。
    """
    if not stream_fps:
        raise ValueError("stream_fps 是空的，無法規劃路由。")
    if shards < 1:
        raise ValueError(f"分片數必須 >= 1，收到 {shards}。")
    effective = min(shards, len(stream_fps))
    if effective < shards:
        logger.info(
            "分片數多於路數，clamp 到路數",
            configured_shards=shards,
            num_streams=len(stream_fps),
        )
    loads = [0.0] * effective
    route = [0] * len(stream_fps)
    for stream_id in sorted(
        range(len(stream_fps)), key=lambda i: (-stream_fps[i], i)
    ):
        # min 對相同權重取索引最小的那片，分配因此與呼叫次數無關
        shard_id = min(range(effective), key=lambda k: loads[k])
        route[stream_id] = shard_id
        loads[shard_id] += stream_fps[stream_id]
    return route


def merge_parts(results_path: Path, part_paths: list[Path]) -> int:
    """把各片的 part 檔合併成下游看的那一個 parquet，成功後清掉 parts 目錄。

    逐檔 `pq.read_table` 再寫進單一 writer：一支 part 是整天的 1/N，逐檔讀入的峰值與
    天數規模無關。全部 part 由同一份 `TRACKING_RESULTS_SCHEMA` 產出，schema 天然一致；
    混入異質 part 時 `write_table` 會拋錯，那是想要的 fail loud。

    **`N=1` 不特例跳過合併**：特例會讓兩條路徑的測試覆蓋分裂。代價是多一次整天檔的
    讀寫，計入下面那行 log 的耗時。

    **順序是先 `replace()` 成正式檔、再清 parts**。清理失敗只記 warning——正式檔已經
    在位，這時中止只會讓一份完整的結果被判成失敗。

    列順序與改動前不同（改動前是九路按到達順序交錯，現在是逐片相接、片內才交錯）。
    下游 zone／line 都走 `group_by` 向量化，不依賴列順序；比對兩次跑批也一律先用
    `(camera_id, timestamp)` 對齊同一格再逐值比（見根 `CLAUDE.md`）。

    最後的 `rmtree` 會把 `.lock` 一起帶走，**這一步不同於認領時的清殘骸**：那時刪掉
    鎖檔會讓另一個執行在新建的 inode 上取得鎖而兩邊都以為自己獨佔，這裡則是正式檔已經
    在位、本次執行的工作已經結束，此後另一個執行接手重跑與循序跑兩次沒有差別。

    Args:
        results_path: 追蹤結果 parquet 的正式輸出路徑。
        part_paths: 各片的 part 檔，依片編號排序。

    Returns:
        合併後的總列數。

    Raises:
        ValueError: `part_paths` 是空的。
        RuntimeError: 任一支 part 檔不存在。正常情況下父進程的子進程檢查會先擋下崩掉
            的那片，走不到這裡。
    """
    if not part_paths:
        raise ValueError("沒有任何 part 檔可合併。")
    missing = [p for p in part_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"合併時有 {len(missing)} 支 part 檔不存在："
            f"{', '.join(str(p) for p in missing)}。"
            "該片的追蹤進程沒有正常收尾，本次結果不完整，不寫出正式檔名。"
        )
    parts_dir = parts_dir_for(results_path)
    merge_tmp = parts_dir / _MERGE_TMP_FILENAME
    started = time.perf_counter()
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for part_path in part_paths:
            table = pq.read_table(part_path)
            if writer is None:
                writer = pq.ParquetWriter(str(merge_tmp), table.schema)
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    merge_tmp.replace(results_path)
    logger.info(
        "part 檔已合併",
        path=str(results_path),
        parts=len(part_paths),
        rows=total_rows,
        elapsed_seconds=round(time.perf_counter() - started, 1),
    )
    try:
        shutil.rmtree(parts_dir)
    except OSError as exc:
        logger.warning(
            "正式檔已寫出，但清理 parts 目錄失敗，下次執行會一併清掉",
            path=str(parts_dir),
            error=str(exc),
        )
    return total_rows
