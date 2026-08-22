import fcntl
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from vfa_observability import StructuredLogger

from video_analyze.config.constants import TRACKING_RESULTS_SCHEMA

logger = StructuredLogger(component="tracking_results")

_SCHEMA = TRACKING_RESULTS_SCHEMA

# 累積這麼多列就 flush 一個 row group，避免整天追蹤明細（數千萬列）常駐記憶體；
# 用列數而非逐段門檻，因多路串流交錯寫入、片段長度不一
_FLUSH_EVERY_ROWS = 200_000

# 認領暫存檔時，「拿到鎖卻發現檔名已經指向別的 inode」最多重來這麼多次。正常情況一次就
# 過；會重來代表剛好撞上另一個執行的 rename 或 unlink，那是瞬間的動作，不會連續發生。
# 給上限而不是無限重試：真的連續撞上代表有東西在同一條路徑上反覆建檔，該讓它 fail loud
_CLAIM_ATTEMPTS = 5


def tmp_path_for(results_path: Path) -> Path:
    """回傳 `results_path` 對應的暫存檔路徑。

    命名規則只寫在這裡：`TrackingResultCollector` 與 `claim_tmp_slot` 都由此取得。
    兩邊各寫一次的話，日後改了後綴會讓 `claim_tmp_slot` 靜默地守著另一個檔名——
    鎖照樣拿得到、殘檔照樣沒清掉，而且不會有任何錯誤訊息。
    """
    return results_path.with_name(results_path.name + ".tmp")


def _identity_of(path: Path) -> tuple[int, int] | None:
    """`path` 此刻指向哪一個檔案，以 `(st_dev, st_ino)` 表示；不存在時回 `None`。

    用 inode 而不是路徑字串當身分：rename 與 unlink 都只改名字，同一個名字前後可以是
    兩個完全不同的檔案，而「這是不是我那一份」正是要靠這個分辨。
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino)


def _is_file_at(fd: int, path: Path) -> bool:
    """`fd` 開著的是不是此刻 `path` 這個名字指向的那個檔案。"""
    by_fd = os.fstat(fd)
    return (by_fd.st_dev, by_fd.st_ino) == _identity_of(path)


def claim_tmp_slot(results_path: Path) -> tuple[int, tuple[int, int]]:
    """認領這條輸出路徑的暫存檔：擋下並行寫入、清掉前一次執行留下的殘檔。

    追蹤進程正常結束會 `save()`、自己拋例外會 `discard()`，兩條都不留暫存檔；但它被
    SIGKILL 或整機斷電時兩條都走不到，`.tmp` 會留在輸出目錄（issue #113）。那種殘檔
    沒有任何進程會回來收，只能由下一次寫同一條路徑的執行順手清掉——本函式就是那一手。

    **判準是「還有沒有人持有這個 inode 的鎖」，不是檔名、不是 mtime**：多個 bucket
    或多個日期並行時各自寫各自的 `.tmp`（輸出路徑帶 bucket 名與日期），本來就碰不到
    彼此；真正需要判斷的是同一條路徑被兩個執行同時寫，而檔名與 mtime 都分不出「上次
    留下的」與「另一個執行正在寫的」。`flock` 由 kernel 在持有者死亡時釋放——SIGKILL
    與整機重啟都算——正好對上「兩個進程都沒機會執行清理」這個情境。

    殘檔是 `ftruncate` 就地清空而不是 `unlink`：刪掉之後這把鎖就留在一個沒有檔名的
    inode 上，另一個執行馬上能在新建的 inode 上取得鎖，兩邊都以為自己獨佔。就地清空
    則 inode 不換，鎖繼續有效，空間也一樣回收得到。

    回傳的 fd **必須持有到暫存檔不再被寫為止**（`run_track_worker` 持有到進程結束）：
    fd 一關鎖就沒了。

    **拿到鎖之後要再確認手上的 inode 還是那個檔名指向的東西**：`os.open` 與 `flock` 之間
    有空窗，另一個執行可能正好在這期間 `save()`（把同一個 inode rename 成正式檔名）或
    `discard()`（unlink）。那時鎖是拿得到的、`fstat` 也看得到非零大小，接著的 `ftruncate`
    會把對方剛寫完的整天結果清成 0 byte，而 log 還寫著「清掉前一次執行留下的暫存檔」。
    對不上就關掉重開——新的一輪要嘛開到對方留下的新檔，要嘛自己建一個空的。

    `flock` 是 BSD 介面（不在 POSIX 內；POSIX 的 advisory lock 是 `fcntl(F_SETLK)`），
    只在都走這個機制的進程之間有效（`pq.ParquetWriter` 照常開檔、照常寫，不受影響）。
    本 repo 只跑 Linux；輸出目錄若日後掛到 NFS，語義要重新確認——Linux 的 NFS 從 2.6.37
    起才用 POSIX lock 去模擬 flock，跨用戶端的行為與本機不同。

    Args:
        results_path: 追蹤結果 parquet 的正式輸出路徑。

    Returns:
        `(fd, identity)`：持有 `flock` 的 file descriptor，以及它鎖住的那個 inode 的
        `(st_dev, st_ino)`。**身分要從這裡帶走、不能之後再用路徑推導一次**——認領完到
        建 `TrackingResultCollector` 之間這個名字仍可能被換掉，那時 `os.stat(path)` 取到
        的是別人的 inode，之後所有的比對都會拿別人的檔當成自己的。

    Raises:
        RuntimeError: 這條路徑的暫存檔正被另一個進程持有，或連續數輪都在拿到鎖的當下
            發現檔名已經指向別的 inode。
        OSError: 開檔、上鎖或 stat 本身失敗（例如 NFS 回 `ENOLCK`、權限不足）。這類
            錯誤原樣往外拋，不轉成 `RuntimeError`——它們與「有人正在寫」是不同的事，
            蓋掉會讓排查方向偏掉。
    """
    tmp_path = tmp_path_for(results_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(_CLAIM_ATTEMPTS):
        fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claimed = _is_file_at(fd, tmp_path)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                f"暫存檔 {tmp_path} 正被另一個進程持有，本次執行中止。"
                "同一條輸出路徑（同一個 bucket 的同一天）不能有兩個執行同時寫——"
                "兩邊會交錯寫進同一個暫存檔，再各自 rename 成正式檔名；要並行請分開"
                "bucket 或分開日期。若確定沒有別的執行在跑，持有者可能是上一次留下的"
                "孤兒追蹤進程（父進程被 SIGKILL 之後它收不到任何訊號，會一直等在"
                "track_queue.get() 上並占住這把鎖）："
                "以 `pgrep -af video_analyze` 找出來收掉之後再重跑。"
            ) from exc
        except BaseException:
            # flock 的其他失敗（例如 NFS 上的 ENOLCK）與 _is_file_at 的 stat 失敗都原樣
            # 往外拋，但 fd 不能跟著漏掉
            os.close(fd)
            raise
        if claimed:
            break
        os.close(fd)  # 等鎖的期間這個 inode 已經不叫 `.tmp` 了，鎖到的是別人的東西
    else:
        raise RuntimeError(
            f"連續 {_CLAIM_ATTEMPTS} 次認領 {tmp_path} 都在拿到鎖的當下發現它已經換了"
            "inode，本次執行中止。同一條輸出路徑上有其他執行正在頻繁地建立與收尾。"
        )
    residue_bytes = os.fstat(fd).st_size
    if residue_bytes:
        # 走到這裡代表上一個持有者已經不在了（否則拿不到鎖），這個檔沒有人會回來收
        os.ftruncate(fd, 0)
        logger.warning(
            "清掉前一次執行留下的暫存檔",
            path=str(tmp_path),
            bytes=residue_bytes,
        )
    claimed = os.fstat(fd)
    return fd, (claimed.st_dev, claimed.st_ino)


class TrackingResultCollector:
    """收集每格的追蹤結果，累積到門檻列數就 flush 成一個 row group 並清空緩衝。

    flush 內容先寫到 `{results_path}.tmp`，只有 save() 成功才原子性改名成正式檔名；
    中途例外改呼叫 discard() 清掉暫存檔，不留下不完整的 parquet（fail-loud）。
    """

    def __init__(
        self, results_path: Path, tmp_identity: tuple[int, int] | None = None
    ):
        """初始化空緩衝，尚未建立任何 parquet writer（惰性建立於首次 flush）。

        Args:
            results_path: 追蹤結果 parquet 的正式輸出路徑；`save()` 成功前
                資料只會寫在同目錄的 `.tmp` 暫存檔。
            tmp_identity: `claim_tmp_slot` 回傳的暫存檔身分 `(st_dev, st_ino)`。
                寫入、刪除、改名前都會拿它跟路徑此刻指向的 inode 比對，確保動到的是
                自己那一份（見 `_require_own_tmp`）。**必須由認領那一步帶進來，不可以
                在這裡用 `os.stat` 重新推導**：認領完到這裡之間名字仍可能被換掉，那時
                推導到的是別人的 inode，之後每一道比對都會通過而動到別人的檔。
                `None` 代表沒有經過認領（例如測試直接建），身分改為首次 flush 建檔時記下。
        """
        self._results_path = results_path
        self._tmp_path = tmp_path_for(results_path)
        self._tmp_identity = tmp_identity
        self._columns: dict[str, list] = {name: [] for name in _SCHEMA}
        self._pending_rows = 0
        self._total_rows = 0
        self._writer: pq.ParquetWriter | None = None

    def add(
        self,
        camera_id: str,
        frame_index: int,
        timestamp: datetime,
        tracks: np.ndarray,
        foot_points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """把某一格的追蹤結果加入緩衝，累積達門檻列數會自動 flush。

        收 `frame_index`／`timestamp` 兩個純量而非整個 `FramePacket`：這裡本來就只用到
        這兩欄，而 `FramePacket` 另外持有的影格是共享記憶體的 view，只在推論進程內有效。

        Args:
            camera_id: 該影格所屬攝影機的 `stream_dirname`。
            frame_index: 該影格在所屬片段內的序號（從 0 起算）。
            timestamp: 該影格的時間戳（台北在地時間，見 `services/video_reader.py`）。
            tracks: `MultiStreamByteTracker.update` 的輸出（列格式定義見該
                函式的 Returns 說明）；空陣列時不新增任何列。
            foot_points: `[N, 2]` 的落腳點，逐列對應 `tracks`（見
                `services/foot_point.py`）。
            frame_width: 該路的影像寬度（像素）。呼叫端的 `frame_shapes` 存的是
                `(height, width)`，順序傳反不會有型別錯誤，只會讓下游的解析度
                換算靜默算錯。
            frame_height: 該路的影像高度（像素）。

        Raises:
            ValueError: `foot_points` 與 `tracks` 的列數不一致——錯位會讓每一列的
                落腳點配到別條軌跡，是下游查不出來的靜默錯誤。
        """
        if len(foot_points) != len(tracks):
            raise ValueError(
                f"落腳點列數（{len(foot_points)}）與追蹤結果列數（{len(tracks)}）"
                "不一致，兩者必須逐列對應。"
            )
        for track, foot in zip(tracks, foot_points, strict=True):
            x1, y1, x2, y2, track_id = track[:5]
            cols = self._columns
            cols["camera_id"].append(camera_id)
            cols["frame_id"].append(frame_index)
            cols["timestamp"].append(timestamp)
            cols["track_id"].append(int(track_id))
            cols["x1"].append(float(x1))
            cols["y1"].append(float(y1))
            cols["x2"].append(float(x2))
            cols["y2"].append(float(y2))
            cols["foot_x"].append(float(foot[0]))
            cols["foot_y"].append(float(foot[1]))
            cols["frame_width"].append(int(frame_width))
            cols["frame_height"].append(int(frame_height))
            self._pending_rows += 1
        if self._pending_rows >= _FLUSH_EVERY_ROWS:
            self._flush()

    def _flush(self) -> None:
        if self._pending_rows == 0:
            return
        table = pl.DataFrame(self._columns, schema=_SCHEMA).to_arrow()
        if self._writer is None:
            self._tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self._require_own_tmp("開始寫入")
            self._writer = pq.ParquetWriter(str(self._tmp_path), table.schema)
            if self._tmp_identity is None:
                # 沒有經過認領（例如測試直接建 collector）：這一份是這裡建出來的。
                # 有認領的話 `ParquetWriter` 是就地 truncate、inode 不換，身分仍然有效
                self._tmp_identity = _identity_of(self._tmp_path)
        self._writer.write_table(table)
        self._total_rows += self._pending_rows
        for col in self._columns.values():
            col.clear()
        self._pending_rows = 0

    def save(self) -> None:
        """全部串流正常跑完後呼叫：flush 剩餘資料，再把暫存檔原子性地改名成正式檔名。"""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._require_own_tmp("改名成正式檔名")
            self._tmp_path.replace(self._results_path)
        else:
            # 全天沒有任何追蹤結果，仍要寫出一個空的 parquet（維持欄位 schema）
            self._results_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(self._columns, schema=_SCHEMA).write_parquet(
                self._results_path
            )
            # 這條分支不經過 rename，`claim_tmp_slot` 建出來的 0 byte 暫存檔要自己收掉，
            # 否則正常跑完的一天也會在輸出目錄留一個看起來像半成品的檔案
            self._remove_tmp()
        logger.info(
            "追蹤結果已寫入",
            path=str(self._results_path),
            rows=self._total_rows,
        )

    def discard(self) -> None:
        """中途例外時呼叫：關閉暫存檔的 writer 並刪除暫存檔，不留下不完整的輸出。

        關 writer 與刪檔是 `try`／`finally`：`close()` 會寫 parquet 的 footer，磁碟滿的
        時候拋錯，而那正是最需要把半成品收掉的時候——不隔開的話刪檔整段被跳過。
        """
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            self._remove_tmp()

    def _require_own_tmp(self, action: str) -> None:
        """動這個暫存檔之前先確認它還是自己認領的那一份，不是的話 fail loud。

        **兩個時機都要問**，而且要在**寫入之前**問第一次：`pq.ParquetWriter` 以路徑開檔、
        帶 `O_TRUNC`，路徑此刻若已經是別人的檔（有人手動 `rm` 掉暫存檔之後另一個執行
        接手），第一次 flush 就會把對方的內容截掉並蓋寫——真正造成破壞的是這一步，等到
        `save()` 才檢查已經來不及，訊息還會把受害者講反。`replace()` 同樣依路徑，不檔的話
        會把對方**寫到一半**的內容改名成正式檔名，log 照樣印「追蹤結果已寫入 rows=N」，
        下游要到 `pl.read_parquet` 才以「must end with PAR1」崩在離原因很遠的地方。

        `_tmp_identity` 為 `None` 代表這個 collector 沒有經過認領（例如測試直接建），
        沒有可比對的身分，放行。

        Args:
            action: 接下來要做的事，寫進錯誤訊息（例如「開始寫入」）。

        Raises:
            RuntimeError: 暫存檔已不是本次執行認領的那一份。
        """
        if self._tmp_identity is None:
            return
        if _identity_of(self._tmp_path) == self._tmp_identity:
            return
        raise RuntimeError(
            f"暫存檔 {self._tmp_path} 已不是本次執行認領的那一份（被刪掉或換成別的檔），"
            f"不能{action}——那會動到別人的檔案。本次執行中止，請確認沒有其他程序在清理"
            "或寫入這個輸出目錄之後重跑。"
        )

    def _remove_tmp(self) -> None:
        """刪掉本次執行的暫存檔；這個路徑此刻若是別份（inode 不同）就一個字節都不動。

        `claim_tmp_slot` 的 flock 擋得住「兩個執行同時認領」，但擋不住有人手動把暫存檔
        `rm` 掉之後另一個執行接手——那時同一個名字底下已經是別人的檔，依路徑刪會把對方
        寫到一半的內容刪掉，而對方要到最後 `replace()` 才會以 `FileNotFoundError` 爆掉。
        """
        if self._tmp_identity is None:
            return  # 從來沒有建立過自己的暫存檔，沒有東西該由這裡刪
        if _identity_of(self._tmp_path) != self._tmp_identity:
            logger.warning(
                "暫存檔已不是本次執行建立的那一份，略過刪除",
                path=str(self._tmp_path),
            )
            return
        self._tmp_path.unlink()
