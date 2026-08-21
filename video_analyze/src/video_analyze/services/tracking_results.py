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


def tmp_path_for(results_path: Path) -> Path:
    """回傳 `results_path` 對應的暫存檔路徑。

    命名規則只寫在這裡：`TrackingResultCollector` 與 `claim_tmp_slot` 都由此取得。
    兩邊各寫一次的話，日後改了後綴會讓 `claim_tmp_slot` 靜默地守著另一個檔名——
    鎖照樣拿得到、殘檔照樣沒清掉，而且不會有任何錯誤訊息。
    """
    return results_path.with_name(results_path.name + ".tmp")


def claim_tmp_slot(results_path: Path) -> int:
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

    `flock` 是 POSIX advisory lock，只在都走這個機制的進程之間有效（`pq.ParquetWriter`
    照常開檔、照常寫，不受影響）。本 repo 只跑 Linux；輸出目錄若日後掛到 NFS，flock 的
    語義要重新確認。

    Args:
        results_path: 追蹤結果 parquet 的正式輸出路徑。

    Returns:
        持有 `flock` 的 file descriptor。

    Raises:
        RuntimeError: 這條路徑的暫存檔正被另一個執行中的進程持有。
    """
    tmp_path = tmp_path_for(results_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            f"暫存檔 {tmp_path} 正被另一個執行中的進程持有，本次執行中止。"
            "同一條輸出路徑（同一個 bucket 的同一天）不能有兩個執行同時寫——"
            "兩邊會交錯寫進同一個暫存檔，再各自 rename 成正式檔名。"
            "要並行請分開 bucket 或分開日期。"
        ) from exc
    residue_bytes = os.fstat(fd).st_size
    if residue_bytes:
        # 走到這裡代表上一個持有者已經不在了（否則拿不到鎖），這個檔沒有人會回來收
        os.ftruncate(fd, 0)
        logger.warning(
            "清掉前一次執行留下的暫存檔",
            path=str(tmp_path),
            bytes=residue_bytes,
        )
    return fd


class TrackingResultCollector:
    """收集每格的追蹤結果，累積到門檻列數就 flush 成一個 row group 並清空緩衝。

    flush 內容先寫到 `{results_path}.tmp`，只有 save() 成功才原子性改名成正式檔名；
    中途例外改呼叫 discard() 清掉暫存檔，不留下不完整的 parquet（fail-loud）。
    """

    def __init__(self, results_path: Path):
        """初始化空緩衝，尚未建立任何 parquet writer（惰性建立於首次 flush）。

        Args:
            results_path: 追蹤結果 parquet 的正式輸出路徑；`save()` 成功前
                資料只會寫在同目錄的 `.tmp` 暫存檔。
        """
        self._results_path = results_path
        self._tmp_path = tmp_path_for(results_path)
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
            self._writer = pq.ParquetWriter(str(self._tmp_path), table.schema)
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
            self._tmp_path.replace(self._results_path)
        else:
            # 全天沒有任何追蹤結果，仍要寫出一個空的 parquet（維持欄位 schema）
            self._results_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(self._columns, schema=_SCHEMA).write_parquet(
                self._results_path
            )
            # 這條分支不經過 rename，`claim_tmp_slot` 建出來的 0 byte 暫存檔要自己收掉，
            # 否則正常跑完的一天也會在輸出目錄留一個看起來像半成品的檔案
            self._tmp_path.unlink(missing_ok=True)
        logger.info(
            "追蹤結果已寫入",
            path=str(self._results_path),
            rows=self._total_rows,
        )

    def discard(self) -> None:
        """中途例外時呼叫：關閉暫存檔的 writer 並刪除暫存檔，不留下不完整的輸出。"""
        if self._writer is not None:
            self._writer.close()
        if self._tmp_path.exists():
            self._tmp_path.unlink()
