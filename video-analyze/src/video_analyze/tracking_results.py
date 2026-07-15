import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from video_analyze.io.video_reader import FramePacket

logger = logging.getLogger(__name__)

_SCHEMA = {
    "camera_id": pl.Utf8,
    "frame_id": pl.Int64,
    # timestamp 為台北在地時間：檔名為 UTC，已在 io/video_reader.py 解析時轉換成
    # 台北（見該檔 _FILENAME_TZ / _LOCAL_TZ），schema 標記需與來源 tzinfo 一致。
    "timestamp": pl.Datetime("us", "Asia/Taipei"),
    "track_id": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
}

# 累積這麼多列就 flush 一個 row group，避免整天追蹤明細（數千萬列）常駐記憶體；
# 用列數而非逐段門檻，因多路串流交錯寫入、片段長度不一
_FLUSH_EVERY_ROWS = 200_000


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
        self._tmp_path = results_path.with_name(results_path.name + ".tmp")
        self._columns: dict[str, list] = {name: [] for name in _SCHEMA}
        self._pending_rows = 0
        self._total_rows = 0
        self._writer: pq.ParquetWriter | None = None

    def add(
        self,
        camera_id: str,
        packet: FramePacket,
        tracks: np.ndarray,
    ) -> None:
        """把某一格的追蹤結果加入緩衝，累積達門檻列數會自動 flush。

        Args:
            camera_id: 該影格所屬攝影機的 `stream_dirname`。
            packet: 該影格的來源資訊（frame_index、timestamp）。
            tracks: `MultiStreamByteTracker.update` 的輸出（列格式定義見該
                函式的 Returns 說明）；空陣列時不新增任何列。
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            cols = self._columns
            cols["camera_id"].append(camera_id)
            cols["frame_id"].append(packet.frame_index)
            cols["timestamp"].append(packet.timestamp)
            cols["track_id"].append(int(track_id))
            cols["x1"].append(float(x1))
            cols["y1"].append(float(y1))
            cols["x2"].append(float(x2))
            cols["y2"].append(float(y2))
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
        logger.info(
            "追蹤結果已寫入 %s（共 %d 列）。", self._results_path, self._total_rows
        )

    def discard(self) -> None:
        """中途例外時呼叫：關閉暫存檔的 writer 並刪除暫存檔，不留下不完整的輸出。"""
        if self._writer is not None:
            self._writer.close()
        if self._tmp_path.exists():
            self._tmp_path.unlink()
