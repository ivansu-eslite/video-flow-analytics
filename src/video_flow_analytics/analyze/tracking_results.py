import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from video_flow_analytics.io.video_reader import FramePacket

logger = logging.getLogger(__name__)

_SCHEMA = {
    "camera_id": pl.Utf8,
    "frame_id": pl.Int64,
    "timestamp": pl.Datetime("us", "UTC"),
    "track_id": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
}

# 每累積這麼多列就 flush 成一個 parquet row group 並清空記憶體緩衝，避免整天的
# 追蹤明細（單日多鏡頭可能有數千萬列）常駐記憶體。以列數為門檻而非「逐段」，
# 是因為多路串流交錯寫入、片段長度不一，列數門檻能給出穩定的記憶體上限。
_FLUSH_EVERY_ROWS = 200_000


class TrackingResultCollector:
    """收集每格的追蹤結果，累積到門檻列數就 flush 成一個 row group 並清空緩衝。

    flush 的內容先寫到 `{results_path}.tmp`；只有 save() 成功時才會把它原子性地
    改名成正式檔名。中途例外時呼叫 discard() 清掉暫存檔，確保不會留下不完整的
    tracking_results.parquet（fail-loud）。
    """

    def __init__(self, results_path: Path):
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
        # tracks 每列為 BYTETracker 輸出 [x1, y1, x2, y2, track_id, score, cls, idx]
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
