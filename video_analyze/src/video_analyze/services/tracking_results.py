from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from vfa_observability import StructuredLogger

from video_analyze.config.constants import TRACKING_RESULTS_SCHEMA
from video_analyze.services.video_reader import FramePacket

logger = StructuredLogger(component="tracking_results")

_SCHEMA = TRACKING_RESULTS_SCHEMA

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
        foot_points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """把某一格的追蹤結果加入緩衝，累積達門檻列數會自動 flush。

        Args:
            camera_id: 該影格所屬攝影機的 `stream_dirname`。
            packet: 該影格的來源資訊（frame_index、timestamp）。
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
            cols["frame_id"].append(packet.frame_index)
            cols["timestamp"].append(packet.timestamp)
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
