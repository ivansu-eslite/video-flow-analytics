"""非 Pydantic 的靜態常數：輸出根、追蹤結果檔名與 parquet schema。

與各套件慣例一致，把輸出契約（路徑、檔名、欄位 schema）從邏輯模組抽出集中於此；
模組私有的調校常數（湊批等待、環形緩衝 slot 數、flush 門檻等）仍留在各自模組。
"""

from pathlib import Path

import polars as pl

# 輸出根目錄（cwd 相對）：實際輸出會再掛上 bucket 名稱，避免不同 bucket 互相覆蓋。
OUTPUT_ROOT = Path("outputs")

# 追蹤結果 parquet 的檔名。
TRACKING_RESULTS_FILENAME = "tracking_results.parquet"

TRACKING_RESULTS_SCHEMA = {
    "camera_id": pl.Utf8,
    "frame_id": pl.Int64,
    # timestamp 為台北在地時間：檔名為 UTC，已在 services/video_reader.py 解析時轉換成
    # 台北（見該檔 _FILENAME_TZ / _LOCAL_TZ），schema 標記需與來源 tzinfo 一致。
    "timestamp": pl.Datetime("us", "Asia/Taipei"),
    "track_id": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
}
