"""計數線進出人數統計的非 Pydantic 靜態常數。

從 `services/line_map.py` 抽出的寫死值集中於此，讓輸出檔名與 parquet schema
這類契約性的定義與程式邏輯分離。
"""

from pathlib import Path

import polars as pl

# 輸出根目錄；為 cwd 相對路徑（見 README「執行位置」）。
OUTPUT_ROOT = Path("outputs")

# 輸入／輸出檔名。
TRACKING_RESULTS_FILENAME = "tracking_results.parquet"
LINE_COUNTS_FILENAME = "line_counts.parquet"
REGISTRY_SNAPSHOT_FILENAME = "camera_registry_used.yaml"
# 寫檔採「先寫 .tmp 再 rename」以避免半寫入的檔案。
TMP_SUFFIX = ".tmp"

# 空輸出也寫出正確 schema 的 parquet；time_bucket tz 沿用 timestamp——上游
# tracking_results.parquet 的 timestamp 已是台北在地時間，見 README 的檔案契約
LINE_COUNTS_SCHEMA = {
    "camera_id": pl.Utf8,
    "line": pl.Utf8,
    "time_bucket": pl.Datetime("us", "Asia/Taipei"),
    "in_count": pl.Int64,
    "out_count": pl.Int64,
}
