"""Zone 人流統計的非 Pydantic 靜態常數。

從 `services/zone_map.py` 抽出的寫死值集中於此，讓輸出檔名與 parquet schema
這類契約性的定義與程式邏輯分離。
"""

from pathlib import Path

import polars as pl

# 輸出根目錄；為 cwd 相對路徑（見 README「執行位置」）。
OUTPUT_ROOT = Path("outputs")

# 像素參數的基準解析度寬度：設定檔的 `boundary_band_px_1080p` 是以 1080p（寬 1920）
# 表示的值，執行時各攝影機依 `frame_width / 1920` 換算成自己的實際像素（見 ADR-004）。
BASELINE_FRAME_WIDTH = 1920

# 線段區域寬度的預設值（1080p 基準像素），由實測挑出，見 README「已知限制」。
# `ZoneConfig` 與 `map_zones_daily` 的簽名共用這一份，避免「預設」在同一個套件裡
# 有兩個意思；`config.toml` 無法引用常數，那一份由 tests/test_config.py 鎖住一致性。
DEFAULT_BOUNDARY_BAND_PX_1080P = 25

# 追蹤結果 parquet 必須具備的欄位；缺任一欄代表是舊版 video_analyze 的產物，
# 無法換算像素參數，直接 fail-loud（見 ADR-004、ADR-006）。`frame_height` 本次的換算
# 用不到，仍列為必要：兩欄同時寫入，只有其中一欄代表產物不完整，此時放行等於接受一份
# 來源不明的 parquet。
REQUIRED_TRACKING_COLUMNS = ("frame_width", "frame_height")

# 輸入／輸出檔名。
TRACKING_RESULTS_FILENAME = "tracking_results.parquet"
ZONE_COUNTS_FILENAME = "zone_counts.parquet"
# 寫檔採「先寫 .tmp 再 rename」以避免半寫入的檔案。
TMP_SUFFIX = ".tmp"

# 空輸出也寫出正確 schema 的 parquet；time_bucket tz 沿用 timestamp——上游
# tracking_results.parquet 的 timestamp 已是台北在地時間，見 README 的檔案契約
ZONE_COUNTS_SCHEMA = {
    "camera_id": pl.Utf8,
    "zone": pl.Utf8,
    "time_bucket": pl.Datetime("us", "Asia/Taipei"),
    "unique_visitors": pl.Int64,
    "entries": pl.Int64,
}
