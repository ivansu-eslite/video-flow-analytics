"""Zone 人流 Excel 報表的非 Pydantic 靜態常數。

從 `services/report.py`／`services/stats.py` 抽出的寫死值集中於此，讓報表結構
（分頁名、表頭、排序欄、閥值）與程式邏輯分離。
"""

from pathlib import Path

# 輸出根目錄；為 cwd 相對路徑（見 README「執行位置」）。
OUTPUT_ROOT = Path("outputs")

# 輸入／輸出檔名。
ZONE_COUNTS_FILENAME = "zone_counts.parquet"
REGISTRY_SNAPSHOT_FILENAME = "camera_registry_used.yaml"
REPORT_FILENAME = "report.xlsx"
# 寫檔採「先寫 .tmp 再 rename」以避免半寫入的檔案。
TMP_SUFFIX = ".tmp"

# Excel 工作表名稱。
SHEET_HOURLY = "每小時人流"
SHEET_PEAK = "每日尖峰"
SHEET_EVENTS = "活動事件"

# Excel 報表標頭定義。
HOURLY_HEADERS = ["日期", "星期", "小時", "區域", "人流量"]
PEAK_HEADERS = ["日期", "星期", "區域", "尖峰時段", "尖峰人流", "每日提醒"]
EVENTS_HEADERS = [
    "日期",
    "星期",
    "開始時間",
    "結束時間",
    "區域",
    "活動名稱",
    "活動類型",
]

# 各分頁欄寬。
COLUMN_WIDTH = 14

# 排序用的鍵值組合（使用欄位名稱而非索引，避免欄位順序調整時忘記同步改數字索引）。
HOURLY_SORT_COLUMNS = ("日期", "小時", "區域")
PEAK_SORT_COLUMNS = ("日期", "區域")

# 用餐時段提醒閥值：(開始小時, 結束小時, 提醒文字)。
MEAL_THRESHOLDS = (
    (11, 14, "加強午餐動線"),
    (17, 20, "加強晚餐動線"),
)
