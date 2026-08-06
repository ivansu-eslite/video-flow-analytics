"""人流 Excel 報表的非 Pydantic 靜態常數。

從 `services/report.py`／`services/stats.py` 抽出的寫死值集中於此，讓報表結構
（分頁名、表頭、排序欄、閥值）與程式邏輯分離。
"""

from pathlib import Path

# 輸出根目錄；為 cwd 相對路徑（見 README「執行位置」）。
OUTPUT_ROOT = Path("outputs")

# 輸入／輸出檔名。
ZONE_COUNTS_FILENAME = "zone_counts.parquet"
LINE_COUNTS_FILENAME = "line_counts.parquet"
REPORT_FILENAME = "report.xlsx"
# 寫檔採「先寫 .tmp 再 rename」以避免半寫入的檔案。
TMP_SUFFIX = ".tmp"

# Excel 工作表名稱。zone 兩頁在加入 line 分頁時一併改名（原為「每小時人流」
# 「每日尖峰」），讓兩組分頁的命名對稱；既有 report.xlsx 的舊分頁不做遷移。
SHEET_ZONE_HOURLY = "各區域人流"
SHEET_ZONE_PEAK = "各區域每日尖峰"
SHEET_LINE_HOURLY = "各出入口人流"
SHEET_LINE_PEAK_IN = "各出入口每日進場尖峰"
SHEET_LINE_PEAK_OUT = "各出入口每日出場尖峰"
SHEET_EVENTS = "活動事件"

# Excel 報表標頭定義。兩個 line 尖峰分頁的表頭相同，只差尖峰時段取自 in_count
# 還是 out_count，故共用同一份 LINE_PEAK_HEADERS。
ZONE_HOURLY_HEADERS = ["日期", "星期", "小時", "區域", "人流量"]
ZONE_PEAK_HEADERS = ["日期", "星期", "區域", "尖峰時段", "尖峰人流", "每日提醒"]
LINE_HOURLY_HEADERS = [
    "日期",
    "星期",
    "小時",
    "群組",
    "計數線",
    "進場人數",
    "出場人數",
    "淨進出",
]
LINE_PEAK_HEADERS = [
    "日期",
    "星期",
    "群組",
    "計數線",
    "尖峰時段",
    "尖峰進場",
    "尖峰出場",
    "每日提醒",
]
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
# 區域／計數線名稱都是全域唯一，故不必把「群組」也放進鍵值。
ZONE_HOURLY_SORT_COLUMNS = ("日期", "小時", "區域")
ZONE_PEAK_SORT_COLUMNS = ("日期", "區域")
LINE_HOURLY_SORT_COLUMNS = ("日期", "小時", "計數線")
LINE_PEAK_SORT_COLUMNS = ("日期", "計數線")

# 用餐時段提醒閥值：(開始小時, 結束小時, 提醒文字)。
MEAL_THRESHOLDS = (
    (11, 14, "加強午餐動線"),
    (17, 20, "加強晚餐動線"),
)
