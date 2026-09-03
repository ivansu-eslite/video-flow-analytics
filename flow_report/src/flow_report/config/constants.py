"""人流 Excel 報表的非 Pydantic 靜態常數。

從 `services/report.py` 抽出的寫死值集中於此，讓報表結構（分頁名、表頭、排序欄）
與程式邏輯分離。
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
SHEET_ZONE_DWELL = "各區域停留人次"
SHEET_LINE_HOURLY = "各出入口人流"
SHEET_LINE_PEAK_IN = "各出入口每日進場尖峰"
SHEET_LINE_PEAK_OUT = "各出入口每日出場尖峰"
SHEET_EVENTS = "活動事件"

# Excel 報表的欄位定義：(`services/stats.py` 輸出的資料欄名, 中文表頭)。
#
# 兩者寫在同一個序對而非兩份清單，是因為寫檔時是**按欄名取值**（見 report.py 的
# `_append_rows`）：資料側改了欄名或少了欄位，寫檔階段就會拋錯，不會靜默把值填進
# 錯的表頭底下。序對的順序即分頁的欄序，與資料側的 `select` 順序無關。
#
# 兩個 line 尖峰分頁的欄位相同，只差尖峰時段取自 in_count 還是 out_count，故共用
# 同一份 LINE_PEAK_COLUMNS。
ZONE_HOURLY_COLUMNS = (
    ("date", "日期"),
    ("weekday", "星期"),
    ("period", "小時"),
    ("zone", "區域"),
    ("value", "人流量"),
)
# 停留分頁的前四欄與 ZONE_HOURLY_COLUMNS 相同，只有值欄不同：它彙總的是
# `dwell_events`（達到停留門檻的**人次**，同一人同一時段內兩段都達標計 2），
# 與「人流量」不是子集關係，故表頭不共用、也不可相減。門檻秒數只在
# `zone_mapping` 的 config.toml，報表側無從得知，因此不出現在表頭（見 ADR-017）。
ZONE_DWELL_COLUMNS = (
    ("date", "日期"),
    ("weekday", "星期"),
    ("period", "小時"),
    ("zone", "區域"),
    ("value", "停留人次"),
)
ZONE_PEAK_COLUMNS = (
    ("date", "日期"),
    ("weekday", "星期"),
    ("zone", "區域"),
    ("peak_period", "尖峰時段"),
    ("peak_value", "尖峰人流"),
)
LINE_HOURLY_COLUMNS = (
    ("date", "日期"),
    ("weekday", "星期"),
    ("period", "小時"),
    ("line_group", "群組"),
    ("line", "計數線"),
    ("in_count", "進場人數"),
    ("out_count", "出場人數"),
    ("net", "淨進出"),
)
LINE_PEAK_COLUMNS = (
    ("date", "日期"),
    ("weekday", "星期"),
    ("line_group", "群組"),
    ("line", "計數線"),
    ("peak_period", "尖峰時段"),
    ("peak_in", "尖峰進場"),
    ("peak_out", "尖峰出場"),
)
# 活動事件的寫入者是其他來源，本階段只建表頭、沒有資料側，故維持純表頭清單。
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
# 停留分頁的排序欄與 ZONE_HOURLY_SORT_COLUMNS 內容相同，但分開一份：兩頁的值欄
# 語義不同，日後任一頁改欄序時不該被迫連動另一頁。
ZONE_DWELL_SORT_COLUMNS = ("日期", "小時", "區域")
ZONE_PEAK_SORT_COLUMNS = ("日期", "區域")
LINE_HOURLY_SORT_COLUMNS = ("日期", "小時", "計數線")
LINE_PEAK_SORT_COLUMNS = ("日期", "計數線")
