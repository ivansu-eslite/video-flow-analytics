"""Zone 人流報表的核心演算法：時區轉換、期間彙總、尖峰計算、用餐時段規則。

所有函式皆為純運算（不做任何檔案 I/O），方便單元測試；I/O 與 orchestration
在 report/pipeline.py。
"""

import datetime

import polars as pl

_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def weekday_zh(d: datetime.date) -> str:
    return _WEEKDAY_ZH[d.weekday()]


def to_taipei(df: pl.DataFrame, column: str = "time_bucket") -> pl.DataFrame:
    """新增 local_time 欄位：column 轉為台北時區（固定 +8 小時，台北無 DST）。"""
    return df.with_columns(
        (pl.col(column).dt.replace_time_zone(None) + pl.duration(hours=8)).alias(
            "local_time"
        )
    )
