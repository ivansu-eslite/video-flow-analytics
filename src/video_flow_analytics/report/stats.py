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


def rollup_by_period(
    df: pl.DataFrame, period_minutes: int, metric: str
) -> pl.DataFrame:
    """把已轉為本地時間的 zone 人流資料，依 period_minutes 彙總成期間×區域的統計。

    輸入需含 local_time（naive datetime）、zone、metric 指定的欄位。
    輸出欄位：date（字串 YYYY-MM-DD）、weekday（中文）、period（字串 HH:MM，
    該期間起始時間）、zone、value（Int64）。
    """
    rolled = (
        df.with_columns(
            pl.col("local_time").dt.truncate(f"{period_minutes}m").alias("period_start")
        )
        .group_by(["zone", "period_start"])
        .agg(pl.col(metric).sum().alias("value"))
        .with_columns(
            pl.col("period_start").dt.strftime("%Y-%m-%d").alias("date"),
            pl.col("period_start").dt.strftime("%H:%M").alias("period"),
        )
        .select("date", "period", "zone", "value")
        .sort(["date", "period", "zone"])
    )
    weekdays = [
        weekday_zh(datetime.date.fromisoformat(d)) for d in rolled["date"].to_list()
    ]
    return rolled.with_columns(pl.Series("weekday", weekdays)).select(
        "date", "weekday", "period", "zone", "value"
    )
