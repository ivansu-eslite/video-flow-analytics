"""Zone 人流報表的核心演算法：時區轉換、期間彙總、尖峰計算、用餐時段規則。

所有函式皆為純運算（不做任何檔案 I/O），方便單元測試；I/O 與 orchestration
在 report/pipeline.py。
"""

import datetime

import polars as pl

_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def weekday_zh(d: datetime.date) -> str:
    """把日期轉成中文星期幾。

    Args:
        d: 要轉換的日期。

    Returns:
        「星期一」～「星期日」其中之一。
    """
    return _WEEKDAY_ZH[d.weekday()]


def to_taipei(df: pl.DataFrame, column: str = "time_bucket") -> pl.DataFrame:
    """新增 local_time 欄位：直接沿用 column 的 wall-clock 值，不做時區位移。

    `column`（來自 zone_counts.parquet 的 time_bucket）在 schema 上正確標記為
    Asia/Taipei（見 io/video_reader.py 的 `_RECORDING_TZ`），本身就已經是台北
    時間的 wall-clock 值，因此這裡不能再額外加 8 小時，否則會造成雙重位移；
    只需要去掉 tz 標記、保留原本的 wall-clock 數值即可。

    Args:
        df: 含 `column` 欄位的資料表。
        column: 來源時間欄位名稱，需標記為 Asia/Taipei。

    Returns:
        新增 `local_time`（naive datetime）欄位後的資料表。
    """
    return df.with_columns(
        pl.col(column).dt.replace_time_zone(None).alias("local_time")
    )


def rollup_by_period(
    df: pl.DataFrame, period_minutes: int, metric: str
) -> pl.DataFrame:
    """依 period_minutes 把已轉本地時間的 zone 人流資料彙總成期間×區域統計。

    輸入需含 local_time（naive datetime）、zone、metric 指定的欄位。

    metric='unique_visitors' 是近似值：跨 bucket 用 sum() 彙總會讓同一人跨相鄰
    bucket 停留時被重複計入（track_id 未保留到這層無法消除重複）；'entries' 不受影響。

    Args:
        df: 含 `local_time`（naive datetime）、`zone`、`metric` 指定欄位的
            資料表（見 `to_taipei`）。
        period_minutes: 彙總的時段粒度（分鐘）。
        metric: 要彙總的欄位名稱（`"entries"` 或 `"unique_visitors"`）。

    Returns:
        依 `date`／`weekday`／`period`／`zone` 排序的彙總表。輸出欄位：date
        （字串 YYYY-MM-DD）、weekday（中文）、period（字串 HH:MM，該期間
        起始時間）、zone、value（Int64）。
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


def meal_time_reminder(hour: int) -> str:
    """依尖峰時段所在小時給出用餐時段提醒文字。

    Args:
        hour: 尖峰期間起始的小時（0-23）。

    Returns:
        午餐/晚餐時段提醒文字；不在用餐時段則為「無」。
    """
    if 11 <= hour < 14:
        return "加強午餐動線"
    if 17 <= hour < 20:
        return "加強晚餐動線"
    return "無"


def peak_per_day(rollup_df: pl.DataFrame) -> pl.DataFrame:
    """每個 (date, zone) 取 value 最大的期間；並列時取時間較早的期間。

    Args:
        rollup_df: `rollup_by_period` 的輸出。

    Returns:
        每個 (date, zone) 一列的尖峰統計，含 `peak_period`／`peak_value`／
        `reminder` 欄位。
    """
    sorted_df = rollup_df.sort(
        ["date", "zone", "value", "period"],
        descending=[False, False, True, False],
    )
    peaks = sorted_df.group_by(["date", "zone"], maintain_order=True).first()
    reminders = [
        meal_time_reminder(int(period.split(":")[0]))
        for period in peaks["period"].to_list()
    ]
    return peaks.with_columns(pl.Series("reminder", reminders)).select(
        "date",
        "weekday",
        "zone",
        pl.col("period").alias("peak_period"),
        pl.col("value").alias("peak_value"),
        "reminder",
    )
