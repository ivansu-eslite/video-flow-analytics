"""人流報表的核心演算法：時區轉換、期間彙總、尖峰計算。

所有函式皆為純運算（不做任何檔案 I/O），方便單元測試；I/O 與 orchestration
在 services/report.py。zone（區域佔用）與 line（計數線進出）的彙總各有一組
函式，共用時區轉換與星期規則。
"""

import polars as pl

_WEEKDAY_NAMES = {
    1: "星期一", 2: "星期二", 3: "星期三",
    4: "星期四", 5: "星期五", 6: "星期六",
    7: "星期日",
}


def _with_weekday(df: pl.DataFrame) -> pl.DataFrame:
    """依 `date`（`YYYY-MM-DD` 字串）欄補上中文 `weekday` 欄。"""
    return df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
        .dt.weekday()
        .replace_strict(_WEEKDAY_NAMES)
        .alias("weekday")
    )


def to_taipei(df: pl.DataFrame, column: str = "time_bucket") -> pl.DataFrame:
    """新增 local_time 欄位：直接沿用 column 的 wall-clock 值，不做時區位移。

    `column`（來自 zone_counts.parquet／line_counts.parquet 的 time_bucket）
    在 schema 上正確標記為
    Asia/Taipei——上游交棒進來時就已經是台北時間的 wall-clock 值（見 README 的
    「輸入 / 輸出檔案」檔案契約），因此這裡不能再額外加 8 小時，否則會造成雙重
    位移；只需要去掉 tz 標記、保留原本的 wall-clock 數值即可。

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

    三個可彙總的欄位分兩類：`'entries'` 與 `'dwell_events'` 是事件型計數，上游只在
    事件發生的那一個 bucket 記一次，跨 bucket 用 sum() 加總不會重複計；
    `'unique_visitors'` 則是近似值，同一人跨相鄰 bucket 停留時會被重複計入
    （track_id 未保留到這層無法消除重複）。

    Args:
        df: 含 `local_time`（naive datetime）、`zone`、`metric` 指定欄位的
            資料表（見 `to_taipei`）。
        period_minutes: 彙總的時段粒度（分鐘）。
        metric: 要彙總的欄位名稱（`"entries"`、`"unique_visitors"` 或
            `"dwell_events"`）。

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
    return _with_weekday(rolled).select("date", "weekday", "period", "zone", "value")


def rollup_lines_by_period(df: pl.DataFrame, period_minutes: int) -> pl.DataFrame:
    """依 period_minutes 把已轉本地時間的計數線進出資料彙總成期間×計數線統計。

    `in_count`／`out_count` 都是可疊加的事件次數，跨 bucket 加總不像
    `unique_visitors` 有重複計入的近似問題。`net`（淨進出）在這裡算出，
    `line_counts.parquet` 本身不帶這一欄。

    Args:
        df: 含 `local_time`（naive datetime）、`line_group`、`line`、
            `in_count`、`out_count` 欄位的資料表（見 `to_taipei`）。
        period_minutes: 彙總的時段粒度（分鐘）。

    Returns:
        依 `date`／`period`／`line` 排序的彙總表。輸出欄位：date（字串
        YYYY-MM-DD）、weekday（中文）、period（字串 HH:MM，該期間起始時間）、
        line_group、line、in_count、out_count、net（Int64）。
    """
    rolled = (
        df.with_columns(
            pl.col("local_time").dt.truncate(f"{period_minutes}m").alias("period_start")
        )
        .group_by(["line_group", "line", "period_start"])
        .agg(pl.col("in_count").sum(), pl.col("out_count").sum())
        .with_columns(
            pl.col("period_start").dt.strftime("%Y-%m-%d").alias("date"),
            pl.col("period_start").dt.strftime("%H:%M").alias("period"),
        )
        .sort(["date", "period", "line"])
    )
    return (
        _with_weekday(rolled)
        .with_columns((pl.col("in_count") - pl.col("out_count")).alias("net"))
        .select(
            "date", "weekday", "period", "line_group", "line",
            "in_count", "out_count", "net",
        )
    )


def peak_per_day(rollup_df: pl.DataFrame) -> pl.DataFrame:
    """每個 (date, zone) 取 value 最大的期間；並列時取時間較早的期間。

    Args:
        rollup_df: `rollup_by_period` 的輸出。

    Returns:
        每個 (date, zone) 一列的尖峰統計，含 `peak_period`／`peak_value` 欄位。
    """
    sorted_df = rollup_df.sort(
        ["date", "zone", "value", "period"],
        descending=[False, False, True, False],
    )
    # maintain_order=True 不可省略：unique() 預設不保證輸出列順序，而本函式的結果
    # 會直接寫入 report.xlsx，列序需在重跑間穩定（見同檔的重跑列序回歸測試）。
    peaks = sorted_df.unique(subset=["date", "zone"], keep="first", maintain_order=True)
    return peaks.select(
        "date",
        "weekday",
        "zone",
        pl.col("period").alias("peak_period"),
        pl.col("value").alias("peak_value"),
    )


def peak_lines_per_day(rollup_df: pl.DataFrame, metric_column: str) -> pl.DataFrame:
    """每個 (date, line) 取 metric_column 最大的期間；並列時取時間較早的期間。

    進場尖峰與出場尖峰是同一個計算、只差取哪個量決定尖峰時段，故以
    `metric_column` 參數化而非寫成兩個函式；兩者的輸出都同時帶該期間的
    進場與出場人數。

    Args:
        rollup_df: `rollup_lines_by_period` 的輸出。
        metric_column: 決定尖峰時段的欄位（`"in_count"` 或 `"out_count"`）。

    Returns:
        每個 (date, line) 一列的尖峰統計，含 `peak_period`／`peak_in`／
        `peak_out` 欄位。
    """
    sorted_df = rollup_df.sort(
        ["date", "line", metric_column, "period"],
        descending=[False, False, True, False],
    )
    # maintain_order=True 不可省略，理由同 peak_per_day：列序需在重跑間穩定。
    peaks = sorted_df.unique(subset=["date", "line"], keep="first", maintain_order=True)
    return peaks.select(
        "date",
        "weekday",
        "line_group",
        "line",
        pl.col("period").alias("peak_period"),
        pl.col("in_count").alias("peak_in"),
        pl.col("out_count").alias("peak_out"),
    )
