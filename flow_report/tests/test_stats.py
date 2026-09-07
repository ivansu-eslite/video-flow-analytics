import datetime
from zoneinfo import ZoneInfo

import polars as pl

from flow_report.services.stats import (
    peak_lines_per_day,
    peak_per_day,
    rollup_by_period,
    rollup_lines_by_period,
    to_taipei,
)


def test_to_taipei_keeps_wall_clock_time_unchanged():
    # time_bucket 正確標記為 Asia/Taipei，wall-clock 值已是台北時間，
    # to_taipei 不應再額外加 8 小時，否則會造成雙重位移
    df = pl.DataFrame(
        {
            "time_bucket": [
                datetime.datetime(2026, 5, 1, 11, 0, tzinfo=ZoneInfo("Asia/Taipei"))
            ]
        }
    )
    result = to_taipei(df)
    assert result["local_time"][0] == datetime.datetime(2026, 5, 1, 11, 0)


def _make_zone_counts(rows):
    return pl.DataFrame(
        rows,
        schema={
            "local_time": pl.Datetime("us"),
            "zone": pl.Utf8,
            "entries": pl.Int64,
            "unique_visitors": pl.Int64,
            "dwell_events": pl.Int64,
        },
        orient="row",
    )


def test_rollup_by_period_sums_entries_within_hour():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80, 7),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40, 3),
            (datetime.datetime(2026, 5, 1, 19, 45), "checkout", 30, 20, 2),
            (datetime.datetime(2026, 5, 1, 20, 0), "checkout", 5, 5, 1),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="entries")
    checkout_19 = result.filter(
        (pl.col("period") == "19:00") & (pl.col("zone") == "checkout")
    )
    assert checkout_19["value"].to_list() == [180]
    assert checkout_19["date"].to_list() == ["2026-05-01"]
    assert checkout_19["weekday"].to_list() == ["星期五"]


def test_rollup_by_period_supports_unique_visitors_metric():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80, 7),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40, 3),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="unique_visitors")
    assert result["value"].to_list() == [120]


def test_rollup_by_period_sums_dwell_events_within_hour():
    """`dwell_events` 與 `entries` 同型（事件型），跨 bucket 相加不重複計。

    值刻意與同幾列的 `entries`／`unique_visitors` 不同：彙總時取錯欄位就會露餡。
    """
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80, 7),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40, 3),
            (datetime.datetime(2026, 5, 1, 20, 0), "checkout", 5, 5, 1),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="dwell_events")
    values_by_period = dict(zip(result["period"].to_list(), result["value"].to_list()))
    assert values_by_period == {"19:00": 10, "20:00": 1}


def test_rollup_by_period_keeps_zones_separate_for_dwell_events():
    """停留人次也是逐 zone 彙總，不會把別的區域的人次算進來。"""
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80, 7),
            (datetime.datetime(2026, 5, 1, 19, 0), "entrance", 10, 8, 2),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="dwell_events")
    values_by_zone = dict(zip(result["zone"].to_list(), result["value"].to_list()))
    assert values_by_zone == {"checkout": 7, "entrance": 2}


def test_rollup_by_period_keeps_zones_separate():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80, 7),
            (datetime.datetime(2026, 5, 1, 19, 0), "entrance", 10, 8, 2),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="entries")
    values_by_zone = dict(zip(result["zone"].to_list(), result["value"].to_list()))
    assert values_by_zone == {"checkout": 100, "entrance": 10}


def _make_rollup(rows):
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Utf8,
            "weekday": pl.Utf8,
            "period": pl.Utf8,
            "zone": pl.Utf8,
            "value": pl.Int64,
        },
        orient="row",
    )


def test_peak_per_day_picks_max_value_per_zone():
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "18:00", "checkout", 776),
            ("2026-05-01", "星期五", "19:00", "checkout", 1246),
            ("2026-05-01", "星期五", "20:00", "checkout", 300),
            ("2026-05-01", "星期五", "11:00", "entrance", 282),
        ]
    )
    result = peak_per_day(df).sort("zone")
    checkout = result.filter(pl.col("zone") == "checkout").row(0, named=True)
    assert checkout["peak_period"] == "19:00"
    assert checkout["peak_value"] == 1246

    entrance = result.filter(pl.col("zone") == "entrance").row(0, named=True)
    assert entrance["peak_period"] == "11:00"
    assert entrance["peak_value"] == 282


def test_peak_per_day_ties_pick_earlier_period():
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "09:00", "entrance", 100),
            ("2026-05-01", "星期五", "15:00", "entrance", 100),
        ]
    )
    result = peak_per_day(df)
    assert result.row(0, named=True)["peak_period"] == "09:00"


def test_peak_per_day_rerun_produces_identical_row_order():
    # 同輸入重跑兩次，輸出（含列序）須逐值一致：peak_per_day 的結果會直接寫入
    # report.xlsx，若列序不穩定，同一天資料在不同次執行間會產生無意義的 diff。
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "18:00", "checkout", 776),
            ("2026-05-01", "星期五", "19:00", "checkout", 1246),
            ("2026-05-01", "星期五", "11:00", "entrance", 282),
            ("2026-05-02", "星期六", "09:00", "entrance", 50),
            ("2026-05-02", "星期六", "15:00", "entrance", 50),
        ]
    )
    first = peak_per_day(df)
    second = peak_per_day(df)
    assert first.equals(second)


def _make_line_counts(rows):
    return pl.DataFrame(
        rows,
        schema={
            "local_time": pl.Datetime("us"),
            "line_group": pl.Utf8,
            "line": pl.Utf8,
            "in_count": pl.Int64,
            "out_count": pl.Int64,
        },
        orient="row",
    )


def test_rollup_lines_by_period_sums_both_directions_within_hour():
    df = _make_line_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "四樓書店", "主入口", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 15), "四樓書店", "主入口", 50, 40),
            (datetime.datetime(2026, 5, 1, 20, 0), "四樓書店", "主入口", 5, 5),
        ]
    )
    result = rollup_lines_by_period(df, period_minutes=60)
    hour_19 = result.filter(pl.col("period") == "19:00").row(0, named=True)
    assert hour_19["in_count"] == 150
    assert hour_19["out_count"] == 120
    assert hour_19["date"] == "2026-05-01"
    assert hour_19["weekday"] == "星期五"
    assert hour_19["line_group"] == "四樓書店"


def test_rollup_lines_by_period_net_is_in_minus_out():
    # 淨進出在彙總時算出（line_counts.parquet 沒有這一欄），出多於進時為負數
    df = _make_line_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "四樓書店", "主入口", 10, 4),
            (datetime.datetime(2026, 5, 1, 19, 0), "四樓書店", "側門", 3, 12),
        ]
    )
    result = rollup_lines_by_period(df, period_minutes=60)
    net_by_line = dict(zip(result["line"].to_list(), result["net"].to_list()))
    assert net_by_line == {"主入口": 6, "側門": -9}


def test_rollup_lines_by_period_keeps_lines_separate():
    df = _make_line_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "四樓書店", "主入口", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 0), "三樓文具", "側門", 10, 8),
        ]
    )
    result = rollup_lines_by_period(df, period_minutes=60)
    assert result.height == 2
    groups = dict(zip(result["line"].to_list(), result["line_group"].to_list()))
    assert groups == {"主入口": "四樓書店", "側門": "三樓文具"}


def test_rollup_lines_by_period_handles_empty_input():
    # 當日無任何跨越事件時 line_counting 會寫出 0 列的正常產物，不可因此崩潰
    result = rollup_lines_by_period(_make_line_counts([]), period_minutes=60)
    assert result.height == 0
    assert result.columns == [
        "date", "weekday", "period", "line_group", "line",
        "in_count", "out_count", "net",
    ]


def _make_line_rollup(rows):
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Utf8,
            "weekday": pl.Utf8,
            "period": pl.Utf8,
            "line_group": pl.Utf8,
            "line": pl.Utf8,
            "in_count": pl.Int64,
            "out_count": pl.Int64,
            "net": pl.Int64,
        },
        orient="row",
    )


def test_peak_lines_per_day_metric_column_decides_peak_period():
    # 同一份資料，進場尖峰與出場尖峰落在不同時段：兩個分頁的差別只有這個
    df = _make_line_rollup(
        [
            ("2026-05-01", "星期五", "12:00", "四樓書店", "主入口", 200, 30, 170),
            ("2026-05-01", "星期五", "19:00", "四樓書店", "主入口", 50, 400, -350),
        ]
    )
    peak_in = peak_lines_per_day(df, "in_count").row(0, named=True)
    assert peak_in["peak_period"] == "12:00"
    assert (peak_in["peak_in"], peak_in["peak_out"]) == (200, 30)

    peak_out = peak_lines_per_day(df, "out_count").row(0, named=True)
    assert peak_out["peak_period"] == "19:00"
    assert (peak_out["peak_in"], peak_out["peak_out"]) == (50, 400)


def test_peak_lines_per_day_ties_pick_earlier_period():
    df = _make_line_rollup(
        [
            ("2026-05-01", "星期五", "09:00", "四樓書店", "主入口", 100, 5, 95),
            ("2026-05-01", "星期五", "15:00", "四樓書店", "主入口", 100, 7, 93),
        ]
    )
    result = peak_lines_per_day(df, "in_count")
    assert result.row(0, named=True)["peak_period"] == "09:00"


def test_peak_lines_per_day_one_row_per_date_and_line():
    df = _make_line_rollup(
        [
            ("2026-05-01", "星期五", "09:00", "四樓書店", "主入口", 100, 5, 95),
            ("2026-05-01", "星期五", "15:00", "四樓書店", "側門", 20, 7, 13),
            ("2026-05-02", "星期六", "09:00", "四樓書店", "主入口", 30, 1, 29),
        ]
    )
    result = peak_lines_per_day(df, "in_count")
    assert result.height == 3
    assert result.columns == [
        "date", "weekday", "line_group", "line",
        "peak_period", "peak_in", "peak_out",
    ]


def test_peak_lines_per_day_rerun_produces_identical_row_order():
    # 理由同 peak_per_day：結果直接寫入 report.xlsx，列序需在重跑間穩定
    df = _make_line_rollup(
        [
            ("2026-05-01", "星期五", "18:00", "四樓書店", "主入口", 776, 700, 76),
            ("2026-05-01", "星期五", "19:00", "四樓書店", "主入口", 1246, 900, 346),
            ("2026-05-02", "星期六", "09:00", "四樓書店", "側門", 50, 50, 0),
            ("2026-05-02", "星期六", "15:00", "四樓書店", "側門", 50, 60, -10),
        ]
    )
    assert peak_lines_per_day(df, "in_count").equals(peak_lines_per_day(df, "in_count"))


def test_peak_lines_per_day_handles_empty_input():
    # 當日無任何跨越事件時 line_counting 會寫出 0 列的正常產物；欄位仍須完整，
    # 少了任何一欄都會讓 report.py 的 _append_rows 取值時拋 ColumnNotFoundError。
    result = peak_lines_per_day(_make_line_rollup([]), "in_count")
    assert result.height == 0
    assert result.columns == [
        "date", "weekday", "line_group", "line",
        "peak_period", "peak_in", "peak_out",
    ]
