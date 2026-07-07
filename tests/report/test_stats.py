import datetime

import polars as pl

from video_flow_analytics.report.stats import (
    meal_time_reminder,
    peak_per_day,
    rollup_by_period,
    to_taipei,
    weekday_zh,
)


def test_weekday_zh_matches_known_friday():
    # 2026-05-01 是星期五（與 Sample Report.xlsx 的資料一致）
    assert weekday_zh(datetime.date(2026, 5, 1)) == "星期五"


def test_weekday_zh_covers_full_week():
    base = datetime.date(2026, 5, 4)  # 星期一
    expected = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    for offset, name in enumerate(expected):
        assert weekday_zh(base + datetime.timedelta(days=offset)) == name


def test_to_taipei_keeps_wall_clock_time_unchanged():
    # 攝影機錄影時鐘本身就是台北時間（UTC+8），time_bucket 雖然被標記成 UTC，
    # 實際 wall-clock 值已經是台北時間，to_taipei 不應該再額外加 8 小時，
    # 否則會造成雙重位移。
    df = pl.DataFrame(
        {
            "time_bucket": [
                datetime.datetime(2026, 5, 1, 11, 0, tzinfo=datetime.timezone.utc)
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
        },
        orient="row",
    )


def test_rollup_by_period_sums_entries_within_hour():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40),
            (datetime.datetime(2026, 5, 1, 19, 45), "checkout", 30, 20),
            (datetime.datetime(2026, 5, 1, 20, 0), "checkout", 5, 5),
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
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="unique_visitors")
    assert result["value"].to_list() == [120]


def test_rollup_by_period_keeps_zones_separate():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 0), "entrance", 10, 8),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="entries")
    values_by_zone = dict(zip(result["zone"].to_list(), result["value"].to_list()))
    assert values_by_zone == {"checkout": 100, "entrance": 10}


def test_meal_time_reminder_boundaries():
    cases = {
        10: "無",
        11: "加強午餐動線",
        13: "加強午餐動線",
        14: "無",
        16: "無",
        17: "加強晚餐動線",
        19: "加強晚餐動線",
        20: "無",
    }
    for hour, expected in cases.items():
        assert meal_time_reminder(hour) == expected, hour


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
    assert checkout["reminder"] == "加強晚餐動線"

    entrance = result.filter(pl.col("zone") == "entrance").row(0, named=True)
    assert entrance["peak_period"] == "11:00"
    assert entrance["reminder"] == "加強午餐動線"


def test_peak_per_day_ties_pick_earlier_period():
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "09:00", "entrance", 100),
            ("2026-05-01", "星期五", "15:00", "entrance", 100),
        ]
    )
    result = peak_per_day(df)
    assert result.row(0, named=True)["peak_period"] == "09:00"
