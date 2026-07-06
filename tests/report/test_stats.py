import datetime

import polars as pl

from video_flow_analytics.report.stats import rollup_by_period, to_taipei, weekday_zh


def test_weekday_zh_matches_known_friday():
    # 2026-05-01 是星期五（與 Sample Report.xlsx 的資料一致）
    assert weekday_zh(datetime.date(2026, 5, 1)) == "星期五"


def test_weekday_zh_covers_full_week():
    base = datetime.date(2026, 5, 4)  # 星期一
    expected = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    for offset, name in enumerate(expected):
        assert weekday_zh(base + datetime.timedelta(days=offset)) == name


def test_to_taipei_adds_eight_hours():
    df = pl.DataFrame(
        {
            "time_bucket": [
                datetime.datetime(2026, 5, 1, 11, 0, tzinfo=datetime.timezone.utc)
            ]
        }
    )
    result = to_taipei(df)
    assert result["local_time"][0] == datetime.datetime(2026, 5, 1, 19, 0)


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
