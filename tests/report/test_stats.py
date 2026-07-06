import datetime

import polars as pl

from video_flow_analytics.report.stats import to_taipei, weekday_zh


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
