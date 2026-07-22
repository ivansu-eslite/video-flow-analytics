"""Zone 人流 Excel 報表：離線下游步驟的 CLI 進入點（`flow-report`）。

從 `config.toml` 取參數後呼叫 `export_report_daily`，把 `zone_counts.parquet`
彙總成跨日累加更新的 `report.xlsx`。核心邏輯在 `services/report.py`。
"""

from flow_report.models.config import settings
from flow_report.services.report import export_report_daily


def main() -> None:
    """`flow-report` 的進入點：從 `config.toml` 取參數後呼叫 `export_report_daily`。

    Raises:
        ValueError: `config.toml` 的 `[input].date` 未設定。
    """
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")

    export_report_daily(
        date=settings.input.date,
        bucket_dir=settings.input.bucket_dir,
        period_minutes=settings.report.period_minutes,
        metric=settings.report.metric,
        on_duplicate_date=settings.report.on_duplicate_date,
        bucket_minutes=settings.zone.bucket_minutes,
    )


if __name__ == "__main__":
    main()
