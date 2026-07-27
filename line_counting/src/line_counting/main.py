"""Line Counting：離線下游步驟的 CLI 進入點（`line_counting`）。

從 `config.toml` 取參數後呼叫 `count_lines_daily`，把 `tracking_results.parquet`
依各攝影機的計數線幾何轉成 `line_counts.parquet`。核心邏輯在
`services/line_map.py`。
"""

from line_counting.models.config import settings
from line_counting.services.line_map import count_lines_daily


def main() -> None:
    """`line_counting` 的進入點：從 `config.toml` 取參數後呼叫 `count_lines_daily`。

    Raises:
        ValueError: `config.toml` 的 `[input].date` 未設定。
    """
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")

    count_lines_daily(
        date=settings.input.date,
        bucket_dir=settings.input.bucket_dir,
        bucket_minutes=settings.line.bucket_minutes,
        crossing_band_px=settings.line.crossing_band_px,
    )


if __name__ == "__main__":
    main()
