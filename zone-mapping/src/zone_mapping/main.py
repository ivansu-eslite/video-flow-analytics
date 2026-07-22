"""Zone Mapping：離線下游步驟的 CLI 進入點（`zone-mapping`）。

從 `config.toml` 取參數後呼叫 `map_zones_daily`，把 `tracking_results.parquet`
依各攝影機的 zone 幾何轉成 `zone_counts.parquet`。核心邏輯在
`services/zone_map.py`。
"""

from zone_mapping.models.config import settings
from zone_mapping.services.zone_map import map_zones_daily


def main() -> None:
    """`zone-mapping` 的進入點：從 `config.toml` 取參數後呼叫 `map_zones_daily`。

    Raises:
        ValueError: `config.toml` 的 `[input].date` 未設定。
    """
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")

    map_zones_daily(
        date=settings.input.date,
        bucket_dir=settings.input.bucket_dir,
        bucket_minutes=settings.zone.bucket_minutes,
        entry_debounce_frames=settings.zone.entry_debounce_frames,
    )


if __name__ == "__main__":
    main()
