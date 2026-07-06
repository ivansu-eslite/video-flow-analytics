"""Zone 人流 Excel 報表：離線下游步驟（CLI 的 `report` 子命令）。

讀 `outputs/{bucket}/{date}/zone_counts.parquet`，彙總成跨日累加更新的
`outputs/{bucket}/report.xlsx`。實際的期間彙總／尖峰計算在 report/stats.py；
這裡負責讀檔、驗證、orchestration 與 Excel 讀寫。
"""

import datetime
import logging
from pathlib import Path

import polars as pl

from video_flow_analytics.core.registry import CameraRegistry, load_registry
from video_flow_analytics.report.stats import peak_per_day, rollup_by_period, to_taipei

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")


def _validate_unique_zone_names(registry: CameraRegistry) -> None:
    """報表以 zone 名稱（不含 camera_id）分組，因此要求整份 registry 的 zone
    名稱全域唯一；此驗證只在產報表時檢查，不影響 analyze_daily / zone_mapping。
    """
    names = [zone.name for cam in registry.cameras for zone in cam.parsed_zones()]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise ValueError(
            "camera_registry.yaml 中有跨攝影機重複的 zone 名稱，報表需要 zone "
            f"名稱全域唯一（不只同一攝影機內唯一）: {dupes}"
        )


def _build_report_frames(
    date: datetime.date,
    bucket_dir: str,
    period_minutes: int,
    metric: str,
    bucket_minutes: int,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if period_minutes % bucket_minutes != 0:
        raise ValueError(
            f"report.period_minutes（{period_minutes}）必須是 "
            f"zone.bucket_minutes（{bucket_minutes}）的倍數。"
        )

    bucket_path = Path(bucket_dir)
    registry = load_registry(bucket_path)
    _validate_unique_zone_names(registry)

    counts_path = (
        output_root / bucket_path.name / date.isoformat() / "zone_counts.parquet"
    )
    if not counts_path.exists():
        raise FileNotFoundError(
            f"找不到 zone 人流統計 {counts_path}，"
            "請先執行 map_zones_daily 產生當日 parquet。"
        )

    df = to_taipei(pl.read_parquet(counts_path))
    hourly_df = rollup_by_period(df, period_minutes, metric)
    peak_df = peak_per_day(hourly_df)
    return hourly_df, peak_df
