"""Zone Mapping：離線下游步驟（CLI 的 `zone-map` 子命令）。

讀 `outputs/{bucket}/{date}/tracking_results.parquet`，套上人工維護的 zone 幾何
（`zones.yaml`），輸出每個時段每個區域的人流統計到同層的 `zone_counts.parquet`，
並把當下套用的 zone 定義快照成 `zones_used.yaml` 以供回溯。

實際的 point-in-polygon 判定與聚合演算法在 `zone_mapping/stats.py`；這裡只負責
讀檔、逐攝影機/逐 zone 呼叫演算法、寫檔與快照。
"""

import datetime
import logging
import shutil
from pathlib import Path

import polars as pl

from video_flow_analytics.core.config import settings
from video_flow_analytics.zone_mapping.stats import (
    count_zone_visits,
    validate_zone_cameras,
)
from video_flow_analytics.zone_mapping.zones import load_zones

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")

# 空輸出時仍寫出帶正確欄位 schema 的 parquet，讓下游讀取行為一致。
_ZONE_COUNTS_SCHEMA = {
    "camera_id": pl.Utf8,
    "zone": pl.Utf8,
    "time_bucket": pl.Datetime("us", "UTC"),
    "unique_visitors": pl.Int64,
    "entries": pl.Int64,
}


def map_zones_daily(
    date: datetime.date,
    bucket_dir: str,
    zones_path: Path,
    bucket_minutes: int,
    entry_debounce_frames: int = 1,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """執行單日 zone mapping，回傳 zone_counts.parquet 路徑。"""
    output_dir = output_root / Path(bucket_dir).name / date.isoformat()
    results_path = output_dir / "tracking_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"找不到追蹤結果 {results_path}，請先執行 analyze_daily 產生當日 parquet。"
        )

    zone_registry = load_zones(zones_path)
    df = pl.read_parquet(results_path)
    validate_zone_cameras(zone_registry.cameras, set(df["camera_id"].unique()))

    df = df.with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("foot_x"),
        pl.col("y2").alias("foot_y"),
        pl.col("timestamp").dt.truncate(f"{bucket_minutes}m").alias("time_bucket"),
    )

    frames: list[pl.DataFrame] = []
    for camera_id, cam_zones in zone_registry.cameras.items():
        cam_sub = df.filter(pl.col("camera_id") == camera_id)
        for zone in cam_zones.zones:
            counts = count_zone_visits(
                cam_sub, zone, entry_debounce_frames
            ).with_columns(
                pl.lit(camera_id).alias("camera_id"),
                pl.lit(zone.name).alias("zone"),
            )
            frames.append(counts)

    if frames:
        result = (
            pl.concat(frames)
            .select(list(_ZONE_COUNTS_SCHEMA))
            .sort("camera_id", "zone", "time_bucket")
        )
    else:
        result = pl.DataFrame(schema=_ZONE_COUNTS_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / "zone_counts.parquet"
    tmp_path = counts_path.with_name(counts_path.name + ".tmp")
    result.write_parquet(tmp_path)
    tmp_path.replace(counts_path)

    # 快照當下套用的 zone 定義，讓這份 zone_counts 自帶當天的 zone 依據、可回溯。
    shutil.copyfile(zones_path, output_dir / "zones_used.yaml")

    logger.info(
        "Zone 人流統計已寫入 %s（%d 台攝影機、共 %d 列時段×區域）。",
        counts_path,
        len(zone_registry.cameras),
        result.height,
    )
    return counts_path


def run_zone_map() -> None:
    """zone-map 子命令：從 config.toml 取參數後呼叫 map_zones_daily。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")

    zones_path = Path(settings.zone.zones_path)
    if not zones_path.is_absolute():
        # 相對路徑以 repo 根為基準
        # （此檔為 src/video_flow_analytics/zone_mapping/pipeline.py）
        zones_path = Path(__file__).resolve().parents[3] / zones_path

    map_zones_daily(
        date=settings.input.date,
        bucket_dir=settings.input.bucket_dir,
        zones_path=zones_path,
        bucket_minutes=settings.zone.bucket_minutes,
        entry_debounce_frames=settings.zone.entry_debounce_frames,
    )


if __name__ == "__main__":
    run_zone_map()
