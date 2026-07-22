"""Zone Mapping 的核心編排：讀檔、逐攝影機/逐 zone 套用演算法、寫檔與快照。

讀 `outputs/{bucket}/{date}/tracking_results.parquet`，套上人工維護在
`camera_registry.yaml` 各攝影機底下的 zone 幾何，輸出每個時段每個區域的人流統計
到同層的 `zone_counts.parquet`，並把當下套用的 camera_registry.yaml 快照成
`camera_registry_used.yaml` 以供回溯。

實際的 point-in-polygon 判定與聚合演算法在 `services/stats.py`。
"""

import datetime
import shutil
from pathlib import Path

import polars as pl
from vfa_observability import StructuredLogger
from vfa_registry import load_registry, parse_and_validate_zones, registry_path

from zone_mapping.config.constants import (
    OUTPUT_ROOT,
    REGISTRY_SNAPSHOT_FILENAME,
    TMP_SUFFIX,
    TRACKING_RESULTS_FILENAME,
    ZONE_COUNTS_FILENAME,
    ZONE_COUNTS_SCHEMA,
)
from zone_mapping.services.stats import count_zone_visits, validate_zone_cameras

logger = StructuredLogger(component="zone_map")


def map_zones_daily(
    date: datetime.date,
    bucket_dir: str,
    bucket_minutes: int,
    entry_debounce_frames: int = 1,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """讀取當日追蹤結果，依 `camera_registry.yaml` 的 zone 定義統計人流。

    純 CPU 向量化運算，不需重跑 GPU 偵測；輸出前會先用
    `validate_zone_cameras` fail-loud 檢查 camera 是否對得上當天資料，再對
    每台攝影機呼叫 `parsed_zones()` 解析驗證 zone 幾何。

    Args:
        date: 要統計的日期，需已有對應的 `tracking_results.parquet`。
        bucket_dir: 本機模擬 GCS bucket 的根目錄。
        bucket_minutes: 人流統計的時段粒度（分鐘）。
        entry_debounce_frames: 連續幾格都在區域內才算一次「進入」。
        output_root: 輸出根目錄。

    Returns:
        `zone_counts.parquet` 的路徑。

    Raises:
        FileNotFoundError: 當日 `tracking_results.parquet` 不存在，或
            `bucket_dir` 底下找不到 `camera_registry.yaml`。
        ValueError: `camera_registry.yaml` 定義了 zone 的攝影機在當天追蹤
            結果中查無資料，或任一 zone 定義不合法。
    """
    output_dir = output_root / Path(bucket_dir).name / date.isoformat()
    results_path = output_dir / TRACKING_RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"找不到追蹤結果 {results_path}，請先執行 analyze_daily 產生當日 parquet。"
        )

    bucket_path = Path(bucket_dir)
    registry = load_registry(bucket_path)
    zone_entries = {
        entry.stream_dirname: entry
        for entry in registry.cameras
        if entry.participates_in_zone_mapping
    }

    df = pl.read_parquet(results_path)
    # 先驗證 camera 對得上當天資料再解析 zone，避免陳舊 zone 定義打錯字蓋過更根本錯誤
    validate_zone_cameras(
        {k for k, e in zone_entries.items() if e.zones},
        set(df["camera_id"].unique()),
    )
    zone_cameras = parse_and_validate_zones(zone_entries)

    df = df.with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("foot_x"),
        pl.col("y2").alias("foot_y"),
        pl.col("timestamp").dt.truncate(f"{bucket_minutes}m").alias("time_bucket"),
    )

    frames: list[pl.DataFrame] = []
    for camera_id, zones in zone_cameras.items():
        cam_sub = df.filter(pl.col("camera_id") == camera_id)
        for zone in zones:
            counts = count_zone_visits(cam_sub, zone, entry_debounce_frames).with_columns(
                pl.lit(camera_id).alias("camera_id"),
                pl.lit(zone.name).alias("zone"),
            )
            frames.append(counts)

    if frames:
        result = (
            pl.concat(frames)
            .select(list(ZONE_COUNTS_SCHEMA))
            .sort("camera_id", "zone", "time_bucket")
        )
    else:
        result = pl.DataFrame(schema=ZONE_COUNTS_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / ZONE_COUNTS_FILENAME
    tmp_path = counts_path.with_name(counts_path.name + TMP_SUFFIX)
    result.write_parquet(tmp_path)
    tmp_path.replace(counts_path)

    # 快照當下套用的 camera_registry.yaml，讓這份 zone_counts 自帶當天的 zone 依據可回溯
    shutil.copyfile(registry_path(bucket_path), output_dir / REGISTRY_SNAPSHOT_FILENAME)

    logger.info(
        "Zone 人流統計已寫入",
        path=str(counts_path),
        cameras=len(zone_cameras),
        rows=result.height,
    )
    return counts_path
