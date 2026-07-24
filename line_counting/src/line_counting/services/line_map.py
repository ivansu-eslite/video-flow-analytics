"""Line Counting 的核心編排：讀檔、逐攝影機/逐計數線套用演算法、寫檔與快照。

讀 `outputs/{bucket}/{date}/tracking_results.parquet`，套上人工維護在
`camera_registry.yaml` 各攝影機底下的計數線幾何，輸出每個時段每條計數線的進出人數
到同層的 `line_counts.parquet`，並把當下套用的 camera_registry.yaml 快照成
`camera_registry_used.yaml` 以供回溯。

實際的跨越判定與聚合演算法在 `services/stats.py`。
"""

import datetime
import shutil
from pathlib import Path

import polars as pl
from vfa_observability import StructuredLogger
from vfa_registry import load_registry, parse_and_validate_lines, registry_path

from line_counting.config.constants import (
    LINE_COUNTS_FILENAME,
    LINE_COUNTS_SCHEMA,
    OUTPUT_ROOT,
    REGISTRY_SNAPSHOT_FILENAME,
    TMP_SUFFIX,
    TRACKING_RESULTS_FILENAME,
)
from line_counting.services.stats import count_line_crossings, validate_line_cameras

logger = StructuredLogger(component="line_map")


def count_lines_daily(
    date: datetime.date,
    bucket_dir: str,
    bucket_minutes: int,
    crossing_band_px: float = 0,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """讀取當日追蹤結果，依 `camera_registry.yaml` 的計數線定義統計進出人數。

    純 CPU 向量化運算，不需重跑 GPU 偵測；輸出前會先用 `validate_line_cameras`
    fail-loud 檢查 camera 是否對得上當天資料，再對每台攝影機呼叫 `parsed_lines()`
    解析驗證計數線幾何。**參與判定**：只處理 `lines` 非空的攝影機（不另設參與旗標；
    要停用某台就移除其 `lines`）。

    Args:
        date: 要統計的日期，需已有對應的 `tracking_results.parquet`。
        bucket_dir: 本機模擬 GCS bucket 的根目錄。
        bucket_minutes: 進出人數統計的時段粒度（分鐘）。
        crossing_band_px: 跨越去抖的帶狀死區寬度（像素）；`0` = 細線純零交越。
        output_root: 輸出根目錄。

    Returns:
        `line_counts.parquet` 的路徑。

    Raises:
        FileNotFoundError: 當日 `tracking_results.parquet` 不存在，或
            `bucket_dir` 底下找不到 `camera_registry.yaml`。
        ValueError: `camera_registry.yaml` 定義了計數線的攝影機在當天追蹤
            結果中查無資料，或任一計數線定義不合法。
    """
    output_dir = output_root / Path(bucket_dir).name / date.isoformat()
    results_path = output_dir / TRACKING_RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"找不到追蹤結果 {results_path}，請先執行 analyze_daily 產生當日 parquet。"
        )

    bucket_path = Path(bucket_dir)
    registry = load_registry(bucket_path)
    # 參與判定：以 `lines` 是否非空決定，不看 participates_in_zone_mapping
    line_entries = {
        entry.stream_dirname: entry for entry in registry.cameras if entry.lines
    }

    df = pl.read_parquet(results_path)
    # 先驗證 camera 對得上當天資料再解析計數線，避免陳舊定義打錯字蓋過更根本錯誤
    validate_line_cameras(
        set(line_entries),
        set(df["camera_id"].unique()),
    )
    line_cameras = parse_and_validate_lines(line_entries)

    df = df.with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("foot_x"),
        pl.col("y2").alias("foot_y"),
        pl.col("timestamp").dt.truncate(f"{bucket_minutes}m").alias("time_bucket"),
    )

    frames: list[pl.DataFrame] = []
    for camera_id, lines in line_cameras.items():
        cam_sub = df.filter(pl.col("camera_id") == camera_id)
        for line in lines:
            counts = count_line_crossings(
                cam_sub, line, crossing_band_px
            ).with_columns(
                pl.lit(camera_id).alias("camera_id"),
                pl.lit(line.name).alias("line"),
            )
            frames.append(counts)

    if frames:
        result = (
            pl.concat(frames)
            .select(list(LINE_COUNTS_SCHEMA))
            .sort("camera_id", "line", "time_bucket")
        )
    else:
        result = pl.DataFrame(schema=LINE_COUNTS_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / LINE_COUNTS_FILENAME
    tmp_path = counts_path.with_name(counts_path.name + TMP_SUFFIX)
    result.write_parquet(tmp_path)
    tmp_path.replace(counts_path)

    # 快照當下套用的 camera_registry.yaml，讓這份 line_counts 自帶當天的計數線依據可回溯
    shutil.copyfile(registry_path(bucket_path), output_dir / REGISTRY_SNAPSHOT_FILENAME)

    logger.info(
        "計數線進出人數統計已寫入",
        path=str(counts_path),
        cameras=len(line_cameras),
        rows=result.height,
    )
    return counts_path
