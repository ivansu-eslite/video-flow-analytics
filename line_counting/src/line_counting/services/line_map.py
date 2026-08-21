"""Line Counting 的核心編排：讀檔、逐攝影機/逐計數線套用演算法、寫檔。

讀 `outputs/{bucket}/{date}/tracking_results.parquet`，套上人工維護在
`camera_registry.yaml` 各攝影機底下的計數線幾何，輸出每個時段每條計數線的進出人數
到同層的 `line_counts.parquet`。

實際的跨越判定與聚合演算法在 `services/stats.py`。
"""

import datetime
from pathlib import Path

import polars as pl
from vfa_observability import StructuredLogger
from vfa_registry import load_registry, parse_and_validate_lines

from line_counting.config.constants import (
    BASELINE_FRAME_WIDTH,
    DEFAULT_CROSSING_BAND_PX_1080P,
    LINE_COUNTS_FILENAME,
    LINE_COUNTS_SCHEMA,
    OUTPUT_ROOT,
    REQUIRED_TRACKING_COLUMNS,
    TMP_SUFFIX,
    TRACKING_RESULTS_FILENAME,
)
from line_counting.services.stats import count_line_crossings, validate_line_cameras

logger = StructuredLogger(component="line_map")


def _resolve_band_px(
    camera_id: str, cam_sub: pl.DataFrame, crossing_band_px_1080p: float
) -> float:
    """把 1080p 基準的線段區域寬度換算成該攝影機的實際像素，並記錄換算過程。

    換算只用寬度、線性：`實際 px = 基準值 × frame_width / 1920`。同一台攝影機整天
    解析度固定（上游 `probe_frame_shape` 只探測首格，中途變動會在讀取端逐格核對時
    就中止），因此這裡要求該攝影機的 `frame_width` 只有單一正值——多個值代表拿到的
    是手工拼接的 parquet 或上游語義已改，靜默取其中一個會讓半天的線段區域用錯尺度。

    Args:
        camera_id: 該攝影機的 `camera_id`（僅供錯誤訊息與日誌）。
        cam_sub: 只含該攝影機的追蹤明細，需已含 `frame_width` 欄位且非空。
        crossing_band_px_1080p: 以 1080p（寬 1920）為基準的線段區域寬度。

    Returns:
        換算後的線段區域寬度（實際像素）。

    Raises:
        ValueError: 該攝影機的 `frame_width` 不是單一正值。
    """
    widths = cam_sub["frame_width"].unique().to_list()
    if len(widths) != 1 or widths[0] is None or widths[0] <= 0:
        raise ValueError(
            f"攝影機 {camera_id} 的 frame_width 必須是單一正值"
            f"（同一台整天解析度固定），實際為 {widths}。"
        )
    frame_width = widths[0]
    scale = frame_width / BASELINE_FRAME_WIDTH
    band_px = crossing_band_px_1080p * scale
    logger.info(
        "線段區域依解析度換算",
        camera_id=camera_id,
        frame_width=frame_width,
        baseline_width=BASELINE_FRAME_WIDTH,
        scale=round(scale, 4),
        crossing_band_px_1080p=crossing_band_px_1080p,
        crossing_band_px=round(band_px, 2),
    )
    return band_px


def count_lines_daily(
    date: datetime.date,
    bucket_dir: str,
    bucket_minutes: int,
    crossing_band_px_1080p: float = DEFAULT_CROSSING_BAND_PX_1080P,
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
        crossing_band_px_1080p: 跨越去抖的線段區域寬度，以 1080p（寬 1920）為
            基準的像素值；逐攝影機依其 `frame_width` 換算成實際像素後才進判定
            （見 ADR-004）。`0` = 細線純零交越，且換算後仍是 0；預設 25 取自
            實測（見 README「已知限制」）。
        output_root: 輸出根目錄。

    Returns:
        `line_counts.parquet` 的路徑。

    Raises:
        FileNotFoundError: 當日 `tracking_results.parquet` 不存在，或
            `bucket_dir` 底下找不到 `camera_registry.yaml`。
        ValueError: 追蹤結果缺少影像尺寸或落腳點欄位（舊版 `video_analyze` 的
            產物）、`camera_registry.yaml` 定義了計數線的攝影機在當天追蹤結果中
            查無資料，或任一計數線定義不合法。
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
    missing = [c for c in REQUIRED_TRACKING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{results_path} 缺少欄位 {missing}：這是舊版 video_analyze 的產物。"
            "沒有影像尺寸就無法把 crossing_band_px_1080p 換算成各攝影機的實際像素；"
            "沒有 foot_x／foot_y 就沒有落腳點可判定（本套件不再自行從 bbox 推算，"
            "見 ADR-009）。請以現行版本的 video_analyze 重跑該日產生新的 "
            "tracking_results.parquet。"
        )
    # 先驗證 camera 對得上當天資料再解析計數線，避免陳舊定義打錯字蓋過更根本錯誤
    validate_line_cameras(
        set(line_entries),
        set(df["camera_id"].unique()),
    )
    line_cameras = parse_and_validate_lines(line_entries)

    # 落腳點直接讀上游欄位——推算需要 head 框，而 head 不在追蹤結果裡（見 ADR-009）
    df = df.with_columns(
        pl.col("timestamp").dt.truncate(f"{bucket_minutes}m").alias("time_bucket"),
    )

    frames: list[pl.DataFrame] = []
    for camera_id, lines in line_cameras.items():
        cam_sub = df.filter(pl.col("camera_id") == camera_id)
        # 換算在此算好，`count_line_crossings` 收到的一律是實際像素——判定邏輯不需要
        # 知道「基準解析度」這個概念，`stats.py` 維持純幾何
        band_px = _resolve_band_px(camera_id, cam_sub, crossing_band_px_1080p)
        for line in lines:
            counts = count_line_crossings(cam_sub, line, band_px).with_columns(
                pl.lit(camera_id).alias("camera_id"),
                pl.lit(line.name).alias("line"),
                pl.lit(line.line_group).alias("line_group"),
            )
            frames.append(counts)

    if frames:
        result = (
            pl.concat(frames)
            .select(list(LINE_COUNTS_SCHEMA))
            .sort("line_group", "camera_id", "line", "time_bucket")
        )
    else:
        result = pl.DataFrame(schema=LINE_COUNTS_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / LINE_COUNTS_FILENAME
    tmp_path = counts_path.with_name(counts_path.name + TMP_SUFFIX)
    result.write_parquet(tmp_path)
    tmp_path.replace(counts_path)

    logger.info(
        "計數線進出人數統計已寫入",
        path=str(counts_path),
        cameras=len(line_cameras),
        rows=result.height,
    )
    return counts_path
