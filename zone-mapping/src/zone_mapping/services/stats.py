"""Zone Mapping 的核心演算法：point-in-polygon 判定與人流聚合統計。

人流指標：
- unique_visitors：該時段內腳底落在區域內的不重複 track_id 數（不重複訪客）。
- entries：每個 track 依時間序偵測「區域外 → 區域內」的轉換次數，歸戶到轉換發生
  那格的時段（同一人離開再進入算多次；首次出現即在區域內也算一次進入）。

判定「人是否在區域內」用 bbox 腳底中心點 ((x1+x2)/2, y2)。
"""

import numpy as np
import polars as pl
from vfa_registry import Zone


def points_in_polygon(
    xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """向量化 ray casting：回傳長度為 N 的布林陣列，標記每個點是否落在多邊形內。

    對每條邊判斷「向右水平射線是否穿越該邊」，穿越次數為奇數即在內部。邊界上
    的點結果依實作而定，對人流統計無實質影響。

    Args:
        xs: 待判定點的 x 座標，長度 N。
        ys: 待判定點的 y 座標，長度 N。
        polygon: 多邊形頂點座標，shape 為 `(M, 2)`。

    Returns:
        長度為 N 的布林陣列，`True` 代表該點落在多邊形內。
    """
    px = polygon[:, 0]
    py = polygon[:, 1]
    inside = np.zeros(len(xs), dtype=bool)
    j = len(polygon) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(polygon)):
            crosses = (py[i] > ys) != (py[j] > ys)  # 邊 (j->i) 是否在 y 方向跨越目標點
            # crosses 為 False 的列不會用到 x_cross，故 py[i]==py[j] 的除零結果無影響
            x_cross = (px[j] - px[i]) * (ys - py[i]) / (py[j] - py[i]) + px[i]
            inside ^= crosses & (xs < x_cross)
            j = i
    return inside


def count_zone_visits(
    cam_sub: pl.DataFrame, zone: Zone, entry_debounce_frames: int = 1
) -> pl.DataFrame:
    """對單一攝影機的追蹤明細套一個 zone，回傳每個 time_bucket 的人流統計。

    輸入 cam_sub 需已含 foot_x / foot_y / time_bucket 欄位，且只包含該攝影機的列。

    entry_debounce_frames 控制「連續幾格都在區域內才算一次進入」，用來過濾腳底點
    在區域邊界附近來回抖動造成的假進入；預設 1 = 不去抖（一格在內就算），數字越大
    越能濾掉抖動，代價是進入事件會延遲 (N-1) 格才被計入、歸戶到較晚的 time_bucket。

    Args:
        cam_sub: 單一攝影機的追蹤明細，需已含 `foot_x`／`foot_y`／
            `time_bucket`／`track_id`／`timestamp` 欄位。
        zone: 要套用的區域定義。
        entry_debounce_frames: 連續幾格都在區域內才算一次「進入」。

    Returns:
        依 `time_bucket` 聚合的 `unique_visitors`／`entries` 統計表。
    """
    inside = points_in_polygon(
        cam_sub["foot_x"].to_numpy(),
        cam_sub["foot_y"].to_numpy(),
        np.asarray(zone.polygon, dtype=float),
    )
    z = cam_sub.with_columns(pl.Series("in_zone", inside)).sort(
        "track_id", "timestamp"
    )

    unique_visitors = (
        z.filter(pl.col("in_zone"))
        .group_by("time_bucket")
        .agg(pl.col("track_id").n_unique().alias("unique_visitors"))
    )

    # 「確認進入」= 連續 entry_debounce_frames 格都在區域內；預設值 1 時等同單純 in_zone
    confirmed_in = (
        pl.col("in_zone")
        .cast(pl.Int8)
        .rolling_sum(
            window_size=entry_debounce_frames, min_samples=entry_debounce_frames
        )
        .over("track_id")
        # 前 N-1 格湊不滿窗格 = 未確認；留 null 會經 shift 汙染 _prev_confirmed
        .fill_null(0)
        == entry_debounce_frames
    )
    entries = (
        z.with_columns(confirmed_in.alias("_confirmed_in"))
        .with_columns(
            pl.col("_confirmed_in")
            .shift(1, fill_value=False)
            .over("track_id")
            .alias("_prev_confirmed")
        )
        .filter(pl.col("_confirmed_in") & ~pl.col("_prev_confirmed"))
        .group_by("time_bucket")
        .agg(pl.len().alias("entries"))
    )

    return unique_visitors.join(
        entries, on="time_bucket", how="full", coalesce=True
    ).with_columns(
        pl.col("unique_visitors").fill_null(0).cast(pl.Int64),
        pl.col("entries").fill_null(0).cast(pl.Int64),
    )


def validate_zone_cameras(zone_camera_ids: set[str], data_cameras: set[str]) -> None:
    """fail-loud：camera_registry.yaml 定義了 zone 的每個 camera 都要在當天
    tracking_results 中出現。

    攝影機改名或 key 打錯時，這裡會直接報錯中止，而不是靜默略過那台攝影機、
    默默算出漏掉區域的人流。

    Args:
        zone_camera_ids: `camera_registry.yaml` 中定義了 zone 的攝影機
            `camera_id` 集合。
        data_cameras: 當天 `tracking_results.parquet` 實際出現的 `camera_id`
            集合。

    Raises:
        ValueError: `zone_camera_ids` 中有任一 ID 不在 `data_cameras` 內。
    """
    unknown = sorted(zone_camera_ids - data_cameras)
    if unknown:
        raise ValueError(
            "camera_registry.yaml 定義了這些 camera 的 zone，"
            f"但當天 tracking_results 沒有對應資料（camera 改名或 key 打錯？）: "
            f"{unknown}。當天實際的 camera_id: {sorted(map(str, data_cameras))}"
        )
