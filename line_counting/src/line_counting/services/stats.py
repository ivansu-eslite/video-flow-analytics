"""Line Counting 的核心演算法：方向性計數線跨越判定與進出人數聚合。

人流指標（每條計數線、每個 time_bucket）：
- in_count：該時段內 track 由外側跨越到內側（往 `inside_point` 那一側）的次數。
- out_count：該時段內 track 由內側跨越到外側的次數。

判定「人在線的哪一側」用 bbox 腳底中心點 ((x1+x2)/2, y2) 到計數線的帶號垂直距離。
跨越偵測用「帶死區的 Schmitt-trigger」：`crossing_band_px` 把細線加粗成帶狀死區，
濾除腳底點在線附近的抖動／駐留；`= 0` 退化為細線純零交越。
"""

import numpy as np
import polars as pl
from vfa_registry import Line


def signed_distance_to_polyline(
    xs: np.ndarray,
    ys: np.ndarray,
    points: np.ndarray,
    inside_point: tuple[float, float],
) -> np.ndarray:
    """向量化：每個點對 polyline 算帶號垂直距離，正值 = 與 `inside_point` 同側。

    對 polyline 的每一段（M 個頂點共 M-1 段）算「點到線段」的有限距離（端點外會夾
    到端點），取最近的一段；輸出該最近段**無限直線**的帶號垂直距離，符號以
    `inside_point` 相對同一段無限直線的側別定為正。

    側別只錨定被選中的最近段（局部直線），不建全域 signed-side——凹角（reflex 頂點
    朝 `inside_point`）理論上有 medial-axis 幽靈翻轉風險；門口計數線的凸／直線
    barrier（含包住 `inside_point` 的ㄇ形，屬凸向 inside）不受影響。詳見 README
    「已知限制」。**勿改成全域 signed-side**：那會在包住 inside 的ㄇ形 barrier 上算錯。

    Args:
        xs: 待判定點的 x 座標，長度 N。
        ys: 待判定點的 y 座標，長度 N。
        points: polyline 頂點座標，shape 為 `(M, 2)`，`M >= 2`。
        inside_point: 場內參考點 `(x, y)`；跨越往這一側為正（`in`）。

    Returns:
        長度為 N 的帶號垂直距離陣列，正值代表該點與 `inside_point` 同側。
    """
    p = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    pts = np.asarray(points, dtype=float)
    a = pts[:-1]  # (S, 2) 各段起點
    b = pts[1:]  # (S, 2) 各段終點
    ab = b - a  # (S, 2)
    len2 = np.einsum("si,si->s", ab, ab)  # (S,)
    # 零長度段（連續重複頂點）在 registry 已被擋下；此處避免除零留一手
    safe_len2 = np.where(len2 == 0, 1.0, len2)
    seg_len = np.sqrt(np.where(len2 == 0, 1.0, len2))  # (S,)

    ap = p[:, None, :] - a[None, :, :]  # (N, S, 2)
    t = np.clip(np.einsum("nsi,si->ns", ap, ab) / safe_len2, 0.0, 1.0)  # (N, S)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (N, S, 2)
    diff = p[:, None, :] - closest  # (N, S, 2)
    dist2 = np.einsum("nsi,nsi->ns", diff, diff)  # (N, S)
    nearest = np.argmin(dist2, axis=1)  # (N,)

    # 各段無限直線的帶號垂直距離：cross(ab, ap) / |ab|
    cross = ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]  # (N, S)
    perp = cross / seg_len[None, :]  # (N, S) 帶號

    # inside_point 相對各段無限直線的側別，用來把「同側」定為正
    inside = np.asarray(inside_point, dtype=float)
    ai = inside[None, :] - a  # (S, 2)
    cross_in = ab[:, 0] * ai[:, 1] - ab[:, 1] * ai[:, 0]  # (S,)
    sign_in = np.sign(cross_in)  # (S,) registry 已擋 inside_point 共線，故非 0

    signed = perp * sign_in[None, :]  # (N, S) 正 = 與 inside_point 同側
    return signed[np.arange(p.shape[0]), nearest]


def count_line_crossings(
    cam_sub: pl.DataFrame, line: Line, crossing_band_px: float = 0
) -> pl.DataFrame:
    """對單一攝影機的追蹤明細套一條計數線，回傳每個 time_bucket 的進出人數。

    輸入 cam_sub 需已含 foot_x / foot_y / time_bucket 欄位，且只包含該攝影機的列。

    以「帶死區的 Schmitt-trigger」偵測跨越：帶號距離 `d > band` 判內側（`+1`）、
    `d < -band` 判外側（`-1`）、落在 `[-band, band]` 帶內為死區（沿用前一個已確認
    側別，hysteresis）。committed 側別翻轉即一次跨越——翻到內側計 `in`、翻到外側計
    `out`；track 起始就在某側（前一格為 null）本身不算跨越（注意：這一點與
    `zone_mapping` **相反**——`zone_mapping` 首次即在區內會算一次 entry；計數線只認
    「側別翻轉」，起始側不構成翻轉）。`crossing_band_px = 0` 時死區退化為單點，等同
    幾何零交越。

    Args:
        cam_sub: 單一攝影機的追蹤明細，需已含 `foot_x`／`foot_y`／
            `time_bucket`／`track_id`／`timestamp` 欄位。
        line: 要套用的計數線定義。
        crossing_band_px: 跨越去抖的帶狀死區寬度（像素）；`0` = 細線純零交越。

    Returns:
        依 `time_bucket` 聚合的 `in_count`／`out_count` 統計表。
    """
    d = signed_distance_to_polyline(
        cam_sub["foot_x"].to_numpy(),
        cam_sub["foot_y"].to_numpy(),
        np.asarray(line.points, dtype=float),
        line.inside_point,
    )
    band = crossing_band_px
    z = (
        cam_sub.with_columns(pl.Series("_d", d))
        .sort("track_id", "timestamp")
        # 死區（帶內）留 null，交給 forward_fill 沿用前一個已確認側別
        .with_columns(
            pl.when(pl.col("_d") > band)
            .then(1)
            .when(pl.col("_d") < -band)
            .then(-1)
            .otherwise(None)
            .alias("_side")
        )
        .with_columns(
            pl.col("_side").forward_fill().over("track_id").alias("_committed")
        )
        .with_columns(
            pl.col("_committed").shift(1).over("track_id").alias("_prev")
        )
    )

    # committed 側別翻轉 = 一次跨越；起始側（_prev 為 null）不算
    crossings = z.filter(
        pl.col("_committed").is_not_null()
        & pl.col("_prev").is_not_null()
        & (pl.col("_committed") != pl.col("_prev"))
    )
    in_counts = (
        crossings.filter(pl.col("_committed") == 1)
        .group_by("time_bucket")
        .agg(pl.len().alias("in_count"))
    )
    out_counts = (
        crossings.filter(pl.col("_committed") == -1)
        .group_by("time_bucket")
        .agg(pl.len().alias("out_count"))
    )

    return in_counts.join(
        out_counts, on="time_bucket", how="full", coalesce=True
    ).with_columns(
        pl.col("in_count").fill_null(0).cast(pl.Int64),
        pl.col("out_count").fill_null(0).cast(pl.Int64),
    )


def validate_line_cameras(line_camera_ids: set[str], data_cameras: set[str]) -> None:
    """fail-loud：camera_registry.yaml 定義了計數線的每個 camera 都要在當天
    tracking_results 中出現。

    攝影機改名或 key 打錯時，這裡會直接報錯中止，而不是靜默略過那台攝影機、
    默默算出漏掉出入口的進出人數。

    Args:
        line_camera_ids: `camera_registry.yaml` 中定義了計數線的攝影機
            `camera_id` 集合。
        data_cameras: 當天 `tracking_results.parquet` 實際出現的 `camera_id`
            集合。

    Raises:
        ValueError: `line_camera_ids` 中有任一 ID 不在 `data_cameras` 內。
    """
    unknown = sorted(line_camera_ids - data_cameras)
    if unknown:
        raise ValueError(
            "camera_registry.yaml 定義了這些 camera 的計數線，"
            f"但當天 tracking_results 沒有對應資料（camera 改名或 key 打錯？）: "
            f"{unknown}。當天實際的 camera_id: {sorted(map(str, data_cameras))}"
        )
