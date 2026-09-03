"""Zone Mapping 的核心演算法：point-in-polygon 判定與人流聚合統計。

人流指標（三者刻意用不同判定，見 ADR-006、ADR-016）：
- unique_visitors：該時段內腳底落在區域內的不重複 track_id 數（不重複訪客）。純粹
  用 `points_in_polygon` 的布林值，不吃線段區域的黏著狀態——佔用型指標若吃了黏著
  狀態，走出區域後停在邊界外線段區域內的人會在其後每個時段都被算成區內訪客。
- entries：每個 track 依時間序偵測「區域外 → 區域內」的轉換次數，歸戶到轉換發生
  那格的時段（同一人離開再進入算多次；首次出現即在區域內也算一次進入）。事件型
  指標，用線段區域（Schmitt-trigger）濾掉腳底點在邊界附近抖動的重複計數。
- dwell_events：該時段內有幾**人次**在區域內連續停留達到 `dwell_threshold_seconds`。
  判定基礎與 `unique_visitors` 同源（生的 `in_zone`，不吃黏著狀態），指標型態與
  `entries` 同型（事件型、跨時段相加不重複）。同一人在同一時段內兩段都達標會計 2，
  因此**不是 `unique_visitors` 的子集**，下游不可寫 `dwell_events <= unique_visitors`
  這類 sanity check。

**首格語義與 `line_counting` 相反**：這裡首次出現即在區內算一次 entry（人可能從
畫面外直接走進區內，沒有「先在區外被看到」那一格）；`line_counting` 的起始側不算
跨越（見 `docs/adr/line_counting/001-line-crossing-detection.md`）。

判定「人是否在區域內」用落腳點 (foot_x, foot_y)。落腳點由上游 `video_analyze` 算好寫進
`tracking_results.parquet`（由 head 框推算，推不出來才退回 bbox 底邊中點，見 ADR-009），
本套件不自己從 bbox 推算。
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


def signed_distance_to_polygon(
    xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """帶號距離：正 = 在多邊形內，絕對值 = 該點到多邊形邊界的最短距離（像素）。

    對封閉多邊形的每一邊算「點到線段」的有限距離（投影落在端點外時夾到端點），逐邊
    取 running min；符號直接由 `points_in_polygon` 給，不像開放 polyline 版需要參考
    點錨定側別（`line_counting` 的做法與此刻意不同，見 ADR-006）。

    多邊形自動閉合：`Zone.polygon` 不重複首點，最後一點連回第一點的那條邊也要算，
    否則收尾邊附近的線段區域會整段失效。

    Args:
        xs: 待判定點的 x 座標，長度 N。
        ys: 待判定點的 y 座標，長度 N。
        polygon: 多邊形頂點座標，shape 為 `(M, 2)`，`M >= 3`（自動閉合）。

    Returns:
        `(signed_d, inside)`：長度均為 N 的帶號距離與內外布林陣列。一併回傳
        `inside` 是為了讓呼叫端沿用同一個 `points_in_polygon` 結果，不必為了
        `unique_visitors` 再跑一次逐邊迴圈；也不可改用 `signed_d > 0` 反推，
        邊界上 `signed_d == 0` 的點會與 `points_in_polygon` 的結果不一致。
    """
    px = np.asarray(xs, dtype=float)
    py = np.asarray(ys, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    dist = np.full(px.shape, np.inf)
    # 逐邊迴圈取 running min（與 points_in_polygon 同風格）：峰值記憶體維持 O(N)，
    # 不用 (N, S, 2) 廣播——距離對每個 zone 都要重算一次，全天列數下差別很大。
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]  # 最後一點連回第一點
        abx, aby = bx - ax, by - ay
        len2 = abx * abx + aby * aby
        apx, apy = px - ax, py - ay
        if len2 == 0:  # 重複頂點的防呆：退化成到該點的距離
            t = np.zeros_like(px)
        else:
            t = np.clip((apx * abx + apy * aby) / len2, 0.0, 1.0)
        np.minimum(dist, np.hypot(apx - t * abx, apy - t * aby), out=dist)

    inside = points_in_polygon(px, py, poly)
    return np.where(inside, dist, -dist), inside


def count_zone_visits(
    cam_sub: pl.DataFrame,
    zone: Zone,
    boundary_band_px: float = 0,
    *,
    dwell_threshold_seconds: float,
    dwell_gap_seconds: float,
) -> pl.DataFrame:
    """對單一攝影機的追蹤明細套一個 zone，回傳每個 time_bucket 的人流統計。

    輸入 cam_sub 需已含 foot_x / foot_y / time_bucket 欄位，且只包含該攝影機的列。

    `entries` 用線段區域的 Schmitt-trigger：帶號距離 `> band` 確認在內、
    `< -band` 確認在外、落在帶內則沿用前一個已確認狀態，只有已確認狀態由外翻內才算
    一次進入。腳底點在邊界附近抖動時整段維持同一個已確認狀態，因此只計一次；
    `band = 0` 退化成純內外判定。

    `unique_visitors` 與 `dwell_events` 都不吃已確認狀態，用的是當格的
    `points_in_polygon` 布林值（見 ADR-016）。

    `dwell_events` 把每個 track 在區內的列依時間切段——相鄰兩次在區內的間隔超過
    `dwell_gap_seconds` 就開新段——段內 `最後一格 − 第一格` 達到
    `dwell_threshold_seconds` 即計一次人次，歸戶到該段**首次跨過門檻**那一格的時段。
    一段只計一次；同一人的兩段各自達標則計兩次。

    Args:
        cam_sub: 單一攝影機的追蹤明細，需已含 `foot_x`／`foot_y`／
            `time_bucket`／`track_id`／`timestamp` 欄位。
        zone: 要套用的區域定義。
        boundary_band_px: 線段區域的半寬，**實際像素**（1080p 基準值的換算
            在 `services/zone_map.py` 完成，本模組維持純幾何）。
        dwell_threshold_seconds: 停留門檻（秒），必須 > 0。keyword-only 且**刻意
            不給預設值**：`boundary_band_px = 0` 有中性的退化語義（純內外判定），
            這兩個參數沒有——沒有預設值就不可能有人拿到錯的預設值。
        dwell_gap_seconds: 同一段停留可容忍的中斷（秒），必須 > 0。

    Returns:
        依 `time_bucket` 聚合的 `unique_visitors`／`entries`／`dwell_events`
        統計表。
    """
    signed_d, inside = signed_distance_to_polygon(
        cam_sub["foot_x"].to_numpy(),
        cam_sub["foot_y"].to_numpy(),
        np.asarray(zone.polygon, dtype=float),
    )
    z = cam_sub.with_columns(
        pl.Series("in_zone", inside), pl.Series("_signed_d", signed_d)
    ).sort("track_id", "timestamp")

    # unique_visitors 與 dwell_events 共用同一份「生 in_zone」子集
    in_z = z.filter(pl.col("in_zone"))

    unique_visitors = in_z.group_by("time_bucket").agg(
        pl.col("track_id").n_unique().alias("unique_visitors")
    )

    # 時間一律用整數微秒比較：timestamp 是 Datetime("us")，由 segment.start +
    # frame_index / fps 算出，30 fps 的相鄰間隔在 33333／33334 µs 之間跳動，用秒的
    # 浮點數比會在除不盡的間隔上分岔。
    dwell_us = round(dwell_threshold_seconds * 1_000_000)
    gap_us = round(dwell_gap_seconds * 1_000_000)
    _gap_us = pl.col("timestamp").diff().over("track_id").dt.total_microseconds()

    dwell_events = (
        in_z.with_columns(
            # 同一 track 相鄰兩次「在區內」的間隔 > T 就開新段；該 track 的第一列
            # diff 為 null，`is_null()` 那一半要顯式納入（`null > x` 得 null，
            # cum_sum 會把 null 傳染給整個 track，最後一列不剩——與 entries 的
            # `_prev != 1` 同一個坑）。兩個 `.over("track_id")` 在目前實作下是防禦
            # 性的：下游 group_by 已把 track_id 納入分組鍵，拿掉輸出不變。保留是為
            # 了讓 `_dwell_seg` 的語義（段號在 track 內編號）不依賴下游怎麼分組。
            (_gap_us.is_null() | (_gap_us > gap_us))
            .cum_sum()
            .over("track_id")
            .alias("_dwell_seg")
        )
        .with_columns(
            # 分組鍵必須帶 track_id：段號是各 track 各自從 1 起算的，少了它會拿別人
            # 的段起點當起算點，是**超計**方向的錯（晚到的人繼承早到者的起算時間）。
            # 用 min 而非 first：不依賴列順序，成本相同。
            (
                pl.col("timestamp")
                - pl.col("timestamp").min().over("track_id", "_dwell_seg")
            )
            .dt.total_microseconds()
            .alias("_dwell_us")
        )
        .filter(pl.col("_dwell_us") >= dwell_us)
        # 過濾後每段最早的一列 = 首次跨過門檻的那一格；一段只計一次，歸到那一格的 bucket
        .group_by("track_id", "_dwell_seg")
        .agg(pl.col("time_bucket").sort_by("timestamp").first())
        .group_by("time_bucket")
        .agg(pl.len().alias("dwell_events"))
    )

    entries = (
        z.with_columns(
            pl.when(pl.col("_signed_d") > boundary_band_px)
            .then(1)
            .when(pl.col("_signed_d") < -boundary_band_px)
            .then(-1)
            .otherwise(None)  # 落在線段區域內：留 null 交給 forward_fill 沿用前一格
            .alias("_side")
        )
        .with_columns(
            pl.col("_side").forward_fill().over("track_id").alias("_committed")
        )
        .with_columns(pl.col("_committed").shift(1).over("track_id").alias("_prev"))
        # `_prev` 為 null（該 track 的第一格，或前面整段都還在帶內）視為區外，首次
        # 確認在區內即算一次進入。不可寫成 `_prev != 1`：polars 的 `null != 1` 得
        # null，filter 會丟掉該列，首格即在區內的那次進入會靜默消失。
        .filter(
            (pl.col("_committed") == 1)
            & (pl.col("_prev").is_null() | (pl.col("_prev") == -1))
        )
        .group_by("time_bucket")
        .agg(pl.len().alias("entries"))
    )

    # 三方 join 後一律 fill_null(0)：漏掉會在 parquet 留 null，而 pl.len() 是
    # UInt32，不 cast 會讓 flow_report 的 sum() 拿到不同型別
    return (
        unique_visitors.join(entries, on="time_bucket", how="full", coalesce=True)
        .join(dwell_events, on="time_bucket", how="full", coalesce=True)
        .with_columns(
            pl.col("unique_visitors").fill_null(0).cast(pl.Int64),
            pl.col("entries").fill_null(0).cast(pl.Int64),
            pl.col("dwell_events").fill_null(0).cast(pl.Int64),
        )
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
