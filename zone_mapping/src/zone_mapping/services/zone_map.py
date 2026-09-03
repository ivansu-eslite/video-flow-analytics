"""Zone Mapping 的核心編排：讀檔、逐攝影機/逐 zone 套用演算法、寫檔。

讀 `outputs/{bucket}/{date}/tracking_results.parquet`，套上人工維護在
`camera_registry.yaml` 各攝影機底下的 zone 幾何，輸出每個時段每個區域的人流統計
到同層的 `zone_counts.parquet`。

實際的 point-in-polygon 判定與聚合演算法在 `services/stats.py`。
"""

import datetime
from pathlib import Path

import numpy as np
import polars as pl
from vfa_observability import StructuredLogger
from vfa_registry import Zone, load_registry, parse_and_validate_zones

from zone_mapping.config.constants import (
    BASELINE_FRAME_WIDTH,
    DEFAULT_BOUNDARY_BAND_PX_1080P,
    DEFAULT_DWELL_GAP_SECONDS,
    DEFAULT_DWELL_THRESHOLD_SECONDS,
    OUTPUT_ROOT,
    REQUIRED_TRACKING_COLUMNS,
    TMP_SUFFIX,
    TRACKING_RESULTS_FILENAME,
    ZONE_COUNTS_FILENAME,
    ZONE_COUNTS_SCHEMA,
)
from zone_mapping.services.stats import (
    count_zone_visits,
    signed_distance_to_polygon,
    validate_zone_cameras,
)

logger = StructuredLogger(component="zone_map")

# 內切半徑取樣的格點密度：步長取 band/8，且長短邊各至少取這麼多點（小 zone 才不會
# 只取到幾個格點）。
_INRADIUS_MIN_SAMPLES = 32

# 估算取樣間隔所需的最少樣本數。各路的幀間隔極穩定（實測 p90 = p50），少量樣本就
# 夠估中位數；這個下限擋的是「每個 track 只有一兩列」那種算不出中位數的退化情形。
_GAP_MIN_SAMPLES = 8


def _resolve_band_px(
    camera_id: str, cam_sub: pl.DataFrame, boundary_band_px_1080p: float
) -> tuple[float, float]:
    """把 1080p 基準的線段區域寬度換算成該攝影機的實際像素，並記錄換算過程。

    換算只用寬度、線性：`實際 px = 基準值 × frame_width / 1920`。同一台攝影機整天
    解析度固定（上游 `probe_frame_shape` 只探測首格，中途變動會在讀取端逐格核對時
    就中止），因此這裡要求該攝影機的 `frame_width` 只有單一正值——多個值代表拿到的
    是手工拼接的 parquet 或上游語義已改，靜默取其中一個會讓半天的線段區域用錯尺度。

    Args:
        camera_id: 該攝影機的 `camera_id`（僅供錯誤訊息與日誌）。
        cam_sub: 只含該攝影機的追蹤明細，需已含 `frame_width` 欄位且非空。
        boundary_band_px_1080p: 以 1080p（寬 1920）為基準的線段區域寬度。

    Returns:
        `(band_px, scale)`：換算後的線段區域寬度（實際像素）與換算比例。一併回傳
        `scale` 是為了讓窄區域的錯誤訊息能把建議上限換算回使用者實際在改的那個
        1080p 基準值（`line_counting` 沒有這道檢查，故其對應函式只回 `band_px`）。

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
    band_px = boundary_band_px_1080p * scale
    logger.info(
        "線段區域依解析度換算",
        camera_id=camera_id,
        frame_width=frame_width,
        baseline_width=BASELINE_FRAME_WIDTH,
        scale=round(scale, 4),
        boundary_band_px_1080p=boundary_band_px_1080p,
        boundary_band_px=round(band_px, 2),
    )
    return band_px, scale


def _validate_zone_fits_band(
    camera_id: str, zone: Zone, band_px: float, scale: float
) -> None:
    """fail-loud：多邊形要寬到容得下線段區域，否則該 zone 的 entries 會恆為 0。

    最窄處小於 `2 × band` 的多邊形，內部可能沒有任何點滿足「帶號距離 > band」，
    狀態機永遠翻不到「確認在區內」。這種靜默的恆為 0 正是 fail-loud 要擋的那類錯誤，
    寫在 README 攔不住日後改 zone 幾何的人。

    內切半徑用格點取樣求：範圍取多邊形自身的 bounding box（不需要影像尺寸），步長取
    `band/8`，長短邊各至少 `_INRADIUS_MIN_SAMPLES` 點。**取樣求出的是內切半徑的
    下界**（真正的內切圓心不一定落在格點上），所以這道檢查會略微低估、可能誤擋邊緣
    案例；步長取 `band/8` 讓低估幅度遠小於判定所需的餘裕。

    Args:
        camera_id: 該攝影機的 `camera_id`（僅供錯誤訊息）。
        zone: 要檢查的區域定義。
        band_px: 該攝影機換算後的線段區域寬度（實際像素）。
        scale: `frame_width / 1920`，用來把建議上限換算回 1080p 基準值。

    Raises:
        ValueError: 該 zone 的內切半徑小於 `band_px`。
    """
    if band_px <= 0:
        return
    poly = np.asarray(zone.polygon, dtype=float)
    mins = poly.min(axis=0)
    maxs = poly.max(axis=0)
    step = max(1.0, band_px / 8)
    counts = [
        max(_INRADIUS_MIN_SAMPLES, int(np.ceil((maxs[i] - mins[i]) / step)) + 1)
        for i in (0, 1)
    ]
    gx, gy = np.meshgrid(
        np.linspace(mins[0], maxs[0], counts[0]),
        np.linspace(mins[1], maxs[1], counts[1]),
    )
    signed_d, _ = signed_distance_to_polygon(gx.ravel(), gy.ravel(), poly)
    inradius = float(signed_d.max())
    if inradius < band_px:
        suggested = inradius / scale
        # 內切半徑小到建議上限會四捨五入成 0 時，「調小 band」不是可執行的路
        # （boundary_band_px_1080p 有 ge=0），只剩改幾何一條
        remedy = (
            f"兩條路擇一：把 [zone].boundary_band_px_1080p 調到 {suggested:.1f} 以下"
            "（1080p 基準值），或把 camera_registry.yaml 的 zone 多邊形畫寬一點。"
            if round(suggested, 1) > 0
            else "這個 zone 窄到放不下任何正的線段區域，只能把 camera_registry.yaml 的 "
            "zone 多邊形畫寬一點。"
        )
        raise ValueError(
            f"攝影機 {camera_id} 的 zone「{zone.name}」放不下線段區域："
            f"內切半徑約 {inradius:.1f}px、線段區域 {band_px:.1f}px（皆為該攝影機的"
            f"實際像素，frame_width/{BASELINE_FRAME_WIDTH} = {scale:.4g}）。"
            "線段區域會吃掉整個區域內部，這個 zone 的 entries 會恆為 0。"
            f"{remedy}內切半徑為格點取樣的下界，會略微低估。"
        )


def _validate_gap_tolerance(
    camera_id: str, cam_sub: pl.DataFrame, dwell_gap_seconds: float
) -> None:
    """fail-loud：容忍窗小於該攝影機的取樣間隔時，`dwell_events` 會恆為 0。

    停留判定把相鄰兩次在區內的間隔超過 `dwell_gap_seconds` 的地方切成兩段。容忍窗
    小於幀間隔時**每一格都會被切開**，段長恆為 0，達不到任何正的門檻——輸出檔照樣
    產出、其餘兩個指標照樣正確，只有這一欄整天是 0。

    取樣間隔用「同一 `track_id` 內相鄰兩列的間隔」估，**不是該攝影機的
    `timestamp` diff**。後者有兩個致命問題：同一格畫面多人時 diff = 0
    （`(camera_id, timestamp)` 不是唯一鍵），忙碌的攝影機中位數會直接變 0、檢查永遠
    不觸發；沒有偵測的時段完全沒有列，冷清的攝影機中位數會膨脹到幾十秒而**誤擋**
    （實測 camera-wide 間隔 max 有 61 秒，per-track 只有 2.1 秒）。

    per-track 序列另有一個結構性的好性質：ByteTrack 的 `max_frames_lost` 就是
    `video_analyze` 的 `[tracker].track_buffer`，所以同一 `track_id` 內的空洞有上界
    `track_buffer / fps`，中位數必落在 `[1/fps, track_buffer/fps]`，最壞情況也只
    高估到約 1–2 秒，誤擋風險被限制住。

    ⚠ 該上界不是常數：`track_buffer` 預設 30 格但可調，而本套件既讀不到也驗不到它。
    上游調大它（或改抽幀）之後，這裡的中位數上界跟著放大，固定的
    `dwell_gap_seconds` 預設值就可能誤擋——代價是該日整份報表都產不出來。

    Args:
        camera_id: 該攝影機的 `camera_id`（僅供錯誤訊息與日誌）。
        cam_sub: 只含該攝影機的追蹤明細，需已含 `track_id`／`timestamp` 欄位。
        dwell_gap_seconds: 同一段停留可容忍的中斷（秒）。

    Raises:
        ValueError: `dwell_gap_seconds` 小於估出的取樣間隔中位數。
    """
    gaps = (
        cam_sub.sort("track_id", "timestamp")
        .select(
            pl.col("timestamp")
            .diff()
            .over("track_id")
            .dt.total_microseconds()
            .alias("_gap_us")
        )["_gap_us"]
        .drop_nulls()
    )
    if len(gaps) < _GAP_MIN_SAMPLES:
        # 證據不足時放行，與 `_validate_zone_fits_band` 對「沒有 zone 就不檢查」
        # 同一個立場：誤擋的代價是該日整份報表都產不出來。
        logger.warning(
            "取樣間隔樣本不足，跳過容忍窗檢查",
            camera_id=camera_id,
            gap_samples=len(gaps),
            min_samples=_GAP_MIN_SAMPLES,
            dwell_gap_seconds=dwell_gap_seconds,
        )
        return
    median_us = float(gaps.median())
    logger.info(
        "取樣間隔估算（同一 track 相鄰列）",
        camera_id=camera_id,
        gap_samples=len(gaps),
        median_gap_ms=round(median_us / 1000, 2),
        dwell_gap_seconds=dwell_gap_seconds,
    )
    if dwell_gap_seconds * 1_000_000 < median_us:
        # median_us 必然 > 0：走到這裡代表 dwell_gap_seconds * 1e6 < median_us，
        # 而 map_zones_daily 的前置檢查已保證 dwell_gap_seconds > 0
        fps = 1_000_000 / median_us
        raise ValueError(
            f"攝影機 {camera_id} 的 [zone].dwell_gap_seconds ="
            f" {dwell_gap_seconds} 秒小於該路的取樣間隔"
            f"（中位數約 {median_us / 1000:.1f} ms，換算約 {fps:.1f} fps，"
            f"取自同一 track 相鄰列的 {len(gaps)} 個間隔）。"
            "每一格都會被切成獨立的停留段，這台攝影機的 dwell_events 會恆為 0。"
            "兩條路擇一：把 dwell_gap_seconds 調到取樣間隔以上"
            "（預設 3.0 秒涵蓋 15–30 fps），或確認上游 video_analyze 的 fps 與 "
            "[tracker].track_buffer 是否異常——後者決定同一 track 內空洞的上界"
            "（track_buffer / fps），調大它會把這個中位數一起拉高。"
        )


def map_zones_daily(
    date: datetime.date,
    bucket_dir: str,
    bucket_minutes: int,
    boundary_band_px_1080p: float = DEFAULT_BOUNDARY_BAND_PX_1080P,
    dwell_threshold_seconds: float = DEFAULT_DWELL_THRESHOLD_SECONDS,
    dwell_gap_seconds: float = DEFAULT_DWELL_GAP_SECONDS,
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
        boundary_band_px_1080p: `entries` 判定的線段區域寬度，以 1080p
            （寬 1920）為基準的像素值；逐攝影機依其 `frame_width` 換算成實際像素後
            才進判定（見 ADR-004、ADR-006）。`0` = 純內外判定，且換算後仍是 0；
            預設 25 取自實測（見 README「已知限制」）。
        dwell_threshold_seconds: `dwell_events` 的停留門檻（秒），必須 > 0。與像素
            參數不同，這是絕對時間、不隨解析度換算（見 ADR-016）。
        dwell_gap_seconds: 同一段停留可容忍的中斷（秒），必須 > 0。
        output_root: 輸出根目錄。

    Returns:
        `zone_counts.parquet` 的路徑。

    Raises:
        FileNotFoundError: 當日 `tracking_results.parquet` 不存在，或
            `bucket_dir` 底下找不到 `camera_registry.yaml`。
        ValueError: 追蹤結果缺少影像尺寸或落腳點欄位（舊版 `video_analyze` 的
            產物）、`camera_registry.yaml` 定義了 zone 的攝影機在當天追蹤結果中
            查無資料、任一 zone 定義不合法、有 zone 窄到放不下線段區域，或任一
            攝影機的 `dwell_gap_seconds` 小於其取樣間隔。
    """
    # 直接呼叫本函式時不經 pydantic，這兩個門檻的 gt=0 要自己擋一次：0 會讓每個
    # 在區內的段都計一次，數字看起來合理但量的是別的東西。
    for name, value in (
        ("dwell_threshold_seconds", dwell_threshold_seconds),
        ("dwell_gap_seconds", dwell_gap_seconds),
    ):
        if value <= 0:
            # 兩個參數都沒有可用的退化語義，但 0 的後果剛好相反：門檻 0 讓每個段都
            # 達標（超計到底），容忍窗 0 把每一格切成獨立的段（恆為 0）。分開講，
            # 誤設的人才知道自己的數字往哪個方向錯。
            consequence = (
                "0 會讓每個在區內的段都達標，dwell_events 量到的就不再是停留"
                if name == "dwell_threshold_seconds"
                else "0 會把每一格都切成獨立的停留段，dwell_events 會恆為 0"
            )
            raise ValueError(
                f"{name} 必須大於 0（收到 {value}）。停留門檻與容忍窗都沒有可用的"
                "退化語義（不像 boundary_band_px_1080p 的 0 = 純內外判定）："
                f"{consequence}（見 ADR-016）。"
            )

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
    missing = [c for c in REQUIRED_TRACKING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{results_path} 缺少欄位 {missing}：這是舊版 video_analyze 的產物。"
            "沒有影像尺寸就無法把 boundary_band_px_1080p 換算成各攝影機的實際像素；"
            "沒有 foot_x／foot_y 就沒有落腳點可判定（本套件不再自行從 bbox 推算，"
            "見 ADR-009）。請以現行版本的 video_analyze 重跑該日產生新的 "
            "tracking_results.parquet。"
        )
    # 先驗證 camera 對得上當天資料再解析 zone，避免陳舊 zone 定義打錯字蓋過更根本錯誤
    validate_zone_cameras(
        {k for k, e in zone_entries.items() if e.zones},
        set(df["camera_id"].unique()),
    )
    zone_cameras = parse_and_validate_zones(zone_entries)

    # 落腳點直接讀上游欄位——推算需要 head 框，而 head 不在追蹤結果裡（見 ADR-009）
    df = df.with_columns(
        pl.col("timestamp").dt.truncate(f"{bucket_minutes}m").alias("time_bucket"),
    )

    frames: list[pl.DataFrame] = []
    for camera_id, zones in zone_cameras.items():
        # 參與名單看的是 participates_in_zone_mapping，不是「zones 非空」，所以這裡
        # 會拿到沒有 zone 的攝影機；它當天可能完全沒有資料，換算會拿不到 frame_width。
        # 換算與檢查一律放在「該攝影機真的有 zone」之後。
        if not zones:
            continue
        cam_sub = df.filter(pl.col("camera_id") == camera_id)
        # 換算在此算好，`count_zone_visits` 收到的一律是實際像素——判定邏輯不需要
        # 知道「基準解析度」這個概念，`stats.py` 維持純幾何
        band_px, scale = _resolve_band_px(camera_id, cam_sub, boundary_band_px_1080p)
        # 容忍窗檢查是 camera 級（估的是該路的取樣間隔），與 zone 幾何無關，故在
        # zone 迴圈之前只做一次
        _validate_gap_tolerance(camera_id, cam_sub, dwell_gap_seconds)
        for zone in zones:
            _validate_zone_fits_band(camera_id, zone, band_px, scale)
            counts = count_zone_visits(
                cam_sub,
                zone,
                band_px,
                dwell_threshold_seconds=dwell_threshold_seconds,
                dwell_gap_seconds=dwell_gap_seconds,
            ).with_columns(
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

    logger.info(
        "Zone 人流統計已寫入",
        path=str(counts_path),
        cameras=len(zone_cameras),
        rows=result.height,
    )
    return counts_path
