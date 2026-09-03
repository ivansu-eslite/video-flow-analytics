import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from zone_mapping.config.constants import ZONE_COUNTS_SCHEMA
from zone_mapping.services.zone_map import _resolve_band_px, map_zones_daily

_TAIPEI = ZoneInfo("Asia/Taipei")

# 邊長 200 的正方形，內切半徑 100：容得下預設的 25 px 線段區域（不傳 band 參數的
# 測試才不會被「窄區域 fail loud」擋在到達待測邏輯之前）。
_SQUARE_200 = [[0, 0], [200, 0], [200, 200], [0, 200]]


def _write_registry(path: Path, cameras: list[dict]) -> None:
    data = {
        "bucket_name": "bucket_test",
        "storage": {},
        "cameras": cameras,
    }
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_tracking_results(
    path: Path,
    camera_id: str,
    frame_width: int = 1920,
    frame_height: int = 1080,
    tracks: dict[int, list[float]] | None = None,
) -> None:
    """寫出追蹤明細；`frame_width`／`frame_height` 逐列補成同一個值（上游即如此寫）。

    預設 1920×1080 = 基準解析度，換算係數 1，讓不測換算的案例維持原本的判定尺度。
    落腳點 (100, 100) 在 `_SQUARE_200` 的正中央，離邊界 100 px。

    `tracks` 把 `track_id` 對到該軌跡逐格的秒數偏移（可為小數、不必等距），停留與
    取樣間隔的案例需要真正的時間序列；不給時維持單一 `track_id`、單一時間點。
    每個 track 的落腳點都在正中央，時間軸才是這些案例唯一的變因。

    各案例刻意**只給 `foot_x`／`foot_y`、不給 bbox 欄位**：落腳點改由上游算好寫進
    parquet 後（ADR-009），本套件只讀這兩欄；若有人把判定改回從 bbox 現算，這些
    測試會因為缺欄位而爆，而不是靜默沿用舊公式。
    """
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    series = tracks if tracks is not None else {1: [0.0]}
    track_ids = [tid for tid, offsets in series.items() for _ in offsets]
    stamps = [
        base + datetime.timedelta(seconds=s)
        for offsets in series.values()
        for s in offsets
    ]
    n = len(track_ids)
    df = pl.DataFrame(
        {
            "camera_id": [camera_id] * n,
            "timestamp": stamps,
            "track_id": track_ids,
            "foot_x": [100.0] * n,
            "foot_y": [100.0] * n,
            "frame_width": [frame_width] * n,
            "frame_height": [frame_height] * n,
        }
    )
    df.write_parquet(path)


def _one_zone_bucket(tmp_path: Path) -> tuple[Path, Path]:
    """建一個「單一攝影機、單一 zone」的 bucket 與輸出目錄，回傳兩者的路徑。

    停留相關的案例只差在追蹤明細的時間軸，registry 與目錄結構完全一樣。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )
    output_dir = tmp_path / "outputs" / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    return bucket_dir, output_dir


def test_map_zones_daily_ignores_missing_data_for_camera_without_zones(tmp_path):
    """`participates_in_zone_mapping=True` 但 `zones: []` 的攝影機當天無資料
    不該中止全天：這種攝影機沒有任何 zone 人流可漏，validate_zone_cameras 的
    fail-loud 保護的東西根本不存在。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [],
            },
            {
                "camera_id": "cam002",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [
                    {
                        "name": "zone_a",
                        "polygon": _SQUARE_200,
                    }
                ],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    # 只有 cam002（有定義 zone）當天有資料；cam001（zones: []）當天無資料
    _write_tracking_results(output_dir / "tracking_results.parquet", "loc_cam002")

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        output_root=output_root,
    )

    result = pl.read_parquet(counts_path)
    assert result["camera_id"].to_list() == ["loc_cam002"]
    assert result["zone"].to_list() == ["zone_a"]


def test_map_zones_daily_writes_only_parquet(tmp_path):
    """輸出目錄只留 parquet：本階段不再複製 registry 快照（見 ADR-007）。

    下游 `flow_report` 改讀 `bucket_dir` 當下的 `camera_registry.yaml`，這裡再寫
    `camera_registry_used.yaml` 只會留下沒人讀的檔案，且與部署端的輸出結構不一致。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    _write_tracking_results(output_dir / "tracking_results.parquet", "loc_cam001")

    map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        output_root=output_root,
    )

    assert not (output_dir / "camera_registry_used.yaml").exists()
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "tracking_results.parquet",
        "zone_counts.parquet",
    ]


def test_map_zones_daily_still_fails_loud_for_camera_with_zones_missing_data(
    tmp_path,
):
    """回歸鎖：真正定義了 zone 的攝影機當天查無資料時仍須 fail-loud，
    證明 A-2 只是縮小驗證範圍，不是關掉這道保護。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam002",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [
                    {
                        "name": "zone_a",
                        "polygon": _SQUARE_200,
                    }
                ],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    # cam002 定義了 zone，但當天 tracking_results 完全沒有它的資料
    _write_tracking_results(output_dir / "tracking_results.parquet", "loc_other")

    with pytest.raises(ValueError, match="loc_cam002"):
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_boundary_band_scales_with_each_camera_frame_width(tmp_path):
    """同一個 1080p 基準值在不同解析度上要換算成不同的實際寬度。

    兩台攝影機給完全相同的 zone 與軌跡（腳底離邊界 15 px），
    `boundary_band_px_1080p = 10`：
    - cam001（1920）換算後線段區域 10 px < 15 → 確認在區內，算一次進入。
    - cam002（3840）換算後線段區域 20 px > 15 → 整段都在線段區域內，沒有已確認的進入。

    `unique_visitors` 兩台都是 1（腳底點確實落在多邊形內），順帶鎖住「線段區域只影響
    entries」。沒換算（兩台都用 10）或換算方向寫反（3840 用 5）時 cam002 會多算一次。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
            {
                "camera_id": "cam002",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_b", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    # 逐列的 frame_width 隨攝影機不同，故不走 _write_tracking_results 的單一尺寸
    pl.DataFrame(
        {
            "camera_id": ["loc_cam001", "loc_cam002"],
            "timestamp": [base, base],
            "track_id": [1, 2],
            "foot_x": [15.0, 15.0],
            "foot_y": [100.0, 100.0],  # 腳底 (15, 100)：離左邊界 15 px（區內）
            "frame_width": [1920, 3840],
            "frame_height": [1080, 2160],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        boundary_band_px_1080p=10,
        output_root=output_root,
    )

    result = pl.read_parquet(counts_path).sort("camera_id")
    assert result.select("camera_id", "unique_visitors", "entries").rows() == [
        ("loc_cam001", 1, 1),
        ("loc_cam002", 1, 0),
    ]


def test_band_scales_linearly_for_non_standard_resolution():
    """換算是線性比例，不是「1080p 或 4K」的查表：2560 寬得到 2560/1920 ≈ 1.333 倍。"""
    cam_sub = pl.DataFrame({"frame_width": [2560, 2560]})

    band_px, scale = _resolve_band_px("loc_cam001", cam_sub, 12)

    assert band_px == pytest.approx(16.0)
    assert scale == pytest.approx(2560 / 1920)


def test_zero_band_stays_zero_at_any_resolution(tmp_path):
    """基準值 0 換算後仍是 0：非基準解析度的攝影機設 0 時也是純內外判定，貼邊界的
    腳底點照算一次進入，行為與時間去抖時代 `entry_debounce_frames = 1` 一致。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    pl.DataFrame(
        {
            "camera_id": ["loc_cam001"],
            "timestamp": [base],
            "track_id": [1],
            "foot_x": [0.5],
            "foot_y": [100.0],  # 腳底 (0.5, 100)：僅離邊界 0.5 px
            "frame_width": [3840],
            "frame_height": [2160],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        boundary_band_px_1080p=0,
        output_root=output_root,
    )

    assert pl.read_parquet(counts_path)["entries"].to_list() == [1]


def test_zone_narrower_than_band_fails_loud(tmp_path):
    """內切半徑小於線段區域的 zone 要報錯：線段區域吃掉整個區域內部後，該 zone 的
    entries 會恆為 0，靜默產出一份全是 0 的統計比中止危險得多。

    訊息要帶得出算出的半徑與建議上限（1080p 基準值），誤擋時操作的人才知道往哪調。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                # 邊長 10 的正方形，內切半徑僅 5 px，放不下預設的 25 px 線段區域
                "zones": [
                    {"name": "zone_a", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
                ],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    _write_tracking_results(output_dir / "tracking_results.parquet", "loc_cam001")

    with pytest.raises(ValueError, match="內切半徑") as excinfo:
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )

    message = str(excinfo.value)
    assert "zone_a" in message
    assert "boundary_band_px_1080p" in message


def test_tracking_results_without_frame_size_fails_loud(tmp_path):
    """舊版 video_analyze 產出的 parquet 沒有影像尺寸欄位：直接報錯中止，不可
    退回「把設定值當成實際像素」——那會在 4K 攝影機上靜默套用只有一半寬的線段區域。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    # 刻意不經 _write_tracking_results：這裡要的正是缺欄位的舊格式
    pl.DataFrame(
        {
            "camera_id": ["loc_cam001"],
            "timestamp": [base],
            "track_id": [1],
            "foot_x": [100.0],
            "foot_y": [100.0],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    with pytest.raises(ValueError, match="frame_width"):
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_tracking_results_without_foot_point_fails_loud(tmp_path):
    """有影像尺寸、但只有 bbox 沒有落腳點欄位（issue #63 之後、#72 之前的產物）：
    同樣要報錯中止。這種 parquet 通過了尺寸那道檢查，若不擋下就會走到判定時才因
    缺欄位炸在 polars 層，或被日後補上的 bbox fallback 靜默套回舊公式。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    # 刻意不經 _write_tracking_results：這裡要的正是「有 bbox、無 foot」的舊格式
    pl.DataFrame(
        {
            "camera_id": ["loc_cam001"],
            "timestamp": [base],
            "track_id": [1],
            "x1": [80.0],
            "y1": [20.0],
            "x2": [120.0],
            "y2": [100.0],
            "frame_width": [1920],
            "frame_height": [1080],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    with pytest.raises(ValueError, match="foot_x"):
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_multiple_frame_widths_for_one_camera_fails_loud(tmp_path):
    """同一台攝影機出現多個 frame_width：靜默取其中一個會讓半天的線段區域用錯尺度，
    須報錯（上游整天解析度固定，出現這種資料代表 parquet 是拼接出來的）。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [{"name": "zone_a", "polygon": _SQUARE_200}],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    pl.DataFrame(
        {
            "camera_id": ["loc_cam001", "loc_cam001"],
            "timestamp": [base, base + datetime.timedelta(seconds=1)],
            "track_id": [1, 1],
            "foot_x": [100.0, 100.0],
            "foot_y": [100.0, 100.0],
            "frame_width": [1920, 3840],
            "frame_height": [1080, 2160],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    with pytest.raises(ValueError, match="frame_width"):
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_gap_tolerance_below_sampling_interval_fails_loud(tmp_path):
    """容忍窗小於該路的取樣間隔要報錯：每一格都會被切成獨立的停留段，段長恆為 0，
    這台攝影機的 `dwell_events` 整天都是 0，而輸出檔與其餘兩個指標完全正常。

    訊息要帶得出估出的間隔、換算 fps 與「會恆為 0」這句後果，誤擋時操作的人才知道
    要往哪調（比照 `test_zone_narrower_than_band_fails_loud`）。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    # 單一軌跡、每 0.5 秒一格（2 fps），容忍窗 0.01 秒遠小於它
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        "loc_cam001",
        tracks={1: [i * 0.5 for i in range(12)]},
    )

    with pytest.raises(ValueError, match="dwell_gap_seconds") as excinfo:
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            dwell_gap_seconds=0.01,
            output_root=tmp_path / "outputs",
        )

    message = str(excinfo.value)
    assert "loc_cam001" in message
    assert "500.0 ms" in message
    assert "2.0 fps" in message
    assert "恆為 0" in message


def test_gap_tolerance_check_passes_when_window_covers_interval(tmp_path):
    """對照上一支：同一份明細把容忍窗調到取樣間隔以上就放行，證明擋下的是容忍窗
    與間隔的關係，不是「只要有時間序列就報錯」。"""
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        "loc_cam001",
        tracks={1: [i * 0.5 for i in range(12)]},
    )

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        dwell_gap_seconds=0.5,
        output_root=tmp_path / "outputs",
    )

    assert pl.read_parquet(counts_path).height == 1


def test_gap_tolerance_check_skipped_when_samples_insufficient(tmp_path):
    """每個 track 都只有一列時算不出間隔中位數，記 warning 放行、不誤擋。

    證據不足時放行，與 `_validate_zone_fits_band` 對「該攝影機沒有 zone 就不檢查」
    同一個立場：誤擋的代價是該日整份報表都產不出來。這裡的容忍窗 0.01 秒小於任何
    真實幀間隔，檢查若在樣本不足時照跑就會擋下這一天。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    # 十條軌跡各只出現一格：per-track 的 diff 全是 null，樣本數 0
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        "loc_cam001",
        tracks={tid: [float(tid)] for tid in range(1, 11)},
    )

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        dwell_gap_seconds=0.01,
        output_root=tmp_path / "outputs",
    )

    assert pl.read_parquet(counts_path)["unique_visitors"].to_list() == [10]


def test_gap_estimate_not_zeroed_by_multiple_people_in_one_frame(tmp_path):
    """同一格畫面有多人時，取樣間隔的估算值不可被壓成 0。

    `(camera_id, timestamp)` 不是唯一鍵（見 CLAUDE.md），同一格畫面每個目標一列。
    把統計量改回該攝影機的 `timestamp.diff()` 中位數，三個人同框就讓多數 diff 是 0、
    中位數變 0，任何正的容忍窗都通過，這道 fail-loud 從此永遠不觸發——而且是在最
    需要它的忙碌攝影機上失效。

    這裡三條軌跡同框、每 0.5 秒一格：per-track 中位數 0.5 秒，camera-wide 中位數 0。
    容忍窗 0.1 秒介於兩者之間，只有正確的統計量會擋下。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        "loc_cam001",
        tracks={tid: [i * 0.5 for i in range(10)] for tid in (1, 2, 3)},
    )

    with pytest.raises(ValueError, match="dwell_gap_seconds"):
        map_zones_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            dwell_gap_seconds=0.1,
            output_root=tmp_path / "outputs",
        )


def test_map_zones_daily_output_contains_dwell_events(tmp_path):
    """端到端：輸出檔含 `dwell_events` 欄，且達標的停留真的算得出來。

    軌跡 1 在區內連續待 25 秒（> 門檻 20）計 1；軌跡 2 只待 5 秒不計。`unique_visitors`
    兩人都算，鎖住「停留不是把另外兩個指標換掉」。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        "loc_cam001",
        tracks={
            1: [float(i) for i in range(26)],
            2: [float(i) for i in range(6)],
        },
    )

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        dwell_threshold_seconds=20.0,
        dwell_gap_seconds=3.0,
        output_root=tmp_path / "outputs",
    )

    result = pl.read_parquet(counts_path)
    assert result.columns == list(ZONE_COUNTS_SCHEMA)
    assert result.select("unique_visitors", "dwell_events").rows() == [(2, 1)]


def test_map_zones_daily_rejects_non_positive_dwell_parameters(tmp_path):
    """直接呼叫 `map_zones_daily` 不經 pydantic，兩個門檻的 `> 0` 要自己擋一次。

    門檻 0 會讓每個在區內的段都達標，數字看起來合理但量的是別的東西；容忍窗 0 則
    把每一格切成獨立的段。README 明列本函式是正式進入點，這道檢查不能只靠設定層。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    _write_tracking_results(output_dir / "tracking_results.parquet", "loc_cam001")

    for kwargs in (
        {"dwell_threshold_seconds": 0},
        {"dwell_threshold_seconds": -1.0},
        {"dwell_gap_seconds": 0},
    ):
        with pytest.raises(ValueError, match="必須大於 0"):
            map_zones_daily(
                date=datetime.date(2026, 5, 1),
                bucket_dir=str(bucket_dir),
                bucket_minutes=60,
                output_root=tmp_path / "outputs",
                **kwargs,
            )


def test_gap_estimate_uses_median_not_mean_or_max(tmp_path):
    """取樣間隔用**中位數**，不是平均或最大值——偏態的時間軸只有中位數不會誤擋。

    per-track 的間隔序列本來就偏態：多數是幀間隔，少數是漏偵測留下的空洞（上界
    `track_buffer / fps`）。這裡 29 個 0.1 秒的間隔配 3 個 5 秒空洞：中位數 0.1 秒、
    平均約 0.56 秒、最大 5 秒。容忍窗取 **0.4 秒**——遠大於幀間隔，對這台攝影機完全
    夠用（5 秒空洞會被切開，但那本來就不是同一段停留）。三個統計量刻意夾在 0.4 的
    兩側：median（0.1）放行，mean（0.56）與 max（5）都會擋下整天，而誤擋的代價是
    該日整份報表都產不出來。
    """
    bucket_dir, output_dir = _one_zone_bucket(tmp_path)
    offsets, t = [], 0.0
    for i in range(33):
        offsets.append(t)
        t += 5.0 if i in (10, 20, 30) else 0.1
    _write_tracking_results(
        output_dir / "tracking_results.parquet", "loc_cam001", tracks={1: offsets}
    )

    counts_path = map_zones_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        dwell_gap_seconds=0.4,
        output_root=tmp_path / "outputs",
    )

    assert pl.read_parquet(counts_path)["unique_visitors"].to_list() == [1]
