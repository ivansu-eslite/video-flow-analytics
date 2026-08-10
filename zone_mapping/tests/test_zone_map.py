import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from zone_mapping.services.zone_map import _resolve_band_px, map_zones_daily

_TAIPEI = ZoneInfo("Asia/Taipei")

# 邊長 200 的正方形，內切半徑 100：容得下預設的 25 px 緩衝帶（不傳 band 參數的
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
    path: Path, camera_id: str, frame_width: int = 1920, frame_height: int = 1080
) -> None:
    """寫出追蹤明細；`frame_width`／`frame_height` 逐列補成同一個值（上游即如此寫）。

    預設 1920×1080 = 基準解析度，換算係數 1，讓不測換算的案例維持原本的判定尺度。
    落腳點 (100, 100) 在 `_SQUARE_200` 的正中央，離邊界 100 px。

    各案例刻意**只給 `foot_x`／`foot_y`、不給 bbox 欄位**：落腳點改由上游算好寫進
    parquet 後（ADR-009），本套件只讀這兩欄；若有人把判定改回從 bbox 現算，這些
    測試會因為缺欄位而爆，而不是靜默沿用舊公式。
    """
    df = pl.DataFrame(
        {
            "camera_id": [camera_id],
            "timestamp": [datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)],
            "track_id": [1],
            "foot_x": [100.0],
            "foot_y": [100.0],
            "frame_width": [frame_width],
            "frame_height": [frame_height],
        }
    )
    df.write_parquet(path)


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
    - cam001（1920）換算後緩衝帶 10 px < 15 → 確認在區內，算一次進入。
    - cam002（3840）換算後緩衝帶 20 px > 15 → 整段都在緩衝帶內，沒有已確認的進入。

    `unique_visitors` 兩台都是 1（腳底點確實落在多邊形內），順帶鎖住「緩衝帶只影響
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
    """內切半徑小於緩衝帶的 zone 要報錯：緩衝帶吃掉整個區域內部後，該 zone 的
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
                # 邊長 10 的正方形，內切半徑僅 5 px，放不下預設的 25 px 緩衝帶
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
    退回「把設定值當成實際像素」——那會在 4K 攝影機上靜默套用只有一半寬的緩衝帶。"""
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
    """同一台攝影機出現多個 frame_width：靜默取其中一個會讓半天的緩衝帶用錯尺度，
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
