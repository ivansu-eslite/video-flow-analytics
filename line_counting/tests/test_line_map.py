import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from line_counting.services.line_map import _resolve_band_px, count_lines_daily

_TAIPEI = ZoneInfo("Asia/Taipei")

# 計數線 door：水平線 y=100，inside_point 在下方（y=300 側，影像座標下方 = y 大）
_DOOR = {
    "name": "door",
    "points": [[0, 100], [200, 100]],
    "inside_point": [100, 300],
    "line_group": "main_entrance",
}


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
    path: Path, rows: dict, frame_width: int = 1920, frame_height: int = 1080
) -> None:
    """寫出追蹤明細；`frame_width`／`frame_height` 逐列補成同一個值（上游即如此寫）。

    預設 1920×1080 = 基準解析度，換算係數 1，讓不測換算的案例維持原本的判定尺度。
    """
    n = len(next(iter(rows.values())))
    pl.DataFrame(
        {**rows, "frame_width": [frame_width] * n, "frame_height": [frame_height] * n}
    ).write_parquet(path)


def test_count_lines_daily_counts_crossing_and_ignores_camera_without_lines(tmp_path):
    """happy path：定義了計數線且當天有資料的攝影機正確算出跨越；沒有 `lines` 的
    攝影機（不在參與集合）不因當天無資料而中止。

    track 由外側（y2=50，y<100）跨到內側（y2=150，往 inside_point 那側）→ in=1。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            # cam001：定義 door，當天有資料
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR],
            },
            # cam002：沒有 lines（不參與），當天亦無資料——不該中止全天
            {
                "camera_id": "cam002",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        {
            "camera_id": ["loc_cam001", "loc_cam001"],
            "timestamp": [base, base + datetime.timedelta(seconds=1)],
            "track_id": [1, 1],
            "x1": [90.0, 90.0],
            "y1": [40.0, 140.0],
            "x2": [110.0, 110.0],
            "y2": [50.0, 150.0],  # 腳底 y：50（外側）→ 150（內側）
        },
    )

    counts_path = count_lines_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        output_root=output_root,
    )

    result = pl.read_parquet(counts_path)
    assert result["camera_id"].to_list() == ["loc_cam001"]
    assert result["line"].to_list() == ["door"]
    assert result["line_group"].to_list() == ["main_entrance"]
    assert result["in_count"].to_list() == [1]
    assert result["out_count"].to_list() == [0]
    assert result["time_bucket"].to_list() == [base]

    # 快照有寫出，供下游以「產生此份資料時的計數線定義」為準
    assert (output_dir / "camera_registry_used.yaml").exists()


def test_count_lines_daily_maps_line_group_to_its_own_line(tmp_path):
    """兩條計數線分屬不同 `line_group`：輸出各列的 `line_group` 須正確對應到
    自己那條線，不能被另一條線的群組污染。刻意不測「加總等於各群組之和」——
    那是 polars 的 `group_by().sum()`，不是本套件的邏輯（見設計章）。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    door_b = {
        "name": "door_b",
        "points": [[0, 500], [200, 500]],
        "inside_point": [100, 700],
        "line_group": "back_entrance",
    }
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR, door_b],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        {
            "camera_id": ["loc_cam001", "loc_cam001", "loc_cam001", "loc_cam001"],
            "timestamp": [
                base,
                base + datetime.timedelta(seconds=1),
                base + datetime.timedelta(seconds=2),
                base + datetime.timedelta(seconds=3),
            ],
            "track_id": [1, 1, 2, 2],
            "x1": [90.0, 90.0, 90.0, 90.0],
            "y1": [40.0, 140.0, 440.0, 640.0],
            "x2": [110.0, 110.0, 110.0, 110.0],
            # track 1 跨 door（y=100）；track 2 跨 door_b（y=500）
            "y2": [50.0, 150.0, 450.0, 650.0],
        },
    )

    counts_path = count_lines_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        output_root=output_root,
    )

    result = pl.read_parquet(counts_path).sort("line")
    assert result.select("line", "line_group").rows() == [
        ("door", "main_entrance"),
        ("door_b", "back_entrance"),
    ]


def test_crossing_band_scales_with_each_camera_frame_width(tmp_path):
    """同一個 1080p 基準值在不同解析度上要換算成不同的實際帶寬。

    兩台攝影機給完全相同的線與軌跡（腳底 y 85 → 115，離線 15 px），
    `crossing_band_px_1080p = 10`：
    - cam001（1920）換算後帶寬 10 px < 15 → 兩格分判外／內側，算一次 in。
    - cam002（3840）換算後帶寬 20 px > 15 → 兩格都落在死區內，不算跨越。

    沒換算（兩台都用 10）或換算方向寫反（3840 用 5）時，cam002 都會多出一列，
    斷言會失敗。
    """
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    door_b = {**_DOOR, "name": "door_b"}
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR],
            },
            {
                "camera_id": "cam002",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [door_b],
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
            "camera_id": ["loc_cam001", "loc_cam001", "loc_cam002", "loc_cam002"],
            "timestamp": [
                base,
                base + datetime.timedelta(seconds=1),
                base,
                base + datetime.timedelta(seconds=1),
            ],
            "track_id": [1, 1, 2, 2],
            "x1": [90.0, 90.0, 90.0, 90.0],
            "y1": [75.0, 105.0, 75.0, 105.0],
            "x2": [110.0, 110.0, 110.0, 110.0],
            "y2": [85.0, 115.0, 85.0, 115.0],  # 腳底離線 15 px（外側 → 內側）
            "frame_width": [1920, 1920, 3840, 3840],
            "frame_height": [1080, 1080, 2160, 2160],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    counts_path = count_lines_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        crossing_band_px_1080p=10,
        output_root=output_root,
    )

    result = pl.read_parquet(counts_path)
    assert result.select("camera_id", "line", "in_count", "out_count").rows() == [
        ("loc_cam001", "door", 1, 0)
    ]


def test_band_scales_linearly_for_non_standard_resolution():
    """換算是線性比例，不是「1080p 或 4K」的查表：2560 寬得到 2560/1920 ≈ 1.333 倍。"""
    cam_sub = pl.DataFrame({"frame_width": [2560, 2560]})

    assert _resolve_band_px("loc_cam001", cam_sub, 12) == pytest.approx(16.0)


def test_zero_band_stays_zero_at_any_resolution(tmp_path):
    """基準值 0 換算後仍是 0：非基準解析度的攝影機設 0 時也是純零交越，貼線的
    小幅跨越照算，行為與換算機制上線前一致。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        {
            "camera_id": ["loc_cam001", "loc_cam001"],
            "timestamp": [base, base + datetime.timedelta(seconds=1)],
            "track_id": [1, 1],
            "x1": [90.0, 90.0],
            "y1": [89.0, 91.0],
            "x2": [110.0, 110.0],
            "y2": [99.0, 101.0],  # 腳底離線僅 1 px
        },
        frame_width=3840,
        frame_height=2160,
    )

    counts_path = count_lines_daily(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        bucket_minutes=60,
        crossing_band_px_1080p=0,
        output_root=output_root,
    )

    assert pl.read_parquet(counts_path)["in_count"].to_list() == [1]


def test_tracking_results_without_frame_size_fails_loud(tmp_path):
    """舊版 video_analyze 產出的 parquet 沒有影像尺寸欄位：直接報錯中止，不可
    退回「把設定值當成實際像素」——那會在 4K 攝影機上靜默套用一半的死區。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR],
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
            "x1": [90.0],
            "y1": [140.0],
            "x2": [110.0],
            "y2": [150.0],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    with pytest.raises(ValueError, match="frame_width"):
        count_lines_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_multiple_frame_widths_for_one_camera_fails_loud(tmp_path):
    """同一台攝影機出現多個 frame_width：靜默取其中一個會讓半天的死區用錯尺度，
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
                "lines": [_DOOR],
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
            "x1": [90.0, 90.0],
            "y1": [40.0, 140.0],
            "x2": [110.0, 110.0],
            "y2": [50.0, 150.0],
            "frame_width": [1920, 3840],
            "frame_height": [1080, 2160],
        }
    ).write_parquet(output_dir / "tracking_results.parquet")

    with pytest.raises(ValueError, match="frame_width"):
        count_lines_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )


def test_count_lines_daily_still_fails_loud_for_camera_with_lines_missing_data(
    tmp_path,
):
    """回歸鎖：真正定義了計數線的攝影機當天查無資料時仍須 fail-loud，
    證明參與判定只是縮小驗證範圍到有 lines 的攝影機，不是關掉這道保護。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        [
            {
                "camera_id": "cam001",
                "location": "loc",
                "ip": "127.0.0.1",
                "lines": [_DOOR],
            },
        ],
    )

    output_root = tmp_path / "outputs"
    output_dir = output_root / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)
    # cam001 定義了計數線，但當天 tracking_results 完全沒有它的資料（只有別台）
    _write_tracking_results(
        output_dir / "tracking_results.parquet",
        {
            "camera_id": ["loc_other"],
            "timestamp": [base],
            "track_id": [1],
            "x1": [90.0],
            "y1": [140.0],
            "x2": [110.0],
            "y2": [150.0],
        },
    )

    with pytest.raises(ValueError, match="loc_cam001"):
        count_lines_daily(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            bucket_minutes=60,
            output_root=output_root,
        )
