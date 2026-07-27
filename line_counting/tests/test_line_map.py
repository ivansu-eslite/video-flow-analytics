import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from line_counting.services.line_map import count_lines_daily

_TAIPEI = ZoneInfo("Asia/Taipei")

# 計數線 door：水平線 y=100，inside_point 在下方（y=300 側，影像座標下方 = y 大）
_DOOR = {
    "name": "door",
    "points": [[0, 100], [200, 100]],
    "inside_point": [100, 300],
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


def _write_tracking_results(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).write_parquet(path)


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
    assert result["in_count"].to_list() == [1]
    assert result["out_count"].to_list() == [0]
    assert result["time_bucket"].to_list() == [base]

    # 快照有寫出，供下游以「產生此份資料時的計數線定義」為準
    assert (output_dir / "camera_registry_used.yaml").exists()


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
