import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from zone_mapping.services.zone_map import map_zones_daily

_TAIPEI = ZoneInfo("Asia/Taipei")


def _write_registry(path: Path, cameras: list[dict]) -> None:
    data = {
        "bucket_name": "bucket_test",
        "storage": {},
        "cameras": cameras,
    }
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_tracking_results(path: Path, camera_id: str) -> None:
    df = pl.DataFrame(
        {
            "camera_id": [camera_id],
            "timestamp": [datetime.datetime(2026, 5, 1, 9, 0, tzinfo=_TAIPEI)],
            "track_id": [1],
            "x1": [4.0],
            "y1": [4.0],
            "x2": [6.0],
            "y2": [6.0],
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
                        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
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
                        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
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
