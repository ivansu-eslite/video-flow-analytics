import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from video_flow_analytics.report.pipeline import _build_report_frames


def _write_registry(path: Path, zones_by_camera: dict[str, list[str]]) -> None:
    data = {
        "bucket_name": "bucket_test",
        "storage": {},
        "cameras": [
            {
                "camera_id": cam_id,
                "location": "loc",
                "ip": "127.0.0.1",
                "zones": [
                    {"name": name, "polygon": [[0, 0], [1, 0], [1, 1]]}
                    for name in names
                ],
            }
            for cam_id, names in zones_by_camera.items()
        ],
    }
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_zone_counts(path: Path) -> None:
    df = pl.DataFrame(
        {
            "camera_id": ["cam001"],
            "zone": ["entrance"],
            "time_bucket": [
                datetime.datetime(2026, 5, 1, 11, 0, tzinfo=datetime.timezone.utc)
            ],
            "unique_visitors": [1],
            "entries": [1],
        },
        schema={
            "camera_id": pl.Utf8,
            "zone": pl.Utf8,
            "time_bucket": pl.Datetime("us", "UTC"),
            "unique_visitors": pl.Int64,
            "entries": pl.Int64,
        },
    )
    df.write_parquet(path)


def test_build_report_frames_validates_snapshot_registry_not_live(tmp_path):
    """_build_report_frames 應該驗證產生 zone_counts.parquet 當時的 registry 快照
    （camera_registry_used.yaml），而不是「當下」的 camera_registry.yaml。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    # 即時 registry：已經修正成不重複
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        {"cam001": ["entrance"], "cam002": ["entrance_2"]},
    )

    output_dir = tmp_path / "outputs" / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    _write_zone_counts(output_dir / "zone_counts.parquet")
    # 快照：產生 parquet 當時兩台攝影機都叫 entrance（重複）
    _write_registry(
        output_dir / "camera_registry_used.yaml",
        {"cam001": ["entrance"], "cam002": ["entrance"]},
    )

    with pytest.raises(ValueError, match="全域唯一"):
        _build_report_frames(
            date=datetime.date(2026, 5, 1),
            bucket_dir=str(bucket_dir),
            period_minutes=60,
            metric="entries",
            bucket_minutes=15,
            output_root=tmp_path / "outputs",
        )


def test_build_report_frames_ignores_live_registry_duplicates(tmp_path):
    """即時 camera_registry.yaml 目前有重複也不該擋下報表，只要產生資料當時的
    快照是唯一的就該正常執行。"""
    bucket_dir = tmp_path / "bucket_test"
    bucket_dir.mkdir()
    # 即時 registry 目前有重複（尚未修正），但不該影響驗證結果
    _write_registry(
        bucket_dir / "camera_registry.yaml",
        {"cam001": ["entrance"], "cam002": ["entrance"]},
    )

    output_dir = tmp_path / "outputs" / "bucket_test" / "2026-05-01"
    output_dir.mkdir(parents=True)
    _write_zone_counts(output_dir / "zone_counts.parquet")
    # 快照：產生 parquet 當時是全域唯一的
    _write_registry(
        output_dir / "camera_registry_used.yaml",
        {"cam001": ["entrance"], "cam002": ["entrance_2"]},
    )

    hourly_df, peak_df = _build_report_frames(
        date=datetime.date(2026, 5, 1),
        bucket_dir=str(bucket_dir),
        period_minutes=60,
        metric="entries",
        bucket_minutes=15,
        output_root=tmp_path / "outputs",
    )
    assert hourly_df.height == 1
    assert peak_df.height == 1
