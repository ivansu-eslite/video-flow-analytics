import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from vfa_registry import Line

from line_counting.services.stats import (
    count_line_crossings,
    signed_distance_to_polyline,
    validate_line_cameras,
)

# 水平計數線 y=10（x=0..20），inside_point 在下方（y=0 側）：跨到下方 = in
_LINE = Line(name="door", points=[(0, 10), (20, 10)], inside_point=(10, 0))


def _make_cam_sub(
    foot: list[tuple[float, float]], track_id: str = "t1"
) -> pl.DataFrame:
    """合成單一 track 的追蹤明細，foot 為逐格腳底點 `(foot_x, foot_y)`。"""
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    n = len(foot)
    return pl.DataFrame(
        {
            "track_id": [track_id] * n,
            "timestamp": [base + datetime.timedelta(seconds=i) for i in range(n)],
            "time_bucket": [base] * n,
            "foot_x": [p[0] for p in foot],
            "foot_y": [p[1] for p in foot],
        }
    )


def _totals(result: pl.DataFrame) -> tuple[int, int]:
    return int(result["in_count"].sum()), int(result["out_count"].sum())


def test_signed_distance_positive_on_inside_side():
    # inside_point 在 y=0 側：y<10 的點帶號距離為正、y>10 為負，量值為垂直距離
    d = signed_distance_to_polyline(
        np.array([10.0, 10.0]),
        np.array([0.0, 20.0]),
        np.asarray(_LINE.points, dtype=float),
        _LINE.inside_point,
    )
    assert d.tolist() == [10.0, -10.0]


def test_straight_crossing_into_inside_counts_one_in():
    # 由外側（y=20）單向穿越到內側（y=0）：in=1、out=0
    result = count_line_crossings(_make_cam_sub([(10, 20), (10, 20), (10, 0)]), _LINE)
    assert _totals(result) == (1, 0)


def test_straight_crossing_to_outside_counts_one_out():
    # 反向：由內側穿越到外側：out=1、in=0
    result = count_line_crossings(_make_cam_sub([(10, 0), (10, 0), (10, 20)]), _LINE)
    assert _totals(result) == (0, 1)


def test_no_crossing_when_track_folds_back_before_line():
    # 未過線就折返（一直在 y=20 外側附近晃）：兩者皆 0，且不產生任何列
    result = count_line_crossings(_make_cam_sub([(10, 20), (10, 15), (10, 20)]), _LINE)
    assert _totals(result) == (0, 0)
    assert result.height == 0


def test_start_already_on_a_side_is_not_counted():
    # track 起始就在某側（前一格為 null），只有之後的翻轉才計——起始側本身不算跨越
    result = count_line_crossings(_make_cam_sub([(10, 0), (10, 20)]), _LINE)
    assert _totals(result) == (0, 1)


def test_polyline_crossing_counts_and_directs_correctly():
    # V 形彎折 barrier，inside_point 在下方；由上方外側穿越彎折到內側：in=1
    line = Line(name="v", points=[(0, 10), (10, 20), (20, 10)], inside_point=(10, 0))
    result = count_line_crossings(_make_cam_sub([(10, 30), (10, 15), (10, 5)]), line)
    assert _totals(result) == (1, 0)


def test_band_filters_jitter_within_dead_zone():
    # band=3：腳底在死區內（|d|<3）來回抖動不產生跨越
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 9), (10, 20), (10, 20)]),
        _LINE,
        crossing_band_px=3,
    )
    assert _totals(result) == (0, 0)
    assert result.height == 0


def test_band_zero_counts_every_zero_crossing():
    # band=0：同一抖動每次零交越都計——進 y=9（內側）再出 y=20：in=1、out=1
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 9), (10, 20), (10, 20)]),
        _LINE,
        crossing_band_px=0,
    )
    assert _totals(result) == (1, 1)


def test_band_still_counts_crossing_larger_than_band():
    # 回歸鎖：band=3 濾的是死區抖動，不是把偵測關掉；幅度 > band 的真跨越仍計
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 0)]), _LINE, crossing_band_px=3
    )
    assert _totals(result) == (1, 0)


def test_band_boundary_distance_stays_in_dead_zone():
    # 邊界鎖：腳底恰在 d == band（y=7 → d=+3）時屬死區、不確認側別，committed 沿用
    # 前一格的外側（y=20 → d=-10）→ 不產生跨越。守護 `> band`（而非 `>= band`）：
    # 改成 >= 會把邊界點誤判為內側而多算一次 in。
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 7)]), _LINE, crossing_band_px=3
    )
    assert _totals(result) == (0, 0)


def test_forward_fill_carries_side_through_multi_frame_dead_zone():
    # hysteresis 鎖：跨越途中連續 2 格落在死區（y=9 → d=+1 < band=3）才落定對側，
    # forward_fill 讓死區沿用前一個已確認側別，最終仍正確判為一次 in。拿掉
    # forward_fill 會讓 committed 在死區變 null、經 shift 汙染 _prev，使跨越被靜默漏計。
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 9), (10, 9), (10, 0)]),
        _LINE,
        crossing_band_px=3,
    )
    assert _totals(result) == (1, 0)


def test_multiple_tracks_and_buckets_aggregate_separately():
    # 兩個 track 各穿越一次（一進一出），time_bucket 相同 → 聚合為 in=1、out=1
    cam = pl.concat(
        [
            _make_cam_sub([(10, 20), (10, 0)], track_id="t1"),
            _make_cam_sub([(10, 0), (10, 20)], track_id="t2"),
        ]
    )
    result = count_line_crossings(cam, _LINE)
    assert _totals(result) == (1, 1)


def test_multiple_lines_do_not_pollute_each_other():
    # 同一份追蹤明細套兩條不同計數線，各自獨立統計；只跨越 door（y=10）、未及 far（y=100）
    door = Line(name="door", points=[(0, 10), (20, 10)], inside_point=(10, 0))
    far = Line(name="far", points=[(0, 100), (20, 100)], inside_point=(10, 0))
    cam = _make_cam_sub([(10, 20), (10, 0)])
    assert _totals(count_line_crossings(cam, door)) == (1, 0)
    assert _totals(count_line_crossings(cam, far)) == (0, 0)


def test_validate_line_cameras_reports_value_error_when_data_cameras_has_none():
    # camera_id 為 nullable Utf8，data_cameras 含 None 時排序不應炸成 TypeError，
    # 蓋掉本該報出的診斷訊息
    with pytest.raises(ValueError, match="cam_missing"):
        validate_line_cameras({"cam_missing"}, {"cam001", None})
