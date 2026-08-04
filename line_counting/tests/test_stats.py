import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from vfa_registry import Line

from line_counting.services.stats import (
    count_line_crossings,
    segment_crosses_polyline,
    signed_distance_to_polyline,
    validate_line_cameras,
)

# 水平計數線 y=10（x=0..20），inside_point 在下方（y=0 側）：跨到下方 = in
_LINE = Line(
    name="door", points=[(0, 10), (20, 10)], inside_point=(10, 0), line_group="g"
)


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


def test_side_in_the_wedge_beyond_a_bend_is_decided_by_the_first_segment():
    # 轉角外側的楔形區是「兩段等距」的區域：ㄥ 形線的轉角在原點、兩臂分別沿 -x 與
    # -y，第一象限任一點對兩段都夾到同一個轉角頂點，距離相同。此時由哪一段定號是
    # 實質選擇——第 0 段的無限直線是 y=0、第 1 段的是 x=0，inside_point 放在
    # (50, -50) 會讓兩段對同一個點給出相反的側別。
    #
    # 這區域不是巧合而是有面積的常態區域（且無界），門口計數線只要有彎折就會有。
    # 若定號的段在此翻來翻去，同一條軌跡在轉角外側的側別會忽正忽負，Schmitt-trigger
    # 會把它算成反覆進出。實作固定取索引較小的段，故結果恆為第 0 段的 -y。
    #
    # 這條測試同時釘住 signed_distance_to_polyline 內距離的算式寫法：改成
    # `(p - a) - t*ab` 會讓第 0 段的 d2 多一次捨入、在 1 ulp 上輸給第 1 段，
    # 下面約 4% 的點會改由第 1 段定號而翻成 +x。
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.5, 800.0, 2000)
    ys = rng.uniform(0.5, 800.0, 2000)
    d = signed_distance_to_polyline(
        xs, ys, np.array([[-100.0, 0.0], [0.0, 0.0], [0.0, -100.0]]), (50.0, -50.0)
    )
    # 側別全部為負（第 0 段），不會有點改由第 1 段定號而翻成正
    assert np.all(d < 0)
    # 量值是到第 0 段所在直線 y=0 的距離，不是到第 1 段的 x=0；不比逐位元相等，
    # perp 的 `cross / seg_len` 本來就可能與 y 差 1 ulp
    assert np.allclose(-d, ys, rtol=1e-12, atol=0.0)


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
    line = Line(
        name="v",
        points=[(0, 10), (10, 20), (20, 10)],
        inside_point=(10, 0),
        line_group="g",
    )
    result = count_line_crossings(_make_cam_sub([(10, 30), (10, 15), (10, 5)]), line)
    assert _totals(result) == (1, 0)


def test_band_filters_jitter_within_line_region():
    # band=3：腳底在線段區域內（|d|<3）來回抖動不產生跨越
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
    # 回歸鎖：band=3 濾的是線段區域內的抖動，不是把偵測關掉；幅度 > band 的真跨越仍計
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 0)]), _LINE, crossing_band_px=3
    )
    assert _totals(result) == (1, 0)


def test_band_boundary_distance_stays_in_line_region():
    # 邊界鎖：腳底恰在 d == band（y=7 → d=+3）時屬線段區域內、不確認側別，committed 沿用
    # 前一格的外側（y=20 → d=-10）→ 不產生跨越。守護 `> band`（而非 `>= band`）：
    # 改成 >= 會把邊界點誤判為內側而多算一次 in。
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 7)]), _LINE, crossing_band_px=3
    )
    assert _totals(result) == (0, 0)


def test_forward_fill_carries_side_through_multi_frame_line_region():
    # hysteresis 鎖：跨越途中連續 2 格落在線段區域內（y=9 → d=+1 < band=3）才落定對側，
    # forward_fill 讓線段區域內沿用前一個已確認側別，最終仍正確判為一次 in。拿掉
    # forward_fill 會讓 committed 在線段區域內變 null、經 shift 汙染 _prev，使跨越被靜默漏計。
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
    door = Line(
        name="door", points=[(0, 10), (20, 10)], inside_point=(10, 0), line_group="g"
    )
    far = Line(
        name="far", points=[(0, 100), (20, 100)], inside_point=(10, 0), line_group="g"
    )
    cam = _make_cam_sub([(10, 20), (10, 0)])
    assert _totals(count_line_crossings(cam, door)) == (1, 0)
    assert _totals(count_line_crossings(cam, far)) == (0, 0)


# --- 有限線段閘門（ADR-003）：線的長度是實質判準 ---


@pytest.mark.parametrize("band", [0, 3])
@pytest.mark.parametrize("x", [25, 5000])
def test_walking_past_beyond_the_endpoints_is_not_counted(x, band):
    # 線只有 x=0..20：在端點之外（擦邊的 x=25 與極遠的 x=5000）由 y=20 走到 y=0，
    # 側別確實會翻轉（側別是最近段的無限直線定的），但沒穿過線段本體 → 不計數
    result = count_line_crossings(
        _make_cam_sub([(x, 20), (x, 0)]), _LINE, crossing_band_px=band
    )
    assert _totals(result) == (0, 0)


def test_going_around_the_endpoint_is_not_counted_before_or_after_a_real_crossing():
    # 繞端點造成的翻轉不計，不論它發生在真正穿門之前或之後，兩種順序都只計那一次 in。
    # 後者是閘門「事件級」的鎖：閘門若寫成「整條 track 有穿過就放行」（`_cum > 0`），
    # 進門後再繞端點外出的那次翻轉會被放行而多算成 out=1。
    around_then_through = count_line_crossings(
        _make_cam_sub([(10, 0), (25, 0), (25, 20), (10, 20), (10, 0)]), _LINE
    )
    through_then_around = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 0), (25, 0), (25, 20)]), _LINE
    )
    assert _totals(around_then_through) == (1, 0)
    assert _totals(through_then_around) == (1, 0)


def test_collinear_displacement_outside_segment_is_not_an_intersection():
    # 判別式鎖：退化檢查必須是「d == 0 且落在 bounding box 內」，不可把 proper 相交
    # 直接放寬成 `<=`。沿著線的延伸方向在端點外走（共線但不重疊）、以及靜止站在
    # 端點外，放寬成 `<=` 都會誤判為相交；只有真的踩在線段上才算。
    points = np.asarray(_LINE.points, dtype=float)
    hit = segment_crosses_polyline(
        np.array([25.0, 25.0, 10.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([30.0, 25.0, 10.0]),
        np.array([10.0, 10.0, 10.0]),
        points,
    )
    assert hit.tolist() == [False, False, True]


@pytest.mark.parametrize("band", [0, 3])
def test_diagonal_crossing_counts_even_when_previous_frame_is_beyond_the_endpoint(band):
    # 防過度修正：斜穿門，穿越前一格 (25,20) 的垂足已落在線段之外（夾到端點），
    # 但位移線段在 x=17.5 真的穿過線段本體 → 仍計 1 in。
    # 「逐格垂足有效性閘門」會在這裡漏算，事件級閘門不會。
    result = count_line_crossings(
        _make_cam_sub([(25, 20), (10, 0)]), _LINE, crossing_band_px=band
    )
    assert _totals(result) == (1, 0)


def test_dwelling_in_line_region_before_entering_still_counts():
    # 防過度修正：外側接近 → 門口線段區域內駐留數格 → 進場（band=3）。相交發生在線段區域內的
    # 那一格，確認側別翻轉則晚好幾格才發生；閘門比對的是「上次確認側別以來」的累計
    # 相交數，故仍計 1 in。門口駐留正是 ADR-001 引入 Schmitt-trigger 要解的情境。
    result = count_line_crossings(
        _make_cam_sub([(10, 20), (10, 11), (10, 9), (10, 9), (10, 0)]),
        _LINE,
        crossing_band_px=3,
    )
    assert _totals(result) == (1, 0)


def test_entering_diagonally_from_outside_a_corner_of_a_u_shaped_line_counts():
    # 防過度修正：ㄇ 形 barrier（三段包住 inside_point），人從左上轉角外的楔形區域
    # 斜穿左段進場 → 計 1 in
    line = Line(
        name="u",
        points=[(0, 0), (0, 20), (30, 20), (30, 0)],
        inside_point=(15, 10),
        line_group="g",
    )
    result = count_line_crossings(_make_cam_sub([(-10, 30), (10, 5)]), line)
    assert _totals(result) == (1, 0)


def test_stepping_exactly_on_the_line_counts():
    # 防過度修正：位移端點恰落在線上（踩線走過）。d == 0 屬線段區域內、不確認側別，
    # 但兩段位移都被判為相交，落定內側時閘門放行 → 計 1 in
    result = count_line_crossings(_make_cam_sub([(10, 20), (10, 10), (10, 0)]), _LINE)
    assert _totals(result) == (1, 0)


def test_validate_line_cameras_reports_value_error_when_data_cameras_has_none():
    # camera_id 為 nullable Utf8，data_cameras 含 None 時排序不應炸成 TypeError，
    # 蓋掉本該報出的診斷訊息
    with pytest.raises(ValueError, match="cam_missing"):
        validate_line_cameras({"cam_missing"}, {"cam001", None})
