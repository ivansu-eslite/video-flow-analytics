import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from vfa_registry import Zone

from zone_mapping.services.stats import (
    count_zone_visits,
    points_in_polygon,
    signed_distance_to_polygon,
    validate_zone_cameras,
)

_SQUARE = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
# 凹多邊形（L 形）：右上角 (4,4)-(10,10) 那塊被挖掉，(4, 4) 是凹頂點
_L_SHAPE = np.array(
    [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]
)

_ZONE = Zone(name="zone_a", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
_DEEP_INSIDE = (5.0, 5.0)  # 距邊界 5
_DEEP_OUTSIDE = (-5.0, 5.0)  # 距邊界 5（區外）
_JITTER_INSIDE = (0.5, 5.0)  # 距邊界 0.5（區內），band=2 時落在緩衝帶
_JITTER_OUTSIDE = (-0.5, 5.0)  # 距邊界 0.5（區外），band=2 時落在緩衝帶
_BAND = 2.0


def _make_cam_sub(
    points: list[tuple[float, float]],
    track_id: str = "t1",
    bucket_index: list[int] | None = None,
) -> pl.DataFrame:
    """組出 `count_zone_visits` 需要的單攝影機明細；`bucket_index` 可分到不同時段。"""
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    n = len(points)
    buckets = bucket_index or [0] * n
    return pl.DataFrame(
        {
            "track_id": [track_id] * n,
            "timestamp": [base + datetime.timedelta(seconds=i) for i in range(n)],
            "time_bucket": [base + datetime.timedelta(hours=b) for b in buckets],
            "foot_x": [p[0] for p in points],
            "foot_y": [p[1] for p in points],
        }
    )


def _entries_total(result: pl.DataFrame) -> int:
    return int(result["entries"].sum())


def test_signed_distance_positive_inside_negative_outside():
    """符號代表內外、絕對值是到邊界的最短距離。"""
    signed_d, inside = signed_distance_to_polygon(
        np.array([5.0, -3.0]), np.array([5.0, 5.0]), _SQUARE
    )

    assert signed_d == pytest.approx([5.0, -3.0])
    assert inside.tolist() == [True, False]


def test_signed_distance_is_zero_on_boundary():
    """恰在邊界上的點距離為 0；內外符號由 points_in_polygon 給（邊界結果依實作而定），
    因此 band=0 時這種點會落在緩衝帶內而沿用前一格狀態。"""
    signed_d, inside = signed_distance_to_polygon(
        np.array([0.0]), np.array([5.0]), _SQUARE
    )

    assert signed_d == pytest.approx([0.0])
    assert inside.tolist() == points_in_polygon(
        np.array([0.0]), np.array([5.0]), _SQUARE
    ).tolist()


def test_signed_distance_accounts_for_closing_edge():
    """多邊形要自動閉合：靠近「最後一點連回第一點」那條邊的點，距離要算到該邊。

    `_SQUARE` 的收尾邊是左邊 x=0。點 (1, 5) 離它 1 px；漏算收尾邊會退而取到其他邊的
    5 px，緩衝帶在該邊整段失效（該邊附近的抖動又會被計成進入）。
    """
    signed_d, _ = signed_distance_to_polygon(
        np.array([1.0]), np.array([5.0]), _SQUARE
    )

    assert signed_d == pytest.approx([1.0])


def test_signed_distance_handles_concave_vertex():
    """凹角：內外由 points_in_polygon 全域決定，凹頂點附近不會翻錯符號。

    (5, 5) 落在被挖掉的缺口裡（區外，離凹角的兩條邊各 1 px）；(3, 6) 在區內，最近的
    邊界是凹角的直立邊 x=4，距離 1 px。
    """
    signed_d, inside = signed_distance_to_polygon(
        np.array([5.0, 3.0]), np.array([5.0, 6.0]), _L_SHAPE
    )

    assert signed_d == pytest.approx([-1.0, 1.0])
    assert inside.tolist() == [False, True]


def test_signed_distance_inside_flag_matches_points_in_polygon():
    """回傳的 inside 必須就是 points_in_polygon 的結果——`unique_visitors` 靠它維持
    與改動前逐值一致，改用 `signed_d > 0` 反推會在邊界點上分岔。"""
    xs = np.array([5.0, -3.0, 0.0, 10.0, 9.9])
    ys = np.array([5.0, 5.0, 5.0, 5.0, 0.1])

    _, inside = signed_distance_to_polygon(xs, ys, _SQUARE)

    assert inside.tolist() == points_in_polygon(xs, ys, _SQUARE).tolist()


def test_entries_counted_once_when_track_starts_inside():
    """首格即在區內算一次進入（人可能從畫面外直接走進區內，沒有「先在區外被看到」
    那一格）。`_prev` 為 null 的守衛若寫成 `_prev != 1`，這次進入會靜默消失。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * 3)

    result = count_zone_visits(cam_sub, _ZONE, _BAND)

    assert _entries_total(result) == 1


def test_boundary_jitter_inside_band_counts_single_entry():
    """腳底點在邊界附近反覆進出（皆落在緩衝帶內）只算一次進入。

    序列：確認在外 → 帶內來回四次 → 確認在內 → 帶內再來回 → 確認在內。緩衝帶讓
    已確認狀態黏著，只有第一次真的越過帶才翻轉。
    """
    cam_sub = _make_cam_sub(
        [
            _DEEP_OUTSIDE,
            _JITTER_INSIDE,
            _JITTER_OUTSIDE,
            _JITTER_INSIDE,
            _JITTER_OUTSIDE,
            _DEEP_INSIDE,
            _JITTER_OUTSIDE,
            _JITTER_INSIDE,
            _DEEP_INSIDE,
        ]
    )

    result = count_zone_visits(cam_sub, _ZONE, _BAND)

    assert _entries_total(result) == 1


def test_same_jitter_without_band_counts_every_crossing():
    """對照組：同一段抖動在 band=0 下每次跨越邊界都被計入（本次要消除的行為）。"""
    cam_sub = _make_cam_sub(
        [
            _DEEP_OUTSIDE,
            _JITTER_INSIDE,
            _JITTER_OUTSIDE,
            _JITTER_INSIDE,
            _JITTER_OUTSIDE,
            _DEEP_INSIDE,
        ]
    )

    result = count_zone_visits(cam_sub, _ZONE, 0)

    assert _entries_total(result) == 3


def test_leaving_beyond_band_and_returning_counts_two_entries():
    """真的離開（越過緩衝帶外側）再回來算兩次進入——緩衝帶不能把重複造訪也吃掉。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE, _DEEP_OUTSIDE, _DEEP_INSIDE])

    result = count_zone_visits(cam_sub, _ZONE, _BAND)

    assert _entries_total(result) == 2


def test_zero_band_matches_plain_inside_outside_transitions():
    """band=0 退化成純內外判定：兩段「區域外 → 區域內」各算一次進入
    （與時間去抖時代 `entry_debounce_frames=1` 的行為一致）。"""
    cam_sub = _make_cam_sub(
        [_DEEP_OUTSIDE, _DEEP_INSIDE, _DEEP_INSIDE, _DEEP_OUTSIDE, _DEEP_INSIDE]
    )

    result = count_zone_visits(cam_sub, _ZONE, 0)

    assert _entries_total(result) == 2


def test_unique_visitors_ignores_committed_state():
    """`unique_visitors` 不吃緩衝帶的黏著狀態：走出區域後停在邊界外緩衝帶內的人，
    在其後的時段不算區內訪客（否則佔用型指標會被事件型判定汙染）。

    第二個時段那格的已確認狀態仍是「在區內」（腳底點還在緩衝帶內），但 `in_zone`
    為 False，該時段完全沒有統計列；吃了黏著狀態的話會多出一列 unique_visitors=1。
    """
    cam_sub = _make_cam_sub([_DEEP_INSIDE, _JITTER_OUTSIDE], bucket_index=[0, 1])

    result = count_zone_visits(cam_sub, _ZONE, _BAND).sort("time_bucket")

    assert result.select("unique_visitors", "entries").rows() == [(1, 1)]


def test_validate_zone_cameras_reports_value_error_when_data_cameras_has_none():
    # camera_id 為 nullable Utf8，data_cameras 含 None 時排序不應炸成
    # TypeError，蓋掉本該報出的診斷訊息
    with pytest.raises(ValueError, match="cam_missing"):
        validate_zone_cameras({"cam_missing"}, {"cam001", None})

