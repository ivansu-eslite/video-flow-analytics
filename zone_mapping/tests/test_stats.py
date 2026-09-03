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
_JITTER_INSIDE = (0.5, 5.0)  # 距邊界 0.5（區內），band=2 時落在線段區域
_JITTER_OUTSIDE = (-0.5, 5.0)  # 距邊界 0.5（區外），band=2 時落在線段區域
_BAND = 2.0

# 停留判定的兩個門檻。`count_zone_visits` 刻意不給預設值，測試自己備一組。
_DWELL = 20.0
_GAP = 3.0


def _make_cam_sub(
    points: list[tuple[float, float]],
    track_id: str = "t1",
    bucket_index: list[int] | None = None,
    offsets: list[float] | None = None,
) -> pl.DataFrame:
    """組出 `count_zone_visits` 需要的單攝影機明細；`bucket_index` 可分到不同時段。

    `offsets` 是每格相對起點的秒數（可為小數、不必等距），停留測試多半需要非等距的
    時間軸（一個 5 秒的洞、一個 2 秒的洞）；不給時維持每格相隔 1 秒。

    `time_bucket` 由 `bucket_index` 直接指定、與 `timestamp` 無關（刻意不從
    `timestamp` 推算），這樣才能用少量列數測「達標那一格落在哪個時段」。
    """
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    n = len(points)
    buckets = bucket_index or [0] * n
    seconds = offsets if offsets is not None else list(range(n))
    assert len(seconds) == n, "offsets 要與 points 一樣長"
    return pl.DataFrame(
        {
            "track_id": [track_id] * n,
            "timestamp": [base + datetime.timedelta(seconds=s) for s in seconds],
            "time_bucket": [base + datetime.timedelta(hours=b) for b in buckets],
            "foot_x": [p[0] for p in points],
            "foot_y": [p[1] for p in points],
        }
    )


def _visits(
    cam_sub: pl.DataFrame,
    boundary_band_px: float = _BAND,
    *,
    dwell: float = _DWELL,
    gap: float = _GAP,
) -> pl.DataFrame:
    """對 `_ZONE` 呼叫 `count_zone_visits`，把兩個無預設值的門檻收斂在一處。"""
    return count_zone_visits(
        cam_sub,
        _ZONE,
        boundary_band_px,
        dwell_threshold_seconds=dwell,
        dwell_gap_seconds=gap,
    )


def _entries_total(result: pl.DataFrame) -> int:
    return int(result["entries"].sum())


def _dwell_total(result: pl.DataFrame) -> int:
    return int(result["dwell_events"].sum())


def test_signed_distance_positive_inside_negative_outside():
    """符號代表內外、絕對值是到邊界的最短距離。"""
    signed_d, inside = signed_distance_to_polygon(
        np.array([5.0, -3.0]), np.array([5.0, 5.0]), _SQUARE
    )

    assert signed_d == pytest.approx([5.0, -3.0])
    assert inside.tolist() == [True, False]


def test_signed_distance_is_zero_on_boundary():
    """恰在邊界上的點距離為 0；內外符號由 points_in_polygon 給（邊界結果依實作而定），
    因此 band=0 時這種點會落在線段區域內而沿用前一格狀態。"""
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
    5 px，線段區域在該邊整段失效（該邊附近的抖動又會被計成進入）。
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

    result = _visits(cam_sub)

    assert _entries_total(result) == 1


def test_boundary_jitter_inside_band_counts_single_entry():
    """腳底點在邊界附近反覆進出（皆落在線段區域內）只算一次進入。

    序列：確認在外 → 帶內來回四次 → 確認在內 → 帶內再來回 → 確認在內。線段區域讓
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

    result = _visits(cam_sub)

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

    result = _visits(cam_sub, 0)

    assert _entries_total(result) == 3


def test_leaving_beyond_band_and_returning_counts_two_entries():
    """真的離開（越過線段區域外側）再回來算兩次進入——線段區域不能把重複造訪也吃掉。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE, _DEEP_OUTSIDE, _DEEP_INSIDE])

    result = _visits(cam_sub)

    assert _entries_total(result) == 2


def test_zero_band_matches_plain_inside_outside_transitions():
    """band=0 退化成純內外判定：兩段「區域外 → 區域內」各算一次進入
    （與時間去抖時代 `entry_debounce_frames=1` 的行為一致）。"""
    cam_sub = _make_cam_sub(
        [_DEEP_OUTSIDE, _DEEP_INSIDE, _DEEP_INSIDE, _DEEP_OUTSIDE, _DEEP_INSIDE]
    )

    result = _visits(cam_sub, 0)

    assert _entries_total(result) == 2


def test_unique_visitors_ignores_committed_state():
    """`unique_visitors` 不吃線段區域的黏著狀態：走出區域後停在邊界外線段區域內的人，
    在其後的時段不算區內訪客（否則佔用型指標會被事件型判定汙染）。

    第二個時段那格的已確認狀態仍是「在區內」（腳底點還在線段區域內），但 `in_zone`
    為 False，該時段完全沒有統計列；吃了黏著狀態的話會多出一列 unique_visitors=1。
    """
    cam_sub = _make_cam_sub([_DEEP_INSIDE, _JITTER_OUTSIDE], bucket_index=[0, 1])

    result = _visits(cam_sub).sort("time_bucket")

    assert result.select("unique_visitors", "entries", "dwell_events").rows() == [
        (1, 1, 0)
    ]


def test_committed_state_does_not_leak_across_tracks():
    """已確認狀態依 `track_id` 分組，不會從前一個 track 洩漏到下一個。

    三個 track：`t1`／`t2` 各自全程在區內（各 1 次進入），`t3` 前兩格都在線段區域內、
    最後一格才確認在區外（0 次進入），正確答案是 2。輸入按時間交錯，模擬多人同時在
    畫面裡的實際明細。少了 `over("track_id")` 兩處都會靜默算錯：

    - `forward_fill` 少分組 → `t3` 首格被 `t2` 的「在區內」填滿，多算一次進入（3）。
    - `shift` 少分組 → `t2` 首格的前一個狀態變成 `t1` 的「在區內」，少算一次（1）。
    """
    cam_sub = pl.concat(
        [
            _make_cam_sub([_DEEP_INSIDE] * 2, track_id="t1"),
            _make_cam_sub([_DEEP_INSIDE], track_id="t2"),
            _make_cam_sub(
                [_JITTER_INSIDE, _JITTER_OUTSIDE, _DEEP_OUTSIDE], track_id="t3"
            ),
        ]
    ).sort("timestamp")

    result = _visits(cam_sub)

    assert _entries_total(result) == 2


def test_dwell_counted_once_for_a_single_qualifying_stay():
    """一段連續 25 秒的停留計 1 人次，不是「每格都達標就計一次」。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * 26)  # 每格 1 秒，elapsed = 25

    result = _visits(cam_sub)

    assert _dwell_total(result) == 1


def test_dwell_counts_stay_exactly_at_threshold():
    """elapsed 恰等於門檻要計入（判定是 `>=`，不是 `>`）。

    寫成 `>` 會讓「剛好待滿 20 秒」這種邊界案例整批消失，而且從輸出檔看不出來。
    """
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * 21)  # elapsed = 20 = 門檻

    result = _visits(cam_sub)

    assert _dwell_total(result) == 1


def test_dwell_not_counted_one_frame_below_threshold():
    """差一格不到門檻就不計——對照上一支，鎖住門檻真的有在比。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * 20)  # elapsed = 19 < 20

    result = _visits(cam_sub)

    assert _dwell_total(result) == 0


def test_dwell_gap_within_tolerance_stays_one_segment():
    """中斷不超過容忍窗視為同一段停留：漏偵測（區內完全沒有列）不該把停留切碎。

    區內出現的時間點是 0–10 與 13–25：中間空了 3 秒（= 容忍窗，未超過），整段
    elapsed = 25 秒達標。容忍窗若失效，兩段分別是 10 與 12 秒，都不到門檻。
    """
    offsets = [float(i) for i in range(11)] + [float(i) for i in range(13, 26)]
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * len(offsets), offsets=offsets)

    result = _visits(cam_sub)

    assert _dwell_total(result) == 1


def test_dwell_gap_beyond_tolerance_splits_into_two_short_segments():
    """中斷超過容忍窗要切成兩段，兩段各自不達標就是 0（不可把兩段接起來算成一次）。

    區內出現的時間點是 0–14 與 20–34：中間空了 5 秒 > 容忍窗 3 秒。兩段各 14 秒，
    接起來算會變成 34 秒、錯計一次停留。
    """
    offsets = [float(i) for i in range(15)] + [float(i) for i in range(20, 35)]
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * len(offsets), offsets=offsets)

    result = _visits(cam_sub)

    assert _dwell_total(result) == 0


def test_dwell_counts_two_events_for_the_same_track():
    """同一個人兩段都達標算 2：`dwell_events` 是人次，不是人數。

    也就是說它**不是 `unique_visitors` 的子集**（那一欄對這個人只算 1），下游不可
    寫 `dwell_events <= unique_visitors` 這類 sanity check。
    """
    offsets = [float(i) for i in range(22)] + [float(i) for i in range(27, 49)]
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * len(offsets), offsets=offsets)

    result = _visits(cam_sub)

    assert _dwell_total(result) == 2
    assert int(result["unique_visitors"].sum()) == 1


def test_dwell_attributed_to_the_bucket_of_the_crossing_frame():
    """一段停留歸戶到「首次跨過門檻」那一格的時段，不是起點也不是終點那一格。

    26 格（elapsed 0–25 秒），達標的是 index 20–25 那六格。時段切換點刻意放在
    **達標區間之內**（第 24 格起換到下一個時段）：首次跨過門檻的 index 20 在第一個
    時段，末格 index 25 在第二個。切換點若放在達標區間之外（例如第 21 格），
    `.first()` 與 `.last()` 會給同一個答案，這條語義就沒被釘住。
    """
    n = 26
    cam_sub = _make_cam_sub(
        [_DEEP_INSIDE] * n, bucket_index=[0] * 23 + [1] * (n - 23)
    )

    result = _visits(cam_sub).sort("time_bucket")

    assert result["dwell_events"].to_list() == [1, 0]


def test_dwell_counts_two_tracks_separately():
    """兩個不同的人各自達標算 2：段的分組鍵必須帶 `track_id`。

    段號是各 track 各自從 1 起算的（`cum_sum().over("track_id")`），所以
    `group_by("track_id", "_dwell_seg")` 少了 `track_id` 會把兩人的第 1 段併成同一
    組，兩次停留只計一次——**漏計**方向的錯，而輸出檔完全正常。時間錯開讓兩人不在
    同一格，避免測試依賴列順序。
    """
    cam_sub = pl.concat(
        [
            _make_cam_sub([_DEEP_INSIDE] * 26, track_id="t1"),
            _make_cam_sub(
                [_DEEP_INSIDE] * 26,
                track_id="t2",
                offsets=[30.0 + i for i in range(26)],
            ),
        ]
    ).sort("timestamp")

    result = _visits(cam_sub)

    assert _dwell_total(result) == 2


def test_dwell_uses_raw_in_zone_not_committed_state():
    """停留判定用生的 `in_zone`，不可沿用 `entries` 的黏著狀態（見 ADR-016）。

    序列：兩格確實在區內，之後 30 秒都停在區外的線段區域內。`_committed` 全程都是
    「確認在區內」（帶內的列沿用前一格），照它判會算出一次 30 秒的停留；正確答案是
    0——這個人只有頭兩格真的在區域裡。
    """
    n_outside = 31
    cam_sub = _make_cam_sub(
        [_DEEP_INSIDE] * 2 + [_JITTER_OUTSIDE] * n_outside,
        offsets=[float(i) for i in range(2 + n_outside)],
    )

    result = _visits(cam_sub)

    assert _dwell_total(result) == 0


def test_dwell_bridges_boundary_jitter_into_one_segment():
    """腳底點在邊界抖進抖出時，容忍窗要把碎掉的停留接回一段。

    每隔一秒交替內外，區內的列因此每 2 秒一次（< 容忍窗 3 秒）。整段 elapsed = 24
    秒達標，計 1；沒有容忍窗的話每一格自成一段，全部是 0。

    這也是採用容忍窗的代價：在邊界上徘徊滿 20 秒的人同樣會被算成一次完整停留
    （見 ADR-016）。
    """
    points = [_DEEP_INSIDE if i % 2 == 0 else _DEEP_OUTSIDE for i in range(25)]
    cam_sub = _make_cam_sub(points)

    result = _visits(cam_sub)

    assert _dwell_total(result) == 1


def test_dwell_elapsed_start_does_not_leak_across_tracks():
    """段起算點依 `track_id` 分組：晚到的人不可繼承早到者的起算時間。

    `t1` 從 0 秒待到 25 秒（達標），`t2` 只在 26–27 秒出現（2 秒）。`.min().over`
    少了 `track_id` 會讓 `t2` 拿 `t1` 的段起點當起算點，算出 26 秒的停留而多計一次
    ——這是**超計**方向的錯，比漏計難發現。正確答案是 1。
    """
    cam_sub = pl.concat(
        [
            _make_cam_sub([_DEEP_INSIDE] * 26, track_id="t1"),
            _make_cam_sub([_DEEP_INSIDE] * 2, track_id="t2", offsets=[26.0, 27.0]),
        ]
    ).sort("timestamp")

    result = _visits(cam_sub)

    assert _dwell_total(result) == 1


def test_dwell_single_row_track_is_zero_not_error():
    """只出現一格的軌跡：elapsed = 0，不計也不炸（`diff` 的首列是 null）。"""
    cam_sub = _make_cam_sub([_DEEP_INSIDE])

    result = _visits(cam_sub)

    assert _dwell_total(result) == 0


def test_dwell_is_zero_not_null_for_buckets_without_a_stay():
    """有 unique_visitors 但沒有停留的時段，`dwell_events` 要是 0 而不是 null。

    漏掉 `fill_null(0)` 會在 parquet 留下 null，下游 `flow_report` 的 `sum()` 拿到
    的就不是數字。
    """
    cam_sub = _make_cam_sub([_DEEP_INSIDE] * 3)

    result = _visits(cam_sub)

    assert result["dwell_events"].to_list() == [0]
    assert result["dwell_events"].null_count() == 0
    assert result["dwell_events"].dtype == pl.Int64


def test_validate_zone_cameras_reports_value_error_when_data_cameras_has_none():
    # camera_id 為 nullable Utf8，data_cameras 含 None 時排序不應炸成
    # TypeError，蓋掉本該報出的診斷訊息
    with pytest.raises(ValueError, match="cam_missing"):
        validate_zone_cameras({"cam_missing"}, {"cam001", None})
