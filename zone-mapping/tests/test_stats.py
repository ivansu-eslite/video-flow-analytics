import datetime
from zoneinfo import ZoneInfo

import polars as pl

from zone_mapping.registry import Zone
from zone_mapping.stats import count_zone_visits

_ZONE = Zone(name="zone_a", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
_INSIDE = (5.0, 5.0)
_OUTSIDE = (-5.0, -5.0)


def _make_cam_sub(in_zone_pattern: list[bool], track_id: str = "t1") -> pl.DataFrame:
    base = datetime.datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    n = len(in_zone_pattern)
    points = [_INSIDE if inside else _OUTSIDE for inside in in_zone_pattern]
    return pl.DataFrame(
        {
            "track_id": [track_id] * n,
            "timestamp": [base + datetime.timedelta(seconds=i) for i in range(n)],
            "time_bucket": [base] * n,
            "foot_x": [p[0] for p in points],
            "foot_y": [p[1] for p in points],
        }
    )


def _entries_total(result: pl.DataFrame) -> int:
    return int(result["entries"].sum())


def test_count_zone_visits_counts_entry_present_from_first_frame_with_debounce():
    # track 從第一格起連續在區域內；entry_debounce_frames=2 仍應算一次進入
    # （修正前因 null 經 shift 洩漏而永久遺失，entries 會是 0）
    cam_sub = _make_cam_sub([True] * 5)
    result = count_zone_visits(cam_sub, _ZONE, entry_debounce_frames=2)
    assert _entries_total(result) == 1


def test_count_zone_visits_debounce_still_filters_single_frame_jitter():
    # 單格抖動（一格進、隨即出）在 entry_debounce_frames=2 下仍應被濾掉，
    # 證明修的是「遺失」而非把 debounce 關掉
    cam_sub = _make_cam_sub([False, True, False, False, False])
    result = count_zone_visits(cam_sub, _ZONE, entry_debounce_frames=2)
    assert _entries_total(result) == 0


def test_count_zone_visits_debounce_one_matches_pre_fix_behavior():
    # entry_debounce_frames=1（config.toml 現值）：行為與修改前完全一致，
    # 兩段各自的「區域外 -> 區域內」轉換各算一次進入
    cam_sub = _make_cam_sub([False, True, True, False, True])
    result = count_zone_visits(cam_sub, _ZONE, entry_debounce_frames=1)
    assert _entries_total(result) == 2
