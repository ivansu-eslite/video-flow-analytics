"""計數線幾何、方向定號與名稱唯一性的驗證。

與 `test_zone_validation.py` 同理由：這幾條規則的編輯點在本 lib，改壞了只跑消費
套件（line-counting）的測試才會爆，故在 lib 內直接釘住。
"""

import pytest

from vfa_registry import CameraEntry, Line, parse_and_validate_lines

_POINTS = [(0, 10), (20, 10)]
_INSIDE = (10, 0)


def test_line_rejects_polyline_with_fewer_than_two_vertices():
    """一個頂點構不成折線，無法定義任何線段。"""
    with pytest.raises(ValueError, match="至少需要 2 個頂點"):
        Line(name="l", points=[(0, 10)], inside_point=_INSIDE)


def test_line_rejects_zero_length_segment():
    """連續重複頂點造成零長度段，無法定號、也會讓距離計算除零。"""
    with pytest.raises(ValueError, match="零長度線段"):
        Line(name="l", points=[(0, 10), (0, 10), (20, 10)], inside_point=_INSIDE)


def test_line_rejects_inside_point_on_segment_extension():
    """inside_point 落在某段的無限延伸線上（外積 ≈ 0）→ side 無法定號 → fail-loud。

    這裡 inside_point (50, 10) 與 y=10 這條線共線；缺這道檢查，帶號距離的符號會
    退化成 0，方向（進/出）判定失效卻不報錯。
    """
    with pytest.raises(ValueError, match="延伸線"):
        Line(name="l", points=[(0, 10), (20, 10)], inside_point=(50, 10))


def _entry(camera_id: str, line_names: list[str]) -> CameraEntry:
    return CameraEntry(
        camera_id=camera_id,
        location="loc",
        ip="127.0.0.1",
        lines=[
            {"name": name, "points": _POINTS, "inside_point": _INSIDE}
            for name in line_names
        ],
    )


def test_parsed_lines_rejects_duplicate_names_within_one_camera():
    """同機同名計數線會在該攝影機的統計裡被算兩次，須 fail-loud。"""
    with pytest.raises(ValueError, match="同一攝影機的計數線 name 不可重複"):
        _entry("cam001", ["door", "door"]).parsed_lines()


def test_parse_and_validate_lines_rejects_names_duplicated_across_cameras():
    """跨機同名是最容易漏掉的一種：各機自己看都合法。

    下游報表依 line 名稱分組彙總、不含 camera_id，同名計數線會讓兩台攝影機的
    進出人數被靜默合併成同一列——沒有這道檢查，錯誤只會變成一個看起來正常但
    數字偏大的報表。
    """
    entries = {
        "loc_cam001": _entry("cam001", ["共用名稱"]),
        "loc_cam002": _entry("cam002", ["共用名稱"]),
    }
    with pytest.raises(ValueError, match="跨攝影機重複的計數線名稱"):
        parse_and_validate_lines(entries)


def test_parse_and_validate_lines_returns_parsed_lines_per_camera():
    entries = {
        "loc_cam001": _entry("cam001", ["line_a"]),
        "loc_cam002": _entry("cam002", ["line_b"]),
    }
    result = parse_and_validate_lines(entries)

    assert {k: [ln.name for ln in v] for k, v in result.items()} == {
        "loc_cam001": ["line_a"],
        "loc_cam002": ["line_b"],
    }
    assert result["loc_cam001"][0].points == [(0.0, 10.0), (20.0, 10.0)]
    assert result["loc_cam001"][0].inside_point == (10.0, 0.0)


def test_polyline_with_three_vertices_is_accepted():
    """彎折 polyline（≥ 3 頂點、凸向 inside_point）為合法定義。"""
    line = Line(
        name="v", points=[(0, 10), (10, 20), (20, 10)], inside_point=(10, 0)
    )
    assert len(line.points) == 3
