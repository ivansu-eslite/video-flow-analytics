"""zone 幾何與名稱唯一性的驗證。

這幾條規則的編輯點在本 lib，但先前本 lib 沒有直接測試、只有消費套件的測試會間接
碰到；改壞了單獨跑本 lib 的測試仍會全綠，要跑到消費套件（flow-report／zone-mapping）
才會爆。故在 lib 內直接釘住。
"""

import pytest

from vfa_registry import CameraEntry, Zone, parse_and_validate_zones

_TRIANGLE = [[0, 0], [10, 0], [10, 10]]


def test_zone_rejects_polygon_with_fewer_than_three_vertices():
    """兩點構不成區域，ray-casting 會退化成一條線段、永遠判不出「在區域內」。"""
    with pytest.raises(ValueError, match="至少需要 3 個頂點"):
        Zone(name="z", polygon=[(0, 0), (10, 10)])


def _entry(camera_id: str, zone_names: list[str]) -> CameraEntry:
    return CameraEntry(
        camera_id=camera_id,
        location="loc",
        ip="127.0.0.1",
        zones=[{"name": name, "polygon": _TRIANGLE} for name in zone_names],
    )


def test_parsed_zones_rejects_duplicate_names_within_one_camera():
    """同機同名 zone 會在該攝影機的統計裡被算兩次，須 fail-loud。"""
    with pytest.raises(ValueError, match="同一攝影機的 zone name 不可重複"):
        _entry("cam001", ["z", "z"]).parsed_zones()


def test_parse_and_validate_zones_rejects_names_duplicated_across_cameras():
    """跨機同名是最容易漏掉的一種：各機自己看都合法。

    下游報表依 zone 名稱分組彙總、不含 camera_id，同名區域會讓兩台攝影機的
    人流被靜默合併成同一列——沒有這道檢查，錯誤不會以例外呈現，只會變成一個
    看起來很正常但數字偏大的報表。
    """
    entries = {
        "loc_cam001": _entry("cam001", ["共用名稱"]),
        "loc_cam002": _entry("cam002", ["共用名稱"]),
    }
    with pytest.raises(ValueError, match="跨攝影機重複的 zone 名稱"):
        parse_and_validate_zones(entries)


def test_parse_and_validate_zones_returns_parsed_zones_per_camera():
    entries = {
        "loc_cam001": _entry("cam001", ["z_a"]),
        "loc_cam002": _entry("cam002", ["z_b"]),
    }
    result = parse_and_validate_zones(entries)

    assert {k: [z.name for z in v] for k, v in result.items()} == {
        "loc_cam001": ["z_a"],
        "loc_cam002": ["z_b"],
    }
    assert result["loc_cam001"][0].polygon == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
