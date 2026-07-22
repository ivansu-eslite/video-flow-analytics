import pytest

from flow_report.models.registry import CameraRegistry, _find_duplicates


def test_find_duplicates_returns_items_appearing_more_than_once():
    assert _find_duplicates(["a", "b", "a", "c", "c", "c"]) == {"a", "c"}


def test_find_duplicates_returns_empty_set_when_all_unique():
    assert _find_duplicates(["a", "b", "c"]) == set()


def _make_registry() -> CameraRegistry:
    return CameraRegistry(
        bucket_name="bucket_test",
        storage={},
        cameras=[
            {"camera_id": "cam001", "location": "loc", "ip": "127.0.0.1"},
            {"camera_id": "cam002", "location": "loc", "ip": "127.0.0.1"},
        ],
    )


def test_resolve_cameras_rejects_duplicate_camera_ids():
    """重複的 camera_id 會讓同一台攝影機在回傳清單中出現多次，須 fail-loud。"""
    registry = _make_registry()
    with pytest.raises(ValueError, match="重複"):
        registry.resolve_cameras(["cam001", "cam001"])


def test_resolve_cameras_rejects_unknown_camera_id():
    registry = _make_registry()
    with pytest.raises(ValueError, match="找不到"):
        registry.resolve_cameras(["cam999"])


def test_resolve_cameras_returns_all_when_none_or_empty():
    registry = _make_registry()
    assert registry.resolve_cameras(None) == list(registry.cameras)
    assert registry.resolve_cameras([]) == list(registry.cameras)


def test_resolve_cameras_preserves_requested_order():
    registry = _make_registry()
    result = registry.resolve_cameras(["cam002", "cam001"])
    assert [cam.camera_id for cam in result] == ["cam002", "cam001"]
