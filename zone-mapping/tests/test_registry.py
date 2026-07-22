import pytest

from zone_mapping.models.registry import (
    CameraRegistry,
    _find_duplicates,
    load_registry_from_path,
)


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


@pytest.mark.parametrize("content", ["", "# 只有註解\n"], ids=["empty", "comment_only"])
def test_load_registry_from_path_rejects_non_mapping_yaml(tmp_path, content):
    """空檔／純註解檔的 safe_load 回傳 None，須報出指向檔案的 ValueError。

    少了這道檢查會直接把 None 丟進 CameraRegistry(**data)，只得到一個沒有檔名
    線索的 TypeError，維運時看不出是哪份 registry 壞掉。
    """
    path = tmp_path / "camera_registry.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="格式不正確或內容為空"):
        load_registry_from_path(path)
