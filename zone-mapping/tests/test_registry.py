from zone_mapping.registry import _find_duplicates


def test_find_duplicates_returns_items_appearing_more_than_once():
    assert _find_duplicates(["a", "b", "a", "c", "c", "c"]) == {"a", "c"}


def test_find_duplicates_returns_empty_set_when_all_unique():
    assert _find_duplicates(["a", "b", "c"]) == set()
