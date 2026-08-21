"""環形緩衝的取用語義測試：`view_slot` 必須是 view，尺寸不符必須擋下。

`view_slot` 免掉的整格複製（4K 約 24MB）是推理進程 14% 的時間，但它換來的是
「slot 在推論完成前不能歸還」這條生命週期約束（見 ADR-010）。有人把它改回
`.copy()` 時，所有既有測試都會照過、輸出也完全正常，只有速度靜默掉回去——第一支
測試釘的就是這件事。
"""

import numpy as np
import pytest

from video_analyze.services import frame_ring
from video_analyze.services.frame_ring import FrameRing, create_ring_buffer

_HEIGHT = 4
_WIDTH = 6
_NUM_SLOTS = 3


def _make_ring() -> FrameRing:
    buffer = create_ring_buffer(_NUM_SLOTS, _HEIGHT, _WIDTH)
    return FrameRing(buffer, _NUM_SLOTS, _HEIGHT, _WIDTH)


def _frame(value: int) -> np.ndarray:
    return np.full((_HEIGHT, _WIDTH, 3), value, dtype=np.uint8)


def test_view_slot_reflects_a_later_write_to_the_same_slot():
    """view 會跟著共享記憶體變——改回回傳副本這支就會失敗。

    這正是「reader 覆寫仍在用的 slot 會靜默換掉推理中的畫面」的機制本身，也是
    歸還必須延後到推論之後的理由。
    """
    ring = _make_ring()
    ring.write_slot(1, _frame(10))

    view = ring.view_slot(1)
    assert np.array_equal(view, _frame(10))

    ring.write_slot(1, _frame(200))  # 模擬 reader 覆寫同一格

    assert np.array_equal(view, _frame(200))


def test_view_slot_does_not_alias_other_slots():
    """只有同一格會連動：寫別格不會改到手上的 view（釘住 reshape 的維度沒接反）。"""
    ring = _make_ring()
    ring.write_slot(0, _frame(10))
    ring.write_slot(2, _frame(20))

    view = ring.view_slot(0)
    ring.write_slot(2, _frame(99))

    assert np.array_equal(view, _frame(10))


def test_allocation_still_works_where_the_shm_diagnostics_are_unavailable(monkeypatch):
    """`/dev/shm` 不存在（非 Linux／受限環境）時照樣配置得出來，只是少了診斷欄位。

    `shm_available_mb` 與 `backing_dirs` 是給人看的訊號，不是配置的前提條件；查不到就
    留空，不能反過來讓整條 pipeline 起不來。
    """
    monkeypatch.setattr(frame_ring, "_SHM_DIR", "/nonexistent-shm")

    assert frame_ring._shm_available_mb() is None

    ring = FrameRing(
        create_ring_buffer(_NUM_SLOTS, _HEIGHT, _WIDTH), _NUM_SLOTS, _HEIGHT, _WIDTH
    )
    ring.write_slot(0, _frame(5))
    assert np.array_equal(ring.view_slot(0), _frame(5))


def test_write_slot_rejects_a_frame_of_a_different_shape():
    """尺寸不符直接拋錯（fail-loud）：緩衝是照推論尺寸一次配置的。

    產線上最可能的成因是讀取端漏了 `letterbox()`；`test_letterbox.py` 用真實的推論
    尺寸釘那個組合，這裡只驗這道檢查本身在自訂尺寸下也在。
    """
    ring = _make_ring()

    with pytest.raises(ValueError, match="與環形緩衝"):
        ring.write_slot(0, np.zeros((_HEIGHT + 1, _WIDTH, 3), dtype=np.uint8))
