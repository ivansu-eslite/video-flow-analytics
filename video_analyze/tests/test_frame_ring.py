"""環形緩衝的取用語義測試：`view_slot` 必須是 view，尺寸不符必須擋下。

`view_slot` 免掉的整格複製（4K 約 24MB）是推理進程 14% 的時間，但它換來的是
「slot 在推論完成前不能歸還」這條生命週期約束（見 ADR-010）。有人把它改回
`.copy()` 時，所有既有測試都會照過、輸出也完全正常，只有速度靜默掉回去——第一支
測試釘的就是這件事。
"""

import numpy as np
import pytest

from video_analyze.services import frame_ring
from video_analyze.services.frame_ring import (
    FrameRing,
    create_ring_buffer,
    create_ring_buffers,
    require_shm_capacity,
)

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

    產線上最可能的成因是讀取端漏了 `letterbox_nv12()`；`test_letterbox.py` 用真實的推論
    尺寸釘那個組合，這裡只驗這道檢查本身在自訂尺寸下也在。
    """
    ring = _make_ring()

    with pytest.raises(ValueError, match="與環形緩衝"):
        ring.write_slot(0, np.zeros((_HEIGHT + 1, _WIDTH, 3), dtype=np.uint8))


# --------------------------------------------------------------------------------------
# 配置前的擋（`require_shm_capacity`）
#
# 這一組釘的是「合計 vs 總容量」這個判準本身。判準改成逐塊比對可用空間時，下面兩支
# 期待拋錯的會失敗、放行的兩支照過（實測過）——那正是要防的實作：逐塊比對是 CPython
# 自己已經在做的事，而它的處置是靜默改用 /tmp，不是報錯。
# --------------------------------------------------------------------------------------

_SLOTS, _H, _W = 32, 384, 640
_ONE_BUFFER_BYTES = _SLOTS * _H * _W * 3  # 一塊 22.5 MiB（出貨設定 batch=16 的每路用量）


def _fake_statvfs(total_bytes: int, monkeypatch) -> None:
    """把 `/dev/shm` 的總容量假造成 `total_bytes`（`f_frsize` 固定 1）。

    這裡換掉的是模組看到的 `os.statvfs`，不是同檔其他測試用的 `_SHM_DIR` 指向不存在的
    路徑——那個做法只控制得了「有沒有」，這一組要同時控制 `f_blocks` 與 `f_bavail` 兩個
    值，才問得出「判準取的是哪一個」。
    """

    class _Stat:
        f_blocks = total_bytes
        f_frsize = 1
        # 可用空間刻意留成「單塊放得下」，逐塊比對的實作會因此全數放行
        f_bavail = _ONE_BUFFER_BYTES

    monkeypatch.setattr(frame_ring.os, "statvfs", lambda _path: _Stat())


def test_shm_capacity_passes_when_the_total_fits(monkeypatch):
    """合計放得下就不能擋——這道檢查誤擋的話整條 pipeline 起不來。"""
    _fake_statvfs(_ONE_BUFFER_BYTES * 9, monkeypatch)

    require_shm_capacity(9, _SLOTS, _H, _W)


def test_shm_capacity_blocks_when_only_one_buffer_fits(monkeypatch):
    """單塊放得下、九塊合計放不下——CPython 正是在這種組態下靜默把後幾塊改丟 /tmp。

    `f_bavail` 刻意設成單塊放得下：逐塊比對可用空間的實作在這裡每一塊都會通過、一路
    放行到底，而那正是 CPython 自己做完之後選擇靜默降級的那個判斷。這支失敗代表判準
    被改回逐塊口徑，這道擋就等於不存在。
    """
    _fake_statvfs(_ONE_BUFFER_BYTES * 2, monkeypatch)

    with pytest.raises(RuntimeError, match="共享記憶體不足"):
        require_shm_capacity(9, _SLOTS, _H, _W)


def test_shm_capacity_message_names_both_numbers(monkeypatch):
    """訊息要同時帶「需要多少」與「只有多少」，否則看到錯誤也不知道該調哪個旋鈕。"""
    _fake_statvfs(_ONE_BUFFER_BYTES * 2, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        require_shm_capacity(9, _SLOTS, _H, _W)

    message = str(excinfo.value)
    assert f"{_ONE_BUFFER_BYTES * 9 / (1024 * 1024):.2f} MiB" in message
    assert f"{_ONE_BUFFER_BYTES * 2 / (1024 * 1024):.2f} MiB" in message


def test_shm_capacity_is_a_noop_without_dev_shm(monkeypatch):
    """取不到 /dev/shm 的平台上 `mp.RawArray` 不走 tmpfs，沒有這個失敗模式，不得誤擋。"""
    monkeypatch.setattr(frame_ring, "_SHM_DIR", "/nonexistent-shm")

    assert frame_ring.shm_total_bytes() is None

    require_shm_capacity(9999, _SLOTS, _H, _W)


def test_create_ring_buffers_allocates_one_per_stream(monkeypatch):
    """正常情況下每路一塊，且每塊都能包成可讀寫的 `FrameRing`。"""
    # 假容量只需大於本測試 3 塊 _NUM_SLOTS/_HEIGHT/_WIDTH 的合計（648 bytes）；
    # 與 _ONE_BUFFER_BYTES（32/384/640 的另一組常數）無關，沿用只是取個現成的大數字。
    _fake_statvfs(_ONE_BUFFER_BYTES * 9, monkeypatch)

    buffers = create_ring_buffers(3, _NUM_SLOTS, _HEIGHT, _WIDTH)

    assert len(buffers) == 3
    ring = FrameRing(buffers[0], _NUM_SLOTS, _HEIGHT, _WIDTH)
    ring.write_slot(0, _frame(7))
    assert np.array_equal(ring.view_slot(0), _frame(7))


def test_create_ring_buffers_allocates_nothing_when_the_total_does_not_fit(monkeypatch):
    """裝不下時一塊都不配。

    擋與配置收在同一個函式，是因為分開時「呼叫端漏掉那道擋」不會有任何症狀——測試全綠、
    輸出正確，只有讀寫變成磁碟等級。這支釘的就是「拿得到緩衝 ⇒ 擋跑過了」這個保證：
    真的配下去之後才發現不夠，前幾塊已經落在 /dev/shm 上了。
    """
    _fake_statvfs(_ONE_BUFFER_BYTES * 2, monkeypatch)
    allocated = []
    monkeypatch.setattr(
        frame_ring,
        "create_ring_buffer",
        lambda *args, **kwargs: allocated.append(args) or object(),
    )

    with pytest.raises(RuntimeError, match="共享記憶體不足"):
        create_ring_buffers(9, _SLOTS, _H, _W)

    assert allocated == []
