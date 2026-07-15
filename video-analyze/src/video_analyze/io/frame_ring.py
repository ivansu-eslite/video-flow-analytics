import ctypes
import multiprocessing as mp

import numpy as np

# 每路環形緩衝的 slot 數，即 reader 能領先推理進程的影格數上限（等同背壓深度）。
# 記憶體用量 = RING_SLOTS × 每格位元組 × 路數（如 16 × 6.2MB × 4 路 ≈ 397MB）。
RING_SLOTS = 16

_CHANNELS = 3  # BGR


def create_ring_buffer(num_slots: int, height: int, width: int):
    """在父進程建立可跨 fork 子進程共享的環形緩衝底層記憶體。

    回傳的 RawArray 作為 Process 參數傳給 reader 與推理進程；fork 下三方共享同一塊
    匿名 mmap，寫入互相可見（不經 pickle）。

    Args:
        num_slots: 緩衝的 slot 數（見 `RING_SLOTS`）。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。

    Returns:
        `mp.RawArray`，可傳給子進程建構 `FrameRing`。
    """
    return mp.RawArray(ctypes.c_uint8, num_slots * height * width * _CHANNELS)


class FrameRing:
    """單一路的共享記憶體環形緩衝，避免每格 6MB 影格走 pickle + pipe（實測該 IPC
    佔推理進程時間約 60%，是搬走影片編碼後的新瓶頸）。

    假設同一路整天解析度固定（緩衝依首格尺寸一次配置）；尺寸不符時 write_slot
    直接拋 ValueError（fail-loud），不會靜默寫壞。
    """

    def __init__(self, buffer, num_slots: int, height: int, width: int):
        """包裝 `create_ring_buffer` 建立的共享記憶體為可讀寫的環形緩衝。

        Args:
            buffer: `create_ring_buffer` 回傳的 `mp.RawArray`。
            num_slots: 緩衝的 slot 數，需與 `buffer` 建立時的 `num_slots` 一致。
            height: 影格高度（pixel），需與 `buffer` 建立時一致。
            width: 影格寬度（pixel），需與 `buffer` 建立時一致。
        """
        self.num_slots = num_slots
        self.frame_shape = (height, width, _CHANNELS)
        self._slots = np.frombuffer(buffer, dtype=np.uint8).reshape(
            num_slots, height, width, _CHANNELS
        )

    def write_slot(self, slot: int, frame: np.ndarray) -> None:
        """把一格影格寫入指定 slot。

        Args:
            slot: 目標 slot 索引。
            frame: 要寫入的影格，形狀須與緩衝建立時的 `frame_shape` 一致。

        Raises:
            ValueError: `frame.shape` 與緩衝的 `frame_shape` 不符。
        """
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"影格解析度 {frame.shape} 與環形緩衝 {self.frame_shape} 不符"
                "（假設單一攝影機整天解析度固定）"
            )
        np.copyto(self._slots[slot], frame)

    def read_slot(self, slot: int) -> np.ndarray:
        """讀出指定 slot 的影格副本。

        複製成私有陣列，呼叫端即可立刻歸還 slot 供 reader 覆寫。

        Args:
            slot: 要讀取的 slot 索引。

        Returns:
            該 slot 影格內容的獨立副本（非 view）。
        """
        return self._slots[slot].copy()
