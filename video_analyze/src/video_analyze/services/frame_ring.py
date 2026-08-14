import ctypes
import multiprocessing as mp

import numpy as np

from video_analyze.models.config import settings

# 每路環形緩衝的 slot 數，即 reader 能領先推理進程的影格數上限（等同背壓深度）。
# 由 `settings.model.batch` 推導而非寫死：推理進程改成整批推論完才歸還 slot（見
# `view_slot`）之後，格數的約束來自批次大小，只調 batch 而沒同步調格數時 reader 會
# 在整個推論期間拿不到空位而停擺，且沒有任何錯誤訊息。
#
# 兩個 ×2 的來源不同：
# - 其一：ultralytics 對 in-memory list source 一次 forward 整個 list（`batch=` 只對
#   檔案來源的 LoadImagesAndVideos 有效），故單次批次是 `settings.model.batch` 的 2 倍，
#   即 `InferencePipeline._target_batch`。
# - 其二：扣住一批之外要再留同量空位給 reader 備下一批。「2 × 一批」正是「單一路供批
#   時 reader 完全不因缺 slot 而停」的最小值——issue #100 只改湊批的輪替起點，內層
#   `while` 仍會把起點那一路取到滿批才換手，所以「整批來自同一路」是常態。
#
# 記憶體用量 = RING_SLOTS × 每格位元組 × 路數；batch = 8（32 格）時九路合計 3.34 GiB
# （4K 三路每格 23.73 MiB、1080p 六路每格 5.93 MiB）。`MODEL__BATCH` 的環境變數覆寫
# 會連帶放大緩衝，記憶體隨 batch 線性成長。
RING_SLOTS = settings.model.batch * 4

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

    def view_slot(self, slot: int) -> np.ndarray:
        """取得指定 slot 的 view，不複製。

        整格複製（4K 約 24MB）實測佔推理進程 14% 的時間，而 YOLO 前處理本來就會把
        影格 letterbox 成新陣列再上傳 GPU，這份副本純屬多餘。

        **代價是 slot 不能立刻歸還**：回傳的是共享記憶體本身，reader 一旦覆寫該
        slot，這個 view 的內容就跟著變。呼叫端必須等到不再需要影格內容（推論完成）
        才歸還 slot。

        Args:
            slot: 要讀取的 slot 索引。

        Returns:
            指向共享記憶體的 view（非副本）。
        """
        return self._slots[slot]
