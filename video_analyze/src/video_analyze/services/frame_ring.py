import ctypes
import multiprocessing as mp
import os

import numpy as np
from vfa_observability import StructuredLogger

from video_analyze.models.config import settings

logger = StructuredLogger(component="frame_ring")

# `mp.RawArray` 在 Linux 是這個目錄下的檔案 mmap（tmpfs，速度等同 RAM）
_SHM_DIR = "/dev/shm"

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
# 記憶體用量 = RING_SLOTS × 每格位元組 × 路數。影格於讀取端就縮成推論尺寸（640×384，
# 見 `services/letterbox.py`），每格一律 0.70 MiB 而與來源解析度無關，batch = 8（32 格）
# 時九路合計 202.5 MiB。縮放前移之前存的是原始解析度（4K 每格 23.73 MiB、1080p 每格
# 5.93 MiB，九路合計 3.34 GiB）。`MODEL__BATCH` 的環境變數覆寫會連帶放大緩衝，記憶體隨
# batch 線性成長。
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
    total_bytes = num_slots * height * width * _CHANNELS
    # CPython 逐塊比對「`/dev/shm` 剩餘空間 >= 本塊大小」，不夠就**靜默**改用 /tmp（磁碟上
    # 的 mmap，讀寫崩掉但輸出完全正常、log 也不會有訊號）。九路各配一塊、逐塊判斷，因此
    # 可能前幾路進 /dev/shm、後幾路掉出去。
    #
    # `backing_dirs` 記的是這塊**實際**落在哪：比對配置前後的 arena 映射，新出現的那個就是
    # 本次配置的。不用「可用空間 vs 大小」去推論，是因為那是預測（同機其他行程也在動用
    # /dev/shm，且檢查與配置之間有時間差），而這裡讀得到已發生的事實。`shm_available_mb`
    # 仍保留，但作用是容量餘裕（還剩多少、下次調大 batch 會不會爆），不是判斷落點。
    # 主動擋下是 issue #106，本函式只負責留下訊號。
    arenas_before = _arena_paths()
    buffer = mp.RawArray(ctypes.c_uint8, total_bytes)
    logger.info(
        "配置共享記憶體環形緩衝",
        num_slots=num_slots,
        height=height,
        width=width,
        size_mb=round(total_bytes / (1024 * 1024), 2),
        shm_available_mb=_shm_available_mb(),
        backing_dirs=sorted(
            {os.path.dirname(path) for path in _arena_paths() - arenas_before}
        ),
    )
    return buffer


def _shm_available_mb() -> float | None:
    """`/dev/shm` 目前的可用空間（MiB）；非 Linux 或該路徑不存在時回傳 `None`。"""
    try:
        stat = os.statvfs(_SHM_DIR)
    except OSError:
        return None
    return round(stat.f_bavail * stat.f_frsize / (1024 * 1024), 2)


def _arena_paths() -> set[str]:
    """本行程目前的 multiprocessing arena 映射路徑（檔名帶 `pym-` 前綴）。

    這些檔案配置後隨即被 `unlink`，故 `/proc/self/maps` 上標著 `(deleted)`，但**目錄仍
    看得到**，落在 `/dev/shm` 或 `/tmp` 一目了然。無 `/proc` 的平台回傳空集合，呼叫端
    因而得到空的 `backing_dirs`（不是錯誤——這條資訊本來就只有 Linux 有）。
    """
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
    except OSError:
        return set()
    paths = set()
    for line in lines:
        fields = line.split()
        # 格式：address perms offset dev inode pathname [(deleted)]
        if len(fields) >= 6 and "/pym-" in fields[5]:
            paths.add(fields[5])
    return paths


class FrameRing:
    """單一路的共享記憶體環形緩衝，避免每格 6MB 影格走 pickle + pipe（實測該 IPC
    佔推理進程時間約 60%，是搬走影片編碼後的新瓶頸）。

    緩衝依**推論尺寸**一次配置（讀取端寫入前已 letterbox），因此尺寸不再隨來源解析度
    變動；尺寸不符時 write_slot 直接拋 ValueError（fail-loud），不會靜默寫壞。
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
            ValueError: `frame.shape` 與緩衝的 `frame_shape` 不符。緩衝照推論尺寸配置，
                所以最可能的成因是呼叫端漏了 `letterbox()`（其次是同一路整天解析度
                並非固定，那是配置時的假設）。
        """
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"影格解析度 {frame.shape} 與環形緩衝 {self.frame_shape} 不符："
                "寫入前需先 letterbox 成推論尺寸（見 services/letterbox.py）"
            )
        np.copyto(self._slots[slot], frame)

    def view_slot(self, slot: int) -> np.ndarray:
        """取得指定 slot 的 view，不複製。

        整格複製（4K 約 24MB）實測佔推理進程 14% 的時間，而 YOLO 前處理本來就會把
        影格搬成新陣列再上傳 GPU，這份副本純屬多餘。該量測是在縮放前移之前做的（slot
        存原始解析度），現在每格只有 0.70 MiB、省下來的絕對時間小得多，但下面那條生命
        週期約束不變。

        **代價是 slot 不能立刻歸還**：回傳的是共享記憶體本身，reader 一旦覆寫該
        slot，這個 view 的內容就跟著變。呼叫端必須等到不再需要影格內容（推論完成）
        才歸還 slot。

        Args:
            slot: 要讀取的 slot 索引。

        Returns:
            指向共享記憶體的 view（非副本）。
        """
        return self._slots[slot]
