import ctypes
import multiprocessing as mp
import os

import numpy as np
from vfa_observability import StructuredLogger

logger = StructuredLogger(component="frame_ring")

# `mp.RawArray` 在 Linux 是這個目錄下的檔案 mmap（tmpfs，速度等同 RAM）
_SHM_DIR = "/dev/shm"

_CHANNELS = 3  # BGR


def require_shm_capacity(
    num_buffers: int, num_slots: int, height: int, width: int
) -> None:
    """配置任何一塊之前，先擋下「全部緩衝合計裝不進 `/dev/shm`」的組態。

    **失敗模式是靜默降級，不是崩潰。** `mp.RawArray` 的整數路徑在配置當下就
    `ctypes.memset` 寫滿整塊，tmpfs 空間是真的被佔掉的；而 `heap.py` 的 `_choose_dir`
    每配一塊比一次 `st.f_bavail * st.f_frsize >= size`，不夠就改用 `util.get_temp_dir()`
    （`/tmp`，磁碟上的 mmap），**不拋例外、不留 log**。因為 memset 讓前幾塊的佔用真的
    看得到，CPython 那個逐塊檢查本身是有效的，不會 over-commit 到執行中才 SIGBUS。
    問題出在它的**處置**：程式照跑、輸出完全正確，只有讀寫成本悄悄變成磁碟等級。

    同機制的容器內實測（CPython 3.12.3、`--shm-size=64m`、九塊各 22.50 MiB）：第 1、2
    塊落在 `/dev/shm`（剩餘 41.5 → 19.0 MiB），第 3 塊起 19.0 < 22.5 就全部落到 `/tmp`，
    九塊都寫得滿、行程 exit 0。合計 202.50 MiB > 64 MiB 這件事沒有任何訊號會主動報錯。

    **這道擋是必要條件，不是充分條件。** 它比的是「合計需求 vs 總容量」（`f_blocks`），
    而 CPython 降級的判準是逐塊的**可用**空間（`f_bavail`）——同機其他行程佔著
    `/dev/shm` 時，合計沒超過總容量也可能降級。所以 `create_ring_buffer` 的
    `backing_dirs` 不是被這道擋取代的裝飾，而是**唯一**能確認實際落點的手段：這裡擋掉
    「一定裝不下」的組態，那裡回答「這次實際落在哪」，兩者互補、都要留。

    也因此判準不能改成逐塊比對 `f_bavail`——那是 CPython 自己已經在做的事，再做一次
    不會多擋到任何東西，只會讓這道擋看起來存在。

    擋的是**未來**：有人調大 `[model].batch`、增加路數或換執行環境時，失敗方式應該是
    啟動時 fail loud，而不是靜默慢下來。

    取不到 `/dev/shm`（`os.statvfs` 拋 `OSError`）時直接放行：沒有這個目錄的環境上
    `mp.RawArray` 本來就不走 tmpfs，沒有這個失敗模式。⚠ Windows 連 `os.statvfs` 都沒有，
    那裡會是 `AttributeError` 而不是放行——與同檔 `_shm_available_mb` 的既有處置一致，
    這條 pipeline 只跑 Linux，不為此加分支。

    Args:
        num_buffers: 要配置幾塊（= 路數）。
        num_slots: 每塊的 slot 數。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。

    Raises:
        RuntimeError: 合計需求超過 `/dev/shm` 總容量。
    """
    total_bytes = shm_total_bytes()
    if total_bytes is None:
        return
    required = num_buffers * num_slots * height * width * _CHANNELS
    if required > total_bytes:
        raise RuntimeError(
            f"共享記憶體不足：{num_buffers} 塊環形緩衝合計需要 "
            f"{required / (1024 * 1024):.2f} MiB，但 {_SHM_DIR} 總容量只有 "
            f"{total_bytes / (1024 * 1024):.2f} MiB。"
            "調小 [model].batch（RING_SLOTS 由它推導，見 services/batching.py）、"
            "減少同時處理的路數，或把執行環境的 /dev/shm 調大"
            "（docker run --shm-size、k8s 的 emptyDir: {medium: Memory}）。"
        )


def create_ring_buffer(num_slots: int, height: int, width: int):
    """在父進程建立可跨 fork 子進程共享的環形緩衝底層記憶體。

    回傳的 RawArray 作為 Process 參數傳給 reader 與推理進程；fork 下三方共享同一塊
    匿名 mmap，寫入互相可見（不經 pickle）。

    Args:
        num_slots: 緩衝的 slot 數（見 `services/batching.py` 的 `RING_SLOTS`）。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。

    Returns:
        `mp.RawArray`，可傳給子進程建構 `FrameRing`。

    Note:
        **多路一次配置請用 `create_ring_buffers`**——配置前的擋在那裡，直接逐塊呼叫本
        函式會繞過它。
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
    # 「合計一定裝不下」的組態由 `require_shm_capacity` 在配置前擋掉，本函式只負責留下
    # 訊號——那道擋是必要非充分（比的是總容量，擋不掉同機其他行程佔用造成的降級），
    # 落點仍然只有這裡回答得了。
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


def create_ring_buffers(
    num_buffers: int, num_slots: int, height: int, width: int
) -> list[ctypes.Array]:
    """配置 `num_buffers` 塊環形緩衝，配置第一塊之前先跑 `require_shm_capacity`。

    擋與配置刻意收在同一個函式：兩者分開時，呼叫端漏掉那道擋**不會有任何症狀**——測試
    照樣全綠、輸出完全正確，只有讀寫成本悄悄變成磁碟等級，正是這道擋本身要防的失敗
    模式。收在一起之後，能拿到緩衝就代表擋跑過了。

    Args:
        num_buffers: 要配置幾塊（= 路數）。
        num_slots: 每塊的 slot 數。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。

    Returns:
        `num_buffers` 個 `mp.RawArray`。

    Raises:
        RuntimeError: 合計需求超過 `/dev/shm` 總容量（一塊都不會配）。
    """
    require_shm_capacity(num_buffers, num_slots, height, width)
    return [create_ring_buffer(num_slots, height, width) for _ in range(num_buffers)]


def shm_total_bytes() -> int | None:
    """`/dev/shm` 的**總容量**（bytes）；該路徑不存在（`statvfs` 拋 `OSError`）時回傳 `None`。

    與 `_shm_available_mb`（剩餘空間）刻意分開：容量是這個環境給的上限、不隨其他行程
    變動，是配置前檢查唯一能用的基準；剩餘空間只適合當事後的餘裕訊號。
    """
    try:
        stat = os.statvfs(_SHM_DIR)
    except OSError:
        return None
    return stat.f_blocks * stat.f_frsize


def _shm_available_mb() -> float | None:
    """`/dev/shm` 目前的可用空間（MiB）；該路徑不存在（`statvfs` 拋 `OSError`）時回傳 `None`。"""
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
