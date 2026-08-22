"""批次相關常數：單次推理批次，與由它推導的兩個緩衝容量。

三個常數彼此有推導關係，定義處卻都不是使用處——`RING_SLOTS` 原本定義在
`frame_ring.py`（該檔自己不用它，生產消費端是 `pipeline.py`），`TRACK_QUEUE_SLOTS`
原本定義在 `track_worker.py`（同樣只有 `pipeline.py` 在用）。分屬三個檔的後果是改一處
不會同步另兩處，故收斂於此。

`_FILL_MAX_WAIT`／`_FILL_POLL`（湊批等待）不在這裡：那兩個確實只有 `inference.py`
自己在用，留在該模組。
"""

from video_analyze.models.config import settings

# 單次推理批次，即一次 `detector.predict` 送進去的影格數（`InferencePipeline._target_batch`）。
# ultralytics 對 in-memory list source 一次 forward 整個 list（`batch=` kwarg 只對檔案來源的
# `LoadImagesAndVideos` 有效，對 list 是 no-op），所以設定值就是實際的 forward 批次，中間
# 不再有換算；`tools/build_engine.py --batch` 綁的引擎最大批次要容得下這個值，兩者同尺度。
TARGET_BATCH = settings.model.batch

# 每路環形緩衝的 slot 數，即 reader 能領先推理進程的影格數上限（等同背壓深度）。
# 由批次大小推導而非寫死：推理進程改成整批推論完才歸還 slot（見 `frame_ring.view_slot`）
# 之後，格數的約束來自批次大小，只調 batch 而沒同步調格數時 reader 會在整個推論期間拿不到
# 空位而停擺，且沒有任何錯誤訊息。
#
# 係數 2：扣住一批之外要再留同量空位給 reader 備下一批。「2 × 一批」正是「單一路供批時
# reader 完全不因缺 slot 而停」的最小值——issue #100 只改湊批的輪替起點，內層 `while` 仍會
# 把起點那一路取到滿批才換手，所以「整批來自同一路」是常態。
#
# 記憶體用量 = RING_SLOTS × 每格位元組 × 路數。影格於讀取端就縮成推論尺寸（640×384，
# 見 `services/letterbox.py`），每格一律 0.70 MiB 而與來源解析度無關，batch = 16（32 格）
# 時九路合計 202.5 MiB。縮放前移之前存的是原始解析度（4K 每格 23.73 MiB、1080p 每格
# 5.93 MiB，九路合計 3.34 GiB）。`MODEL__BATCH` 的環境變數覆寫會連帶放大緩衝，記憶體隨
# batch 線性成長。
RING_SLOTS = TARGET_BATCH * 2

# `track_queue` 的容量上限（payload 個數）。**這個上限是背壓，不是調校參數**——它擋的
# 兩件事都沒有其他機制在擋：
#
# - **backlog 無上限成長**。影格側的背壓是「reader 拿不到空 slot 就阻塞」，而 slot 在
#   predict 完成當下就歸還（ADR-010），所以那條保護只覆蓋到推論為止。追蹤搬出去之後，
#   追蹤只要比推論慢，payload 就會以 Python 物件的形式堆在推論進程裡（OS pipe 只緩衝
#   約 64 KB，其餘都在 feeder thread 的緩衝），整天數百萬格可以堆到 GB 級而全程沒有訊號。
#   給上限之後 `put` 會阻塞推論迴圈 → 推論不再消費 slot → reader 跟著阻塞，整條 pipeline
#   收斂到最慢的階段，與追蹤還在推論進程內時的行為一致。
# - **`TRACK_FAILED` 送不到**。它是排在同一條 FIFO 尾端的 in-band 訊號，送達延遲與
#   backlog 成正比；而推論進程一死，父進程約 0.5 秒內就 `_terminate_all`。backlog 大到
#   消化不完那幾秒的量時，追蹤進程會在還沒讀到訊號時就被 SIGTERM 收掉。**暫存檔仍然清得
#   掉**（issue #113 之後 SIGTERM 有 handler，走的是同一個 `collector.discard()`），但
#   那條 in-band 的失效路徑等於沒作用，而且會把「上游崩潰」與「被 terminate」混成同一種
#   結束方式。有了上限，backlog 至多這麼多格——預設 batch（16，即上限 64 格）下約 70 ms
#   的工作量，遠小於父進程的偵測延遲，訊號趕得上。
#   ⚠ **這是「backlog 有界」的推論，不是時序保證**：上限隨 `MODEL__BATCH` 線性成長，而
#   父進程的偵測延遲不隨它成長，所以把 batch 調得很大時這個邊際會變薄（例如
#   `MODEL__BATCH=64` → 上限 256 格）。調大 batch 時要一併重新評估這條路徑。
#
# 係數 4：一批推論完會連續 put 一整批，留幾批的鬆弛才不會讓正常抖動變成兩個進程互等。
#
# 代價：追蹤進程先死時，推論進程的 `put` 會阻塞而不再是立即返回。這與 reader 卡在
# `free_queue.get()` 是同一種收斂方式——父進程的 `_raise_if_abnormal` 偵測到非零 exitcode
# 後 `_terminate_all` 收掉，不會 hang；而那種情況下暫存檔已由追蹤進程自己的 `discard()` 清掉。
TRACK_QUEUE_SLOTS = TARGET_BATCH * 4
