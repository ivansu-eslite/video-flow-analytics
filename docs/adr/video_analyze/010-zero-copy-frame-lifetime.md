# ADR-010: 推理主迴圈免複製消費共享記憶體，slot 延後到整批推論後歸還

## Status

Accepted

## Context

`video_analyze` 處理一天九支攝影機需 42.2 小時，需求是 9 小時。瓶頸在單一推理進程的
序列成本，其中「把畫面從共享記憶體複製出來」（`FrameRing.read_slot`）佔每張畫面 3.07 ms、
全部 21.57 ms 的 **14.2%**，是非 GPU 成本裡最大的一筆。

這筆複製沒有換來任何東西：

- 實測 `cv2.resize` 從共享記憶體讀與從私有陣列讀同速（4K 皆 0.31 ms），複製沒有換到
  快取效益。
- YOLO 前處理本來就會把畫面 letterbox 成新陣列（`LetterBox.apply_image` 對 3 通道無條件
  走 `cv2.copyMakeBorder`），等於在 `preprocess` 結束前像素已經被複製兩次。
- 反推 3.07 ms 的記憶體流量得 8.5 GB/s，正是 Skylake 單執行緒 memcpy 的典型頻寬——它就是
  一筆乾淨的 memcpy，不是別的成本被算在這裡。

改成回傳 view 本身只是一行，本 ADR 要記的是它換來的東西：**畫面的生命週期從此綁在 slot
上**。slot 一歸還，reader 隨時可以覆寫同一塊記憶體，而覆寫進去的不是雜訊，是同一支攝影機
幾格之後的正常畫面——偵測數不會崩、輸出檔完全正常，只有座標靜默偏移。這種錯誤沒有訊號，
所以「什麼時候可以歸還」必須是一條寫下來的約束，而不是靠讀程式碼推出來。

前置條件是 issue #102（PR #103）已移除標註影片輸出：`FramePacket.frame` 的消費者收斂成
`detector.predict` 一處，推論之後不再有人需要畫面。

## Options Considered

### Option A：維持整格複製

零風險，但 14% 的推理進程時間是純浪費，且它隨解析度線性成長（4K 每格約 24 MB）——
攝影機升級只會讓這筆變大。

### Option B：改回 view，但維持「取出後立刻歸還」

一行改動、看似不影響任何東西，實際是最糟的選項：覆寫窗口的長度等於 `predict` 的耗時
（T4 上一批 16 格約 254 ms），reader 在這段時間繞完一圈就會覆寫仍在推論的畫面。錯誤靜默
（見 Context），排除。

### Option C：view ＋ 整批推論完成後才歸還（採用）

歸還點延後到 `predict` 回傳之後。代價是同一路最多有「一批」的 slot 被扣住不還，因此環形
緩衝的格數必須跟著批次大小走（見 Decision 3）。

### Option D：把畫面縮放移到讀取端

效益更大（約 4.75 vs 3.07 ms），因為連解碼後的尺寸都縮小了。但框會還原到 640×360 座標系，
下游 `zone_mapping`／`line_counting` 的幾何是照原始影像座標畫的，重縮放算錯不會有任何訊號。
屬正確性面的改動，需要另立 plan 與控制組驗證，不併進本次。

### Option E：畫面不離開 GPU（NVDEC 解碼）

上限最高，但是架構級變更（讀取進程、環形緩衝、批次組成全要重做），另立 plan 與 ADR。

## Decision

### 1. `read_slot` 改名為 `view_slot`，兩者不並存

唯一的生產呼叫端是 `services/inference.py` 的 `_collect_batch`，改完即無殘留。刻意不留
`read_slot`：留著會讓人以為兩種取用模式可以互換，而免複製的整個代價（本 ADR 這條生命週期
約束）正是「不可互換」的理由。需要在推論之後仍持有畫面的路徑（例如輸出標註影片）已於
PR #103 移除；日後若要加回來，是要重新引入取副本的介面，不是切換一個參數。

### 2. 歸還點卡在 `predict` 之後、逐格 tracking 迴圈之前

三個方向都是硬的：

- **不能更早**：`predict` 的前處理還在讀共享記憶體，早還會讓 reader 邊寫邊被讀。
- **不能更晚**：逐格 tracking 迴圈在 T4 上一批 16 格要數十 ms，拖過去只是白讓 reader 空等，
  把原本可與推論重疊的解碼變成序列。
- **`results[i].orig_img` 就是這些 slot 的 view**：ultralytics `engine/results.py` 是
  `self.orig_img = orig_img`，沒有 `.copy()`（已動態驗證 `Results.orig_img is view` 為
  True）。歸還之後該欄位的內容不再可信。

歸還的職責因此整個歸呼叫端：`_collect_batch` 只把取用的 `(stream_id, slot)` 回報上去，
**連 `free_queues` 都不再收**——拿不到就不可能提早歸還，比留著參數再用測試盯強。

> **（issue #142 後更新）歸還點前移到前處理之後。** 偵測器的介面拆成 `preprocess` ＋
> `infer` 兩步（ADR-013），像素在 `preprocess` 的 `np.stack` 當下就複製進新的 CPU 陣列、
> H2D 也在它回傳前完成（pageable 記憶體的 `.to()` 是同步的），所以「不能更早」的界線
> 從「`predict` 回來」移到「`preprocess` 回來」，reader 早一整段 forward 拿回空位。
> 上面第三點的 `results[i].orig_img` 一併作廢——`Results` 已經不在正式推論路徑上。

### 3. `RING_SLOTS` 由 `[model].batch` 推導，不寫死

```python
RING_SLOTS = TARGET_BATCH * 2          # TARGET_BATCH = settings.model.batch
```

延後歸還之後，格數的約束來源是批次大小，而批次大小在另一個檔案（`models/config.py` 的
`settings`）。維持寫死常數的話，只調 `model.batch` 而沒同步調格數時 reader 會在整個推論
期間拿不到空位、整條變慢，**且沒有任何錯誤訊息**。推導即消除雙來源。

係數 2 的理由：扣住一批之外要再留同量空位給 reader 備下一批。「2 × 一批」正是「單一路
供批時 reader 完全不因缺 slot 而停」的最小值——issue #100 只改湊批的輪替起點，內層
`while` 仍會把起點那一路取到滿批才換手，「整批來自同一路」是常態而非例外。

> **（issue #116 後更新）本節原本寫成 `settings.model.batch * 4`**，因為當時 config 的
> `batch` 是**湊批目標**而非單次批次：ultralytics 對 in-memory list source 一次 forward
> 整個 list（`batch=` kwarg 只對檔案來源的 `LoadImagesAndVideos` 有效，對 list 是 no-op），
> 所以單次批次是設定值的 2 倍，`4 = 2 × 2` 是兩個來源不同的係數摺在一起。issue #116 把
> `[model].batch` 的語義改成直接等於單次批次（設定值同步由 8 改 16），那個換算用的 `×2`
> 消失，只剩上面這一個。**最終值未變**（16 → 32 格），改的是推導式的表示法。

推導式現在收在 `services/batching.py`，`RING_SLOTS`／`TRACK_QUEUE_SLOTS` 都從那裡匯出。
**這與本節原本否決的「搬進 `config/constants.py`」不是同一件事**，當時的兩條理由在新位置
都不成立：

- **循環 import**：`config/constants.py` 被 `models/config.py` import，把讀 `settings` 的
  常數搬進去會成環；`services/batching.py` 不在那條鏈上（它 import `models/config.py`，
  反向沒有）。
- **「模組私有的調校常數留在各自模組」**：這個前提本來就不成立——`RING_SLOTS` 定義在
  `frame_ring.py` 而該檔自己不用它，`TRACK_QUEUE_SLOTS` 定義在 `track_worker.py` 也一樣，
  兩者唯一的生產消費端都是 `pipeline.py`。留在原處不是「私有」，只是分散。真正私有的
  `_FILL_MAX_WAIT`／`_FILL_POLL` 仍留在 `inference.py`。

### 4. 兩道 fail-loud

**其一：歸還前把 `packet.frame` 與 `results[i].orig_img` 都設為 `None`。** 這兩處是**本
專案自己持有**的、指向已放行 slot 的參照，清掉之後「日後有人在歸還之後讀畫面」從靜默讀到
別格畫面變成當場拋錯。代價是 `FramePacket.frame` 的型別放寬為 `np.ndarray | None`。

> **（issue #142 後更新）只剩 `packet.frame` 那半條，而下一段的「擋不住 ultralytics
> 內部」已不再適用。** 自建前處理與後處理之後（ADR-013）不再呼叫 `predict`，
> `predictor.dataset.im0` 與 `predictor.batch[1]` 兩個活別名連同 `Results` 一起從正式
> 路徑消失，推論輸出（N×6 的 numpy）根本不攜帶影格參照。**這不是保護變弱**：那條保護
> 防的是「日後有人在歸還之後讀影格」，而現在已經沒有可讀的參照留在下游。

**這道保護不覆蓋 ultralytics 內部。** `LoadPilAndNumpy._single_check` 對 numpy 輸入原樣
回傳，所以 `predictor.dataset.im0` 與 `predictor.batch[1]` 存的就是我們傳進去的那批 view；
predictor 掛在 model 上，那兩處要到**下一次** `predict` 才被替換，期間橫跨歸還迴圈、
tracking 迴圈與下一輪 `_collect_batch`。刻意不去清它們——動第三方套件的內部狀態，風險大於
收益。也就是說，這道保護擋的是「我們自己的程式碼日後誤用」，不是所有可能的讀取路徑。

`orig_img` 清得掉，是因為 `Results.__init__` 已把 `orig_shape` 另存一份、`Boxes` 的座標系
也綁在 `orig_shape` 上，本迴圈之後只用 `boxes`。實測 ultralytics 8.4.75：清空後
`orig_shape`、`boxes.orig_shape`、`boxes.xyxy` 皆不受影響，而 `plot()` 直接拋
`AttributeError`——正是希望它發生的時機。

**清參照要排在歸還之前**：反過來的話，兩個迴圈之間會有一段「slot 已放行、活別名還在」的
狀態，正是這道保護要消滅的東西。

**這道保護是活別名唯一擋得住的地方**：測試與控制組比對都擋不住它，因為讀到的是內容正常
的畫面。

**其二：`start_loop` 開頭的格數不變量檢查**（任一路 `ring.num_slots < 2 × _target_batch`
即拋 `ValueError`）。Decision 3 的推導式已保證它成立，所以它防的不是「有人調 batch」，而是
**日後兩處推導公式漂移**。正常情況永遠不觸發，成本一行。

> **（issue #116 後更新）這道檢查已移除。** 它防的「兩處公式漂移」在三個常數收斂到
> `services/batching.py` 之後不再有第二處可漂移，同一條關係改由 `tests/test_batching.py`
> 釘住。代價是若日後有人在 `pipeline.py` 直接傳自訂 `num_slots` 給 `FrameRing`（不走
> `RING_SLOTS`），就沒有執行期的守門了——那條路徑目前不存在，接受此代價。

### 5. 例外路徑刻意不加 `try/finally`

`READER_FAILED` 或 `predict` 拋出時 `held_slots` 不會歸還，該路 reader 卡在
`free_queue.get()`。**不會 hang**：推理進程死亡 → `pipeline.py` 的 `_raise_if_abnormal`
偵測到非零 exitcode → `_terminate_all` 殺掉所有 reader。這條推理依賴 `analyze_daily` 的
收尾邏輯，故在程式碼註解裡寫明，不留給下一個人自己推。包 `try/finally` 得讓 `held_slots`
的作用域橫跨 `_collect_batch` 與 `start_loop` 兩個函式，代價高於收益。

### 6. `orig_img` 安全是本版消費路徑的性質，不是 ultralytics 的保證

歸還之後 `results[i].orig_img` 仍指著已放行的 slot。今天不出事，是因為這一版的消費路徑
一處都沒碰它：

- 前處理已把像素複製兩次（`LetterBox` ＋ 上傳 GPU 的 tensor）；
- 後處理只從 `orig_img` 取 `shape`；
- BYTETracker 收到的 `img` 是 `None`（`update(results, img=None)`；`img` 只有 BOTSORT 的
  `gmc` 分支會用）；
- 結果收集只用 `frame_index` 與 `timestamp`。

四條裡任何一條變動都會讓這個依賴浮上來：開 `verbose=True`、呼叫 `plot()`、改用 BOTSORT、
或升級 ultralytics（`orig_img` 的處理是實作細節，不在其 API 契約裡）。屆時要改回取副本，
而不是調歸還時機——Decision 4 把 `orig_img` 清成 `None`，就是要讓那一刻以 `AttributeError`
的形式出現，而不是以「座標偏了幾十 px 的正常輸出」的形式出現。

> **（issue #142 後更新）這整條依賴已經不存在。** 前處理與後處理改成本套件自己算
> （ADR-013），`Results` 不再被建出來，推論輸出是 N×6 的 numpy——「歸還之後仍指著已放行
> slot 的欄位」沒有了對象。上面四條裡的前兩條連同 predictor 一起消失；BYTETracker 與
> 結果收集那兩條搬到追蹤進程（issue #109），行為不變。換句話說，這一節記的是**當時**
> 為何不必取副本，不再是現行程式碼的約束。

## Consequences

Positive

- 回收每張畫面約 2.5–3.1 ms（T4 預估 21.57 → 18.5–19.1 ms，整體 12–14%），且不是搬到別處
  ——那筆 memcpy 直接消失。
- 效益隨解析度成長：4K 那幾路省得比 1080p 多。
- 環形緩衝格數不再是要人工同步的第二份設定。

Negative

- **合併時正確性只有靜態論證與單元測試撐著**。真正的驗收是 T4 端到端控制組比對，排在效能
  主文件的 B 線；比對須照「控制組相對條件」做（未改動的程式自己重跑兩次取得偏離基準），
  並回報 p99.9、最大值、偏差 > 50 px 的畫面數。**不可用偵測數判讀**：讀到被覆寫的 slot
  拿到的是同一路幾格之後的正常畫面，偵測數不會崩，只會靜默偏移。
- **地端 5090 跑過不等於 T4 安全**：覆寫窗口的長度等於 `predict` 耗時，5090 上短得多，
  reader 來不及繞完一圈覆寫。
- **`Results.orig_img` 的活別名依賴無法由測試擋住**（見 Decision 6）：測試不會「讀到錯格
  畫面」，因為那需要真實的 reader 併發覆寫。擋得住它的只有註解、本 ADR，以及歸還後把
  `packet.frame` 與 `results[i].orig_img` 清成 `None` 這道保護——後者能讓誤用當場拋錯，但
  無法讓誤用者事先知道為什麼不能那樣寫。
- **記憶體由 1.67 GiB 增為 3.34 GiB**（九路各 32 格）。`mp.RawArray` 在 Linux 優先落
  `/dev/shm`，空間不足時**靜默** fallback 到磁碟（變成檔案 mmap，效能崩掉但不報錯），
  部署前要在容器內確認 `df -h /dev/shm` ≥ 4 GiB。CPython 是逐塊判斷的，九路各配一塊，
  因此可能前幾路進 `/dev/shm`、後幾路掉出去。`create_ring_buffer` 逐塊 log 出 `size_mb`、
  `shm_available_mb` 與 **`backing_dirs`**——最後一項比對配置前後 `/proc/self/maps` 上的
  arena 映射，記的是這塊**實際**落在哪，不是由可用空間推論（同機其他行程也在動用
  `/dev/shm`，推論會失準）。~~**主動擋下**是 issue #106，本次只留訊號、不擋。~~
  **（issue #106 後更新）配置前的擋已補上**：`create_ring_buffers` 在配置第一塊之前呼叫
  `require_shm_capacity`（擋與配置收在同一個函式——分開時呼叫端漏掉那道擋不會有任何
  症狀，正是它要防的失敗模式），比
  「全部路合計 vs `/dev/shm` 總容量」，不足即拋 `RuntimeError`。判準刻意不用逐塊比對
  可用空間——那是 CPython 自己已經在做的事，而它的處置是靜默改用 `/tmp`。這道擋是必要
  非充分（同機其他行程佔用時，合計沒超過總容量也可能降級），`backing_dirs` 仍是唯一能
  確認實際落點的手段，兩者互補、都要留。
- **`MODEL__BATCH` 的環境變數覆寫會連帶放大緩衝**，記憶體隨 batch 線性成長（batch = 16 →
  64 格 → 6.7 GiB）。這是推導的預期行為，但部署端要知道這兩個旋鈕已經綁在一起。
- **（issue #108／#116 後更新）上面兩條的容量數字已不成立**：影格縮放移到讀取端之後，
  緩衝依推論尺寸 640×384 配置，每格一律 0.70 MiB 而與來源解析度無關——出貨設定
  `batch = 16`（32 格）九路合計 202.5 MiB，`MODEL__BATCH=32`（64 格）九路合計 405 MiB。
  （這兩個數字對應的格數與更新前相同；`batch` 的語義在 issue #116 改成直接等於單次批次，
  同一個格數現在對應的設定值是原本的 2 倍。）`df -h /dev/shm` 的部署門檻隨之從 4 GiB 降
  一個量級到 256 MiB，但容器預設值（Docker 不給 `--shm-size` 時是 64 MB）**仍然不夠**。
  逐塊判斷造成的部分落磁碟也沒有消失，反而因為每塊縮到 22.5 MiB 而更容易發生——64 MB 夠塞
  前兩三路、其餘掉出去，正是最難歸因的情況。
- **`frame_ring.py` 從零依賴模組變成 import `settings`**。目前無循環（`models/config.py`
  只 import `config/constants.py`，反向沒有），但這條依賴日後要維持單向。
  **（issue #116 後更新）`frame_ring.py` 已改回零依賴**：`RING_SLOTS` 搬到
  `services/batching.py`，那條 `settings` 依賴跟著搬過去，維持單向的要求不變。
- **argus 的兩份拷貝不隨本次同步**：`pipelines/vertex_ai/` 那份仍有 `save_video`，**必須
  保留 `read_slot` 與其條件判斷**，不可照搬本次的簡化版（標註影片的編碼是背景執行緒非同步
  進行，slot 早被覆寫了）；onprem 拷貝的同步另議。
