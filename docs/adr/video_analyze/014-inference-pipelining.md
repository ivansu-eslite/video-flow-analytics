# ADR-014：推論進程改成兩批深度的流水，正式路徑改吃本套件自建的 TensorRT runner

- 狀態：已採用
- 日期：2026-08-29
- 影響範圍：`video_analyze`（`services/trt_runner.py`、`services/detector.py`、
  `services/inference.py`、`services/fps_meter.py`）

## Context

推論進程是 T4 上端到端的序列瓶頸。自建前處理與後處理之後（ADR-013），T4 上推論進程每格
3.76 ms（`detection_fps` 265.64，2 分鐘版），GPU 穩態 `sm%` 88；同一顆引擎、同樣批次 16
的 forward-only 滿載是 301.92 張/秒＝每格 3.31 ms（功耗牆下量到的 GPU 自身上限）。兩者
差 0.45 ms/格（12%），那就是 GPU 在等 host 的時間。

原因是主迴圈一件接一件做：湊批 → `np.stack` → 同步 H2D → `execute_v2`（阻塞到算完）→
同步 D2H → numpy 過濾 → 一批 16 次 `queue.put`。host 做事時 GPU 閒著，GPU 做事時 host
閒著，而這兩段本來沒有資料相依——第 k 批在 GPU 上算的時候，host 可以去湊第 k+1 批。

要讓它們重疊，`AutoBackend.forward` 的介面就不夠用了。它把「送出」與「拿回」綁成一次
呼叫（`execute_v2` 阻塞），而且輸出寫在 backend 自己持有、下一次 forward 就地覆寫的
binding buffer 上——非同步之後那塊 buffer 會在還沒被讀走時就被下一批覆寫。兩件事都不是
加個參數能解決的。

## Decision

### 1. 正式推論路徑改吃 `services/trt_runner.py`，不再經 `AutoBackend`

`TrtRunner` 只做四件事：讀 ultralytics 檔頭之後 `deserialize_cuda_engine`、建一個
execution context、從 optimization profile 讀 `max_batch` 與輸入高寬、
`enqueue(im, out, stream)`（`execute_async_v3`，**不等它算完**）。輸出寫到哪一塊由呼叫端
指定，這正是 ping-pong 緩衝的前提。

檔頭的解析與 `services/engine_metadata.py` 共用同一支 `read_metadata_length`：引擎仍由
ultralytics 的 exporter 產出，格式不是我們定的，各寫一份只會漂移。

> **2026-08-31（[ADR-015](015-narrow-engine-profile.md)）**：引擎改由
> `tools/build_engine.py` 自建 builder 產出，檔頭也由 `engine_metadata` 自己寫，所以
> 「引擎仍由 ultralytics 的 exporter 產出」這句已不成立。**共用同一支
> `read_metadata_length` 的理由不變**，而且更強了——現在讀與寫都在同一個模組。

載入期四道 fail loud，判準一律取**引擎自己宣告的東西**，不是 JSON metadata（後者改了
不會改變引擎）：

1. I/O 張量必須恰好一入一出。多輸入只會綁到其中一個，其餘 binding 停在未設定的位址。
2. 輸入 binding 的 dtype 必須是 FP32。FP16 引擎的 I/O binding 本來就是 FP32；真的是別的
   dtype 代表送進去的位元組會被重新解讀，而輸出仍是形狀正常的框。
3. optimization profile 的 opt 高寬必須是 640×384。高寬是 dynamic axes，opt 不對的引擎
   照樣跑得完、框也對，只是所有 kernel 都是為別的尺寸挑的——**症狀只有變慢**。
   （**ADR-015 起改驗三個界**：min／opt／max 的空間維都必須是 640×384，batch 維必須是
   `1`–上界。收窄 profile 之後，「opt 對但 min／max 沒收窄」的舊引擎是同一類失效——
   跑得完、框也對，只是慢 2.3%～6.1% 且每個 execution context 多吃約 1 GB。）
4. 輸出 binding 的最後一維必須是 6（end2end）。這一項從 ADR-013 的「載入末端跑一次
   zeros forward 看輸出形狀」移到這裡：兩者的資訊來源是同一個（binding 宣告決定實跑的
   輸出形狀），但在這裡驗不必先配置緩衝，而且測得到——CI 上沒有 GPU 也沒有引擎，
   假的 `trt.Runtime` 就能覆蓋。

`enqueue` 的三個 TensorRT 呼叫的回傳值都要檢查：**`set_input_shape` 超出 profile 只回
`False` 不拋錯，接著 `execute_async_v3` 仍回 `True` 並用上一個 shape 跑**——輸出形狀正常、
內容是別批的。這三個 `if` 是那條路徑唯一的訊號。

### 2. 深度 2 的流水，每一塊緩衝的重用都由 event 明著守住

介面從 `preprocess` ＋ `infer` 改成 `submit` ＋ `collect`：

| 方法 | 回傳 | 做什麼 |
|---|---|---|
| `submit(frames)` | `int`（批次序號） | `stack_frames` → H2D → 前處理 → forward → D2H 全部排進 stream，**不等** |
| `collect()` | `(序號, 逐格框)` | 等最舊那一批的 D2H，過濾成逐格的 N×6 |
| `in_flight` | `int` | 已送出、還沒取回的批數 |

每一種緩衝有兩份，第 k 批用第 `k % 2` 份：

| 緩衝 | 位置 | 誰寫 | 誰讀 | 重用前要等 |
|---|---|---|---|---|
| `_pinned_in` | host pinned | `stack_frames`（CPU） | H2D（copy stream） | `_h2d_done` |
| `_dev_u8` | device | H2D | 前處理（compute stream） | `_fwd_done` |
| `_dev_out` | device | forward | D2H（copy stream） | `_d2h_done` |
| `_pinned_out` | host pinned | D2H | `collect`（CPU） | 深度上限本身 |

最後一列不必等 event：深度上限逼得第 k 批的 `submit` 之前必須先 `collect` 第 k−2 批，
而 `collect` 讀完就把結果複製成新陣列了（boolean mask 產生新陣列，與 buffer 無別名）。

**前三列（上一輪的緩衝重用）在現行迴圈下三條都是傳遞性可推的**：深度上限逼得第 k 批的
`submit` 之前必先 `collect` 第 k−2 批，那裡 host 同步過 `_d2h_done`，而 copy stream 是
循序的。仍然明著寫出來：漏一條不會報錯，只會讓某一批讀到已被覆寫的記憶體，而輸出檔完全
正常，深度、stream 數或收批時機一改傳遞性就沒了。

**真正 load-bearing 的是另外兩條**——本輪的跨 stream 生產／消費依賴：前處理讀
`_dev_u8[b]` 前等本輪的 `_h2d_done[b]`、D2H 讀 `_dev_out[b]` 前等本輪的 `_fwd_done[b]`。
`collect` 只同步得到上一輪，這兩條沒有任何東西可以替代。`submit` 的 docstring 把五個
同步點分成這兩類列出，日後稽核照那份清單對。

**`np.stack` 改成寫進常駐的 pinned buffer**（`stack_frames(frames, out)`）而不是每批配一塊
新陣列：pageable 記憶體的 `copy_(non_blocking=True)` 會被 CUDA 悄悄降級成同步複製，流水
直接退化回「host 做事時 GPU 閒著」。緩衝一律照引擎的 `max_batch` 配置、只寫前綴——湊不滿
批是常態，整塊覆寫會讓上一批殘留的影格被當成本批的資料。

**`preprocess_batch` 拆成 `stack_frames`（CPU）與 `to_infer_tensor`（GPU）**，GPU 那半的
語義逐字沿用 ADR-013 Decision 1（`permute` 排在通道索引之前、`.float()` 不是 `.half()`）。
`postprocess_batch` 完全不動。

### 3. 兩條 stream，都必須是非預設 stream

複製與運算各一條。`stream_handle=0` 會讓 TensorRT 插入 `cudaDeviceSynchronize`，流水退化
成序列而**輸出檔一模一樣、只有吞吐掉回改動前**，所以建構時與 `enqueue` 各擋一次。

H2D 每批 11.8 MB 約 1 ms，對 T4 每批 43 ms 的 forward 只佔 2%——單一 stream 也能拿到絕大
部分收益（host 與 GPU 的重疊不靠第二條 stream）。仍然做兩條：定序的複雜度只多兩個
`wait_event`，而 D2H 與下一批的 forward 重疊是免費的。

### 4. 單一 execution context、由呼叫端序列化呼叫

shape 與 binding 位址是在 `execute_async_v3` 當下擷取的，同一執行緒依序 enqueue 多批
（中間不同步）不在 TensorRT 的未定義行為條款內。反過來說「兩個 context 各綁一組 buffer」
才是錯的——同時在途的多個 context 必須各用一個 optimization profile，而本專案的引擎只有
一個。地端 5090 已實證：同一 context 連續 16→3→7→16→1 改 shape 與位址、中途不同步，
輸出與 `execute_v2` 逐位元相同。

### 5. 「哪一批的結果配到哪一批的影格」用序號核對，不靠 FIFO 默契

`submit` 回傳序號，`collect` 回傳序號，主迴圈的 `_pending` 與偵測器的在途佇列同進退，
不符即拋錯。

這是本次改動最危險的失效方式：第 k 批的框配到第 k+1 批的影格，**輸出檔完全正常**，每一
格的框都是真的，只有時間戳對到隔壁批；逐值比對也只表現為配對率下降，看不出是錯配還是
偵測變了。

主迴圈另外兩件事跟著改：沒有新影格時，有在途批就先 `collect`（不讓結果等到下一批湊成
才回收），所有路讀完後 drain 剩下的在途批再送 `TRACK_DONE`（漏 drain 就是 parquet 少
幾百格，而檔案完全正常）。

### 6. slot 歸還點不變，`detection_fps` 的口徑變了

歸還仍在「影格的像素複製走的當下」（現在是寫進 pinned buffer），呼叫端在 `submit` 回來
之後歸還。ADR-010 Decision 2、ADR-013 Decision 5 記的那個位置不變，變的只是複製的目的地。
**推論進程任一時刻仍只扣一批 slot**，所以 `RING_SLOTS` 的推導與係數不動。

`add_detection_time` 改為累計 `submit` 與 `collect` 兩段 wall time 的合計。`collect` 內等
GPU 的時間仍算在內，但**被 host 工作蓋掉的那段 GPU 時間不算**——`detection_fps` 因此從
「一批從送出到拿回的時間」變成「host 為偵測付出的時間」，數字不可與改動前直接相減。log
欄位與格式不變（`tools/bench_e2e.py` 讀的是 `overall_fps`，不受影響），README 註明口徑。

## Consequences

Positive

- host 的湊批、堆疊、過濾、送 payload 與 GPU 的 forward 重疊，`sm%` 的空檔由下一批填上。
- 輸出緩衝由呼叫端指定，ADR-013 Negative 記的「TRT backend 回傳可重用的 binding buffer，
  後處理延後就會拿到別批的輸出」這條限制消失——重用改由 event 與深度上限管。
- 每批 16 次隱含同步在 ADR-013 已收成 1 次，本次再把那 1 次從 `.cpu()`（等整個 stream）
  改成等單一 event，不會連帶等掉還在途的下一批。
- 引擎的 I/O 宣告有了自己的 fail loud，而且 CI 上測得到（假的 `trt.Runtime`）。

Negative

- **對 TensorRT Python API 的直接依賴**（`deserialize_cuda_engine`、`set_input_shape`、
  `set_tensor_address`、`execute_async_v3`）。ultralytics 那層原本吸收了版本差異（TRT 7-9
  的 binding API 對 10/11 的 named tensor API）；本模組只支援後者，升 TensorRT 時要看的
  是這四個呼叫。版本已 pin 在 `tensorrt-cu12==10.13.3.9`。
- **多了一類「錯批」的失效方式**：緩衝重用競態與序號錯配，兩者的症狀都是「輸出檔完全
  正常，只是某些格的框是別批的」。防線是 event、序號核對，與 GPU 上的定序測試（假 runner
  以 `torch.cuda._sleep` 放慢，讓競態可觀察）；CI 沒有 GPU，那支測試在 CI 上是 skip 的。
- **`detection_fps` 與改動前不可比**（見 Decision 6）。跨改動的比較只剩 `overall_fps`。
- **常駐緩衝多佔約 47 MB**（批次 16 下每份影格緩衝 11.8 MB，pinned 兩份、device 兩份；
  輸出側每份不到 0.12 MB），另有兩份在途的中間 f32 張量各 47 MB。相對於引擎本身的
  顯存不顯著。
- **`tools/build_engine.py`／`tools/compare_backend.py` 與正式路徑的距離又遠了一步**：
  它們仍走 ultralytics 的完整 `predict`。ADR-013 已記這件事，本次再加上「連 backend 都
  不是同一個」。要驗自建 runner，看的是 `test_trt_runner.py` 對 `AutoBackend.forward` 的
  逐位元比對與端到端的 parquet 比對。

## 驗收

- 同一顆引擎、同一批輸入，`TrtRunner` 的原始輸出與 `AutoBackend.forward` **逐位元相同**
  （`test_trt_runner.py`，地端 5090 實跑；CI 沒有 GPU 與引擎檔，該支 skip）。
- 地端 40 秒版與 2 分鐘版改動前後的 `tracking_results.parquet`，以 `(camera_id, timestamp)`
  對齊同一格後 `x1`/`y1`/`x2`/`y2`/`foot_x`/`foot_y` 六欄逐值相同。判準取最嚴的一種，理由
  同 ADR-013 Decision 6：引擎與每一步的算式都沒有變，這條改動沒有任何理由改變像素值或
  框的座標。
- T4 四象限的吞吐門檻與結論記在效能報告 5.1 D（不進版控）。

## Related

- ADR-010（推理主迴圈免複製消費共享記憶體）：Decision 2 的歸還點不變，複製的目的地改為
  pinned buffer。
- ADR-011（正式推論路徑只有 TensorRT 引擎一條）：本次不改變這條，只再一次改變取得
  backend 的方式。
- ADR-013（自建前後處理）：本次取代其 Decision 3（保留 `AutoBackend.forward`）、
  Decision 4（載入期以實跑輸出形狀驗 end2end）與 Decision 5 的介面表，並解掉它
  Negative 裡「binding buffer 會被下一次 forward 覆寫」那一條。
- ADR-012／PR #141（追蹤進程依攝影機分片）：分片讓追蹤側跟得上，本次才量得出推論側的
  改善。
