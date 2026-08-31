# ADR-013：推論進程自建前處理與後處理，ultralytics 只留 `AutoBackend` 的 forward

- 狀態：已採用
- 日期：2026-08-29
- 影響範圍：`video_analyze`（`services/detector.py`、`services/inference.py`、
  `services/track_worker.py`）

## Context

推論進程是這條 pipeline 的序列瓶頸。2026-08-28 在 T4 上逐格拆解，每格 4.24 ms 裡
preprocess 佔 1.06 ms（25%，在 CPU 上）、forward 2.70 ms（64%）、postprocess 0.48 ms
（11%）。

preprocess 那 25% 裡可以拿掉的**不是全部**——ultralytics `engine/predictor.py:154-176`
送上 GPU 的已經是 uint8，`.float()` 與 `/255` 本來就在 GPU 上做。真正可以省的是 CPU
側三趟整批記憶體搬動裡的兩趟：

1. `pre_transform` 的 LetterBox。影格在讀取端就縮成 384×640 了（issue #108），這個
   letterbox 在本專案是恆等（不 resize、四邊 padding 全 0），但 `data/augment.py` 的
   `apply_image` 對 3 通道**無條件**走 `cv2.copyMakeBorder`，照樣逐格複製一份。
2. `np.stack` 把整批堆成連續陣列（保留，H2D 要一塊連續來源）。
3. `[..., ::-1]`（BGR→RGB）＋ `transpose(0,3,1,2)` ＋ `np.ascontiguousarray`——一趟
   strided 複製，三者中局部性最差的一趟。

第 1 與第 3 趟都與九個解碼進程搶同一批 CPU 核。

postprocess 則是**逐格**在 GPU 上做 boolean index 再逐格 `.cpu()`：一批 16 格等於 16
次隱含同步，而實際搬的資料只有 16×300×6×4 ≈ 115 KB。

兩段都不是 GPU 算力的問題，是「工作放錯邊」與「同步次數過多」。要動它們就得繞過
`predictor.preprocess`／`postprocess`，而繞過的前提是把它們的語義逐行釘死——這正是本次
改動的風險所在：**前處理寫錯（例如送 FP16）或後處理的過濾順序寫錯，結果會安靜地變快
又變得不一樣，而輸出檔的欄位、列數、格式全部正常**。

## Decision

### 1. 前處理在 GPU 上做，且必須是 `.float()`

`preprocess_batch(frames, device)`：`np.stack` → 一次 H2D（uint8）→ `permute(0,3,1,2)`
→ `[:, [2,1,0]]` → `.float()` → `div_(255)`。

- **通道與軸的重排排在還是 uint8 的時候**：GPU 上搬一批 11.8 MB，轉成 float32 之後是
  47 MB。BGR→RGB 與 permute 的先後不影響任何元素的值。
- **`permute` 要排在通道索引之前**。`[..., [2,1,0]]` 再 `permute` 逐值相同，但結果
  **不是 contiguous**（本機實測 0.074 ms vs 0.079 ms，那點差距換不到補一趟
  `.contiguous()`），而 TensorRT 取的是 `im.data_ptr()`、不看 stride——非連續張量會被
  當成連續的讀進去，等於餵了一張錯位的影像。
- **是 `.float()` 不是 `.half()`**。FP16 引擎的 I/O binding 仍是 FP32
  （`trt_runner.py` 的 dtype 檢查與 `engine_metadata.validate_engine_precision` 的
  docstring 已記這件事），送 FP16 進去會變快又變得不一樣。
  `AutoBackend` 建構時傳的 `fp16=False` 不是這件事的執行機制：engine 這條路徑上
  `TensorRTBackend.load_model` 會先把 `self.fp16` 設回 False、再依 input binding 的 dtype
  覆寫，`AutoBackend.forward` 讀的是覆寫後的 `self.backend.fp16`，建構子傳進來的值用不到。
  填 False 只是不製造相反的訊號。

H2D 的資料量與 GPU 上的算式都沒有變，省下來的就是 Context 的第 1 與第 3 趟。

### 2. 後處理整批一次 D2H，之後用 numpy，順序不可調動

`postprocess_batch(raw, conf, classes, max_det)`：呼叫端先對整批輸出做一次
`.cpu().numpy()`（115 KB），再逐格套上 ultralytics `utils/nms.py` end2end 分支的三行
＋ `construct_result` 的 `scale_boxes`。三件事釘死：

1. **順序是 conf 過濾 → `[:max_det]` 截斷 → classes 過濾**，且 **conf 是嚴格大於**。
   把 classes 提到截斷之前，同一批輸入會多留下原本被擠出前 300 名的目標類別框。
2. **`scale_boxes` 這一步保留**。在本專案它退化成 clip：`orig_img` 就是推論尺度的
   影格，`gain = min(384/384, 640/640) = 1`、`pad = round(0/2 - 0.1) = 0`，等價於
   `clip_boxes(pred, (384, 640))`。成本近零，拿掉會讓超出邊界的框流到下游。
3. **`iou` 與 `agnostic_nms` 在 end2end 分支不被使用**（NMS 已經在引擎裡跑完），所以
   自建版本沒有它們的對應物。

`conf = 0.25`（`BasePredictor.__init__` 在 `args.conf is None` 時填入）與
`max_det = 300`（`cfg/default.yaml` 的預設）寫成 `services/detector.py` 的模組常數，
**刻意不進設定面**：它們是「與過去的輸出可比」的錨，不是可調參數。

boolean mask 產生的是新陣列，因此每格的 N×6 與整批那塊 buffer 無別名關係。

### 3. 直接建 `AutoBackend`，不經 `YOLO`／`predictor`

`YOLODetector.__init__` 的載入檢查全部保留、順序不變，末段的 `YOLO(str(path))` 換成
`AutoBackend(str(path), device=torch.device("cuda:0"), fp16=False)`。

- predictor 連同它的 `args`（`conf`／`iou`／`classes`／`max_det`）整組消失。留著會有
  一份「看起來還在生效、其實我們自己算」的參數面。
- ADR-010 Decision 4 明寫「這道保護擋不住 ultralytics 內部的活別名」——
  `predictor.dataset.im0` 與 `predictor.batch[1]` 存的就是我們傳進去的 view，要到下一
  次 `predict` 才被替換。不再呼叫 `predict`，那兩個別名一併消失。
- 保留 `AutoBackend.forward`（而不是直接打 TRT backend）是為了留住它對 backend `fp16`
  與多輸出的處理。

代價是引擎 deserialize 從第一批推論提前到 `__init__`。兩者都在推論子進程內，
`pipeline.py` 的錯誤路徑不變，而提前等於失敗得更早。

> **（issue #147 後取代）`AutoBackend` 也拿掉了。** `AutoBackend.forward` 走
> `execute_v2`（阻塞到 GPU 算完）、輸出寫在它自己持有的 binding buffer 上，兩件事都讓
> host 與 GPU 無法重疊。正式路徑改吃 `services/trt_runner.py`（`execute_async_v3`、
> 輸出寫到呼叫端指定的緩衝），見 ADR-014 Decision 1。本節保留 `AutoBackend` 的理由
> （「留住它對 backend `fp16` 與多輸出的處理」）在自建 runner 裡由兩道載入期檢查取代：
> 輸入 binding 的 dtype 必須是 FP32、I/O 張量必須恰好一入一出。

### 4. 載入期以實跑的輸出形狀驗 end2end

`__init__` 末端跑一次 zeros forward（批次取 `min([model].batch, 引擎的 max batch)`），
既是 warmup，也是 fail loud：輸出不是單一張量、或最後一維不是 6 就拋錯。

**判準用實際跑出來的形狀，不讀 metadata 的 `end2end` 欄位**：後者是匯出時寫進檔頭的
一個值，改它不會改變引擎，而這裡要驗的正是引擎本身。沒有內建 NMS 的引擎吐的是
`(B, 4 + nc, num_anchors)`，Decision 2 的三行照樣跑得完——只是把某個類別分數當成 conf、
另一個當成 cls，得到一堆座標是 xywh 的框，而列數、欄位、格式全部正常。

批次取 `min` 是為了不搶走 `InferencePipeline` 在 `start_loop` 開頭的批次上限檢查：那裡
的訊息指得出要改哪一邊（`[model].batch` 還是引擎），warmup 先炸只會得到 ultralytics 的
assert。

### 5. 介面拆成 `preprocess` ＋ `infer`，slot 歸還點前移到兩者之間

| 方法 | 回傳 | 用途 |
|---|---|---|
| `preprocess(frames)` | `torch.Tensor` | 整批 `np.stack` ＋ H2D ＋ GPU 前處理 |
| `infer(im)` | `list[np.ndarray]` | forward ＋ 整批 D2H ＋ numpy 後處理 |

主迴圈因此是「前處理 → **歸還 slot** → forward ＋ 後處理 → 送 payload」。像素在
`np.stack` 當下就已經複製進新的 CPU 陣列，`preprocess` 回傳時 H2D 也已完成（pageable
記憶體的 `.to()` 是同步的），所以歸還點可以從 ADR-010 Decision 2 的「`predict` 回來」
提前到「`preprocess` 回來」——不需要 `cuda.synchronize()` 也不需要 event。這讓 reader
早一整段 forward 的時間拿回空位。

**`predict` 不保留為兩者的合成方法**：兩種取用模式並存會讓人以為可以互換，而歸還點
前移的整個前提正是「不可互換」（與 ADR-010 Decision 1 拿掉 `read_slot` 同一個理由）。
誤用由 `infer` 入口的張量檢查擋下——dtype 不是 float32、不是 contiguous、不是 4D、不在
CUDA 上，四項都是「跑得完但結果不對」而不會自己拋錯的情況。高寬的核對
（`_check_infer_shape`）也從「讀 TRT backend 的 `bindings["images"]`」改成驗我們自己
組出來、即將交給 `execute_v2` 的那個張量，語義更直接；「同一個形狀只記一次 log」的
行為保留。

`Results` 消失後，ADR-010 Decision 4 的「歸還前把 `result.orig_img` 設為 None」失去
對象，主迴圈只剩 `packet.frame = None` 那半條。這**不是保護變弱**：那條保護防的是
「日後有人在歸還之後讀影格」，而現在推論輸出根本不再攜帶影格參照。ADR-010 的
Decision 2、4、6 已就地補註。

### 6. 驗收判準是「逐值相同」，不是「差異不大」

論據是「uint8 → float32 → `/255` 每個元素獨立，GPU 與 CPU 都 round-to-nearest」，而
forward 與引擎完全沒有變。若實測有差，要查的是寫成了 `.half()`、或過濾順序被調動，
**不可放寬判準**——這條改動沒有任何理由改變像素值。

這與根 `CLAUDE.md` 記的「改變送進模型的畫面時改用輸入擾動當參照組」不衝突：那條講的是
換解碼器、改縮放或色彩轉換路徑那類**必然**改變像素的改動；本次的每一步都有逐值相同的
論證，所以判準回到最嚴的那一種。

實測（2026-08-29，地端 5090、`bucket_20260801_perf40` 九路 40 秒版、同一顆引擎）：

- `compare_tracking.py`：配對率 100%、偏差 p50／p99／p99.9／最大值皆 0.0、逐格偵測數
  0 差、偏差 > 50 px 的畫面 0 個。
- `compare_tracking_exact.py`（本次新增，補逐列相等）：17,884 格、143,729 列，
  `x1`/`y1`/`x2`/`y2`/`foot_x`/`foot_y` 六欄**全部逐值相同**。
- 吞吐：`detection_fps` 864.24 → 1301.12（＋51%），`overall_fps` 763.57 → 869.55
  （＋13.9%）。地端不設門檻——射程是在 T4 上量的，這裡只驗方向與逐值相同。

## Consequences

Positive

- 前處理的兩趟 CPU 複製消失（不是搬到別處），後處理的每批 16 次隱含同步收成 1 次。
- 正式推論路徑不再依賴 ultralytics 的 predictor，連帶消滅 ADR-010 Decision 4 記的
  「擋不住的 ultralytics 內部活別名」。
- slot 歸還點前移一整段 forward，reader 的空等變短。
- 前後處理成為模組層純函式，**測試在 CPU 上跑**，不需要 GPU 也不需要引擎。

Negative

- **對 ultralytics 內部的依賴比改動前更深**：`AutoBackend` 的建構子、TRT backend 的
  輸出約定、end2end 分支的過濾語義。版本已 pin 在 8.4.75；升級時要重跑逐值比對，
  不能只看測試綠。
- **TRT backend 的 forward 回傳的是可重用的 binding buffer**
  （`nn/backends/tensorrt.py` 的 `return [self.bindings[x].data for x in ...]`），下一
  次 forward 就地覆寫。本次的後處理立刻 `.cpu()` 複製走，安全；日後若把後處理延後到
  下一批之後，會靜默拿到別批的輸出。這條寫在 `infer` 的 docstring 裡。
- **`tools/build_engine.py`／`tools/compare_backend.py` 從此不再等同正式路徑**。兩支
  工具不隨 wheel 出貨，仍走 ultralytics 的完整 `predict`；`compare_backend.py` 比的是
  「Torch FP32 對 TensorRT FP16」，那個比較在改動後不再涵蓋本套件實際跑的前後處理。
  要驗前後處理，看的是 `test_detector.py` 的逐值比對與端到端的 parquet 比對。
- **本次的吞吐增益在九路全開時可能被追蹤進程蓋住**（追蹤側已量到會先撞到上限），所以
  逐格時間的下降未必等比反映在端到端；判讀 T4 那批時要同時看 `tracking_fps` 與逐片
  `tracking_fps ÷ overall_fps`。

## Related

- ADR-010（推理主迴圈免複製消費共享記憶體）——本次修訂其 Decision 2 與 4。
- ADR-011（正式推論路徑只有 TensorRT 引擎一條）——本次不改變這條，只改變取得 backend
  的方式。
- ADR-012／PR #141（追蹤進程依攝影機分片）：本線的分支基準。
- issue #108（影格縮放移到讀取端）：讓 `pre_transform` 的 LetterBox 變成恆等，是本次
  能整段拿掉前處理的前提。
