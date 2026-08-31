# ADR-015：建置工具接管 TensorRT builder，optimization profile 的空間維釘成 384×640

- 狀態：已採用
- 日期：2026-08-31
- 影響範圍：`video_analyze`（`tools/build_engine.py`、`tools/compare_backend.py`、
  `services/engine_metadata.py`、`services/trt_runner.py`）

## Context

`tools/build_engine.py` 原本走 ultralytics 的引擎匯出（`model.export(format="engine")`
→ `utils/export/engine.py::onnx2engine`），optimization profile 由 ultralytics 訂成

```
min=(1, 3,  32,  32)   opt=(16, 3, 384, 640)   max=(16, 3, 768, 1280)
```

上下界都不是誰選的：

- **上界是 `workspace` 的副產物**。`onnx2engine` 算的是
  `max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))`，
  倍數下限就是 2，所以 max 恆為 opt 的兩倍以上。傳 `workspace` 改不出 1.0，而且那個
  參數同時經 `set_memory_pool_limit` 當記憶體上限。
- **下界是一個永遠跑不起來的形狀**。32×32 的輸入不到 300 個 anchor，end2end head 的
  `TopK(K=300)` 在下界形狀上不成立——TensorRT 建置期就明著警告
  `TopK: length of reduction axis is smaller than K (300)`（5090 與 T4 都複現）。

而執行期的高寬只有一種：影格在讀取端就縮成 384×640，`trt_runner._check_infer_shape_of`
對任何別的高寬 fail loud。所以這兩個界不對應任何實際會送進去的形狀，只是在擴大
TensorRT autotuning 要涵蓋的範圍。

## Decision

### 1. `build_engine.py` 自己建 builder／network／profile／config，不再走 ultralytics 的引擎匯出

ONNX 中繼檔仍由 ultralytics 匯出（`format="onnx"`、`half=False`、`dynamic=True`、
`simplify=True`、`imgsz=(384, 640)`），之後的 builder 全部在
`build_engine.build_serialized_engine`：FP16 builder flag、單一 optimization profile、
`build_serialized_network`。設定刻意複製 `onnx2engine` 在本工具的呼叫下實際生效的那一組，
只改 profile 一項。

**`half=False` 是必要的。** 在 `format="engine"` 下 `half=True` 只作用於 builder flag
（`exporter.py` 只對 `fmt in {onnx, torchscript}` 呼叫 `model.half()`），所以 ultralytics
的中繼 ONNX 本來就是 FP32；改走 `format="onnx"` 之後同一個參數會真的把模型 `.half()`，
得到 FP16 輸入 binding 的另一種引擎——2026-08-30 在 5090 上量過，那顆是 0.987×（更慢）。

**不設 `set_memory_pool_limit`。** 這與 ultralytics 現況一致（`workspace` 預設 `None` →
整段略過，TensorRT 的預設上限是整張卡可用記憶體），而且是刻意的：workspace 上限同時是
**tactic 的篩選條件**，設一個保守常數會排除吃 workspace 的 tactic，而「解鎖 tactic」正是
收窄 profile 量到增益的機制。T4 是 16 GB、5090 是 32 GB，預設值本來就不同——這也是倍數
必須在 T4 重量一次的理由之一。

`profiling_verbosity` 取 `DETAILED`（ultralytics 在 FP16 路徑上不動它，預設是
`LAYER_NAMES_ONLY`）：與相位 1 量測的那批引擎同一組設定，並讓日後的逐層 profiling 拿得到
層名以外的資訊。代價是引擎檔略大，不影響 tactic 選擇。

### 2. profile 的空間維三個界全部釘成 384×640，batch 維維持 1–`--batch`

```
min=(1, 3, 384, 640)   opt=(batch, 3, 384, 640)   max=(batch, 3, 384, 640)
```

組裝在 `build_engine.profile_shapes`（純函式），判準在
`trt_runner.check_profile_shapes`（載入端與建置端共用同一支）。

**batch 維不收窄**：湊不滿批是常態而非例外——T4（與正式節點同機型）上實測一次跑完出現
1 到 16 全部 16 種批次大小，8 核的解碼餵不滿批。下界訂在 1 不是保守，是必要條件。

### 3. 引擎檔頭由 `services/engine_metadata.py` 寫

接管 builder 之後，**寫**檔頭這件事第一次進到我們自己手上。三支新函式與
`read_engine_metadata`／`read_metadata_length` 同檔：

- `build_engine_metadata`：把 ONNX 匯出當下攔下的 `exporter.metadata` 改寫成引擎檔頭
  該有的樣子。模型衍生欄位（`stride`／`task`／`names`／`channels`／`end2end`／`imgsz`／
  `description`）一個都不自己重推——那些值算錯的症狀全部是靜默的。只動兩件事：裁掉
  `args` 裡的 `opset`（`format="onnx"` 與 `format="engine"` 的 `fmt_keys` 就差這一個），
  以及把 `args["half"]` 改寫成 `True`（引擎的 FP16 由 builder flag 給，與 ultralytics
  的引擎匯出路徑同義）。
- `validate_engine_header`：必填鍵存在、型別正確，`end2end` 還必須為真。
- `write_engine_with_metadata`：4 bytes little-endian 的長度 ＋ JSON ＋ 引擎位元組。
  **長度寫的是 payload 的位元組數**，不是 `json.dumps` 的字元數——ultralytics 寫的是後者，
  只有在它預設的 `ensure_ascii=True` 下兩者才相等，而本 repo 別處一律
  `ensure_ascii=False`。

### 4. 載入端擋下沒有收窄過的引擎

`trt_runner.check_profile_shapes` 驗三個界，不是只驗 opt。**這會擋下改動之前建的每一顆
引擎**，是刻意的：舊引擎的 opt 高寬照樣是 384×640，會通過其餘全部載入檢查，症狀只有
「forward 慢幾個百分點、每個 execution context 多吃約 1 GB 裝置記憶體」——與同一個
`__init__` 裡別的檢查擋的是同一類失效（引擎跑得完、輸出檔完全正常，只有成本不對）。

**後果是升版順序有向的**：先在目標卡上用新的 `build_engine.py` 重建引擎並發布，才能滾動
映像檔。與 `validate_engine_metadata` 裡 TensorRT 升版那條同型。

### 5. 兩支工具的每一次 `predict` 都要帶 `imgsz=(384, 640)`

`verify_engine` 的 probe 與 `compare_backend` 的 `build_model`／`detect` 三處。理由是
ultralytics predictor 的 **warmup 不走 letterbox**，直接用 `args.imgsz`
（`engine/predictor.py` 的 `self.model.warmup(imgsz=(bs, ch, *self.imgsz))`），而
`setup_model` 只在 backend **不是** dynamic 時才把檔頭的 `imgsz` 抄進 `args.imgsz`——
所以不指定就是預設的 640×640。

收窄之前 640×640 落在 max 界（768×1280）內，這件事沒有症狀。收窄之後它出界，而
`nn/backends/tensorrt.py` 呼叫 `set_input_shape` **不檢查回傳值**，接著就地把
`bindings["images"]` 換成 `im.shape`（於是下一行的 `assert im.shape == s` 必定通過），
最後以 context 上一個 shape 執行。`compare_backend` 逐張送（batch 1），warmup 緩衝只有
1×3×640×640×4 而引擎照 16×3×384×640 讀——**越界讀取**，最好的情況是拿到垃圾，最差是
`CUDA illegal memory access` 毒掉整個 process，而發生時機在約 7 分鐘的建置之後。

帶上 384×640 之後 warmup 與 predict 同形狀。對 16:9 來源這是**行為保持**的改法：
`new_shape` 640×640 與 384×640 在 `auto=True` 的 letterbox 下都收斂到 384×640。

## Consequences

**取得的**（T4 相位 1，2026-08-31）：

- forward 快 **+2.3%～+6.1%**（族群內比較的區間，批次 16／8／4／2 各一格）。每個批次
  的收窄引擎對重建引擎逐對比較零例外：三輪六顆版 24 對全勝，五輪五顆版再 16 對全勝。
- execution context 的裝置記憶體 **1,329 MB → 332 MB**。這是確定性的量，不吃 tactic 噪音。
- `min=(1,3,32,32)` 的 `TopK(K=300)` 建置期警告消失（那是 profile 本身不成立，與快慢無關）。

**放棄的**：

- **ultralytics 的引擎匯出路徑不再是備援。** TensorRT 升版時 profile、精度 flag、記憶體
  上限、`profiling_verbosity` 都要自己維護；`onnx2engine` 日後改了什麼，我們不會自動吃到。
  方向與 ADR-014（自建 runner）一致——正式路徑上 ultralytics 只剩 ONNX 匯出與 ByteTrack。
- **既有的 sm75／sm120 兩顆引擎都要重建**（見 Decision 4 的升版順序）。
- 多 optimization profile（逐批次大小給 opt shape）沒有做，排在本項之後另評估。

**統計上的限制，照實記**：

- 相位 1 的收窄 n=2、重建 n=3，四個批次共用同一批引擎（不是四份獨立證據），引擎層級的
  排列檢定約 **p ≈ 0.1**——那是 3 vs 2 這個樣本數的下限，不是估出來的值。
- **batch 16 是最窄的一格**（分離度 +2.30%／+2.40%），而重建族群三顆的全距是 2.83%。
- 判準 (b)（增益要大於 2× 噪音）**照原文在 batch 2 未通過**，成因是它的分母取了跨族群的
  正式引擎（ultralytics 匯出路徑、另一份獨立匯出的 ONNX、`profiling_verbosity` 不同），
  量到的是「建置路徑差 ＋ tactic 噪音」的混合。族群內重算後 batch 2 反而是四格裡訊號最
  乾淨的（+5.42%／+6.13%）。這是以指揮線身分覆蓋 (b) 的裁定，不是 (b) 通過了。
- 地端 5090 是另一個架構、另一組建置抽樣（重建噪音 ±1.2%），方向一致、+2.6%～4.0%。

## 被否決的替代方案

- **傳 `workspace` 把 max 界壓下來**：倍數下限是 2，改不出 1.0；而且那個參數同時是記憶體
  上限，會篩掉 tactic——正好抵銷收益來源。
- **只收窄 max、留著 `min=(1,3,32,32)`**：下界是那個 `TopK(K=300)` 警告的來源，留著等於
  留一個永遠跑不起來的形狀在限制 tactic 的選擇範圍。
- **把 profile 三個界寫進 `vfa` 檔頭、schema 進位到 3**（取代 Decision 4）：檔頭是可以被
  改寫的宣告，而 `get_tensor_profile_shape` 讀的是引擎自己說的。既然兩者都會擋下既有
  引擎、代價相同，取後者。
- **不擋舊引擎，只在 README 寫「記得重建」**：症狀只有變慢與多吃顯存，正是這份程式碼在
  別處刻意擋掉的那一類失效。

## 驗收

- `video_analyze/tests/test_build_engine.py`：`profile_shapes` 的三個界，以及它的輸出
  餵得過載入端的 `check_profile_shapes`。
- `video_analyze/tests/test_trt_runner.py`：沒收窄過的舊 profile 要被擋下、batch 下界不是
  1 要被擋下、opt batch 不等於上界要被擋下。
- `video_analyze/tests/test_engine_metadata.py`：檔頭組裝（裁 `opset`、`half` 改寫、模型
  衍生欄位照抄）、必填鍵與 `end2end` 的把關、含非 ASCII 值的檔頭往返。
- 建置期驗收（需要 GPU）：`build_engine.py` 跑完四道既有門檻，並以引擎自己宣告的 profile
  確認三個界（`verify_engine` 建一顆 `TrtRunner`）。

## Related

- ADR-011（正式產品只有一套 inference implementation）——本 ADR **修訂**它 Decision 3
  的「代價」那段：那 1.3 GB 顯存與 2× 的 max 形狀是 `workspace` 當倍數的副產物，不是
  dynamic 逼出來的。Decision 7 的執行期形狀核對不變，而且正是收窄的前提。
- ADR-013（自建前處理與後處理）、ADR-014（自建 TensorRT runner）——同一個方向的前兩步。
- 相位 1 的完整數字與判準逐條結果：`outputs/vfa_perf/docs/report.md` 5.4 #23（不進版控）。
