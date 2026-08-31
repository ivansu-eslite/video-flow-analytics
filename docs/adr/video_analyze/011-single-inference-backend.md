# ADR-011: 正式推論路徑收斂為單一實作（TensorRT FP16），Torch FP32 只作為套件外的驗證工具

## Status

Accepted

## Context

`video_analyze` 原本唯一的推論路徑是 Torch FP32（`YOLODetector` 載 `.pt`、`.to("cuda")`、
`predict`）。效能量測的結論是 TensorRT FP16 為最終最佳方案，而使用者裁定：**正式產品只有
一套 inference implementation**。

「換一個檔名」不足以描述這件事，因為 **TensorRT 引擎不是原始碼，是產物**：

- 建置時會對**實際裝置**做 kernel autotuning，引擎因此綁在該 SM 上。`IBuilderConfig`
  沒有指定 target SM 的 API，`HardwareCompatibilityLevel` 只有 `NONE`／`AMPERE_PLUS`／
  `SAME_COMPUTE_CAPABILITY`，而 `AMPERE_PLUS` **不含 Turing**。所以「在開發機建出雲端要
  用的引擎」是封死的，不是效率問題。部署 GPU 實測為 T4（Turing sm75，84/84 筆 CustomJob），
  開發機是 RTX 5090（sm120）——**兩顆引擎、兩台機器、一支工具**。
- 引擎跨版本不保證能 deserialize，精度也烤在裡面。「跑的是哪一顆」這件事在執行期看不出來：
  載得起來、跑得出結果、parquet 的欄位與列數全部正常，只有數值或成本不對。

同時，這次收斂會**順手移走三處既有防護**，每一處都得各自處置（issue #108 的教訓：移走一段
程式碼前先確認它有沒有在擋什麼）：`.to(device)`（引擎不是 PyTorch module，呼叫就崩）、
CPU fallback（引擎路徑下從「慢」變成「一定失敗」）、`_validate_imgsz`（引擎路徑下**語義
失效**）。

## Options Considered

### Option A：維持 Torch FP32 唯一實作

零風險。但既有量測顯示引擎路徑是目前唯一能把一天九路壓進時窗的方案，維持現狀等於放棄整串
優化（#108／#109／本 ADR）的收尾。排除。

### Option B：把後端做成可切換的設定項（`backend = "torch" | "tensorrt"`）

直覺上「保留退路」。實際代價：

- `YOLODetector` 內每一道檢查都要分岔（`.to()` 要不要呼叫、精度怎麼驗、形狀怎麼驗），
  而兩條分支的驗證強度不同——Torch 那條驗不到引擎的身分，引擎那條驗不到權重的 `imgsz`。
  分岔的檢查是**測不到的檢查**：正式環境只跑其中一條，另一條的迴歸只有它自己的單元測試在盯。
- 「退路」在實際失效模式下派不上用場。引擎載不起來的原因（SM 不符、TRT 版本不符、檔案不在）
  都是部署環境的問題，退回 Torch FP32 會讓那一天以**十倍的時間**跑完而沒有人發現配置壞了
  ——把一個會當場失敗的錯誤變成一個只表現為「今天怎麼特別慢」的錯誤。
- 精度變回設定項。引擎的精度是烤進二進位的，設定項與實際值可以不一致而無訊號。

排除。這是本 ADR 要記下來的核心約束。

### Option C：單一引擎實作，Torch FP32 活在套件外的工具裡（採用）

`YOLODetector` 只載 `.engine`，非引擎檔即中止；Torch FP32 只出現在
`video_analyze/tools/build_engine.py`（建置的來源）與 `video_analyze/tools/compare_backend.py`
（驗收的基準），兩支自己載模型、不經過 `YOLODetector`。

### Option D：引擎進版控／烤進映像檔

引擎是幾百 MB 的二進位、且每張卡一顆、每次升 TRT 全部作廢——比 `.pt` 更不該進版控。烤進
映像檔則讓「升引擎」與「滾映像檔」綁死，而**升版順序是有向的**（先建新引擎並發布，才能滾
映像檔；反過來會讓正式環境啟動失敗），綁死會讓那個順序無法執行。排除；引擎走 artifact。

## Decision

### 1. 正式套件內沒有 Torch 權重推論路徑

`YOLODetector.__init__` 的第一道檢查就是副檔名必須是 `.engine`。精度不再是設定項——它由
引擎自帶，並在載入時驗證。

### 2. 分界是「不在 `YOLODetector` 的推論路徑上」，不是「檔案不在套件目錄裡」

建置工具與比對工具放在 `video_analyze/tools/`（跟著套件走，才能 import `settings` 與
`INFER_HEIGHT`／`INFER_WIDTH`，建置參數與執行參數不會各自漂移）。這讓套件資料夾底下出現
載 `.pt` 的程式碼，與「套件內沒有 Torch 推論路徑」的字面表述有張力，因此把分界寫清楚：

- **`tools/` 在 `src/` 之外**。`pyproject.toml` 用 uv_build 的 src layout，只打包
  `src/video_analyze`，所以那兩支**不隨 wheel 出貨**——正式容器裡不存在這些檔案。
- 它們**不經過 `YOLODetector`**，自己 `YOLO(...)`、自己決定裝置與精度。推論路徑上的實作
  仍然只有一套。

日後要放第三支工具進 `tools/` 時，這條分界是判準：它可以載 `.pt`，但不可以被
`src/video_analyze` 底下的任何模組 import。

### 3. `dynamic=True` 不是彈性需求，是批次逼出來的

靜態引擎的 `TensorRTBackend.forward` 對輸入形狀是 `assert im.shape == s`，而**湊不滿批
是常態而非例外**：T4（n1-standard-8，與正式節點同機型）上實測一次跑完出現了 **16 種**
不同的批次大小，1 到 16 全都有——8 核的解碼餵不滿 GPU，實際批次由供料決定。靜態引擎會
在第一個不滿批就當場失敗。

⚠ **一個規劃階段寫錯、實作時才查清楚的理由**：本 ADR 的草稿曾把「靜態引擎會讓形狀退回
640×640」也列為 dynamic 的理由。那是錯的——`predictor.py` 的 `setup_model` 有
`if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
self.args.imgsz = self.model.imgsz`，靜態引擎反而**會**套用 metadata 的 `imgsz`
（也就是 384×640），`auto=False` 的 LetterBox 填到那個尺寸，形狀是對的。

真正的因果是反過來的：**因為選了 dynamic，形狀才不再被 metadata 釘住**。dynamic 引擎
拿不到那行的套用，`args.imgsz` 停在預設的 640，實際形狀改由 `pre_transform` 的 `auto`
決定；`auto` 的條件是 `same_shapes and args.rect and (format == "pt" or dynamic)`，三者
同時成立才保住 640×384。這正是 Decision 7 那道執行期形狀核對存在的理由。

代價：dynamic 引擎的 optimization profile 把 max 形狀取到 2×（`(batch, 3, 768, 1280)`），
建 execution context 實測吃約 1.3 GB 顯存（規劃階段估的是 4.7 GB，偏保守），且不隨實際
批次縮小。T4 有 15 GB，容得下。

> **這段代價已由 [ADR-015](015-narrow-engine-profile.md) 消掉（2026-08-31）。** 那個 2×
> 的 max 是 ultralytics 拿 `workspace` 當倍數的副產物，不是 dynamic 逼出來的——batch 維
> 動態與空間維動態是兩件事。`build_engine.py` 接管 builder 之後，空間維三個界都釘在
> 384×640（batch 維仍是 1–max），execution context 的裝置記憶體降到約 0.33 GB。本節
> 「dynamic ⇒ 顯存代價」的因果不再成立；Decision 7 那道執行期形狀核對仍然成立，而且
> 正是收窄的前提。

### 4. 精度驗 `metadata["args"]["half"]`，不是 backend 的 `fp16` 屬性

**FP16 引擎的 I/O binding 仍是 FP32**，`AutoBackend` 對 FP16 引擎永遠回報 `fp16 = False`。
拿那個值當判準只有兩種結局：永遠失敗，或被拿掉。`args` 是 exporter 把匯出參數原樣寫進
metadata 的那一份，只有它反映建置時的實際設定。

### 5. 注入的 metadata 收在單一 `vfa` key 底下

公開的 `export()` 沒有任意欄位入口，注入走 `on_export_start` callback 改 `exporter.metadata`
（`Exporter.__call__` 先組完 `self.metadata` 才跑該 callback）。讀回端
（`nn/backends/tensorrt.py` 的 `apply_metadata`）會把**每一個 key `setattr` 到 backend 上**，
所以不攤成數個頂層 key——攤開要逐一避開 `fp16`／`dynamic`／`model`／`context`／`bindings`／
`output_names`，而其中幾個是在 `apply_metadata` **之後**才賦值的，注入的值會被無聲蓋掉，
比對於是拿到 backend 自己的值而不是引擎的值。

容器 key 之外另帶 `schema` 版本號：欄位語義改版時舊引擎要被擋下，而不是以「缺欄位」的形式
靜默通過。

### 6. 三項不符即中止（訊息各自寫），驅動版本只記錄

中止：**compute capability**（不是 GPU 型號字串——引擎的實際約束是 SM）、**TensorRT 版本**、
**TensorRT wheel 變體**（`tensorrt-cu12` 與 `tensorrt-cu13` 是不同套件；比的是
`importlib.metadata` 讀到的**實際套件名**，不是 `torch.version.cuda`——那兩個數字可以不一樣，
本機現況就是 torch `2.12.1+cu130` 搭 `tensorrt-cu12`，拿 torch 的 CUDA 建置版本當變體的代理
既擋不到真正的 cu12／cu13 對調，又會在 torch 換版時誤擋既有引擎。**任一端讀不到套件名時
只記 warning 不擋**：TensorRT 不是用 wheel 裝的話 runtime 完全正常卻讀不到名字，
「測不出來」不等於「不相容」），以及**來源權重的 SHA-256**
（`[model].source_weights_sha256` 有釘才比；沒釘只記 warning，讓「沒在驗」在 log 上看得見）。
四條各自拋、訊息各自寫，因為處置完全不同：SM 不符要在目標卡上重建，TRT 版本不符要對齊
映像檔，權重 hash 不符是拿錯了引擎。合成一句「metadata 與環境不符」會讓看 log 的人得自己
重跑一次才知道要修哪裡。

**驅動版本與 torch 的 CUDA 建置版本只記錄、不擋**，前者是本 ADR 對 plan 的一處刻意偏離。引擎的硬約束來自 TensorRT
本身（SM 與 TRT 版本），驅動不在其中；而建置機（GCE T4 VM）與正式節點（Vertex CustomJob
的容器）本來就不保證同一份映像檔與驅動——把它做成致命條件等於堵死唯一可行的建置路徑。
不符時記 warning，留給人判斷。

### 7. `_validate_imgsz` 不照抄，改成「建置期固定 ＋ 執行期逐批核對實際形狀」

`_validate_imgsz`（#108 加的）驗的是權重帶的 `imgsz` 是否等於 `INFER_WIDTH`，防「影格被
縮小再放大回去、召回率靜默下降」。引擎路徑下它**語義失效**：dynamic 引擎不套用 metadata
的 `imgsz`，實際形狀由 `pre_transform` 決定。照抄會得到一個**看起來有在驗、其實驗不到**的
檢查——比沒有更危險，因為它會讓人以為這件事已經被盯住了。

改成兩半：

- **建置期固定**：`tools/build_engine.py` 以 `imgsz=(INFER_HEIGHT, INFER_WIDTH)` 匯出，
  並在產出前用滿批跑一次、核對實際形狀。
- **執行期逐批核對**：`YOLODetector._check_infer_shape` 讀 TensorRT backend **自己的**
  `bindings["images"].shape`（`forward` 每次形狀變動都會更新它，那就是交給
  `context.execute_v2` 的形狀），高寬不是推論尺寸即拋錯；同一個形狀只記一次 log。
  刻意不從輸入影格重新推導一次——那只是把 `pre_transform` 的邏輯抄第二遍，抄錯了兩邊會
  一起錯而檢查照樣通過。它擋的是 Decision 3 那三個 `auto` 條件日後有任何一項翻掉
  （ultralytics 改 `rect` 預設、批次混進不同尺寸的來源）——那時影格會被填到 640×640，
  像素量 1.67 倍，而症狀只有「變慢」。

**`dynamic` 本身也在載入時驗**（`_validate_dynamic`）。少了它，靜態引擎會通過其餘全部
載入檢查，然後在第一個不滿批被 ultralytics 的 assert 擋下，訊息只講「input size 不等於
max model size」；而接手的形狀核對排在 `predict` **之後**，永遠輪不到。

這裡比 plan 多做了一步：plan 寫的是「逐批**記錄**」，實作改成「記錄**並擋下**」。理由是這
個形狀完全由三件不會在執行期變動的事決定（讀取端的輸出尺寸、引擎是不是 dynamic、
ultralytics 的 `pre_transform`），任何偏離都代表環境或版本變了；而它會在第一批就發生，
不是跑到半天才中止。

### 8. CPU fallback 改 fail loud

引擎綁 GPU，CPU 上不是「比較慢」而是一定失敗。留著 fallback 只是把同一個錯誤延後到第一次
predict。

### 9. 引擎檔名帶 SM，引擎不進版控

檔名格式 `<權重 stem>_sm<SM>.engine`（如 `..._sm75.engine`）。理由不只是「好認」：下游
argus 的 promotion 用 `Path(source_model_uri).stem` 當 `model_version`，同一份權重的 T4 與
5090 兩顆引擎**同名會直接撞在一起**。這件事在 vfa 這側就處理，否則同步 argus 那一步要回頭改。

引擎走 artifact，不進版控（`.gitignore` 一併加上 `*.engine` 與匯出中繼的 `*.onnx`）。

### 10. 比對沒過就沒有產物

`build_engine.py` 先把產物落在 `.unverified` 尾綴上，跑完 metadata／精度／形狀驗證與對
Torch FP32 的比對才改成正式檔名；任何一關沒過就把它刪掉。`--skip-compare`（T4 smoke 用）
**也不改名**——只印警告的話磁碟上會留下一顆與通過比對的產物長得一模一樣的引擎，事後無從
分辨；留著尾綴同時讓它載不進正式推論路徑（`YOLODetector` 要求副檔名是 `.engine`）。**不用 `.tmp` 尾綴**——那在本
repo 已經是 `TrackingResultCollector` 的暫存檔語義。

**比對的判準看 p99，不看 max**，這是對 plan 的第二處刻意偏離。plan 寫的是「偏差不超過
既有基準（1.20 px）」，但那兩個既有數字的統計量不同：FP16 的 1.20 px 是 **277 個框的
max**（Torch FP32 對 Torch FP16），TensorRT 已驗的 1.16 px 是 **277 個框的 p99**。而
`max` 是樣本數的函數——本工具的預設取樣是 2000 個框以上，拿 277 個框的 max 當門檻是在比
兩個不同的統計量。實測也印證了這件事：本次 2045 個配對的 p99 是 **1.16 px**（與 T4 已驗
的數字相同）、平均 0.30 px，但 max 是 18.37 px，而那個 max 只有**一個**配對——4K 畫面上
一個 conf 0.255（Torch）對 0.455（引擎）的邊界偵測，模型自己就不確定人腳在哪，不是 FP16
的捨入誤差。全樣本超過 5 px 的配對就那一個（0.05%）。

因此門檻改成四道，都寫在 `compare_backend.check_report`（判準與計算放同一支，改了取樣
方式而沒改門檻才會有訊號）：p99 ≤ 1.20 px、偏差 > 5 px 的配對比例 ≤ 0.5%（尾巴有人管，
但單一離群不擋）、配對率 ≥ 98%、偵測框數差 ≤ ±2%。後兩道擋的是「整批壞掉」——精度或
形狀出錯時偏差不見得變大，但框數與配對率會塌。

（2026-08-31 更新：p99 門檻放寬到 1.3 px——同一份設定重建 TensorRT 引擎，p99 會在
1.183–1.250 px 之間跳，1.20 正好落在這個離散區間裡面，等於有一半的建置會被自己擋下。
上面 1.20 這個數字是本次決策當時的實測值，不因此改寫；新門檻見
`compare_backend.DEFAULT_MAX_FOOT_DEV_P99_PX`。）

比對**強制關閉 TF32**（不開放由 CLI 打開）：PyTorch 預設讓卷積在 Ampere＋用 TF32（尾數
10 bit），開發機的 5090 吃得到而 T4 是 Turing、跑的是真 FP32。在開發機用預設值量等於拿
「TF32 對 FP16」當「FP32 對 FP16」，會**低估**偏差——而低估的方向剛好是「看起來沒問題」。

### 11. 引擎載入失敗要走得到 `TRACK_FAILED`

引擎的載入失敗發生在推論子進程的**啟動階段**，也就是 `InferencePipeline.start_loop` 的
`except` 涵蓋不到的地方，而追蹤進程此刻已經阻塞在 `track_queue.get()`。issue #113 揭露了
它不會自己醒過來的原因：`track_queue` 的 pipe 寫入端 fd 被父進程與九個讀取進程一起繼承，
寫入端永遠不會全部關閉，`get()` 收不到 EOF。因此 `run_inference_pipeline` 的組裝段
（建 `YOLODetector`、建環形緩衝、建 pipeline）包一層 `except` 送 `TRACK_FAILED`——只包組裝
段，`start_loop` 自己那條會重複送。

改吃引擎之後「載入失敗」從罕見變成常見的一類（引擎檔不在、SM 對不上、TRT 版本與映像檔不
一致），所以這條路徑值得補上。

## Consequences

Positive

- 正式路徑只有一套實作，每一道檢查都跑在正式環境會走的那條路上，不存在「只有另一條分支
  會踩到」的迴歸。
- 精度、批次上限、來源權重、SM、TRT 版本全部由引擎自帶並在載入當下驗證，「跑的不是你以為
  的那顆」從無訊號變成當場中止。
- `.pt` 不再需要出現在正式容器裡。

Negative

- **建置流程從此是部署的前置條件**。換權重、升 TensorRT、換 GPU 型號都要重跑
  `build_engine.py`，而且**只能在目標卡上跑**。升版順序是有向的：先建新引擎並發布，才能
  滾映像檔；反過來會讓正式環境啟動失敗。
- **本機建不出正式引擎**。開發機（sm120）只能建自用那顆；T4 那顆要在 T4 機器上建。
  地端引擎產出的結果不能當正式環境的逐值真相（FP16 在不同架構上數值可能有微小差異）。
- **顯存多一筆固定成本**：dynamic 引擎建 execution context 就吃約 4.7 GB，不隨批次縮小。
- **`video_analyze` 多一個 runtime 依賴 `tensorrt-cu12`**，且版本帶是硬的（10.8 ～ 10.16：
  下緣是 sm120 的 kernel 支援，上緣是 11.0.0 移除 `BuilderFlag.FP16`）。升到帶外會讓
  FP16 只能靠額外的量化工具烤進中介格式，連帶把建置環境撐大一個量級。
- **`[model].source_weights_sha256` 沒釘的話，權重身分只記不擋**。這是刻意留的缺口
  （釘 hash 對每個部署是額外的維護成本），但「沒在驗」會以 warning 的形式出現在 log 上。
- **升 TensorRT 版本要重跑 sm75 kernel 檢查**：10.13.3.9 的 `libnvinfer_builder_resource`
  確認含 sm75，版本帶上緣（10.16.x）沒驗過。
- **argus 的兩份拷貝不隨本次同步**：正式節點的映像檔要裝 `tensorrt-cu12`、要能
  取到引擎，`.pt` 的下載鏈也要改成引擎的下載鏈。在那之前，argus 側仍跑 Torch FP32。
