# video_analyze

多路離線影片的偵測與追蹤：以「一天」為單位，讀取多路攝影機的錄影片段，用 YOLO
（CrowdHuman 微調權重，僅偵測 `fbody` 完整人體）搭配 ByteTrack 做多路追蹤，產出每格的
追蹤明細。

## 套件概述

輸入是多路攝影機一整天的錄影片段，輸出是逐格的追蹤明細 parquet，`track_id` 跨片段延續。

資料來源為**本機模擬的 GCS bucket 目錄**，各攝影機的片段依日期分層存放；攝影機清單集中
在該 bucket 根目錄下的 `camera_registry.yaml`。

**進入點是函式呼叫，CLI 只是外殼。** 核心是 `analyze_daily` 函式，CLI 只負責從
`config.toml` 組出參數後呼叫它。兩者分離，要換掉觸發方式時只需替換外殼，pipeline 本身
不必更動。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，整個 workspace 共用單一 root `uv.lock`） |
| GPU | **必要，且必須是建置引擎時的那個 SM**。正式推論路徑是 TensorRT FP16 引擎（[ADR-011](../docs/adr/video_analyze/011-single-inference-backend.md)），引擎綁 GPU 也綁架構，沒有 CPU fallback——CPU 上不是「比較慢」而是一定失敗 |
| 系統相依 | FFmpeg / 影像編解碼器（OpenCV 解 `mkv` 等格式）；`lap` 為 C 擴充，環境無對應 wheel 時需要編譯工具鏈 |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `opencv-python` | 影片片段讀取與解碼 |
| `ultralytics` | YOLO 偵測 |
| `torch` / `torchvision` | 前處理／後處理的張量運算，以及建置與比對工具的 Torch FP32 基準（與 `ultralytics` 一併釘住版本） |
| `tensorrt-cu12` | 正式推論後端。版本帶 **10.8 ～ 10.16**：下緣是 sm120 的 kernel 支援，上緣是 11.0.0 起 strongly-typed 成為預設、移除 `BuilderFlag.FP16`。帶內取已實測的 `10.13.3.9` |
| `lap` | ByteTrack 的線性指派求解 |
| `numpy` | 影格與追蹤結果的陣列運算 |
| `polars` / `pyarrow` | 追蹤明細 parquet 寫出 |
| `pydantic` / `pydantic-settings` | 設定與 registry 的資料模型與驗證；`config.toml`＋環境變數載入 |
| `pyyaml` | `vfa_registry` 讀 `camera_registry.yaml` 用；本包不直接 import，pin 在此是為了與 lib 對齊版本（見根 CLAUDE.md）|
| `vfa_registry` | 共用 lib：`camera_registry.yaml` 的模型（workspace 成員）|
| `vfa_observability` | 共用 lib：`StructuredLogger` 結構化 JSON log（workspace 成員）|
| `vfa_config` | 共用 lib：`[input]` 設定區塊與 `config.toml` 定位（workspace 成員）|

依賴版本以 `==` 釘住，固定推理堆疊。

**模型**：`config.toml` 的 `model_path` 指向一顆 **TensorRT FP16 引擎**（`.engine`），
不是 `.pt`。正式套件內已沒有 Torch 權重的推論路徑（[ADR-011](../docs/adr/video_analyze/011-single-inference-backend.md)），
載到 `.pt` 會當場中止。引擎與權重都不進版控（`.gitignore` 排除 `*.pt`／`*.engine`），
引擎要用 `tools/build_engine.py` 自行建（見[建置 TensorRT 引擎](#建置-tensorrt-引擎)）。

來源權重為 CrowdHuman 微調的 `20260714-153811_yolo26m_baseline.pt`，best epoch 驗證指標
mAP50≈0.806、recall≈0.72、precision≈0.847；這些欄位由建置工具抄進引擎 metadata，
載入時印在 log 裡（引擎本身沒有 `ckpt`，取不到訓練資訊）。

## 安裝與執行

```bash
uv sync --package video_analyze
```

準備下列輸入後即可執行：

1. 一份本機的 `bucket_dir/`，內含各攝影機的影片片段與 `camera_registry.yaml`
   （格式見[設定](#設定)）。
2. 本套件根目錄的 `config.toml`（指定要跑哪個 bucket、哪一天、哪些攝影機與各項參數）。

```bash
uv run --package video_analyze video_analyze
```

此命令不接受任何旗標，所有參數都讀自 `config.toml`。

> **於倉庫根目錄執行**：`bucket_dir`、輸出根目錄 `outputs/` 與 `model_path` 皆為 **cwd
> 相對路徑**，`uv run --package` 不改變 cwd。本套件自己的 `config.toml` 則以共用 lib 的
> `get_toml_path(__file__)`（往上找 `pyproject.toml`）定位，不受 cwd 影響。

## 設定

設定分成兩個檔案，職責清楚切分：

- **`config.toml`** — 描述「這次要怎麼跑」，置於本套件根目錄。找不到此檔時會印出警告並
  回退到各項預設值。
- **`camera_registry.yaml`** — 描述「資料長什麼樣」，置於 `bucket_dir` 根目錄。

### `config.toml`（本次執行參數）

```toml
[tracker]
track_high_thresh = 0.5
track_low_thresh = 0.1
new_track_thresh = 0.6
track_buffer = 30
match_thresh = 0.8
fuse_score = true

[model]
model_path = "20260714-153811_yolo26m_baseline_sm120.engine"   # 開發機那顆；T4 用 _sm75
source_weights_sha256 = "b14302f4…"   # 釘住引擎的來源權重；留空則只記 warning
batch = 16
classes = [0, 2]           # CrowdHuman 類別過濾：0=head, 1=vbody, 2=fbody
                           # fbody 是追蹤目標；head 只用來推算落腳點，不進 tracker

[foot_point]
method = "head"            # "head" = 由頭部位置推算；"bbox_bottom" = 框底邊中點（舊做法）

[input]
bucket_dir = "bucket_name1"
date = 2026-05-01
camera_ids = []            # 空 = camera_registry.yaml 內全部攝影機
```

各區塊的主要欄位與約束：

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[tracker]` | ByteTrack 各項閾值 | 見範例 | `*_thresh` 皆介於 0–1，`track_buffer >= 1` |
| `[model]` | `model_path` | `"…_sm75.engine"` | TensorRT 引擎路徑。**只吃 `.engine`**，指到 `.pt` 直接中止。檔名的 `_sm<SM>` 尾綴是刻意的：引擎綁架構，兩顆同名會在下游 promotion 撞在一起 |
| | `source_weights_sha256` | `""` | 釘住引擎的來源 `.pt` 內容 hash，不符即中止。留空只記 warning——這是唯一擋得下「換成另一顆 id 剛好都存在、語義卻不同的權重」的檢查 |
| | `batch` | `1` | 單次推理批次，`>= 1`（範例用 `16`）；**不得超過引擎建置時綁的最大批次**（`build_engine.py --batch`，同一尺度），超過即中止 |
| | `classes` | `[0, 2]` | 要保留的偵測類別 id；權重類別為 `0=head, 1=vbody, 2=fbody`；至少 1 個元素。載入時會驗證此清單存在於**引擎 metadata 的 `names`**（不是另建 predictor 去讀 `model.names`，那會把引擎多載一次），不符直接拋錯。**必須含 fbody**（追蹤目標）；`method = "head"` 時**還必須含 head**，否則直接拋錯——少了 head 每一列都會退回框底邊中點，改動靜默失效 |
| `[foot_point]` | `method` | `"head"` | 落腳點的推算方式：`"head"` 由頭部位置推算（修正斜向視角下框底邊中點落在人體外的偏移），`"bbox_bottom"` 為改動前的框底邊中點，保留供對照與回退。見 [ADR-009](../docs/adr/shared/009-head-based-foot-point.md) |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（範例用 `bucket_name1`） |
| | `date` | — | 分析日期 |
| | `camera_ids` | `[]` | 要分析的攝影機；空清單 = 全部 |

`[input]` 由共用 lib `vfa_config` 提供、四包同一份定義，故本包也接受 `bucket_minutes`
（只有 `zone_mapping`／`line_counting`／`flow_report` 會讀）。為何不能各包裁剪見
[ADR-008](../docs/adr/shared/008-config-section-namespace.md)。

### `camera_registry.yaml`（資料樣貌）

放在每個 `bucket_dir` 根目錄下。**此檔不進版控**（隨 `bucket_*/` 一起被 `.gitignore`
排除），需依實際部署環境人工維護。

攝影機片段的目錄結構為：

```
<bucket_dir>/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv
```

> **時區處理**：檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，本套件在
> `services/video_reader.py` 解析時即把它轉換成台北在地時間（`Asia/Taipei`，UTC+8）；
> `tracking_results.parquet` 的
> `timestamp` 即為台北在地時間，下游不需要、也不應該再對它做任何 UTC→+8 位移。

格式範例：

```yaml
bucket_name: bucket_name1

storage:
  file_ext: mkv
  target_codec: h265
  segment_strategy: time
  segment_seconds: 1800

cameras:
  - camera_id: cam001
    location: test
    ip: 192.168.104.115
    participates_in_zone_mapping: true
    zones:
      - name: 平擺桌
        polygon: [[640.01, 866.83], [521.34, 938.8], [700.0, 1000.0]]
    lines:
      - name: 出入口161_左側
        points: [[1180.0, 980.0], [1180.0, 1080.0]]
        inside_point: [900.0, 1030.0]
        line_group: 4F賣場
```

欄位規範：

| 層級 | 欄位 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- | --- |
| 頂層 | `bucket_name` | str | 必填 | bucket 名稱 |
| | `storage` | 物件 | 必填 | 片段儲存格式參數 |
| | `cameras` | list | 必填 | 攝影機清單 |
| `storage` | `file_ext` | str | `mkv` | 片段副檔名 |
| | `target_codec` | str | `h265` | 原始錄影編碼 |
| | `segment_strategy` | str | `time` | 分段策略 |
| | `segment_seconds` | int | `1800` | 每段秒數，`>= 1` |
| `cameras[]` | `camera_id` | str | 必填 | 攝影機代碼 |
| | `location` | str | 必填 | 地點名稱 |
| | `ip` | str | 必填 | 攝影機 IP |
| | `participates_in_zone_mapping` | bool | `true` | 接受但不使用（語義由 `zone_mapping` 實作） |
| | `zones` | list | `[]` | 接受但不使用；幾何刻意不在本階段驗證 |
| | `lines` | list | `[]` | 接受但不使用；幾何刻意不在本階段驗證 |

本套件只讀攝影機身份（`camera_id` / `location` / `ip`）用來定位片段目錄與過濾攝影機。
`participates_in_zone_mapping` 與 `zones` 由下游的 `zone_mapping` 使用、`lines` 由
`line_counting` 使用；registry 的資料模型不接受未列出的欄位，故這三個欄位仍須保留於
模型中並忽略其值。`zones[]`／`lines[]` 的子欄位規範見[根 README](../README.md) 的
`camera_registry.yaml` 章節——本套件不解析它們，此處不重複列出。

使用限制（皆為 fail-loud，違反時直接報錯）：

- **`camera_id` 與 `location_camera_id` 皆須唯一**。兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，因此在載入時即擋下。
- `config.toml` 的 `camera_ids` 若含 registry 中查無的 ID，直接報錯。
- `cameras[]` 不接受未列出的欄位（多打的欄位會報錯）。

## 建置 TensorRT 引擎

引擎不是原始碼，是**要在目標 GPU 上建置的產物**（建置時對實際裝置做 kernel
autotuning）。TensorRT 沒有指定 target SM 的 API，`HardwareCompatibilityLevel` 只有
`NONE`／`AMPERE_PLUS`／`SAME_COMPUTE_CAPABILITY` 而 `AMPERE_PLUS` 不含 Turing，所以
**開發機建不出 T4 要用的引擎**：同一份權重、同一支工具、兩台機器、兩顆二進位。

```bash
# 在 repo 根目錄執行；--package 不改變 cwd（--directory 會）
uv run --package video_analyze python video_analyze/tools/build_engine.py \
    --weights 20260714-153811_yolo26m_baseline.pt \
    --output-dir . \
    --batch 16 \
    --bucket bucket_20260801_small
```

`--batch` 是引擎綁的**最大**批次，要容得下 `config.toml` 的 `[model] batch`（兩者同
尺度：ultralytics 對 in-memory list source 一次 forward 整個 list，設定值就是實際的
forward 批次）。產物名為 `<權重 stem>_sm<SM>.engine`。

工具做四件事，**任何一關沒過就不留下產物**（先落在 `.unverified` 尾綴上，全過才改名）：

1. 以 `half=True, dynamic=True, imgsz=(384, 640)` 匯出，並用 `on_export_start` callback
   把來源權重身分、SM、TensorRT／CUDA／驅動版本與訓練追溯資訊注入 metadata。
2. 驗引擎本身：metadata 對得上當下環境、`args.half` 為真、`args.dynamic` 為真、最大批次
   等於 `--batch`，並真的用滿批跑一格確認實際形狀是 384×640。
3. 對 Torch FP32 逐框比對（`tools/compare_backend.py`），比的是**框底邊中點的座標偏差**
   ——下游的落腳點、跨線進出、區域佔用都從這個點算。
4. 套四道門檻（`compare_backend.check_report`）：落腳點偏差 **p99** ≤ 1.20 px、偏差
   > 5 px 的配對比例 ≤ 0.5%、配對率 ≥ 98%、偵測框數差 ≤ ±2%。任一沒過就刪掉產物並以
   非零 exit code 結束。**判準看 p99 不看 max**：既有的 1.20 px 是 277 個框的 max、
   1.16 px 是 277 個框的 p99，而本工具預設取樣 2000 個框以上——max 是樣本數的函數，
   尾巴改由「超過 5 px 的比例」這道門檻管（見 ADR-011）。

比對**強制關閉 TF32**：PyTorch 預設讓卷積在 Ampere＋用 TF32（尾數 10 bit），開發機的
5090 吃得到而 T4 是 Turing、跑的是真 FP32。在開發機用預設值量等於拿「TF32 對 FP16」當
「FP32 對 FP16」，會**低估**偏差——而低估的方向剛好是「看起來沒問題」。

`tools/` 在 `src/` 之外，不隨 wheel 出貨；那兩支工具會載 `.pt`，但不經過
`YOLODetector`。這條分界就是 ADR-011 說的「正式產品只有一套 inference implementation」。

## 函式介面

```python
analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult
```

回傳的 `AnalysisResult` 含 `date` / `camera_ids` / `tracking_results_path`。
`bucket_dir` 以參數傳入（而非讀全域 `settings`），故可重複以不同
bucket 呼叫。

## 輸出檔案

| 路徑 | 內容 |
| --- | --- |
| `outputs/{bucket_name}/{date}/tracking_results.parquet` | 追蹤明細 |

`tracking_results.parquet` 的欄位：

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `camera_id` | str | 該影格所屬攝影機的 `<location>_<camera_id>` |
| `frame_id` | int | **片段內**幀序，跨片段會重複（非全日流水號） |
| `timestamp` | datetime（`Asia/Taipei`） | 該片段檔名時間 ＋ 片段內幀序 / fps |
| `track_id` | int | ByteTrack 指派的追蹤編號，跨片段延續 |
| `x1` / `y1` / `x2` / `y2` | float | 追蹤框的像素座標 |
| `foot_x` / `foot_y` | float | 落腳點（人站在地面的位置）的像素座標；由 head 框推算，配不到頭時沿用該軌跡上次的偏移量，連偏移量都沒有才退回 `((x1+x2)/2, y2)` |
| `frame_width` / `frame_height` | int | 該路的影像尺寸（`probe_frame_shape` 探測首格所得，整天固定）；逐列重複同一個值 |

`frame_width` / `frame_height` 是為下游而存的：`line_counting` 與 `zone_mapping` 都是純
CPU 套件、部署時不掛載影片，拿不到影像尺寸，卻需要它把以 1080p 為基準的像素參數
（`crossing_band_px_1080p`／`boundary_band_px_1080p`）換算成各攝影機的實際像素
（見 [ADR-004](../docs/adr/shared/004-band-resolution-scaling.md) 與
[ADR-006](../docs/adr/zone_mapping/006-zone-boundary-band.md)）。缺這兩欄的舊 parquet 會被這兩包
**都**直接擋下，需以本套件重跑產生。

`foot_x` / `foot_y` 同樣是為下游而存：推算需要 head 框，而 head 不進追蹤結果（它若進
tracker，同一個人會多出一條頭部軌跡），因此只能在本套件算。**推算帶跨幀狀態**：配不到
頭的那幾格沿用該軌跡上次成功推算的偏移量（超過 60 個有軌跡的幀才放棄），避免落腳點在
兩種算法之間彈跳——代價是同一份 detection 在不同歷史下會得到不同的落腳點。**成功推算不
等於會存下偏移**：同一幀內一顆 head 被兩個以上 fbody 框配到時，這些框的推算結果照樣用在
該幀但都不進狀態，共用時分不出哪一邊是錯配，存下來等於把單幀錯配放大成最多 60 個有軌跡
的幀。下游一律讀這兩欄、不再自行從 bbox 推算，缺欄位同樣 fail loud。公式、配對條件與多
候選時的選法見
[ADR-009](../docs/adr/shared/009-head-based-foot-point.md)。

## 架構

### 模組結構

`src/video_analyze/` 依 DDD 分層（與 `flow_report`／`zone_mapping` 及 argus 慣例對齊），
依賴方向單向、無循環：

| 模組 | 職責 |
| --- | --- |
| `main.py` | 薄 CLI 外殼：進入點 `main`，讀 `settings` 組參數後呼叫 `analyze_daily` |
| `models/config.py` | Pydantic-settings 設定模型（`config.toml`＋環境變數）與全域 `settings` 單例 |
| `config/constants.py` | 非 Pydantic 靜態常數（`OUTPUT_ROOT`、輸出檔名與 parquet schema、CrowdHuman 類別 id `HEAD_CLASS_ID`／`FBODY_CLASS_ID`） |
| `services/batching.py` | 單次推理批次 `TARGET_BATCH`，與由它推導的 `RING_SLOTS`／`TRACK_QUEUE_SLOTS` |
| `services/pipeline.py` | `analyze_daily` 與多進程編排（讀取／推理／追蹤子進程生命週期） |
| `services/inference.py` | 推理迴圈（湊批、偵測、把偵測框送往追蹤進程） |
| `services/track_worker.py` | 追蹤進程：偵測框拆分、追蹤、落腳點推算、座標反算、落盤；`TRACK_DONE`／`TRACK_FAILED` 訊號 |
| `services/detector.py` | TensorRT 引擎載入與偵測；載入前的四道檢查與執行期的形狀核對 |
| `services/engine_metadata.py` | 引擎自帶 metadata 的注入格式、讀取與環境比對（建置端與載入端共用同一份） |
| `services/foot_point.py` | `FootPointEstimator`：head 框配對與落腳點推算；自行維護跨幀狀態（每條軌跡上次成功推算的偏移量，共用 head 的那些不存） |
| `services/tracker.py` | 多路追蹤，每路各自獨立的 `BYTETracker` 實例 |
| `services/tracking_results.py` | 追蹤明細累積與 parquet 寫出 |
| `services/fps_meter.py` | 處理 FPS 統計 |
| `services/frame_ring.py` | 共享記憶體環形緩衝 |
| `services/letterbox.py` | 縮放到推論尺寸與座標反算（正反兩半共用同一組參數） |
| `services/video_reader.py` | 逐日掃描片段、讀影格、縮到推論尺寸 |

I/O 邊界（讀寫檔、子進程、影像解碼）依 argus 慣例一律放 `services/`，不另立
頂層 adapter/io 層。log 用共用 lib `vfa_observability` 的 `StructuredLogger`
（`from vfa_observability import StructuredLogger`），輸出單行 JSON。

`camera_registry.yaml` 的 `CameraRegistry` / `CameraEntry` 由四包共用的
[libs/vfa_registry](../libs/vfa_registry) 提供（`from vfa_registry import load_registry`），
以 workspace 成員引用。本包不呼叫 `parse_and_validate_zones`／`parse_and_validate_lines`，
zone 與 line 幾何都不會被驗證。

### 多進程 pipeline

`analyze_daily` 在主進程先 `discover_segments` 掃出當天片段、`probe_frame_shape` 讀首格
解析度，再以多進程拆成 N 個讀取進程 ＋ 1 個推理進程 ＋ 1 個追蹤進程：

- **影格走共享記憶體、不走 pickle**（`frame_ring.py`）：每路一塊固定格數的環形緩衝
  （`mp.RawArray`），queue 只傳 slot 索引，避免每格影格逐格 pickle 的高成本。推理進程
  直接消費 slot 的 view、不複製出私有副本（`view_slot`）。緩衝依**推論尺寸**（640×384）
  配置——影格在讀取端就縮好了，故每格一律 0.70 MiB 而與來源解析度無關（縮放前移之前
  4K 每格是 23.73 MiB）。此設計仍**假設同一攝影機整天解析度固定**（`probe_frame_shape`
  只探測首格）。
- **讀取進程**：解碼後立即 letterbox 成推論尺寸再寫入 slot（`letterbox.py`）。無空 slot
  時阻塞，形成對推理進程的天然背壓——slot 在 predict 完成當下就歸還，所以這條背壓只覆蓋
  到推論為止，追蹤那一段由 `track_queue` 的容量上限接手。**時間戳 = 該片段檔名時間 ＋ 片段內幀序 / fps**
  （逐段計算，不能用全日累計幀數推算）。
- **推理進程**：非阻塞輪詢各路 queue 湊批，維持 GPU 批次效率；每批推論完就把逐格的
  偵測框丟進 `track_queue`，本身不做追蹤。影格既然是共享記憶體的 view，**整批推論完成
  才歸還 slot**（生命週期約束見
  [ADR-010](../docs/adr/video_analyze/010-zero-copy-frame-lifetime.md)），送 payload 排在
  歸還之後（payload 取的是推論輸出，與 slot 無關）。
- **追蹤進程**：逐格經 裁切到內容區 → 拆出 fbody／head（只有 fbody 進 tracker）→ 追蹤
  → 推算落腳點 → **框與落腳點映射回原始解析度** → 累積追蹤結果並落盤。追蹤與 GPU 推論
  之間沒有資料相依（下一批的推論不需要上一批的軌跡），拆成兩個進程即可重疊。
  **跨進程傳的是偵測框而不是影格**：每格幾十個框、幾 KB，而影格在推論完成後已無用途。
  空格（該格沒有任何偵測）照樣送 payload、照樣呼叫 `tracker.update`——`BYTETracker` 的
  `frame_id` 與軌跡老化都靠每格呼叫推進。

**縮放前移的代價**：ultralytics 只知道自己收到 640×384，`orig_shape` 也是那個尺寸，
輸出的框就停在推論尺度上，反算因此變成本包的責任（`letterbox.py` 的
`unscale_boxes_inplace`／`unscale_points_inplace`）。反算**插在落腳點推算之後**：`heads`
是唯一沒有反算的陣列，提早反算 `tracks` 會讓兩者尺度不一致而配不到頭，每列靜默退回框
底邊中點。`probe_frame_shape` 的原始解析度因此有兩個消費端（parquet 的尺寸欄位、反算
參數），兩者都不能換成推論尺寸——**兩者都在追蹤進程**，故 `frame_shapes` 傳給的是
`run_track_worker` 而非推理進程。

### fail-loud 錯誤處理

- 檔名格式錯誤 → `discover_segments` 在主進程直接拋 `ValueError`；各攝影機首段的開檔 /
  讀影格失敗 → `probe_frame_shape` 同樣在主進程拋 `ValueError`（子進程尚未啟動）。
- **片段所在目錄日期與轉換後的台北曆日不同 → `_parse_segment_start` 拋 `ValueError`**。
  目錄 `{YYYY}/{MM}/{DD}` 是 UTC 曆日、檔名時間轉換後是台北時間，UTC 16:00 之後兩者
  分岔（例如 `2026/05/01/160000.000Z.mkv` 屬台北 05/02 00:00）。這代表片段被放進錯誤
  的日期目錄，寧可中止也不靜默寫錯天。此檢查在 `discover_segments` 掃描時、於主進程
  執行，**任一路踩到就整天中止**（不是只跳過該攝影機或該片段）。
- 其餘片段的開檔 / 讀 FPS 失敗 → 讀取子進程拋錯、以非零 exitcode 結束。
- **引擎載入前的四道檢查，任一不過即中止**（`services/detector.py`）：`model_path` 不是
  `.engine`、引擎檔不存在（**這道 `is_file()` 前置檢查不可移除**——ultralytics 的
  `check_file` 在檔案不存在時會遞迴 glob 整個套件目錄找同名檔、把 `gs://` 改寫成公開
  HTTPS 下載、把沒有副檔名的值補成官方權重下載）、引擎 metadata 與當下環境不符
  （compute capability／TensorRT 版本／TensorRT wheel 變體／來源權重 hash，四項各自拋錯、
  訊息指出是哪一項；驅動版本與 torch 的 CUDA 建置版本只記 warning——它們不在 TensorRT 對
  引擎的約束裡）、引擎不是 FP16 建的。沒有 CUDA 也是中止而非 fallback CPU。
- **引擎不是 dynamic 建的 → 拋 `ValueError`**。靜態引擎會通過其餘全部載入檢查，然後在
  第一個沒湊滿的批次上被 ultralytics 的 assert 擋下（訊息看不出原因）——而湊不滿批是
  常態：T4 上實測一次跑完出現 16 種不同的批次大小。
- **實際進入推論的張量形狀不是 640×384 → 拋 `ValueError`**。讀的是 TensorRT backend 自己
  的 binding。這一項取代了 `_validate_imgsz`：dynamic 引擎不套用 metadata 的 `imgsz`
  （`predictor.py` 只在 backend 不是 dynamic 時才套用），照抄那個檢查會得到一個看起來
  有在驗、其實驗不到的檢查（見 ADR-011）。
- **單次批次超過引擎綁的最大批次 → 拋 `ValueError`**（`services/inference.py`，在主迴圈的
  try 內，送得出 `TRACK_FAILED`）。
- **`frame_shapes` 傳成推論尺寸 → `run_track_worker` 拋 `ValueError`**。那代表
  拿到的是縮放後的尺寸而非原始解析度，反算會退化成恆等、parquet 的 `frame_width` 也一起
  寫成 640，而下游 zone／line 只檢查欄位存在（ADR-004／ADR-006）。影格尺寸與緩衝配置
  不符（例如讀取端漏了 `letterbox()`）則由 `write_slot` 擋下。
- `analyze_daily` 以 0.5 秒輪詢所有子進程；任一非零結束 → 先終止所有子進程再拋
  `RuntimeError`；`KeyboardInterrupt` → 終止後以 exit code 130 收斂。
- 追蹤明細 parquet 先寫 `.tmp`、收到 `TRACK_DONE` 才 `rename` 成正式檔名（`rename` 具
  原子性）。推論進程中途例外時送 `TRACK_FAILED`，追蹤進程收到就刪除 `.tmp` 並以非零
  exitcode 結束，正式檔名下不會出現不完整的 parquet。這條路徑**依賴 `track_queue` 的
  容量上限**（`services/batching.py` 的 `TRACK_QUEUE_SLOTS`）：訊號是排在同一條佇列尾端的 in-band
  訊號，backlog 無上限的話它會晚於父進程的 terminate 抵達而靜默失效。這是「backlog
  有界」的推論而非時序保證——上限隨 `MODEL__BATCH` 線性成長，父進程的偵測延遲不隨它
  成長，故調大 batch 時要重新評估。**推論主迴圈啟動之前**就失敗的路徑（建 `YOLODetector`、建環形緩衝）
  由 `run_inference_pipeline` 自己的 `except` 送出同一個訊號——改吃引擎之後「載入失敗」
  從罕見變成常見的一類（引擎檔不在、SM 對不上、TensorRT 版本與映像檔不一致），而
  `track_queue` 的 pipe 寫入端 fd 被父進程與九個讀取進程一起繼承，上游死掉不會讓
  `get()` 收到 EOF（issue #113）。
- **推論進程被 SIGKILL（OOM kill、人工 kill）時連 `TRACK_FAILED` 都送不出來**，追蹤進程
  就卡在上面那個收不到 EOF 的 `get()` 上。它等到的是父進程 `_terminate_all` 的 **SIGTERM**，
  而追蹤進程攔下了這個訊號（`track_worker._raise_on_sigterm` 拋 `SystemExit(143)`），走的
  仍是既有的 `collector.discard()`；不攔的話預設處置不執行 Python 的 `except`／`finally`，
  `.tmp` 會留下。清理期間、以及 `save()` 的關檔＋改名期間，SIGTERM 與 SIGINT 都設成
  `SIG_IGN`——Ctrl+C 走的是 process group 的 SIGINT，本進程進到清理的同時父進程也在送
  SIGTERM，連按兩次 Ctrl+C 又多一個 SIGINT；任何一個落在關 writer 之後、刪檔之前都會留下
  整天的暫存檔，落在 `save()` 的關檔與改名之間則會讓 `discard()` 刪掉一份**已經完整**的
  parquet。這是把窗口**縮到**「例外拋出到設定 `SIG_IGN` 之間」，不是關掉；用訊號就到這裡
  為止，再被 SIGKILL 就靠下面那條收。
- **整機重啟／追蹤進程本身被 SIGKILL** 沒有任何 in-process 的機制擋得住，改由**下一次寫
  同一條路徑的執行**在啟動時清掉（`tracking_results.claim_tmp_slot`）。判準是「還有沒有
  進程持有這個暫存檔的 `flock`」而不是檔名或 mtime：拿得到鎖代表持有者已經不在，殘檔就地
  `ftruncate` 清空（不 `unlink`，否則鎖會留在沒有檔名的 inode 上而守不住）；拿不到鎖代表
  另一個執行正在寫同一條路徑，**當場 fail loud，一個 byte 都不動**。因此同一個 bucket 的
  同一天不能有兩個執行並行（那本來就會兩邊交錯寫進同一個暫存檔），要並行請分開 bucket
  或分開日期。收尾時刪檔與改名都認 inode 而不認路徑：暫存檔若已被換成別份，`discard()`
  留著不動、`save()` 直接 fail loud（那時本次結果已隨原本的檔案一起遺失）。

## 開發

```bash
uv run --directory video_analyze ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory video_analyze pytest         # 執行測試
```

此處用 `--directory`（切換 cwd 進 `video_analyze/`）而非執行分析時的 `--package`：
`--package` 不會改變 cwd，pytest 會從 repo 根遞迴收集到全部套件的 `tests/`，因四包皆有
同名測試檔（如 `test_config.py`）而撞名衝突報錯；`--directory` 才會讓 pytest 只解析
本套件自己的 `tests/`。
