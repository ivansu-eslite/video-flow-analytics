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
| 系統相依 | FFmpeg / 影像編解碼器（PyAV 解 `mkv` 等格式）；`lap` 為 C 擴充，環境無對應 wheel 時需要編譯工具鏈。**讀取層開 NVDEC 硬解後（issue #130 第二階段），解碼本身也要求 CUDA/NVDEC 可用**——`allow_software_fallback=False`，沒有 CPU 軟解 fallback，PyAV wheel／驅動缺 NVDEC 支援時每個片段都會直接失敗 |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `av` | 影片片段讀取與解碼（PyAV，wheel 自帶 FFmpeg）|
| `opencv-python` | 影格縮放（在解碼出的 nv12 兩個平面上）、色彩轉換與 letterbox 補邊（`services/letterbox.py`）|
| `ultralytics` | ByteTrack（`services/tracker.py`）與建置／比對工具（`tools/`，不隨 wheel 出貨）。**正式推論路徑上已經沒有它**：引擎載入與 forward 是 `services/trt_runner.py`（ADR-014），前處理與後處理是本套件自己的程式碼（ADR-013）|
| `torch` / `torchvision` | 前處理的張量運算（在 GPU 上做）、流水的 stream 與 event、pinned 與 device 緩衝，以及建置與比對工具的 Torch FP32 基準（與 `ultralytics` 一併釘住版本） |
| `tensorrt-cu12` | 正式推論後端，本套件直接呼叫（`deserialize_cuda_engine`／`set_input_shape`／`set_tensor_address`／`execute_async_v3`）。版本帶 **10.8 ～ 10.16**：下緣是 sm120 的 kernel 支援，上緣是 11.0.0 起 strongly-typed 成為預設、移除 `BuilderFlag.FP16`。帶內取已實測的 `10.13.3.9` |
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

放在每個 `bucket_dir` 根目錄下。**此檔不進版控**（隨 `bucket_*` 一起被 `.gitignore`
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
2. 驗引擎本身：metadata 對得上當下環境、精度旗標逐項符合宣告（`args.half` 為真、
   `args.int8` 為假；缺欄一律視為不符，見 `engine_metadata.validate_engine_precision`）、
   `args.dynamic` 為真、最大批次等於 `--batch`，並真的用滿批跑一格確認實際形狀是
   384×640。
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

## 端到端效能量測

`tools/bench_e2e.py` 跑一整組矩陣（測試片 × 批次 × 重複）並把產物解析成表。它取代的是
以前每次量測都重寫一遍的臨時腳本；執行組態全部走 CLI，沒有寫死的路徑或日期。

```bash
# 在 repo 根目錄執行（bucket_dir 與 outputs/ 都是 cwd 相對路徑）
uv run --package video_analyze python video_analyze/tools/bench_e2e.py run \
    --buckets bucket_20260801_perf40,bucket_20260801_perf \
    --date 2026-08-01 \
    --batches 16,8 \
    --repeat 2

uv run --package video_analyze python video_analyze/tools/bench_e2e.py report
```

一輪做的事：清空 `outputs/<bucket>/` → 寫 meta（組態、commit、引擎檔與其 sha256、
torch／TensorRT／ultralytics 版本、片段數）→ 背景起 `nvidia-smi dmon` 取樣 → 用
`INPUT__`／`MODEL__`／`FOOT_POINT__` 環境變數跑一次 `video_analyze` → 補寫資源用量與
`exit_status` → 複製該輪的 `tracking_results.parquet`。產物落在 `--runs-dir`
（預設 `outputs/bench_e2e`，已 gitignore），每輪四個檔：

| 檔案 | 內容 |
| --- | --- |
| `<run>.meta` | `key=value` 的組態、環境身分、資源用量（`wall_seconds`／`max_rss_kb`／CPU％／page fault／context switch／fs I/O） |
| `<run>.log` | 該輪的 stdout（結構化 JSON log） |
| `<run>.gpu.log` | `dmon` 取樣（預設 `-s pucm`；欄位隨指標集而變，解析依表頭找欄位） |
| `<run>.parquet` | 該輪寫出的 `tracking_results.parquet` |

**報表的「每秒張數」一律是推論進程的口徑**（`component == "inference"` 的 `FPS 整體`
那行）。每輪 log 裡還有追蹤進程印的 `overall_fps`（`[tracker].shards` 有幾片就有幾行），
但它的分母是該進程自己的 wall clock，不是同一個東西，混用會系統性高估 1–2%。工具用四道
互鎖擋掉取錯（欄位各自帶口徑、追蹤側逐片保存且單一數字只給最小值、統計只吃推論值、缺推論
行不以追蹤值遞補、重複的推論行直接拋錯——追蹤那側因為分片後多行是正常的，改以 `shard_id`
重複認出「兩次執行寫進同一份 log」），且**不提供切換口徑的旗標**：給了開關等於承認兩者
可互換。細節見該檔的模組 docstring。

子進程會**繼承整份環境**，工具只設上述四個變數。環境裡另有 `INPUT__`／`MODEL__`／
`FOOT_POINT__`／`TRACKER__` 開頭的變數時（例如 `INPUT__CAMERA_IDS`），那輪量到的是別的
工作量，而 FPS 表上完全看不出來。工具把這些變數收進 meta 的 `extra_settings_env`，並讓它
**進 `report` 的分組 key**：在不同環境下跑的兩輪，結構上不可能被平均在一起。

落腳點方法**不是矩陣的第四條軸**：對照組只掛在單一格上，當成軸會讓每輪從 5 個 run 變
8 個。要跑對照組就再跑一次工具、寫進同一個產物目錄（`report` 本就按組態分組）。
重複次數在展開時放**最外層**，同組態不連跑——否則熱漂移會集中在單一格，離散度那欄
就不再是量測雜訊。

本工具只用 stdlib、不 import torch，也不經過 `YOLODetector`；與另外兩支一樣在 `src/`
之外、不隨 wheel 出貨（ADR-011 的分界）。

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
| `outputs/{bucket_name}/{date}/tracking_results.parts/` | **只在跑到一半時存在**：各追蹤進程的 part 檔（`shard<k>.parquet`）與那一天的鎖檔（`.lock`）。跑完由主進程合併成上一列那個檔並整個刪掉；中途崩掉留下的由下一次執行認領時清掉 |

`tracking_results.parquet` 的**列順序是「逐片相接、片內才交錯」**，改
`[tracker].shards` 就會變。下游 zone／line 都走 `group_by` 向量化、不依賴列順序；比對
兩份輸出也一律先用 `(camera_id, timestamp)` 對齊同一格再逐值比。

`tracking_results.parquet` 的欄位：

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `camera_id` | str | 該影格所屬攝影機的 `<location>_<camera_id>` |
| `frame_id` | int | **片段內**幀序，跨片段會重複（非全日流水號） |
| `timestamp` | datetime（`Asia/Taipei`） | 該片段檔名時間 ＋ 片段內幀序 / fps |
| `track_id` | int | ByteTrack 指派的追蹤編號，跨片段延續。**唯一性只到「同一路之內」**：分片後各追蹤進程各自持有 ByteTrack 的計數器，同一個 id 值會出現在分屬不同片的兩台攝影機（小 bucket 實測 N=1 時 0 個、N=2 時 2442 個 id 跨路重號）。下游 zone／line 都先 `filter(camera_id == ...)` 再 `.over("track_id")`，不受影響；新的消費端要一起帶 `camera_id` |
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
| `config/constants.py` | 非 Pydantic 靜態常數（`OUTPUT_ROOT`、輸出檔名、parts 目錄與鎖檔名、parquet schema、CrowdHuman 類別 id `HEAD_CLASS_ID`／`FBODY_CLASS_ID`） |
| `services/batching.py` | 單次推理批次 `TARGET_BATCH`，與由它推導的 `RING_SLOTS`／`TRACK_QUEUE_SLOTS` |
| `services/pipeline.py` | `analyze_daily` 與多進程編排（讀取／推理／追蹤子進程生命週期） |
| `services/inference.py` | 推理迴圈（湊批、偵測、把偵測框送往追蹤進程） |
| `services/track_worker.py` | 追蹤進程（N 個）：偵測框拆分、追蹤、落腳點推算、座標反算、寫自己那支 part；`TRACK_DONE`／`TRACK_FAILED` 訊號與 fan-out |
| `services/detector.py` | 偵測器與兩批深度的流水（`submit`／`collect`／`in_flight`）：載入前看 metadata 的檢查、ping-pong 緩衝與 stream／event 的定序、自建的前處理（`stack_frames` 在 CPU、`to_infer_tensor` 在 GPU）與後處理（`postprocess_batch`，整批一次 D2H 後用 numpy 過濾）|
| `services/trt_runner.py` | 引擎的 deserialize、execution context，與「把一批排進指定 stream」（`execute_async_v3`）；載入期驗引擎自己宣告的 I/O，enqueue 前核對張量 |
| `services/engine_metadata.py` | 引擎自帶 metadata 的注入格式、讀取與環境比對（建置端與載入端共用同一份） |
| `services/foot_point.py` | `FootPointEstimator`：head 框配對與落腳點推算，整格一次算完（判準算在 `[N, M]` 矩陣上，不逐框迴圈）；自行維護跨幀狀態（每條軌跡上次成功推算的偏移量，共用 head 的那些不存） |
| `services/tracker.py` | 多路追蹤，每路各自獨立的 `BYTETracker` 實例 |
| `services/tracking_results.py` | 追蹤明細累積與 parquet 寫出 |
| `services/output_parts.py` | `tracking_results.parts/` 的全部知識：認領（`parts/.lock`）、清殘骸、片檔命名、fps 加權路由、合併 |
| `services/fps_meter.py` | 處理 FPS 統計。**`detection_fps` 的口徑在 issue #147 換過**：流水化之後它累計的是 `submit` 與 `collect` 兩段 wall time 的合計，也就是「host 為偵測付出的時間」，被 host 工作蓋掉的那段 GPU 時間不算——log 欄位與格式不變，但數字不可與改動前直接相減，跨改動的比較看 `overall_fps` |
| `services/frame_ring.py` | 共享記憶體環形緩衝 |
| `services/letterbox.py` | 在 nv12 平面上縮放到推論尺寸與座標反算（正反兩半共用同一組參數） |
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
解析度、`probe_stream_fps` 讀各路 fps（路由權重，只讀容器標頭不解碼），接著**認領這一天**
（`output_parts.claim_parts_dir`，鎖是 `tracking_results.parts/.lock`），再以多進程拆成
N 個讀取進程 ＋ 1 個推理進程 ＋ `[tracker].shards` 個追蹤進程：

- **影格走共享記憶體、不走 pickle**（`frame_ring.py`）：每路一塊固定格數的環形緩衝
  （`mp.RawArray`），queue 只傳 slot 索引，避免每格影格逐格 pickle 的高成本。推理進程
  直接消費 slot 的 view、不複製出私有副本（`view_slot`）。緩衝依**推論尺寸**（640×384）
  配置——影格在讀取端就縮好了，故每格一律 0.70 MiB 而與來源解析度無關（縮放前移之前
  4K 每格是 23.73 MiB）。此設計仍**假設同一攝影機整天解析度固定**（`probe_frame_shape`
  只探測首格）。多路配置走 `create_ring_buffers`，它在配置任何一塊之前先以
  `require_shm_capacity` 擋下「全部路合計裝不進
  `/dev/shm`」的組態：空間不夠時 CPython 的處置是靜默改用 `/tmp`（磁碟上的 mmap），
  程式照跑、輸出完全正常，只有讀寫成本悄悄變成磁碟等級，因此這裡要在啟動時就 fail
  loud。這道擋是必要非充分（比的是總容量），實際落點仍由 log 的 `backing_dirs` 回答。
- **讀取進程**：解碼後立即 letterbox 成推論尺寸再寫入 slot（`letterbox.py`）。縮放做在
  **解碼出的 nv12 兩個平面上**，縮完才轉 BGR（issue #130 第三階段）：搬動的資料量只有
  BGR 的一半，色彩轉換也只需處理縮小後的畫面，實測讀取端 CPU 地端 2.03 → 0.88 核、
  T4 6.093 → 1.343 核。代價是色度在 1/2 解析度上插值，畫面與「先轉 BGR 再縮」不逐位元
  相同。無空 slot
  時阻塞，形成對推理進程的天然背壓——slot 在**像素複製進 pinned buffer** 的當下就歸還，
  所以這條背壓只覆蓋到那一步為止，forward 與追蹤那兩段分別由流水深度與 `track_queue` 的
  容量上限接手。**時間戳 = 該片段檔名時間 ＋ 片段內幀序 / fps**
  （逐段計算，不能用全日累計幀數推算）。
- **推理進程**：非阻塞輪詢各路 queue 湊批，維持 GPU 批次效率；每批的結果取回來就把逐格
  的偵測框丟進 `track_queue`，本身不做追蹤。前處理與後處理都是本套件自己算的
  （[ADR-013](../docs/adr/video_analyze/013-self-built-pre-post.md)）：整批 `np.stack`
  ＋一次 H2D 之後在 GPU 上翻通道、換軸、轉 float32 並除以 255；forward 之後整批一次
  D2H，再用 numpy 逐格過濾（conf → 截斷 → classes，順序不可調動）。

  **這些步驟排成兩批深度的流水**（[ADR-014](../docs/adr/video_analyze/014-inference-pipelining.md)）：
  `submit` 把一批排進 GPU 就回來，主迴圈接著湊下一批、歸還 slot、送上一批的 payload，
  `collect` 才取回結果——host 的工作與 GPU 的 forward 因此重疊。引擎由
  `services/trt_runner.py` 直接以 `execute_async_v3` 驅動（不再經 ultralytics 的
  `AutoBackend`），每一種緩衝兩份、複製與運算各一條 stream、重用由 event 守住，
  「哪一批的結果配到哪一批的影格」由批次序號核對。

  影格既然是共享記憶體的 view，**像素複製進 pinned buffer 就歸還 slot**——reader 因此
  不必等整段 forward（生命週期約束見
  [ADR-010](../docs/adr/video_analyze/010-zero-copy-frame-lifetime.md)）；送 payload 排在
  歸還之後（payload 取的是推論輸出，與 slot 無關）。
- **追蹤進程**：逐格經 裁切到內容區 → 拆出 fbody／head（只有 fbody 進 tracker）→ 追蹤
  → 推算落腳點 → **框與落腳點映射回原始解析度** → 累積追蹤結果並落盤。追蹤與 GPU 推論
  之間沒有資料相依（下一批的推論不需要上一批的軌跡），拆成兩個進程即可重疊。
  **跨進程傳的是偵測框而不是影格**：每格幾十個框、幾 KB，而影格在推論完成後已無用途。
  空格（該格沒有任何偵測）照樣送 payload、照樣呼叫 `tracker.update`——`BYTETracker` 的
  `frame_id` 與軌跡老化都靠每格呼叫推進。
- **追蹤進程有 N 個**（`[tracker].shards`，預設 2），各自負責一組固定的攝影機：各路的
  tracker 與落腳點推算器狀態獨立、路與路之間不共享，所以追蹤天然可切。路由在啟動時依
  各路 fps 加權貪婪分配（tie-break 用 stream_id，讓同組態的兩輪量測分到一樣的組合），
  整天不變，並記一行 log。**stream_id 維持全域編號**（每片都收完整的 `stream_names`／
  `frame_shapes`），送錯片的唯一訊號是各片入口的歸屬檢查——那片也有全部路的 tracker，
  照樣追蹤、照樣寫進自己的 part，只是該路的軌跡被切成兩段而合併後的檔案完全正常。
  切不開的是輸出（`pq.ParquetWriter` 是行程內的 handle），故每片寫自己的
  `tracking_results.parts/shard<k>.parquet`，主進程在各片都收尾後合併成正式檔名再刪掉
  整個 parts 目錄（見 [ADR-012](../docs/adr/video_analyze/012-track-worker-sharding.md)）。
  **分片不改變任何一格的結果**，只改列順序。

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
- **引擎載入前的檢查，任一不過即中止**（`services/detector.py`）：`model_path` 不是
  `.engine`、引擎檔不存在（`model_path` 是 cwd 相對路徑，跑錯目錄就是這個症狀；自己先
  擋是為了訊息——下一步讀檔頭時拋的例外只有一個路徑字串。經 `YOLO` 載入時這道檢查還
  擋得住 ultralytics `check_file` 的三種替代來源解析，見 ADR-013）、引擎 metadata 與
  當下環境不符
  （compute capability／TensorRT 版本／TensorRT wheel 變體／來源權重 hash，四項各自拋錯、
  訊息指出是哪一項；驅動版本與 torch 的 CUDA 建置版本只記 warning——它們不在 TensorRT 對
  引擎的約束裡）、引擎的精度旗標與宣告不符（`args.half` 非真或 `args.int8` 非假，缺欄
  一律視為不符——`engine_metadata.validate_engine_precision`，逐項比對而非只驗
  `half`：INT8 引擎可以同時把 FP16 flag 設成真，只驗 `half` 對這種引擎完全照過）。
  沒有 CUDA 也是中止而非 fallback CPU。
- **引擎不是 dynamic 建的 → 拋 `ValueError`**。靜態引擎會通過其餘全部載入檢查，然後在
  第一個沒湊滿的批次上被 `set_input_shape` 擋下（訊息看不出原因）——而湊不滿批是
  常態：T4 上實測一次跑完出現 16 種不同的批次大小。
- **引擎自己宣告的 I/O 不合格 → 拋 `ValueError`**（`services/trt_runner.py`，
  deserialize 之後）：不是恰好一入一出、輸入 binding 的 dtype 不是 FP32、optimization
  profile 的 opt 高寬不是 640×384、輸出 binding 的最後一維不是 6（即不是 end2end）。
  四項一律取引擎自己宣告的值，不讀 JSON metadata 的對應欄位——後者改了不會改變引擎。
  沒有內建 NMS 的引擎吐 `(B, 4 + nc, num_anchors)`，自建後處理那三行照樣跑得完，只是
  把類別分數當成 conf 與 cls；opt 高寬不對的引擎則是框都對、只是所有 kernel 都為別的
  尺寸挑（症狀只有變慢）。見 ADR-014。
- **實際進入推論的張量形狀不是 640×384 → 拋 `ValueError`**，dtype 不是 float32、不是
  contiguous、不是 4D 或不在 CUDA 上亦然。驗的是本套件自己組出來、即將交給
  `execute_async_v3` 的那個張量。形狀這一項取代了 `_validate_imgsz`：dynamic 引擎不套用
  metadata 的 `imgsz`（`predictor.py` 只在 backend 不是 dynamic 時才套用），照抄那個檢查
  會得到一個看起來有在驗、其實驗不到的檢查（見 ADR-011）。dynamic 引擎的高寬維也是
  動態的，錯誤的高寬不會被 TensorRT 擋下。
- **`enqueue` 收到 default stream，或三個 TensorRT 呼叫任一個回 `False` → 拋
  `RuntimeError`**。前者讓 TensorRT 插入 `cudaDeviceSynchronize`、流水退化成序列而輸出檔
  一模一樣；後者裡最危險的是 `set_input_shape`——它只回 `False` 不拋錯，接著
  `execute_async_v3` 仍回 `True` 並沿用**上一批**的 shape 跑。
- **`collect` 回傳的批次序號與主迴圈等的那批對不上 → 拋 `RuntimeError`**
  （`services/inference.py`）。錯配的輸出檔完全正常，只有時間戳配到隔壁批。
- **單次批次超過引擎綁的最大批次 → 拋 `ValueError`**（`services/inference.py`，在主迴圈的
  try 內，送得出 `TRACK_FAILED`）。
- **`frame_shapes` 傳成推論尺寸 → `run_track_worker` 拋 `ValueError`**。那代表
  拿到的是縮放後的尺寸而非原始解析度，反算會退化成恆等、parquet 的 `frame_width` 也一起
  寫成 640，而下游 zone／line 只檢查欄位存在（ADR-004／ADR-006）。影格尺寸與緩衝配置
  不符（例如讀取端漏了 `letterbox_nv12()`）則由 `write_slot` 擋下。
- **解碼出的影格不是 `nv12` → 讀取子進程拋 `ValueError`**。讀取端的縮放綁在 nv12 的
  平面佈局上（前 H 列 Y、其後 H/2 列交錯 UV）；yuv420p 攤成 ndarray 的形狀與 nv12
  完全相同，只是 U、V 分成兩塊而非交錯，當成 nv12 縮不會拋錯、只會讓顏色靜默錯亂。
  `allow_software_fallback=False` 擋的是整條退回軟解，擋不到這裡。
- **等比縮放後的寬高不是偶數 → `letterbox_nv12` 拋 `ValueError`**（非 16:9 來源）。
  色度平面是 2×2 取樣，奇數尺寸拼不回 nv12；此處不把尺寸調成偶數，那會讓實際縮放
  比例與反算用的 `letterbox_params` 分岔，每個座標靜默偏掉而輸出完全正常。
- `analyze_daily` 以 0.5 秒輪詢所有子進程；任一非零結束 → 先終止所有子進程再拋
  `RuntimeError`；`KeyboardInterrupt` → 終止後以 exit code 130 收斂。
- 每片的 part 檔先寫 `.tmp`、收到 `TRACK_DONE` 才 `rename` 成 `shard<k>.parquet`
  （`rename` 具原子性），主進程等各片都到齊才合併成正式檔名。推論進程中途例外時把
  `TRACK_FAILED` **送到每一片**，各片收到就刪除自己的 `.tmp` 並以非零 exitcode 結束，
  正式檔名下不會出現不完整的 parquet。fan-out 給 `TRACK_FAILED` 帶 1 秒上限、逾時記
  warning 續下一片：它是序列的 `put`，前一片若已經死了、它的 queue 又是滿的，無 timeout
  的 `put` 會永久阻塞而**後面的片一個都收不到**；`TRACK_DONE` 反過來不帶上限——走到那裡
  各片都在正常消化，而這個訊號掉了就缺一支 part、合併會 fail loud。這條路徑**依賴 `track_queue` 的
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
- **整機重啟／追蹤進程本身被 SIGKILL** 沒有任何 in-process 的機制擋得住，改由**下一次跑
  同一天的執行**在啟動時清掉（`output_parts.claim_parts_dir`）。判準是「還有沒有進程持有
  `tracking_results.parts/.lock` 的 `flock`」而不是檔名或 mtime：拿得到鎖代表持有者已經
  不在，parts 目錄裡除了鎖檔以外的東西全部清掉；拿不到鎖代表另一個執行正在跑同一天，
  **當場 fail loud，一個 byte 都不動**。因此同一個 bucket 的同一天不能有兩個執行並行，
  要並行請分開 bucket 或分開日期。三個容易踩到的細節：
  - **鎖由主進程持有**，子進程靠 `fork` 繼承同一個 open file description。主進程被
    SIGKILL 之後孤兒子進程仍守著鎖，另一個執行照樣擋得下——改成 spawn 會靜默失去這道
    保護。錯誤訊息叫人 `pgrep -af video_analyze` 找出來收掉再重跑。
  - **清殘骸不刪 `.lock` 自己**。`rmtree` 會把它一起帶走，鎖就留在一個沒有檔名的 inode
    上，另一個執行馬上能在新建的 inode 上取得鎖，兩邊都以為自己獨佔而輸出都完全正常。
  - **改版前的 `tracking_results.parquet.tmp` 殘檔不再有人清**，認領時只記一行 warning
    指出路徑：它可能仍被舊版的孤兒追蹤進程持有著 flock，而新版已經不看那把鎖。確認沒有
    舊版進程還在跑之後手動刪即可；合併的暫存檔也因此改放在 parts 目錄裡，不寫那條路徑。

  收尾時刪檔與改名都認 inode 而不認路徑：part 的暫存檔若已被換成別份，`discard()` 留著
  不動、`save()` 直接 fail loud（那時本次結果已隨原本的檔案一起遺失）。

## 開發

```bash
uv run --directory video_analyze ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory video_analyze pytest         # 執行測試
```

此處用 `--directory`（切換 cwd 進 `video_analyze/`）而非執行分析時的 `--package`：
`--package` 不會改變 cwd，pytest 會從 repo 根遞迴收集到全部套件的 `tests/`，因四包皆有
同名測試檔（如 `test_config.py`）而撞名衝突報錯；`--directory` 才會讓 pytest 只解析
本套件自己的 `tests/`。
