# video-analyze

多路離線影片的偵測與追蹤：以「一天」為單位，讀取多路攝影機的錄影片段，用 YOLO
（僅偵測 `person`）搭配 ByteTrack 做多路追蹤，產出每格的追蹤明細。

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
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，本套件附 `uv.lock`） |
| GPU | 選用。以 `torch.cuda.is_available()` 判斷，無 GPU 時 fallback 到 CPU（明顯變慢） |
| 系統相依 | FFmpeg / 影像編解碼器（OpenCV 解 `mkv` 等格式）；`lap` 為 C 擴充，環境無對應 wheel 時需要編譯工具鏈 |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `opencv-python` | 影片片段讀取與標註影片輸出 |
| `ultralytics` | YOLO 偵測 |
| `torch` / `torchvision` | 推理後端（與 `ultralytics` 一併釘住版本） |
| `lap` | ByteTrack 的線性指派求解 |
| `numpy` | 影格與追蹤結果的陣列運算 |
| `polars` / `pyarrow` | 追蹤明細 parquet 寫出 |
| `pydantic` | 設定與 registry 的資料模型與驗證 |
| `pyyaml` | 讀取 `camera_registry.yaml` |

依賴版本以 `==` 釘住並附 `uv.lock`，固定推理堆疊。

**模型權重**：`config.toml` 的 `model_path`（預設 `yolo26m.pt`）指向的權重檔不進版控
（`.gitignore` 排除所有 `*.pt`）；若本機找不到該檔，ultralytics 會**靜默地自動下載**對應
權重、不會報錯。

## 安裝與執行

```bash
uv sync --project video-analyze
```

準備下列輸入後即可執行：

1. 一份本機的 `bucket_dir/`，內含各攝影機的影片片段與 `camera_registry.yaml`
   （格式見[設定](#設定)）。
2. 本套件根目錄的 `config.toml`（指定要跑哪個 bucket、哪一天、哪些攝影機與各項參數）。

```bash
uv run --project video-analyze video-analyze
```

此命令不接受任何旗標，所有參數都讀自 `config.toml`。

> **於倉庫根目錄執行**：`bucket_dir`、輸出根目錄 `outputs/` 與 `model_path` 皆為 **cwd
> 相對路徑**，`uv run --project` 不改變 cwd。本套件自己的 `config.toml` 則以 `__file__`
> 定位，不受 cwd 影響。

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
gmc_method = "none"

[model]
model_path = "yolo26m.pt"
batch = 8

[output]
save_video = false         # 是否輸出標註影片（開發 / 偵錯輔助）

[input]
bucket_dir = "bucket_name1"
date = 2026-05-01
camera_ids = []            # 空 = camera_registry.yaml 內全部攝影機
```

各區塊的主要欄位與約束：

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[tracker]` | ByteTrack 各項閾值 | 見範例 | `*_thresh` 皆介於 0–1，`track_buffer >= 1` |
| `[model]` | `model_path` | `"yolo26m.pt"` | 權重檔路徑 |
| | `batch` | `1` | YOLO 推理湊批目標，`>= 1`（範例用 `8`）；實際單次推理批次為此值的 2 倍 |
| `[output]` | `save_video` | `false` | 是否輸出標註影片（開發 / 偵錯用途） |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（範例用 `bucket_name1`） |
| | `date` | — | 分析日期 |
| | `camera_ids` | `[]` | 要分析的攝影機；空清單 = 全部 |

### `camera_registry.yaml`（資料樣貌）

放在每個 `bucket_dir` 根目錄下。**此檔不進版控**（隨 `bucket_name*/` 一起被 `.gitignore`
排除），需依實際部署環境人工維護。

攝影機片段的目錄結構為：

```
<bucket_dir>/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv
```

> **時區處理**：檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，本套件在 `io/video_reader.py`
> 解析時即把它轉換成台北在地時間（`Asia/Taipei`，UTC+8）；`tracking_results.parquet` 的
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
| | `participates_in_zone_mapping` | bool | `true` | 接受但不使用（語義由 `zone-map` 實作） |
| | `zones` | list | `[]` | 接受但不使用；幾何刻意不在本階段驗證 |

本套件只讀攝影機身份（`camera_id` / `location` / `ip`）用來定位片段目錄與過濾攝影機。
`participates_in_zone_mapping` 與 `zones` 由下游的 `zone-map` 使用；registry 的資料模型
不接受未列出的欄位，故這兩個欄位仍須保留於模型中並忽略其值。

使用限制（皆為 fail-loud，違反時直接報錯）：

- **`camera_id` 與 `location_camera_id` 皆須唯一**。兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，因此在載入時即擋下。
- `config.toml` 的 `camera_ids` 若含 registry 中查無的 ID，直接報錯。
- `cameras[]` 不接受未列出的欄位（多打的欄位會報錯）。

## 函式介面

```python
analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult
```

回傳的 `AnalysisResult` 含 `date` / `camera_ids` / `tracking_results_path` /
`output_video_paths`。`bucket_dir` 以參數傳入（而非讀全域 `settings`），故可重複以不同
bucket 呼叫。

## 輸出檔案

| 路徑 | 內容 |
| --- | --- |
| `outputs/{bucket_name}/{date}/tracking_results.parquet` | 追蹤明細 |
| `outputs/{bucket_name}/{stream_dirname}/{YYYY}/{MM}/{DD}/…`（鏡射輸入路徑） | 逐片段標註影片，`save_video = true` 時才產出（開發 / 偵錯輔助） |

`tracking_results.parquet` 的欄位：

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `camera_id` | str | 該影格所屬攝影機的 `<location>_<camera_id>` |
| `frame_id` | int | **片段內**幀序，跨片段會重複（非全日流水號） |
| `timestamp` | datetime（`Asia/Taipei`） | 該片段檔名時間 ＋ 片段內幀序 / fps |
| `track_id` | int | ByteTrack 指派的追蹤編號，跨片段延續 |
| `x1` / `y1` / `x2` / `y2` | float | 追蹤框的像素座標 |

## 架構

### 模組結構

`src/video_analyze/` 依職責分層，依賴方向單向、無循環：

| 模組 | 職責 |
| --- | --- |
| `config.py` | Pydantic 設定模型與全域 `settings` 單例（模組載入時建立，各模組直接 import 使用） |
| `registry.py` | `CameraRegistry` / `CameraEntry`，讀 `camera_registry.yaml` |
| `io/video_reader.py` | 逐日掃描片段、讀影格 |
| `io/video_writer.py` | 標註影片輸出 |
| `io/frame_ring.py` | 共享記憶體環形緩衝 |
| `visualization/visualizer.py` | `TrackAnnotator`，畫追蹤框 |
| `detector.py` | YOLO 偵測 |
| `tracker.py` | 多路追蹤，每路各自獨立的 `BYTETracker` 實例 |
| `fps_meter.py` | 處理 FPS 統計 |
| `inference.py` | 推理迴圈（湊批、偵測、追蹤、寫檔） |
| `tracking_results.py` | 追蹤明細累積與 parquet 寫出 |
| `pipeline.py` | `analyze_daily` 與 CLI 進入點 `run_analyze` |

### 多進程 pipeline

`analyze_daily` 在主進程先 `discover_segments` 掃出當天片段、`probe_frame_shape` 讀首格
解析度，再以多進程拆成 N 個讀取進程 ＋ 1 個推理進程：

- **影格走共享記憶體、不走 pickle**（`frame_ring.py`）：每路一塊固定格數的環形緩衝
  （`mp.RawArray`），queue 只傳 slot 索引，避免每格影格逐格 pickle 的高成本。此設計
  **假設同一攝影機整天解析度固定**。
- **讀取進程**：無空 slot 時阻塞，形成對推理進程的天然背壓。**時間戳 = 該片段檔名時間 ＋
  片段內幀序 / fps**（逐段計算，不能用全日累計幀數推算）。
- **推理進程**：非阻塞輪詢各路 queue 湊批，維持 GPU 批次效率；每個 packet 依序經
  偵測 → 追蹤 → 累積追蹤結果 → 畫框 → 寫檔。
- **mp4v 編碼在背景執行緒**，與下一批 GPU 推理重疊。關檔順序有講究：某路尾端影格常與
  該路結束訊號同批出現，必須等這批全部寫完才關檔，否則背景緒會先收尾、之後補寫的影格
  會把檔案截斷。

### fail-loud 錯誤處理

- 檔名格式錯誤 → `discover_segments` 在主進程直接拋 `ValueError`；各攝影機首段的開檔 /
  讀影格失敗 → `probe_frame_shape` 同樣在主進程拋 `ValueError`（子進程尚未啟動）。
- 其餘片段的開檔 / 讀 FPS 失敗 → 讀取子進程拋錯、以非零 exitcode 結束。
- `analyze_daily` 以 0.5 秒輪詢所有子進程；任一非零結束 → 先終止所有子進程再拋
  `RuntimeError`；`KeyboardInterrupt` → 終止後以 exit code 130 收斂。
- 追蹤明細 parquet 先寫 `.tmp`、全部串流結束後才 `rename` 成正式檔名（`rename` 具
  原子性）；中途例外則刪除 `.tmp`，正式檔名下不會出現不完整的 parquet。

## 開發

```bash
uv run --directory video-analyze ruff check .   # lint（line-length = 88，select = ["E", "F", "I", "W"]）
uv run --directory video-analyze pytest         # 執行測試
```

此處用 `--directory`（切換 cwd 進 `video-analyze/`）而非執行分析時的 `--project`，
pytest 才會收集到本套件的 `tests/`。
