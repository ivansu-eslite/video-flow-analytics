# video-flow-analytics

多路離線影片流分析系統：以「一天」為單位，從本機模擬的 GCS bucket 目錄結構讀取各攝影機影片片段，用 YOLO（僅偵測 `person`）搭配 ByteTrack 做多路追蹤，再將追蹤結果轉換成「區域人流統計」與「Excel 人流報表」。

## 功能特色

三個階段刻意獨立、可分別重跑：

1. **analyze（偵測 / 追蹤）**：YOLO 偵測 + ByteTrack 多路追蹤，輸出追蹤明細 Parquet 與逐片段標註影片。GPU、多進程，執行成本高。
2. **zone-map（區域人流統計）**：讀取階段一輸出的 Parquet，依 `camera_registry.yaml` 的 zone 幾何定義統計人流，輸出 `zone_counts.parquet`。純 CPU 向量化運算，不需重跑偵測。
3. **report（Excel 報表）**：讀取 `zone_counts.parquet`，彙總成跨日累加更新的 Excel 人流報表。純 CPU 運算，不需重跑偵測或 zone mapping。

只要調整 zone 幾何定義，只需重跑 `zone-map`；只要調整報表參數，只需重跑 `report`，不必重跑昂貴的 GPU 偵測。

## 設計理念

- **進入點刻意設計成函式呼叫**（`analyze_daily`／`map_zones_daily`／`export_report_daily`），CLI 只是從 `config.toml` 組參數後呼叫，兩者是分離的兩層——未來換掉觸發方式（例如 Airflow）時，只需要換呼叫這些函式的「外殼」，不需動 pipeline 本身。
- **三階段之間不靠記憶體傳遞資料，只靠檔案（parquet／yaml 快照）交接**：下游階段靠檢查上游輸出檔案是否存在來驗證相依關係（例如 `map_zones_daily` 檢查 `tracking_results.parquet`），而非透過回傳值或記憶體物件。這代表任何 orchestrator 都能個別重跑其中一個階段，只要對應輸入檔案還在。
- **所有輸出都走 `.tmp` 檔 + 原子 `rename`**（`tracking_results.parquet`／`zone_counts.parquet`／`report.xlsx` 皆同），加上 `report` 的 `on_duplicate_date="overwrite"` 預設值，讓「同一個階段對同一天重跑」天生冪等、不會產生半成品或重複資料。

## 環境需求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（套件管理與執行）
- GPU（選用）：`YOLODetector` 以 `torch.cuda.is_available()` 判斷，無 GPU 時會 fallback 到 CPU（明顯變慢）

## 安裝

```bash
uv sync
```

## 資料目錄結構

分析目標為本機模擬的 GCS bucket 目錄，格式為：

```
<bucket_dir>/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv
```

檔名的 `Z` 尾綴排版沿用 RFC 3339，但實際時間並非真正的 UTC：攝影機錄影時鐘本身就是台北時間（UTC+8）。

每個 bucket 根目錄下需有一份 `camera_registry.yaml`，描述攝影機清單與各攝影機的 zone 定義，範例：

```yaml
bucket_name: bucket_name

storage:
  file_ext: mkv
  target_codec: h265
  segment_strategy: time
  segment_seconds: 1800

cameras:
  - camera_id: cam001
    location: test
    ip: 192.168.104.115
    zones:
      - name: 平擺桌
        polygon: [[640.01, 866.83], [521.34, 938.8], ...]
```

此檔案不進版控，需依實際部署環境自行維護。

## 設定

執行參數集中在專案根目錄的 `config.toml`：

```toml
[tracker]
track_high_thresh = 0.5
# ... 其餘 ByteTrack 參數

[model]
model_path = "yolo26m.pt"
batch = 8

[output]
save_video = true

[input]
bucket_dir = "bucket_name1"
date = 2026-05-01
camera_ids = []           # 空 = camera_registry.yaml 內全部攝影機

[zone]
bucket_minutes = 15
entry_debounce_frames = 1

[report]
period_minutes = 60
metric = "entries"        # "entries" 或 "unique_visitors"
on_duplicate_date = "overwrite"  # "overwrite" / "append" / "error"
```

`camera_registry.yaml`（資料長什麼樣＋zone 定義）與 `config.toml`（這次要怎麼跑）分工分開。

## 使用方式

```bash
# 偵測/追蹤：讀影片跑 YOLO+ByteTrack，輸出 tracking_results.parquet 與標註影片
uv run video-flow-analytics analyze

# 區域人流統計：讀 tracking_results.parquet 套 zone 幾何，輸出每時段每區域人流
uv run video-flow-analytics zone-map

# Excel 報表：讀 zone_counts.parquet 彙總成跨日累加的人流報表
uv run video-flow-analytics report
```

三個子命令的參數皆讀取 `config.toml` 對應區塊；`analyze` 進入點另外可用 `analyze.pipeline.analyze_daily(date, bucket_dir, camera_ids=None)` 以函式呼叫、支援不同 bucket 重複呼叫。

## 三階段函式介面（輸入／輸出）

### `analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult`

- **輸入**：`bucket_dir/camera_registry.yaml` + 當日各攝影機影片片段
- **輸出**：`tracking_results.parquet`、（依設定）逐片段標註影片，回傳 `AnalysisResult`（`date`／`camera_ids`／`tracking_results_path`／`output_video_paths`）
- **備註**：GPU + 多進程，成本最高

### `map_zones_daily(date, bucket_dir, bucket_minutes, entry_debounce_frames=1) -> Path`

- **輸入**：階段一輸出的 `tracking_results.parquet` + `camera_registry.yaml` 的 zone 定義
- **輸出**：`zone_counts.parquet` + `camera_registry_used.yaml`（快照），回傳 parquet 路徑
- **備註**：純 CPU 向量化運算

### `export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date, bucket_minutes) -> Path`

- **輸入**：階段二輸出的 `zone_counts.parquet` + 對應的 `camera_registry_used.yaml` 快照（不是即時的 `camera_registry.yaml`）
- **輸出**：跨日累加更新的 `report.xlsx`，回傳其路徑
- **備註**：純 CPU 運算

## 輸出

- `outputs/{bucket_name}/{date}/tracking_results.parquet`：追蹤明細
- `outputs/{bucket_name}/{date}/...`：逐片段標註影片（路徑鏡射輸入、根目錄換成 `outputs/{bucket_name}/`）
- `outputs/{bucket_name}/{date}/zone_counts.parquet`：各 zone 人流統計
- `outputs/{bucket_name}/report.xlsx`：跨日累加更新的 Excel 人流報表（含「每小時人流」「每日尖峰」「活動事件」三個分頁）

## 開發

```bash
uv run ruff check .     # lint（line-length=88, select=["E","F","I","W"]）
uv run pytest           # 執行測試
```

## 未來銜接 Airflow 時的考量（給 DevOps 參考）

- **結論**：現有函式介面（顯式參數、檔案驅動相依、原子寫入冪等）本來就是 Airflow 友善的形狀，建議 Airflow 直接呼叫三個 `_daily()` 函式本身（PythonOperator/TaskFlow），**不要**透過 CLI／`run_xxx()`，因為那層綁定模組載入時期建立的全域 `settings` 單例與固定的 `config.toml` `[input].date`，無法對應每個 DAG run 不同的日期/bucket。
- **待處理落差點（與執行環境無關，先修）**：`analyze_daily` 目前只攔截 `KeyboardInterrupt` 與一般 `Exception` 做子進程清理；Airflow 對逾時/手動失敗/DAG clear 送出的是 SIGTERM，預設不會進到這些 except 分支，可能讓讀取/推理子進程變孤兒、繼續佔用 GPU。建議之後導入前先補上 SIGTERM 的訊號處理再串接。
- **待評估落差點（視未來選定的 executor 而定）**：`analyze_daily` 內部用 `mp.Process`（fork）。若 Airflow 用 CeleryExecutor 的 prefork worker，在已經 fork 過的 worker 行程裡再 fork 是已知容易出問題的組合；若用 KubernetesPodOperator 或每個 task 獨立行程/容器執行則無此疑慮。等決定 executor 時要重新檢視。
- **設計上不需要改的地方**：`YOLODetector`/`MultiStreamByteTracker` 讀的 `settings.model`/`settings.tracker`（模型路徑、追蹤器門檻）目前是全域單例、非函式參數；這類「部署層級」調參設定通常不需要每個 DAG run 各自覆寫，目前把「每次執行會變的業務參數」（date/bucket_dir/camera_ids/bucket_minutes/period_minutes/metric/on_duplicate_date）都做成顯式函式參數、把「調參設定」留在 `config.toml` 全域單例，這條界線是對的，不需要為了 Airflow 而重構。

## 架構

套件依職責分為五個子套件（`core`、`io`、`visualization`、`analyze`、`zone_mapping`、`report`），依賴方向單向、無循環。詳細架構、多進程 pipeline 設計、錯誤處理策略與各階段實作細節，請參閱 [CLAUDE.md](CLAUDE.md)。
