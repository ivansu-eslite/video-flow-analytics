# flow-report

人流統計分析與 BI 報表：把「每時段、每區域」的事件統計做跨期間彙總與分析，持續寫入單一
Excel 報表，供 Looker Studio 等 BI 工具接手做長期視覺化。

## 概述

輸入是區域事件統計 `zone_counts.parquet` 與產生它當時的 `camera_registry_used.yaml`
快照；把多個 `bucket_minutes` 併成 `period_minutes`、算出每日尖峰與用餐時段提醒，
**持續 Append** 至同一份 `outputs/{bucket}/report.xlsx`（而非逐日各產一份），讓 BI 工具
對這份不斷累加的資料做長期觀測。

報表含三個分頁：

| 分頁 | 欄位 | 寫入者 |
| --- | --- | --- |
| 每小時人流 | 日期、星期、小時、區域、人流量 | 本階段 |
| 每日尖峰 | 日期、星期、區域、尖峰時段、尖峰人流、每日提醒 | 本階段 |
| 活動事件 | 日期、星期、開始時間、結束時間、區域、活動名稱、活動類型 | 其他來源；本階段只建立標題列、不寫入 |

「每日提醒」依尖峰時段所在小時給出用餐動線提醒（11–14 時為午餐、17–20 時為晚餐，其餘為
「無」）。純 CPU 運算，不需重跑偵測或區域事件統計；只調整報表參數時僅需重跑本階段。

**進入點是函式呼叫，CLI 只是外殼**：核心是
`export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date,
bucket_minutes, output_root=OUTPUT_ROOT) -> Path`，CLI 只是從 `config.toml` 組出參數後
呼叫它。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，附 `uv.lock`） |
| GPU | 不需要，純 CPU |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `openpyxl` | 讀寫 Excel 報表 |
| `polars` / `pyarrow` | parquet 讀取與期間彙總 |
| `pydantic` | 設定與 registry 的資料模型與驗證 |
| `pyyaml` | 讀取 `camera_registry_used.yaml` 快照 |

## 安裝與快速開始

```bash
uv sync --project flow-report
uv run --project flow-report flow-report
```

CLI 不接受任何旗標，所有參數都讀自 `config.toml`。執行前需備妥：

1. 當日的 `outputs/{bucket}/{date}/zone_counts.parquet` 與同層的
   `camera_registry_used.yaml` 快照。
2. `flow-report/config.toml`（指定本次要彙總哪個 bucket、哪一天與報表參數）。

### 執行位置（cwd 約束）

**一律在倉庫根目錄執行，以 `--project` 指定本套件**（`uv run` 不改變 cwd）。下列兩者皆為
**cwd 相對路徑**，並非相對於本資料夾：

| 路徑 | 來源 | 在 `flow-report/` 內執行時 |
| --- | --- | --- |
| `OUTPUT_ROOT = outputs/` | `pipeline.py` 常數 | 去 `flow-report/outputs/` 找輸入而 `FileNotFoundError`，報表也落在錯的樹 |
| `settings.input.bucket_dir` | `config.toml` `[input]` | 只取其目錄名來組路徑，故實務上不會走到；上一列的輸入檢查會先失敗 |

本套件自己的 `config.toml` 以 `__file__` 定位（`parents[2]`），不受 cwd 影響。

## 設定

`config.toml` 置於本套件根目錄（`flow-report/config.toml`），含 `[input]`、`[zone]`、
`[report]` 三個區塊。找不到此檔時會印出警告並回退到各項預設值。範例：

```toml
[input]
bucket_dir = "bucket_name1"
date = 2026-05-01

[zone]
bucket_minutes = 60           # 上游 zone_counts.parquet 的時段粒度（分鐘）

[report]
period_minutes = 60           # 報表彙總粒度；須為 zone.bucket_minutes 的倍數
metric = "entries"            # "entries" 或 "unique_visitors"
on_duplicate_date = "append"  # "overwrite" / "append" / "error"
```

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（cwd 相對）；本階段只取其目錄名來組出 `outputs/{bucket}/` 路徑 |
| | `date` | — | 彙總日期；未設定時報錯 |
| `[zone]` | `bucket_minutes` | `60` | 上游 `zone_counts.parquet` 的時段粒度（分鐘），`>= 1`；須與產生該份 parquet 時的 `zone-mapping/config.toml` 一致 |
| `[report]` | `period_minutes` | `60` | 報表彙總粒度（分鐘），`>= 1`，且**須為 `zone.bucket_minutes` 的倍數**（否則 fail-loud 報錯） |
| | `metric` | `"entries"` | `"entries"` 或 `"unique_visitors"`；決定「人流量」「尖峰人流」用哪個統計量 |
| | `on_duplicate_date` | `"append"` | 同日期重跑的處理：`"overwrite"` / `"append"` / `"error"` |

`on_duplicate_date` 三種模式的行為：

| 模式 | 行為 |
| --- | --- |
| `overwrite` | 先刪除既有相同日期的列再插入，並依日期／區域重新排序；天生冪等 |
| `append` | 直接附加到尾端、不檢查；重跑同一天會產生重複列 |
| `error` | 發現重複日期即整個中止，不寫入任何內容 |

`camera_registry.yaml`（攝影機清單與區域定義，放在 `bucket_dir` 根目錄、不進版控）的完整
格式見根 README。本階段讀的是它在 `outputs/{bucket}/{date}/` 下的快照
`camera_registry_used.yaml`，相關使用限制（皆為 fail-loud，違反時直接報錯）：

- **`zone` 名稱須全域唯一**：本階段以區域名稱、不含 `camera_id` 分組彙總，同名區域會讓
  不同攝影機的人流被合併成同一列，故不只同一攝影機內不可重複，跨攝影機也不可重複。
- **`camera_id` 與 `location_camera_id` 皆須唯一**：兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，載入 registry 時即擋下。
- **`polygon` 至少需要 3 個頂點**，座標為該攝影機固定解析度下的像素座標。
- **`participates_in_zone_mapping = false`** 的攝影機不列入上述 zone 名稱唯一性驗證，
  其 `zones` 內容不影響本階段。
- **`zone_counts.parquet` 內的 `(camera, zone)` 組合須全部存在於快照定義內**：出現快照
  沒有的組合視為資料與定義不一致，直接報錯。正常流程下不會觸發——上游 `zone-mapping`
  已先用同一份快照過濾出參與 zone mapping 的攝影機才產生 parquet。

**驗證對象是快照、不是當下的 `camera_registry.yaml`**：兩者之間若改過區域命名，拿當下的
檔案驗證會通過，但 parquet 裡其實是舊定義，不同攝影機的人流會被靜默合併。

## 輸入 / 輸出檔案

`{bucket}` = `bucket_dir` 的目錄名，皆位於倉庫根目錄的 `outputs/` 下：

| 路徑 | 讀 / 寫 | 內容 |
| --- | --- | --- |
| `outputs/{bucket}/{date}/zone_counts.parquet` | 讀 | 每時段每區域事件統計，欄位 `camera_id` / `zone` / `time_bucket` / `unique_visitors` / `entries`；缺少時報錯 |
| `outputs/{bucket}/{date}/camera_registry_used.yaml` | 讀 | 產生該份 parquet 當時的 registry 快照；缺少時報錯 |
| `outputs/{bucket}/report.xlsx` | 寫 | 跨日累加的 Excel 報表（三個分頁）；不存在時建立，存在時依 `on_duplicate_date` 更新 |

**時區**：`zone_counts.parquet` 的 `time_bucket` 已是台北在地時間（`Asia/Taipei`），本階段
只去掉時區標記、保留原本的 wall-clock 值，不做任何時區位移。

**寫入冪等**：`report.xlsx` 先寫入 `.tmp` 再 `rename` 成正式檔名，藉由 `rename` 的原子性
確保中斷時不會在正式檔名下留下半成品。但**內容層面是否冪等取決於 `on_duplicate_date`**
（見上表）。

## 已知限制

- **`metric = "unique_visitors"` 的彙總為近似值**：`unique_visitors` 是各 bucket 內的不
  重複人數，跨相鄰 bucket 停留的同一人會在彙總時被重複計入；`zone_counts.parquet` 未保留
  原始 `track_id`，本階段無法在彙總時去重。`metric = "entries"` 本身即為可疊加的事件
  次數，不受此影響。

## 開發

```bash
uv run --directory flow-report ruff check .   # lint（line-length = 88，select = ["E", "F", "I", "W"]）
uv run --directory flow-report pytest         # 執行測試
```

> 測試的 cwd 要求與執行 CLI 相反：這裡用 `--directory`（會 chdir 進 `flow-report/`），
> 讓 pytest 的 rootdir 解析到本套件；測試本身不碰 `bucket_dir` 與 `outputs/`。
> 等價寫法：`uv run --project flow-report pytest flow-report/tests`。
