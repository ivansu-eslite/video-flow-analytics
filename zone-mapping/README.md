# zone-mapping

區域事件統計：把一整天的追蹤明細對映到各攝影機的區域（zone）幾何，轉換成「每時段、
每區域」的事件統計。

## 概述

輸入是追蹤明細 `tracking_results.parquet` 與 `camera_registry.yaml` 的區域定義；以每個
track 的腳底中心點 `((x1 + x2) / 2, y2)` 做 ray-casting 判定是否落在區域多邊形內，再依
`time_bucket` 聚合出兩項事件指標，輸出 `zone_counts.parquet`：

| 指標 | 定義 |
| --- | --- |
| `unique_visitors` | 該時段內在區域出現過的不重複 `track_id` 數 |
| `entries` | 「區域外 → 區域內」的轉換次數，`entry_debounce_frames` 控制去抖 |

本階段只做「事件轉化」，不做跨期間彙總或分析。純 CPU 向量化運算，不需 GPU。只調整區域
幾何時僅需重跑本階段。

**進入點是函式呼叫，CLI 只是外殼**：核心是
`map_zones_daily(date, bucket_dir, bucket_minutes, entry_debounce_frames=1,
output_root=OUTPUT_ROOT) -> Path`（在 `services/zone_map.py`），CLI 進入點
`zone_mapping.main:main` 只是從 `config.toml` 組出參數後呼叫它。

原始碼採 DDD 分層（`src/zone_mapping/`）：

| 目錄 | 內容 |
| --- | --- |
| `main.py` | CLI 外殼：讀 `settings` → 組參數 → 呼叫 `map_zones_daily` |
| `config/constants.py` | 非 Pydantic 靜態常數（輸出根目錄、輸入輸出檔名、parquet schema） |
| `models/config.py` | pydantic-settings 設定模型與全域單例 `settings` |
| `services/zone_map.py` | 讀檔、逐攝影機/逐區域套用演算法、寫檔與 registry 快照 |
| `services/stats.py` | point-in-polygon 判定與人流聚合等純函式 |

`camera_registry.yaml` 的模型與 zone 驗證（`vfa_registry`）、單行 JSON 的
`StructuredLogger`（`vfa_observability`）由三包共用的 lib 提供，以 path 依賴引用，
不在本包內：[libs/vfa-registry](../libs/vfa-registry)、
[libs/vfa-observability](../libs/vfa-observability)。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，附 `uv.lock`） |
| GPU | 不需要，純 CPU |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `numpy` | 向量化幾何運算（ray-casting 區域判定） |
| `polars` / `pyarrow` | parquet 讀寫與聚合 |
| `pydantic` / `pydantic-settings` | 設定與 registry 的資料模型與驗證，config 從 `config.toml`／環境變數載入 |
| `pyyaml` | 讀取 `camera_registry.yaml` |
| `vfa-registry` | 共用 lib：`camera_registry.yaml` 的模型與 zone 驗證（path 依賴）|
| `vfa-observability` | 共用 lib：單行 JSON 的 `StructuredLogger`（path 依賴）|

## 安裝與快速開始

```bash
uv sync --project zone-mapping
uv run --project zone-mapping zone-mapping
```

CLI 不接受任何旗標，所有參數都讀自 `config.toml`。執行前需備妥：

1. 當日的 `outputs/{bucket}/{date}/tracking_results.parquet`。
2. 一份本機的 `bucket_dir/`，內含 `camera_registry.yaml`（格式見根 README 的
   [設定](../README.md#設定)）。
3. `zone-mapping/config.toml`（指定本次要跑哪個 bucket、哪一天與統計參數）。

### 執行位置（cwd 約束）

**一律在倉庫根目錄執行，以 `--project` 指定本套件**（`uv run` 不改變 cwd）。下列兩者皆為
**cwd 相對路徑**，並非相對於本資料夾：

| 路徑 | 來源 | 在 `zone-mapping/` 內執行時 |
| --- | --- | --- |
| `OUTPUT_ROOT = outputs/` | `config/constants.py` 常數 | 去 `zone-mapping/outputs/` 找輸入而 `FileNotFoundError`，產出也落在錯的樹 |
| `settings.input.bucket_dir` | `config.toml` `[input]` | 對到不存在的 `zone-mapping/bucket_name1`；實務上不會走到，上一列的輸入檢查會先失敗 |

本套件自己的 `config.toml` 以 `find_project_root`（往上找 `pyproject.toml`）定位，不受
cwd 影響。

## 設定

`config.toml` 置於本套件根目錄（`zone-mapping/config.toml`），只含 `[input]` 與 `[zone]`
兩個區塊，透過 pydantic-settings 載入；**找不到此檔**時會印出警告並以各項預設值啟動，
**此檔存在但參數不合法**則直接報錯（不靜默套用預設值）。同理，出現**未知的頂層區塊**
（例如把 `[zone]` 拼成 `[zones]`）也會直接報錯，而非被靜默忽略。各欄位亦可用環境變數覆寫
（巢狀分隔符 `__`，例如 `ZONE__ENTRY_DEBOUNCE_FRAMES=3`）。注意欄位名未加前綴，
`ZONE`／`INPUT` 這兩個名稱本身也是有效的覆寫來源（設成 JSON 會整段取代該區塊），
在共用的執行環境中留意不要與其他程式的環境變數撞名。範例：

```toml
[input]
bucket_dir = "bucket_name1"
date = 2026-05-01

[zone]
bucket_minutes = 60        # 事件統計時間粒度（分鐘）
entry_debounce_frames = 1  # 進場去抖；1 = 不去抖
```

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（cwd 相對） |
| | `date` | — | 統計日期；未設定時報錯 |
| `[zone]` | `bucket_minutes` | `60` | 事件統計時間粒度（分鐘），`>= 1` |
| | `entry_debounce_frames` | `1` | 連續在區域內幾格才算一次進場，`>= 1`；`1` = 不去抖 |

`camera_registry.yaml`（攝影機清單與區域定義，放在 `bucket_dir` 根目錄、不進版控）的
完整格式見根 README。與本階段相關的使用限制（皆為 fail-loud，違反時直接報錯）：

- **`camera_id` 與 `location_camera_id` 皆須唯一**：兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，載入 registry 時即擋下。
- **`zone` 名稱須全域唯一**：不只同一攝影機內不可重複，跨攝影機也不可重複（下游報表以
  區域名稱、不含 `camera_id` 分組彙總，同名區域會被合併）。
- **`polygon` 至少需要 3 個頂點**，座標為該攝影機固定解析度下的像素座標。
- **`participates_in_zone_mapping = false`** 的攝影機直接跳過，不看其 `zones` 內容。
- **定義了區域的攝影機在當日追蹤明細中必須有資料**：攝影機改名或 key 打錯時直接報錯，
  而非靜默算出漏掉區域的人流。
- **檔案內容須為 YAML mapping**：空檔或只有註解時直接報錯並指出檔案路徑，不會讓後續
  解析拋出看不出原因的 `TypeError`。

區域幾何在載入 registry 時刻意不驗證，而是先確認攝影機對得上當日資料、再解析幾何，讓
「攝影機對不上」這個更根本的錯誤先報出來，不被區域定義的筆誤蓋過。

## 輸入 / 輸出檔案

`{bucket}` = `bucket_dir` 的目錄名，皆位於倉庫根目錄的 `outputs/` 下：

| 路徑 | 讀 / 寫 | 內容 |
| --- | --- | --- |
| `outputs/{bucket}/{date}/tracking_results.parquet` | 讀 | 追蹤明細；缺少時報錯 |
| `{bucket_dir}/camera_registry.yaml` | 讀 | 攝影機清單與區域幾何 |
| `outputs/{bucket}/{date}/zone_counts.parquet` | 寫 | 每時段每區域事件統計，欄位 `camera_id` / `zone` / `time_bucket` / `unique_visitors` / `entries` |
| `outputs/{bucket}/{date}/camera_registry_used.yaml` | 寫 | 本次套用的 `camera_registry.yaml` 快照，供下游以「產生此份資料時的定義」為準做驗證 |

**時區**：`tracking_results.parquet` 的 `timestamp` 已是台北在地時間（`Asia/Taipei`），
`time_bucket` 沿用之，本階段不做任何時區位移。

**重跑冪等**：`zone_counts.parquet` 先寫入 `.tmp` 再 `rename` 成正式檔名，藉由 `rename`
的原子性確保中斷時不會在正式檔名下留下半成品。

## 開發

```bash
uv run --directory zone-mapping ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory zone-mapping pytest         # 執行測試
```

> 測試的 cwd 要求與執行 CLI 相反：這裡用 `--directory`（會 chdir 進 `zone-mapping/`），
> 讓 pytest 的 rootdir 解析到本套件；測試本身不碰 `bucket_dir` 與 `outputs/`。
> 等價寫法：`uv run --project zone-mapping pytest zone-mapping/tests`。
