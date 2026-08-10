# zone_mapping

區域事件統計：把一整天的追蹤明細對映到各攝影機的區域（zone）幾何，轉換成「每時段、
每區域」的事件統計。

## 概述

輸入是追蹤明細 `tracking_results.parquet` 與 `camera_registry.yaml` 的區域定義；以每個
track 的落腳點 `(foot_x, foot_y)` 做 ray-casting 判定是否落在區域多邊形內，再依
`time_bucket` 聚合出兩項事件指標，輸出 `zone_counts.parquet`：

落腳點是上游 `video_analyze` 算好寫進 `tracking_results.parquet` 的欄位（由 head 框推算，
推不出來才退回 bbox 底邊中點），本套件不自行從 bbox 推算；缺這兩欄的舊 parquet 直接
fail loud。理由見 [ADR-009](../docs/adr/009-head-based-foot-point.md)。

| 指標 | 定義 |
| --- | --- |
| `unique_visitors` | 該時段內在區域出現過的不重複 `track_id` 數 |
| `entries` | 「區域外 → 區域內」的轉換次數，由區域邊界緩衝帶（`boundary_band_px_1080p`）濾掉邊界抖動 |

**兩項指標刻意用不同判定**（見 [ADR-006](../docs/adr/006-zone-boundary-band.md)）：
`entries` 是事件型指標，用緩衝帶的 Schmitt-trigger——落腳點的帶號距離 `> band` 才確認
在區內、`< -band` 才確認在區外，落在帶內時沿用前一個已確認狀態，因此在邊界附近來回
徘徊只計一次進入。`unique_visitors` 是佔用型指標，仍用當格的 point-in-polygon 布林值，
不吃這個黏著狀態（否則走出區域後停在邊界外緩衝帶內的人，會在其後每個時段都被算成區內
訪客）。**`entries` 首次出現即在區內算一次進入**，與 `line_counting` 的「起始側不計」
相反，是刻意的。

本階段只做「事件轉化」，不做跨期間彙總或分析。純 CPU 向量化運算，不需 GPU。只調整區域
幾何時僅需重跑本階段。

**進入點是函式呼叫，CLI 只是外殼**：核心是
`map_zones_daily(date, bucket_dir, bucket_minutes, boundary_band_px_1080p=25,
output_root=OUTPUT_ROOT) -> Path`（在 `services/zone_map.py`），CLI 進入點
`zone_mapping.main:main` 只是從 `config.toml` 組出參數後呼叫它。

原始碼採 DDD 分層（`src/zone_mapping/`）：

| 目錄 | 內容 |
| --- | --- |
| `main.py` | CLI 外殼：讀 `settings` → 組參數 → 呼叫 `map_zones_daily` |
| `config/constants.py` | 非 Pydantic 靜態常數（輸出根目錄、輸入輸出檔名、parquet schema） |
| `models/config.py` | pydantic-settings 設定模型與全域單例 `settings` |
| `services/zone_map.py` | 讀檔、逐攝影機/逐區域套用演算法、寫檔 |
| `services/stats.py` | point-in-polygon 判定與人流聚合等純函式 |

下列三者由四包共用的 lib 提供，為 uv workspace 成員，不在本包內：`camera_registry.yaml`
的模型與 zone 驗證（[libs/vfa_registry](../libs/vfa_registry)）、單行 JSON 的
`StructuredLogger`（[libs/vfa_observability](../libs/vfa_observability)）、`[input]` 設定
區塊與 `config.toml` 定位（[libs/vfa_config](../libs/vfa_config)）。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，使用倉庫根目錄單一 `uv.lock`） |
| GPU | 不需要，純 CPU |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `numpy` | 向量化幾何運算（ray-casting 區域判定） |
| `polars` / `pyarrow` | parquet 讀寫與聚合 |
| `pydantic` / `pydantic-settings` | 設定與 registry 的資料模型與驗證，config 從 `config.toml`／環境變數載入 |
| `pyyaml` | 讀取 `camera_registry.yaml` |
| `vfa_registry` | 共用 lib：`camera_registry.yaml` 的模型與 zone 驗證 |
| `vfa_observability` | 共用 lib：單行 JSON 的 `StructuredLogger` |
| `vfa_config` | 共用 lib：`[input]` 設定區塊與 `config.toml` 定位 |

## 安裝與快速開始

```bash
uv sync --package zone_mapping
uv run --package zone_mapping zone_mapping
```

CLI 不接受任何旗標，所有參數都讀自 `config.toml`。執行前需備妥：

1. 當日的 `outputs/{bucket}/{date}/tracking_results.parquet`。
2. 一份本機的 `bucket_dir/`，內含 `camera_registry.yaml`（格式見根 README 的
   [設定](../README.md#設定)）。
3. `zone_mapping/config.toml`（指定本次要跑哪個 bucket、哪一天與統計參數）。

### 執行位置（cwd 約束）

**一律在倉庫根目錄執行，以 `--package` 指定本套件**（`uv run` 不改變 cwd）。下列兩者皆為
**cwd 相對路徑**，並非相對於本資料夾：

| 路徑 | 來源 | 在 `zone_mapping/` 內執行時 |
| --- | --- | --- |
| `OUTPUT_ROOT = outputs/` | `config/constants.py` 常數 | 去 `zone_mapping/outputs/` 找輸入而 `FileNotFoundError`，產出也落在錯的樹 |
| `settings.input.bucket_dir` | `config.toml` `[input]` | 對到不存在的 `zone_mapping/bucket_name1`；實務上不會走到，上一列的輸入檢查會先失敗 |

本套件自己的 `config.toml` 以共用 lib 的 `get_toml_path(__file__)`（往上找
`pyproject.toml`）定位，不受 cwd 影響。

## 設定

`config.toml` 置於本套件根目錄（`zone_mapping/config.toml`），只含 `[input]` 與 `[zone]`
兩個區塊，透過 pydantic-settings 載入；**找不到此檔**時會印出警告並以各項預設值啟動，
**此檔存在但參數不合法**則直接報錯（不靜默套用預設值）。同理，出現**未知的頂層區塊**
（例如把 `[zone]` 拼成 `[zones]`）也會直接報錯，而非被靜默忽略。各欄位亦可用環境變數覆寫
（巢狀分隔符 `__`，例如 `ZONE__BOUNDARY_BAND_PX_1080P=30`）。注意欄位名未加前綴，
`ZONE`／`INPUT` 這兩個名稱本身也是有效的覆寫來源（設成 JSON 會整段取代該區塊），
在共用的執行環境中留意不要與其他程式的環境變數撞名。範例：

```toml
[input]
bucket_dir = "bucket_name1"
date = 2026-05-01
bucket_minutes = 60         # 事件統計時間粒度（分鐘）

[zone]
boundary_band_px_1080p = 25 # entries 的區域邊界緩衝帶（1080p 基準像素）；0 = 純內外判定
```

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（cwd 相對） |
| | `date` | — | 統計日期；未設定時報錯 |
| | `bucket_minutes` | `60` | 事件統計時間粒度（分鐘），`>= 1`；與 `line_counting`／`flow_report` 的同名欄位是同一個口徑，三包要填一致的值 |
| `[zone]` | `boundary_band_px_1080p` | `25` | `entries` 的區域邊界緩衝帶寬度，`>= 0`，以 1080p（寬 1920）為基準的像素；執行時依各攝影機的 `frame_width` 換算成實際像素（`基準值 × frame_width / 1920`，只用寬度、線性），`0` = 純內外判定且換算後仍是 `0` |

`[input]` 由共用 lib `vfa_config` 提供、四包同一份定義，故本包也接受 `camera_ids`
（只有 `video_analyze` 會讀）；`bucket_minutes` 於 issue #79 由 `[zone]` 移到這裡，沿用
舊位置（含 `ZONE__BUCKET_MINUTES`）會直接報錯並指出新位置與新的環境變數名
`INPUT__BUCKET_MINUTES`。理由見 [ADR-008](../docs/adr/008-config-section-namespace.md)。

同一個設定值在 1080p 與 4K 上代表同樣的實際距離，不必為混解析度的 bucket 各調一套；
尺寸來自 `tracking_results.parquet` 的 `frame_width`／`frame_height` 欄位，取捨見
[ADR-004](../docs/adr/004-band-resolution-scaling.md)。舊參數 `entry_debounce_frames`
（時間去抖）已移除，沿用會直接報錯並說明語義變更。

`camera_registry.yaml`（攝影機清單與區域定義，放在 `bucket_dir` 根目錄、不進版控）的
完整格式見根 README。與本階段相關的使用限制（皆為 fail-loud，違反時直接報錯）：

- **`camera_id` 與 `location_camera_id` 皆須唯一**：兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，載入 registry 時即擋下。
- **`zone` 名稱須全域唯一**：不只同一攝影機內不可重複，跨攝影機也不可重複（下游報表以
  區域名稱、不含 `camera_id` 分組彙總，同名區域會被合併）。
- **`polygon` 至少需要 3 個頂點**，座標為該攝影機固定解析度下的像素座標。
- **`polygon` 要寬到容得下緩衝帶**：內切半徑小於該攝影機換算後的 `boundary_band_px_1080p`
  時直接報錯。這種區域內部沒有任何點能滿足「帶號距離 > band」，`entries` 會恆為 0；
  報錯訊息帶算出的半徑與建議上限（半徑為格點取樣的下界，會略微低估）。
- **`participates_in_zone_mapping = false`** 的攝影機直接跳過，不看其 `zones` 內容。
- **定義了區域的攝影機在當日追蹤明細中必須有資料**：攝影機改名或 key 打錯時直接報錯，
  而非靜默算出漏掉區域的人流。
- **檔案內容須為 YAML mapping**：空檔或只有註解時直接報錯並指出檔案路徑，不會讓後續
  解析拋出看不出原因的 `TypeError`。

區域幾何在載入 registry 時刻意不驗證，而是先確認攝影機對得上當日資料、再解析幾何，讓
「攝影機對不上」這個更根本的錯誤先報出來，不被區域定義的筆誤蓋過。

### 已知限制

- **預設值 `25` 是保守值，不是實測最佳值**：實測掃的是 4K 的 30／50／60／115 px
  （分別對應基準值 15／25／30／57.5），資料只來自 2026-07-28 的兩台 4K 攝影機、4 個
  zone。實測建議 60 px（基準值 30），採用的 25 更保守——濾重複計數的力道比實測建議弱。
  **1080p 攝影機的 zone 從未掃過**。數字與取捨見
  [ADR-006](../docs/adr/006-zone-boundary-band.md)。
- **加寬 band 會同時砍掉真實訪客**：基準值 25（4K 50 px）讓四個 zone 的 `entries` 降
  56–81%，但「貢獻過 entry 的不同人數」也降到原本的 57–80%——只在區域邊緣淺淺待過、
  從未進到深處的人不再被計為進入。`unique_visitors` 不受影響（該欄不吃緩衝帶）。
- **內切半徑檢查是必要條件，不是充分條件**：通過檢查（半徑 > band）不代表 band 可用。
  實測那四個 zone 的內切半徑 141–218 px，但 4K 115 px 的 band 仍讓三個代表性 track
  完全不再被計入。
- **內切半徑用格點取樣，求出的是下界**：真正的內切圓心不一定落在格點上，因此檢查會略微
  低估、可能誤擋邊緣案例。誤擋時**整天所有攝影機的 `zone_mapping` 都不會產出**，錯誤
  訊息會列出兩條操作路徑（調小 band 或把 zone 多邊形畫寬）。
- **本階段沒產出時，當日整份報表都不會產出**：`flow_report` 的輸入必要性看
  `bucket_dir/camera_registry.yaml` 的定義，registry 裡有 `zones` 定義就必須有
  `zone_counts.parquet`，缺檔會中止整份報表、連出入口三個分頁也不產（見
  [ADR-005](../docs/adr/005-report-input-requirement-from-snapshot.md)、
  [ADR-007](../docs/adr/007-remove-registry-snapshot.md)）。上面兩道
  fail-loud 誤擋的代價因此不只是區域那兩頁。
- **跨日報表會混到兩種口徑**：`flow_report` 以 `append` 累加各日的 `zone_counts.parquet`，
  改動前後產出的 `entries` 語義不同。要口徑一致就得重跑歷史日期，否則需在報表註明
  改動日期。`unique_visitors` 無此問題（數值完全不變）。

## 輸入 / 輸出檔案

`{bucket}` = `bucket_dir` 的目錄名，皆位於倉庫根目錄的 `outputs/` 下：

| 路徑 | 讀 / 寫 | 內容 |
| --- | --- | --- |
| `outputs/{bucket}/{date}/tracking_results.parquet` | 讀 | 追蹤明細；缺少時報錯。須含 `frame_width`／`frame_height`（緩衝帶寬度的解析度換算靠它），2026-07 之前產出的舊檔沒有這兩欄，會直接報錯要求重跑 `video_analyze` |
| `{bucket_dir}/camera_registry.yaml` | 讀 | 攝影機清單與區域幾何 |
| `outputs/{bucket}/{date}/zone_counts.parquet` | 寫 | 每時段每區域事件統計，欄位 `camera_id` / `zone` / `time_bucket` / `unique_visitors` / `entries` |

**時區**：`tracking_results.parquet` 的 `timestamp` 已是台北在地時間（`Asia/Taipei`），
`time_bucket` 沿用之，本階段不做任何時區位移。

**重跑冪等**：`zone_counts.parquet` 先寫入 `.tmp` 再 `rename` 成正式檔名，藉由 `rename`
的原子性確保中斷時不會在正式檔名下留下半成品。

## 開發

```bash
uv run --directory zone_mapping ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory zone_mapping pytest         # 執行測試
```

> 測試的 cwd 要求與執行 CLI 相反：這裡用 `--directory`（會 chdir 進 `zone_mapping/`），
> 讓 pytest 的 rootdir 解析到本套件；測試本身不碰 `bucket_dir` 與 `outputs/`。
> 等價寫法：`uv run --package zone_mapping pytest zone_mapping/tests`。
