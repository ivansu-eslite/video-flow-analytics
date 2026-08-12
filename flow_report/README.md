# flow_report

人流統計分析與 BI 報表：把「每時段、每區域」的區域事件統計與「每時段、每計數線」的進出
人數做跨期間彙總與分析，持續寫入單一 Excel 報表，供 Looker Studio 等 BI 工具接手做長期
視覺化。

## 概述

輸入是區域事件統計 `zone_counts.parquet`、計數線進出統計 `line_counts.parquet`，以及
`bucket_dir` 下的 `camera_registry.yaml`；把多個 `bucket_minutes` 併成
`period_minutes`、算出每日尖峰，**持續 Append** 至同一份
`outputs/{bucket}/report.xlsx`（而非逐日各產一份），讓 BI 工具對這份不斷累加的資料做長期
觀測。

報表含六個分頁：

| 分頁 | 欄位 | 寫入者 |
| --- | --- | --- |
| 各區域人流 | 日期、星期、小時、區域、人流量 | 本階段 |
| 各區域每日尖峰 | 日期、星期、區域、尖峰時段、尖峰人流 | 本階段 |
| 各出入口人流 | 日期、星期、小時、群組、計數線、進場人數、出場人數、淨進出 | 本階段 |
| 各出入口每日進場尖峰 | 日期、星期、群組、計數線、尖峰時段、尖峰進場、尖峰出場 | 本階段 |
| 各出入口每日出場尖峰 | 同上 | 本階段 |
| 活動事件 | 日期、星期、開始時間、結束時間、區域、活動名稱、活動類型 | 其他來源；本階段只建立標題列、不寫入 |

- 兩個出入口尖峰分頁的表頭相同，差別只在尖峰時段由哪個量決定：進場尖峰看 `in_count`、
  出場尖峰看 `out_count`；並列時都取較早的時段。兩者都同時列出該時段的進場與出場人數。
- 「淨進出」= 進場 − 出場，在彙總時算出，`line_counts.parquet` 本身沒有這一欄。
- `camera_id` 不進報表：區域與計數線名稱都是全域唯一的（見下方使用限制）。

純 CPU 運算，不需重跑偵測、區域事件統計或計數線統計；只調整報表參數時僅需重跑本階段。

> **區域兩個分頁在加入計數線分頁時一併改名**（`每小時人流` → `各區域人流`、
> `每日尖峰` → `各區域每日尖峰`），讓兩組分頁的命名對稱。**既有的 `report.xlsx` 不做
> 遷移**：舊的兩個分頁會留在檔案裡、不再被寫入，新的分頁另建。BI 端若已接舊分頁名需要
> 重接。
>
> **三個尖峰分頁移除了「每日提醒」欄**（原本依尖峰時段所在小時給出用餐動線建議，屬寫死
> 的業務判讀規則，不是統計量）。同樣**不做遷移**：既有 `report.xlsx` 的表頭與歷史值留著，
> 之後 append 的新列該欄留白。BI 端若已接該欄，新資料會是空值。要換成乾淨表頭只能刪掉該
> 檔重建，但 `export_report_daily` 一次只彙總單一日期，得把要保留的每一天逐日重跑才補得
> 回來。

**進入點是函式呼叫，CLI 只是外殼**：核心是
`export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date,
bucket_minutes, output_root=OUTPUT_ROOT) -> Path`（在 `services/report.py`），CLI 進入點
`flow_report.main:main` 只是從 `config.toml` 組出參數後呼叫它。

原始碼採 DDD 分層（`src/flow_report/`）：

| 目錄 | 內容 |
| --- | --- |
| `main.py` | CLI 外殼：讀 `settings` → 組參數 → 呼叫 `export_report_daily` |
| `config/constants.py` | 非 Pydantic 靜態常數（分頁名、欄位定義、排序欄、預設路徑等） |
| `models/config.py` | pydantic-settings 設定模型與全域單例 `settings` |
| `services/report.py` | 報表 orchestration、I/O 與 Excel 讀寫 |
| `services/stats.py` | 時區轉換、期間彙總、尖峰計算等純函式 |

**報表欄位只定義一次**：每個分頁的欄位以 `(資料欄名, 中文表頭)` 序對寫在
`config/constants.py`，寫檔時按欄名取值。要增刪報表欄位改這一處，分頁的欄序也由它決定，
與 `services/stats.py` 的 `select` 順序無關；資料側少了定義中的欄位會在寫檔階段直接拋錯，
不會把值靜默填進錯的表頭底下。

下列三者由四包共用的 lib 提供，以 workspace 成員引用，不在本包內：`camera_registry.yaml`
的模型與 zone／line 驗證（[libs/vfa_registry](../libs/vfa_registry)）、單行 JSON 的
`StructuredLogger`（[libs/vfa_observability](../libs/vfa_observability)）、`[input]` 設定
區塊與 `config.toml` 定位（[libs/vfa_config](../libs/vfa_config)）。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，單一 root `uv.lock`） |
| GPU | 不需要，純 CPU |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `openpyxl` | 讀寫 Excel 報表 |
| `polars` / `pyarrow` | parquet 讀取與期間彙總 |
| `pydantic` / `pydantic-settings` | 設定與 registry 的資料模型與驗證，config 從 `config.toml`／環境變數載入 |
| `pyyaml` | 讀取 `camera_registry.yaml` |
| `vfa_registry` | 共用 lib：`camera_registry.yaml` 的模型與 zone／line 驗證 |
| `vfa_observability` | 共用 lib：單行 JSON 的 `StructuredLogger` |
| `vfa_config` | 共用 lib：`[input]` 設定區塊與 `config.toml` 定位 |

## 安裝與快速開始

```bash
uv sync --package flow_report
uv run --package flow_report flow_report
```

CLI 不接受任何旗標，所有參數都讀自 `config.toml`。執行前需備妥：

1. `bucket_dir` 下的 `camera_registry.yaml`，以及**它所要求的上游 parquet**：registry
   中有攝影機定義了區域就要有 `zone_counts.parquet`、有攝影機定義了計數線就要有
   `line_counts.parquet`（見下方「哪些輸入是必要的」）。
2. `flow_report/config.toml`（指定本次要彙總哪個 bucket、哪一天與報表參數）。

### 執行位置（cwd 約束）

**一律在倉庫根目錄執行，以 `--package` 指定本套件**（`uv run` 不改變 cwd）。下列兩者皆為
**cwd 相對路徑**，並非相對於本資料夾：

| 路徑 | 來源 | 在 `flow_report/` 內執行時 |
| --- | --- | --- |
| `settings.input.bucket_dir` | `config.toml` `[input]` | **最先失敗的就是這個**：`load_registry` 吃完整路徑，去 `flow_report/bucket_name1/` 找 registry 而 `FileNotFoundError: 找不到設備登錄檔` |
| `OUTPUT_ROOT = outputs/` | `config/constants.py` 常數 | 會去 `flow_report/outputs/` 找輸入、報表也落在錯的樹，但上一列已先拋錯，實務上走不到這裡 |

`bucket_dir` 在本階段有兩種用法，只有前者受 cwd 影響：`registry_path`／`load_registry`
吃**完整路徑**（故 cwd 錯了就找不到 registry），輸出目錄則只取 `bucket_path.name`。

本套件自己的 `config.toml` 以共用 lib 的 `get_toml_path(__file__)`（往上找
`pyproject.toml`）定位，不受 cwd 影響。

## 設定

`config.toml` 置於本套件根目錄（`flow_report/config.toml`），含 `[input]`、`[report]`
兩個區塊，透過 pydantic-settings 載入；**找不到此檔**時會印出警告並以各項
預設值啟動，**此檔存在但參數不合法**則直接報錯（不靜默套用預設值）。同理，出現**未知的
頂層區塊**（例如把 `[report]` 拼成 `[reports]`）也會直接報錯，而非被靜默忽略。各欄位亦可
用環境變數覆寫（巢狀分隔符 `__`，例如 `REPORT__PERIOD_MINUTES=30`）。範例：

```toml
[input]
bucket_dir = "bucket_name1"
date = 2026-05-01
bucket_minutes = 60           # 上游兩份 parquet 的時段粒度（分鐘）

[report]
period_minutes = 60           # 報表彙總粒度；須為 input.bucket_minutes 的倍數
metric = "entries"            # "entries" 或 "unique_visitors"
on_duplicate_date = "append"  # "overwrite" / "append" / "error"
```

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（cwd 相對）；本階段有兩種用法——讀 `camera_registry.yaml` 吃**完整路徑**，組 `outputs/{bucket}/` 則只取其目錄名（見上方「執行位置」） |
| | `date` | — | 彙總日期；未設定時報錯 |
| | `bucket_minutes` | `60` | 上游 `zone_counts.parquet`／`line_counts.parquet` 的時段粒度（分鐘），`>= 1`；須與產生這兩份 parquet 時的 `zone_mapping`／`line_counting` 設定一致 |
| `[report]` | `period_minutes` | `60` | 報表彙總粒度（分鐘），`>= 1`，且**須為 `input.bucket_minutes` 的倍數**（否則 fail-loud 報錯） |
| | `metric` | `"entries"` | `"entries"` 或 `"unique_visitors"`；決定「人流量」「尖峰人流」用哪個統計量。**只作用於區域統計**，計數線固定用 `in_count`／`out_count` |
| | `on_duplicate_date` | `"append"` | 同日期重跑的處理：`"overwrite"` / `"append"` / `"error"` |

**`bucket_minutes` 放在 `[input]` 而非 `[report]`**：它描述的是**輸入資料**的粒度（上游
兩包寫出 parquet 時用的值），不是報表的呈現粒度——後者是 `[report] period_minutes`。
區域與計數線共用這一個數字，兩包的設定若不一致，這裡只能對上其中一個（見「已知限制」）。
沿用舊的 `[zone] bucket_minutes` 會直接報錯並指出新位置。環境變數那側**不再有對稱的
警告**（issue #79 移除）：`ZONE__BUCKET_MINUTES` 現在對 `zone_mapping` 也已不是合法設定
（該包的 `bucket_minutes` 同樣移到了 `[input]`），會由該包的搬家提示擋下，本階段不必再
警告一份。三包現在共用單一環境變數 `INPUT__BUCKET_MINUTES`。

`[input]` 由共用 lib `vfa_config` 提供、四包同一份定義，故本階段也接受 `camera_ids`
（只有 `video_analyze` 會讀）。為何不能各包裁剪見
[ADR-008](../docs/adr/shared/008-config-section-namespace.md)。

`on_duplicate_date` 三種模式的行為：

| 模式 | 行為 |
| --- | --- |
| `overwrite` | 先刪除既有相同日期的列再插入，並依日期／區域或計數線重新排序；天生冪等 |
| `append` | 直接附加到尾端、不檢查；重跑同一天會產生重複列 |
| `error` | 發現重複日期即整個中止，不寫入任何內容 |

`overwrite` 與 `error` 的日期集合取**本次彙總的日期，加上區域與計數線兩邊資料實際帶到的
日期**，且作用於本階段寫入的五個分頁（`活動事件` 由其他來源寫入，本階段完全不動它）：

- 本次彙總的日期一律計入，因為上游重跑後該日事件可能清空（0 列是正常產物，不是缺資料）；
  只看資料內容的話這種情況一列都清不到，該日的舊資料會留在報表裡。

- `overwrite` 時聯集日期一律從五個分頁清除，該日沒有資料的分頁**只清不寫**。若改成「沒有
  資料的分頁完全不動」，registry 移除 `lines` 後重跑該日，出入口三頁會留著舊列、區域兩頁
  換成新列，同一天在不同分頁混雜新舊資料。
- `error` 時只要任一分頁的既有日期與聯集日期相交就整個中止，五個分頁都不寫入。

## 哪些輸入是必要的

**由 `bucket_dir/camera_registry.yaml` 的定義決定，不由檔案是否存在決定**
（[ADR-005](../docs/adr/flow_report/005-report-input-requirement-from-snapshot.md)、
[ADR-007](../docs/adr/shared/007-remove-registry-snapshot.md)）：

| registry 的內容 | 缺對應的 parquet 時 |
| --- | --- |
| 有任一攝影機 `participates_in_zone_mapping` 且 `zones` 非空 | 缺 `zone_counts.parquet` 即報錯，訊息指出要先跑 `zone_mapping` |
| 有任一攝影機 `lines` 非空 | 缺 `line_counts.parquet` 即報錯，訊息指出要先跑 `line_counting` |
| 沒有任何 `zones` 定義 | 不報錯，區域兩頁不寫入資料（表頭仍建立） |
| 沒有任何 `lines` 定義 | 不報錯，出入口三頁不寫入資料（表頭仍建立） |
| 兩者都沒有定義 | 報錯：沒有可彙總的統計 |

反向的一種情況也會報錯：**registry 已沒有某一側的定義，當日對應的 parquet 卻有資料**
（例如把所有攝影機的 `lines` 都拿掉了，但 `line_counts.parquet` 還在；區域那側還包含
「`zones` 還在、但 `participates_in_zone_mapping` 全被關掉」）。這代表 registry 在產生
parquet 之後被改過，靜默跳過會讓該類統計整批從報表消失，`on_duplicate_date` 為
`overwrite` 時還會清掉報表裡既有的舊列。0 列的 parquet 不算——那是上游對「這個 bucket
沒有該側定義」的正常產物。

確定不再統計某一側時，正當做法是把該日的 parquet 一併移除（本階段不刪上游產物）。要注意
這之後以 `overwrite` 重跑，報表中該日該側的既有列同樣會被清掉——那正是這道檢查原本要防的
損失，差別只在於這次是明示同意的。

理由：純看檔案的話，「定義了計數線卻忘了跑 `line_counting`」與「這個 bucket 本來就沒有
計數線」無法區分，前者會靜默少三頁。**刻意不提供跳過用的設定旗標**——要跳過的正當做法是
把該攝影機的 `lines` 從 registry 拿掉。代價是 `flow_report` 從此被 `line_counting` 綁住：
registry 只要有任一 `lines`，沒跑 `line_counting` 就連區域兩頁都產不出來。

不論該 bucket 有沒有計數線，六個分頁的表頭一律建立，讓 BI 端接到的 schema 穩定。
`line_counts.parquet` 是 **0 列**（當日無任何跨越事件）則是 `line_counting` 的正常產物、
不是缺資料，不報錯。

## registry 的使用限制

`camera_registry.yaml`（攝影機清單與區域／計數線定義，放在 `bucket_dir` 根目錄、不進版控）
的完整格式見根 README。本階段讀的是 `bucket_dir` 下當下的那一份（缺檔即報錯；為何不是
產生 parquet 當時的快照，見 [ADR-007](../docs/adr/shared/007-remove-registry-snapshot.md)），
相關使用限制（皆為 fail-loud，違反時直接報錯）：

- **`zone` 與 `line` 名稱都須全域唯一**：本階段以區域／計數線名稱、不含 `camera_id`
  分組彙總，同名會讓不同攝影機的數字被合併成同一列，故不只同一攝影機內不可重複，跨攝影機
  也不可重複。`line_group` 則刻意不驗證唯一（見 [ADR-002](../docs/adr/line_counting/002-line-group-semantics.md)），
  它只是報表的一個維度欄位。
- **`camera_id` 與 `location_camera_id` 皆須唯一**：兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，載入 registry 時即擋下。
- **`polygon` 至少需要 3 個頂點、`points` 至少需要 2 個頂點**，座標為該攝影機固定解析度
  下的像素座標。
- **`participates_in_zone_mapping = false`** 的攝影機不列入上述 zone 名稱唯一性驗證，
  其 `zones` 內容不影響本階段。計數線沒有對應的旗標，`lines` 非空即代表參與。
- **兩份 parquet 內的 `(camera_id, zone)`／`(camera_id, line)` 組合須全部存在於 registry
  的定義內**：出現 registry 沒有的組合視為資料與定義不一致，直接報錯。正常流程下不會
  觸發——上游是用同一份 registry 過濾出參與的攝影機才產生 parquet。

**產生 parquet 之後改過 registry 的話**，區域或計數線改名、移除會被這道組合驗證擋下；
整側定義被清空則由上一節的反向檢查擋下。只改幾何座標（名稱不變）擋不下來，見「已知限制」。

## 輸入 / 輸出檔案

`{bucket}` = `bucket_dir` 的目錄名，皆位於倉庫根目錄的 `outputs/` 下：

| 路徑 | 讀 / 寫 | 內容 |
| --- | --- | --- |
| `outputs/{bucket}/{date}/zone_counts.parquet` | 讀 | 每時段每區域事件統計，欄位 `camera_id` / `zone` / `time_bucket` / `unique_visitors` / `entries`；registry 有區域定義時缺少即報錯 |
| `outputs/{bucket}/{date}/line_counts.parquet` | 讀 | 每時段每計數線進出人數，欄位 `line_group` / `camera_id` / `line` / `time_bucket` / `in_count` / `out_count`；registry 有計數線定義時缺少即報錯 |
| `{bucket_dir}/camera_registry.yaml` | 讀 | 攝影機清單與區域／計數線定義；缺少時報錯 |
| `outputs/{bucket}/report.xlsx` | 寫 | 跨日累加的 Excel 報表（六個分頁）；不存在時建立，存在時依 `on_duplicate_date` 更新（缺哪個分頁就補建哪個） |

**時區**：兩份 parquet 的 `time_bucket` 都已是台北在地時間（`Asia/Taipei`），本階段
只去掉時區標記、保留原本的 wall-clock 值，不做任何時區位移。

**寫入冪等**：`report.xlsx` 先寫入 `.tmp` 再 `rename` 成正式檔名，藉由 `rename` 的原子性
確保中斷時不會在正式檔名下留下半成品。但**內容層面是否冪等取決於 `on_duplicate_date`**
（見上表）。

## 已知限制

- **`metric = "unique_visitors"` 的彙總為近似值**：`unique_visitors` 是各 bucket 內的不
  重複人數，跨相鄰 bucket 停留的同一人會在彙總時被重複計入；`zone_counts.parquet` 未保留
  原始 `track_id`，本階段無法在彙總時去重。`metric = "entries"` 與計數線的
  `in_count`／`out_count` 本身即為可疊加的事件次數，不受此影響。
- **`bucket_minutes` 仍靠人工同步**：本階段只有一個數字，卻要對應 `zone_mapping` 與
  `line_counting` 兩包各自的設定。三者不一致時，`period_minutes` 的倍數檢查會拿錯的數字
  驗證，不會被自動抓到。issue #79 讓三包的欄位路徑一致（都在 `[input]`，可用單一
  `INPUT__BUCKET_MINUTES` 一次覆寫），但三份 `config.toml` 仍各填一次；唯一能消除手抄的
  做法（寫進 parquet 檔級 metadata）已在 [ADR-008](../docs/adr/shared/008-config-section-namespace.md)
  否決，**這是已接受的最終狀態，不是待辦**。
- **`line_group` 只是報表的一個維度欄位**：本階段不做範圍層級的加總（例如整個賣場的進出
  合計），出入口三頁都是逐計數線的數字。
- **幾何改過但名稱沒變時不會有任何訊號**：本階段讀的是當下的 registry，不是產生 parquet
  當時的定義。zone／line 改名或移除後拿舊 parquet 彙總仍會 fail-loud（(camera, zone)
  組合驗證與全域唯一驗證），但只調整多邊形或線段座標的話，報表數字仍是舊幾何算出來的，
  本階段看不出來。**只改 `line_group`、不改 `line` 名稱也屬於這個範圍**：組合驗證不看
  `line_group`，報表的「群組」欄取自 `line_counts.parquet`，會沿用舊的群組名。這是移除
  registry 快照時接受的代價，見
  [ADR-007](../docs/adr/shared/007-remove-registry-snapshot.md)。

## 開發

```bash
uv run --directory flow_report ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory flow_report pytest         # 執行測試
```

> 測試的 cwd 要求與執行 CLI 相反：這裡用 `--directory`（會 chdir 進 `flow_report/`），
> 讓 pytest 的 rootdir 解析到本套件；測試本身不碰 `bucket_dir` 與 `outputs/`。
> 等價寫法：`uv run --package flow_report pytest flow_report/tests`。
