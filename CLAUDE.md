# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

`video-flow-analytics` 是三個**各自獨立的 uv 專案**（非單一套件）：`video-analyze/`
（YOLO+ByteTrack 偵測與多路追蹤，GPU、多進程、重）、`zone-mapping/`（zone 人流統計，純
CPU 向量化）、`flow-report/`（彙總成跨日累加的 Excel，純 CPU）。每包各帶自己的
`pyproject.toml`／`config.toml`／`uv.lock`／`src/`／`tests/`，彼此無跨資料夾 import；共用碼
放在 `libs/`（見「三包共用碼的處理方式」），三包以 path 依賴引用。

本 repo 原為單一套件 `src/video_flow_analytics/`，2026-07 拆成上述三包（issue #18）；
**各套件的完整實作細節（模組結構、多進程 pipeline、fail-loud 處理、
演算法、`config.toml` 完整欄位、函式介面）以各自 README 為準**，本檔只記錄跨套件、不易
從單一套件程式碼本身看出的設計決策：

- [video-analyze/README.md](video-analyze/README.md)
- [zone-mapping/README.md](zone-mapping/README.md)
- [flow-report/README.md](flow-report/README.md)

## 常用指令

```bash
uv sync --project video-analyze && uv sync --project zone-mapping && uv sync --project flow-report

uv run --project video-analyze video-analyze   # 偵測/追蹤 → tracking_results.parquet
uv run --project zone-mapping  zone-mapping    # zone 事件統計 → zone_counts.parquet
uv run --project flow-report   flow-report     # 報表彙總 → report.xlsx

uv run --directory <pkg> ruff check .          # lint；<pkg> = video-analyze / zone-mapping / flow-report
uv run --directory <pkg> pytest                # 測試（三包各 5／3／3 支測試檔）

uv run --directory libs/vfa-registry pytest    # 共用 lib 的測試（2 支）自成一套，不在三包底下
uv run --directory libs/vfa-registry ruff check .
```

共用 lib 用 path 依賴（非 uv workspace），因此**上述指令的形式完全不變**：三包仍各有自己的
`.venv` 與 `uv.lock`，`uv sync --project <pkg>` 會順帶把 lib 以 editable 裝進該包的環境。

**執行 cwd 約束**：`bucket_dir` 與 `OUTPUT_ROOT = Path("outputs")` 是**cwd 相對路徑**，
與各套件 `config.toml` 的檔案定位（三包 DDD 重構後皆用 `find_project_root` 往上找
`pyproject.toml`）是兩套機制。三包一律以 `--project`／
`--directory` 指定套件、**在 repo 根目錄執行**（`uv run` 不改變 cwd）；若改在套件資料夾
內執行，`bucket_dir` 會對到不存在的路徑，`outputs/` 也會裂成三棵互不相通的樹，讓階段間
的檔案契約失效。

## 架構

### 三包共用碼的處理方式

- **`registry.py`、`structured_logging.py` → 抽成 `libs/` 共用 lib（issue #48）。**
  `libs/vfa-registry`（`camera_registry.yaml` 的 Pydantic 模型與 zone 驗證，三包都吃）與
  `libs/vfa-observability`（`StructuredLogger`，三包都吃——`video-analyze` 於 issue #50
  一併改用）。三包在自己的 `pyproject.toml` 以
  `[tool.uv.sources]` 的 **path 依賴（editable）** 引用，**不建 root uv workspace**。
- **`config.py`：仍刻意各包分開**，各包只保留自己 `run_*` 實際讀到的區塊（`video-analyze`
  保留 `tracker`/`model`/`output`/`input`；`zone-mapping` 保留 `input`/`zone`；`flow-report`
  保留 `input`/`zone.bucket_minutes`/`report`）。`flow-report`（issue #42）與 `zone-mapping`
  （issue #46）與 `video-analyze`（issue #50）皆已 DDD 重構，三者的 config 都移至
  `models/config.py` 並改用 pydantic-settings（`config.toml`＋環境變數覆寫）、以
  `find_project_root` 定位設定檔。

**為何改成共用 lib（本檔先前主張刻意重複，2026-07-23 推翻）**：舊理由是「三個階段未來可能
各奔不同平台，共用 lib 會在其中一個移走時斷裂」。推翻的兩點——

1. **前提被實際軌跡否定**：三包的既定路線是**一起**移交
   `argus/pipelines/onprem/<pkg>`，不是各奔平台；lib 用 path 依賴、相對路徑
   （`../libs/<lib>`）在兩個 repo 相同，是**跟著一起搬**而非會斷裂的耦合。真要分家，把 lib
   原地複製回該包即可，一次性成本。
2. **重複的成本已經實現**：`load_registry_from_path` 的 yaml 型別防呆補丁，flow-report 先補
   （PR #45），zone-mapping 隔一個工作單元才補（issue #46），video-analyze 直到抽 lib 前
   **從未補上**——空檔或純註解的 registry 在該包會以沒有檔名線索的 `TypeError` 崩潰。
   「改一份時人工同步另外兩份」這個機制實測撐不住。

**選 path 依賴而非 uv workspace** 的理由：workspace 只有單一 root `uv.lock` 與單一 root
`.venv`，`video-analyze` 的 torch／ultralytics／opencv 會外溢到另外兩包，破壞「依賴面收斂」
這個拆包的主要成果（三包 `pyproject.toml` 的版本 pin 註解即以此為前提）；path 依賴則讓
上方「常用指令」與「執行 cwd 約束」兩節完全不受影響。代價是沒有單一 lock 強制三包版本
一致——**這不是倒退**（本來就是三份 lock ＋ 註解手動 pin），但 lib 的 `pydantic`／`pyyaml`
也 pin 成同一組版本，改版時要連同三包一起改，否則消費端解析 lock 會撞版本衝突。

`camera_registry.yaml` 本身**只有一份**（放在 `bucket_dir`，執行時參數傳入，不進版控），
三包讀的是同一份實體檔案。此檔含 `zones`／`participates_in_zone_mapping` 兩個欄位，即使
`video-analyze` 用不到 zone，模型也必須保留這兩個欄位，否則在 `extra="forbid"` 下會直接
解析失敗；`video-analyze` 不呼叫 `parse_and_validate_zones`，因此吃完整版 lib 後 zone 幾何
仍不會被驗證。

### zone 名稱全域唯一

`zone-mapping` 與 `flow-report` 的報表都以 zone 名稱（不含 `camera_id`）分組彙總，因此
`camera_registry.yaml` 的 zone 名稱**跨攝影機也不可重複**（非僅同一攝影機內）。此驗證
的實作是共用 lib `vfa-registry` 的 `parse_and_validate_zones`——`zone-mapping` 與
`flow-report` 都會呼叫（`video-analyze` 不呼叫），**即使當天不會產生報表，`zone-mapping`
本身也會擋下跨攝影機重複的 zone 命名**。`flow-report` 驗證的對象是產生該日 `zone_counts.parquet` 當時的
`camera_registry_used.yaml` **快照**，而非「當下」的 `camera_registry.yaml`——若兩者之間
改過 zone 名稱，用即時檔案驗證會通過，但 parquet 裡的 zone 名稱其實是舊定義，可能讓不同
攝影機的人流被靜默合併。

### 時區不變量（貫穿三包）

檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`video-analyze` 解析時即轉換成台北在地時間
（`Asia/Taipei`，UTC+8）。此後 `tracking_results.parquet` 的 `timestamp`、
`zone_counts.parquet` 的 `time_bucket`、`report.xlsx` 的日期／小時欄位皆為台北在地時間，
下游（`zone-mapping`／`flow-report`）不需要、也不應該再對它們做任何 UTC→+8 位移。

### `tracking_results.parquet` 不可重現（非拆分相關，屬既有特性）

`video-analyze` 的批次跨串流組成受時序影響（非阻塞輪詢湊批）、`bucket_name1` fixture 為
混解析度會讓 letterbox 隨批次組成變動，加上 **ByteTrack 的 `track_id` 指派本身在重跑間
會改變**（同一輸入重跑兩次即可能有數千列 key 對不上、座標差可達數百 px）。因此
`tracking_results.parquet` 逐值比對對「邏輯是否正確」沒有驗收力；`zone_counts.parquet`
經 `time_bucket` 聚合後穩定（同輸入重跑可逐值/byte 級一致），是**交付期／大重構做 golden
回歸比對**時更可靠的標的（vfa 日常改動的把關是各包 pytest、不依賴 golden；golden 產在
交付期、存放於 argus GCS）。若需驗證 `video-analyze` 的推理邏輯未被改壞，不可用固定容差比對
`tracking_results.parquet`——未改動的程式碼自身重跑就可能差上千 px；改用**控制組相對
條件**：改動後對 golden 的偏離，須不大於未改動程式碼自身重跑對同一份 golden 的偏離。
比對時 join key 用 `(camera_id, timestamp, track_id)`，不可用 `frame_id`（片段內幀序、
跨片段重複，會笛卡兒展開而算出假的大幅座標差）。

## 其他注意事項

- `yolo26m*.pt`、`bucket_name*/`、`outputs/` 皆在各包 `.gitignore`，不進版控
  （`camera_registry.yaml` 含 zone 定義，隨 `bucket_name*/` 一起不進版控）。
- 三包版本 pin 成彼此一致（`torch`/`ultralytics`/`numpy`/`opencv` 等推理堆疊、
  `polars`/`pyarrow`/`openpyxl` 等輸出格式相關套件），避免函式庫版本漂移造成非邏輯性的
  輸出差異；新增或升級依賴時留意是否需要三包同步。**`libs/` 底下兩個 lib 的 `pydantic`／
  `pyyaml` 也在此範圍內**——lib 與消費端各自解析 lock，版本不一致會直接讓消費端 `uv sync`
  撞版本衝突。
