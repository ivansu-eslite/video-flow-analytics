# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

`video-flow-analytics`（vfa）是**單一 uv workspace**（repo 根為 workspace root），由四個
成員套件組成：`video_analyze/`（YOLO+ByteTrack 偵測與多路追蹤，GPU、多進程、重）、
`zone_mapping/`（zone 區域佔用人流統計，純 CPU 向量化）、`line_counting/`（方向性計數線
進出人數統計，純 CPU 向量化）、`flow_report/`（彙總成跨日累加的 Excel，純 CPU）。每個成員
各帶自己的 `pyproject.toml`／`config.toml`／`src/`／`tests/`，彼此無跨資料夾 import；共用碼
放在 `libs/`（見「四包共用碼的處理方式」），以 `{ workspace = true }` 引用；單一 root
`uv.lock`／`.venv`。

`zone_mapping` 與 `line_counting` 是概念雙胞胎：輸入完全相同（`tracking_results.parquet` ＋
`camera_registry.yaml`）、都以腳底點做純 CPU 向量化判定，差別只在幾何——zone 判「腳底是否
落在多邊形內」（區域佔用），line 判「腳底是否跨越計數線及其方向」（方向性進出）。

本 repo 原為單一套件 `src/video_flow_analytics/`，2026-07 拆成 `video_analyze`／`zone_mapping`／
`flow_report` 三包（issue #18），同月再收斂成 uv workspace（issue #56），其後新增
`line_counting`（issue #41）；**各套件的完整實作細節（模組結構、多進程 pipeline、fail-loud
處理、演算法、`config.toml` 完整欄位、函式介面）以各自 README 為準**，本檔只記錄跨套件、
不易從單一套件程式碼本身看出的設計決策：

- [video_analyze/README.md](video_analyze/README.md)
- [zone_mapping/README.md](zone_mapping/README.md)
- [line_counting/README.md](line_counting/README.md)
- [flow_report/README.md](flow_report/README.md)

## 常用指令

```bash
uv sync --all-packages                             # 全量同步（含 torch）

uv run --package video_analyze video_analyze       # 偵測/追蹤 → tracking_results.parquet
uv run --package zone_mapping  zone_mapping        # zone 區域佔用統計 → zone_counts.parquet
uv run --package line_counting line_counting       # 計數線進出人數統計 → line_counts.parquet
uv run --package flow_report   flow_report         # 報表彙總 → report.xlsx

uv run --directory <pkg> ruff check .              # lint；<pkg> = video_analyze / zone_mapping / line_counting / flow_report
uv run --directory <pkg> pytest                    # 測試（四包各 5／3／3／3 支測試檔）

uv run --directory libs/vfa_registry pytest        # 共用 lib 的測試（4 支）自成一套，不在四包底下
uv run --directory libs/vfa_registry ruff check .
uv run --directory libs/vfa_observability pytest   # （1 支）
uv run --directory libs/vfa_observability ruff check .
```

**torch 隔離**：workspace 為單一 `.venv`，但 `uv sync --package <pkg>` 只裝該包依賴子樹
（`flow_report`／`zone_mapping`／`line_counting` 不含 torch）；`uv sync --all-packages` 才裝含
torch 的完整環境。部署時各容器以 `uv sync --package <pkg>` 維持 CPU-only 隔離。

**pytest／ruff 用 `--directory`（切換 cwd）而非 `--package`**：`--package` 不改變 cwd，
`pytest` 會從 repo 根遞迴收集到所有套件的測試而撞名衝突（`tests/test_config.py` 等檔名
四包重複）；`--directory` 切進該套件資料夾，才會只解析到該套件自己的 `tests/`。

**執行 cwd 約束**：`bucket_dir` 與 `OUTPUT_ROOT = Path("outputs")` 是**cwd 相對路徑**，
與各套件 `config.toml` 的檔案定位（四包 DDD 重構後皆用 `find_project_root` 往上找
`pyproject.toml`）是兩套機制。四包一律以 `--package`／`--directory` 指定套件、**在 repo
根目錄執行**（`uv run` 不改變 cwd）；若改在套件資料夾內執行，`bucket_dir` 會對到不存在
的路徑，`outputs/` 也會裂成四棵互不相通的樹，讓階段間的檔案契約失效。

## 架構

### 四包共用碼的處理方式

- **`registry.py`、`structured_logging.py` → 抽成 `libs/` 共用 lib（issue #48）。**
  `libs/vfa_registry`（`camera_registry.yaml` 的 Pydantic 模型與 zone／line 驗證，四包都吃）
  與 `libs/vfa_observability`（`StructuredLogger`，四包都吃——`video_analyze` 於 issue #50
  一併改用）。四包在自己的 `pyproject.toml` 以 `[tool.uv.sources]` 的
  `{ workspace = true }` 引用（issue #56 起）。line 支援（`Line` 模型、
  `CameraEntry.lines` 欄位、`parse_and_validate_lines` 跨攝影機全域唯一驗證）於 issue #41
  加在此 lib，是唯一一處 registry 改動——三包經 workspace 依賴自動吃到 `lines` 忽略欄位
  相容，本身無需改碼。
- **`config.py`：仍刻意各包分開**，各包只保留自己 `run_*` 實際讀到的區塊（`video_analyze`
  保留 `tracker`/`model`/`output`/`input`；`zone_mapping` 保留 `input`/`zone`；`line_counting`
  保留 `input`/`line`；`flow_report` 保留 `input`/`zone.bucket_minutes`/`report`）。四包皆已
  DDD 重構（`flow_report` issue #42、`zone_mapping` issue #46、`video_analyze` issue #50、
  `line_counting` issue #41 鏡射 `zone_mapping` 建立），config 都在 `models/config.py` 並改用
  pydantic-settings（`config.toml`＋環境變數覆寫）、以 `find_project_root` 定位設定檔。

**共用 lib 存在的理由**：抽出前，`load_registry_from_path` 的 yaml 型別防呆補丁三包各自
維護、版本各自漂移——flow_report 先補（PR #45），zone_mapping 隔一個工作單元才補
（issue #46），video_analyze 直到抽 lib 前**從未補上**，空檔或純註解的 registry 在該包
會以沒有檔名線索的 `TypeError` 崩潰。改為單一 lib 後同一份實作四包共用，不再需要人工
同步；`line_counting` 直接吃這份 lib，未再重蹈各自漂移的覆轍。

`camera_registry.yaml` 本身**只有一份**（放在 `bucket_dir`，執行時參數傳入，不進版控），
四包讀的是同一份實體檔案。此檔含 `zones`／`lines`／`participates_in_zone_mapping` 三個欄位，
即使 `video_analyze` 用不到 zone 與 line，模型也必須保留這些欄位，否則在 `extra="forbid"`
下會直接解析失敗；`video_analyze` 不呼叫 `parse_and_validate_zones`／`parse_and_validate_lines`，
因此吃完整版 lib 後 zone／line 幾何仍不會被驗證。

### zone／line 名稱全域唯一

`zone_mapping` 與 `flow_report` 的報表都以 zone 名稱（不含 `camera_id`）分組彙總，因此
`camera_registry.yaml` 的 zone 名稱**跨攝影機也不可重複**（非僅同一攝影機內）。此驗證
的實作是共用 lib `vfa_registry` 的 `parse_and_validate_zones`——`zone_mapping` 與
`flow_report` 都會呼叫（`video_analyze` 不呼叫），**即使當天不會產生報表，`zone_mapping`
本身也會擋下跨攝影機重複的 zone 命名**。`flow_report` 驗證的對象是產生該日 `zone_counts.parquet` 當時的
`camera_registry_used.yaml` **快照**，而非「當下」的 `camera_registry.yaml`——若兩者之間
改過 zone 名稱，用即時檔案驗證會通過，但 parquet 裡的 zone 名稱其實是舊定義，可能讓不同
攝影機的人流被靜默合併。

`line_counting` 的計數線名稱有**完全平行**的約束：下游同樣以 line 名稱（不含 `camera_id`）
分組彙總，故 line 名稱跨攝影機也不可重複，由 `vfa_registry` 的 `parse_and_validate_lines`
擋下——**即使當天不會產生報表，`line_counting` 本身也會擋下跨攝影機重複的 line 命名**
（`flow_report` 對 line 的串接、對快照的同型驗證另開 issue，本次只做到 `line_counting`）。

### 時區不變量（貫穿四包）

檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`video_analyze` 解析時即轉換成台北在地時間
（`Asia/Taipei`，UTC+8）。此後 `tracking_results.parquet` 的 `timestamp`、
`zone_counts.parquet`／`line_counts.parquet` 的 `time_bucket`、`report.xlsx` 的日期／小時
欄位皆為台北在地時間，下游（`zone_mapping`／`line_counting`／`flow_report`）不需要、也不
應該再對它們做任何 UTC→+8 位移。

### `tracking_results.parquet` 不可重現（非拆分相關，屬既有特性）

`video_analyze` 的批次跨串流組成受時序影響（非阻塞輪詢湊批）、`bucket_name1` fixture 為
混解析度會讓 letterbox 隨批次組成變動，加上 **ByteTrack 的 `track_id` 指派本身在重跑間
會改變**（同一輸入重跑兩次即可能有數千列 key 對不上、座標差可達數百 px）。因此
`tracking_results.parquet` 逐值比對對「邏輯是否正確」沒有驗收力；`zone_counts.parquet`
經 `time_bucket` 聚合後穩定（同輸入重跑可逐值/byte 級一致），是**交付期／大重構做 golden
回歸比對**時更可靠的標的（vfa 日常改動的把關是各包 pytest、不依賴 golden；golden 產在
交付期、存放於 argus GCS）。若需驗證 `video_analyze` 的推理邏輯未被改壞，不可用固定容差比對
`tracking_results.parquet`——未改動的程式碼自身重跑就可能差上千 px；改用**控制組相對
條件**：改動後對 golden 的偏離，須不大於未改動程式碼自身重跑對同一份 golden 的偏離。
比對時 join key 用 `(camera_id, timestamp, track_id)`，不可用 `frame_id`（片段內幀序、
跨片段重複，會笛卡兒展開而算出假的大幅座標差）。

## 其他注意事項

- `yolo26m*.pt`、`bucket_name*/`、`outputs/` 皆在 `.gitignore`，不進版控
  （`camera_registry.yaml` 含 zone／line 定義，隨 `bucket_name*/` 一起不進版控）。
- 四包版本 pin 成彼此一致（`torch`/`ultralytics`/`numpy`/`opencv` 等推理堆疊、
  `polars`/`pyarrow`/`openpyxl` 等輸出格式相關套件），避免函式庫版本漂移造成非邏輯性的
  輸出差異；`line_counting` 的 `numpy`/`polars`/`pyarrow`/`pydantic`/`pyyaml` 與 `zone_mapping`
  pin 成同版，`libs/` 底下兩個 lib 的 `pydantic`／`pyyaml` 也在此範圍內。單一 root `uv.lock`
  下版本一致由 `uv lock` 自動把關——`==` pin 彼此衝突會直接讓 `uv lock` 解析失敗；新增或
  升級依賴時留意是否需要四包同步。
