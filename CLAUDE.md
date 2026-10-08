# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

`video-flow-analytics`（vfa）是**單一 uv workspace**（repo 根為 workspace root），由四個
成員套件組成：`video_analyze/`（YOLO+ByteTrack 偵測與多路追蹤，GPU、多進程、重）、
`zone_mapping/`（zone 區域佔用人流統計，純 CPU 向量化）、`line_counting/`（方向性計數線
進出人數統計，純 CPU 向量化）、`flow_report/`（彙總成跨日累加的 Excel，純 CPU）。每個成員
各帶自己的 `pyproject.toml`／`config.toml`／`src/`／`tests/`，彼此無跨資料夾 import；共用碼
放在 `libs/`（見 [.claude/rules/shared-code.md](.claude/rules/shared-code.md)），以
`{ workspace = true }` 引用；單一 root `uv.lock`／`.venv`。

`zone_mapping` 與 `line_counting` 的輸入相同（`tracking_results.parquet` ＋
`camera_registry.yaml`），都以落腳點做純 CPU 向量化判定，差別在幾何——zone 判「落腳點是否
落在多邊形內」（區域佔用），line 判「落腳點是否跨越計數線及其方向」（方向性進出）。

本 repo 原為單一套件 `src/video_flow_analytics/`，2026-07 拆成 `video_analyze`／`zone_mapping`／
`flow_report` 三包（issue #18），同月再收斂成 uv workspace（issue #56），其後新增
`line_counting`（issue #41）；**各套件的完整實作細節（模組結構、多進程 pipeline、fail-loud
處理、演算法、`config.toml` 完整欄位、函式介面）以各自 README 為準**，本檔只記錄跨套件、
不易從單一套件程式碼本身看出的設計決策：

- [video_analyze/README.md](video_analyze/README.md)
- [zone_mapping/README.md](zone_mapping/README.md)
- [line_counting/README.md](line_counting/README.md)
- [flow_report/README.md](flow_report/README.md)

**跨專案脈絡**：本 repo 是 `video_analyze` 推論鏈的開發正本，改動會往 argus 的兩份副本
移植（`pipelines/onprem/` 交付副本 → 各雲端 job）。移植方向、測試素材清單、GPU 環境與
權限約束、以及各條線目前的進度**不記在本檔**，見 `~/.claude/playbook/vfa-argus-topology.md`
與 board（`~/.claude/board`，看板由 `wf next` 產生；兩者皆為本機個人筆記，
不進版控，同仁看不到）。

## 常用指令

```bash
uv sync --all-packages                             # 全量同步（含 torch）

uv run --package video_analyze video_analyze       # 偵測/追蹤 → tracking_results.parquet
uv run --package zone_mapping  zone_mapping        # zone 區域佔用統計 → zone_counts.parquet
uv run --package line_counting line_counting       # 計數線進出人數統計 → line_counts.parquet
uv run --package flow_report   flow_report         # 報表彙總 → report.xlsx

uv run --directory <pkg> ruff check .              # lint；<pkg> = video_analyze / zone_mapping / line_counting / flow_report
uv run --directory <pkg> pytest                    # 測試（四包各 18／3／3／3 支測試檔）

uv run --directory libs/vfa_registry pytest        # 共用 lib 的測試（4 支）自成一套，不在四包底下
uv run --directory libs/vfa_registry ruff check .
uv run --directory libs/vfa_observability pytest   # （1 支）
uv run --directory libs/vfa_observability ruff check .
uv run --directory libs/vfa_config pytest          # （1 支）
uv run --directory libs/vfa_config ruff check .

uv run --no-project --with pytest --with pyyaml pytest tests/    # 文件契約測試（釘住文件裡可機械驗證的斷言）
```

**torch 隔離**：workspace 為單一 `.venv`，但 `uv sync --package <pkg>` 只裝該包依賴子樹
（`flow_report`／`zone_mapping`／`line_counting` 不含 torch）；`uv sync --all-packages` 才裝含
torch 的完整環境。部署時各容器以 `uv sync --package <pkg>` 維持 CPU-only 隔離。

**pytest／ruff 用 `--directory`（切換 cwd）而非 `--package`**：`--package` 不改變 cwd，
`pytest` 會從 repo 根遞迴收集到所有套件的測試而撞名衝突（`tests/test_config.py` 等檔名
四包重複）；`--directory` 切進該套件資料夾，才會只解析到該套件自己的 `tests/`。

根層的文件契約測試（`tests/`）要**指定路徑**並帶 `--no-project`：不指定路徑會遞迴收集到
四包而撞名；`--no-project` 則是因為該測試的相依只有 pytest 與 pyyaml（後者用來解析
`.github/workflows/ci.yml`），經 workspace 解析會為了跑文件檢查而裝上 `video_analyze`
的 torch 依賴子樹。**不要改用在根 `pyproject.toml` 填
`testpaths` 的寫法**——那會讓 `uv run --package <pkg> pytest` 的撞名保護消失，變成靜默
只跑文件測試、一支套件測試都沒跑卻回報通過。

**執行 cwd 約束**：`bucket_dir` 與 `OUTPUT_ROOT = Path("outputs")` 是**cwd 相對路徑**，
與各套件 `config.toml` 的檔案定位（四包 DDD 重構後皆用 `find_project_root` 往上找
`pyproject.toml`）是兩套機制。四包一律以 `--package`／`--directory` 指定套件、**在 repo
根目錄執行**（`uv run` 不改變 cwd）；若改在套件資料夾內執行，`bucket_dir` 會對到不存在
的路徑，`outputs/` 也會裂成四棵互不相通的樹，讓階段間的檔案契約失效。

## 架構

技術決策記在 [docs/adr/](docs/adr/)，依影響的模組分子目錄：只動一個套件的放
`docs/adr/<套件名>/`，跨套件的放 `docs/adr/shared/`；編號是全域流水號、與子目錄無關，
新增一律取下一號。ADR 的清單、影響範圍與各自主題見
[README.md 的「架構決策紀錄」](README.md#架構決策紀錄)（唯一索引，本檔不另列一份）。

### 時區不變量（貫穿四包）

檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`video_analyze` 解析時即轉換成台北在地時間
（`Asia/Taipei`，UTC+8）。此後 `tracking_results.parquet` 的 `timestamp`、
`zone_counts.parquet`／`line_counts.parquet` 的 `time_bucket`、`report.xlsx` 的日期／小時
欄位皆為台北在地時間，下游（`zone_mapping`／`line_counting`／`flow_report`）不需要、也不
應該再對它們做任何 UTC→+8 位移。

## 其他注意事項

- `*.pt`（模型權重）、`bucket_*`、`outputs`（皆刻意不帶尾斜線，讓它是 symlink 時也擋得住）
  皆在 `.gitignore`，不進版控
  （`camera_registry.yaml` 含 zone／line 定義，隨 `bucket_*` 一起不進版控）。
  `bucket_*` 涵蓋所有 `bucket_` 開頭的目錄，`bucket_name1` 與
  `bucket_<日期>_<變體>` 兩種命名都擋得住。
- 四包版本 pin 成彼此一致（`torch`/`ultralytics`/`numpy`/`opencv` 等推理堆疊、
  `polars`/`pyarrow`/`openpyxl` 等輸出格式相關套件），避免函式庫版本漂移造成非邏輯性的
  輸出差異；`line_counting` 的 `numpy`/`polars`/`pyarrow`/`pydantic`/`pyyaml` 與 `zone_mapping`
  pin 成同版，`libs/` 底下三個 lib 的 `pydantic`／`pyyaml` 也在此範圍內。單一 root `uv.lock`
  下版本一致由 `uv lock` 自動把關——`==` pin 彼此衝突會直接讓 `uv lock` 解析失敗；新增或
  升級依賴時留意是否需要四包同步。

## 碰到特定目錄才載入的設計說明

下表的檔只在讀到或改到對應路徑時自動載入；「也該先讀」那欄的情況不會自動觸發，要自己開來讀。
自動載入的路徑條件以各檔開頭的 `paths:` 為準，下表那欄只是摘要。

| 檔案 | 讀到或改到這些路徑時自動載入 | 沒碰那些檔也該先讀的時候 |
|---|---|---|
| [frame-size-columns.md](.claude/rules/frame-size-columns.md) | `video_analyze/**`、`zone_mapping/**`、`line_counting/**` | 改 `tracking_results.parquet` 的欄位前 |
| [foot-point.md](.claude/rules/foot-point.md) | `video_analyze/**`、`zone_mapping/**`、`line_counting/**` | 改 `tracking_results.parquet` 的欄位前 |
| [shared-code.md](.claude/rules/shared-code.md) | `libs/**`、各 `pyproject.toml`、各包 `config.toml`、`models/config.py`、`tests/test_config.py` | 新增依賴、新增 `config.toml` 頂層區塊前 |
| [zone-line-names.md](.claude/rules/zone-line-names.md) | `zone_mapping/**`、`line_counting/**`、`flow_report/**`、`libs/vfa_registry/**` | 改 `camera_registry.yaml` 的格式前 |
| [flow-report-inputs.md](.claude/rules/flow-report-inputs.md) | `flow_report/**`、`zone_mapping/**`、`line_counting/**` | 調整四包的執行順序前 |
| [tracking-results-reproducibility.md](.claude/rules/tracking-results-reproducibility.md) | `video_analyze/**`、`zone_mapping/**`、`line_counting/**` | 比對兩次執行的輸出、判斷改動有沒有改變偵測結果前 |
