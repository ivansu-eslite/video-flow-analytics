# video-flow-analytics 拆成三個獨立套件（總體計畫）

> 對應 Issue: #18 — https://github.com/ivansu-eslite/video-flow-analytics/issues/18

> 這是**總體計畫**，對應**父 issue**。三個實作任務各有細項計畫（見 Related Links），
> 對應三個 sub-issue 與三個 PR。

## Context

`video-flow-analytics` 目前是單一 Python 套件（`src/video_flow_analytics/`），內含三個
刻意解耦的離線階段：`analyze`（YOLO+ByteTrack，GPU、多進程、重）、`zone-map`（zone 人流
統計，純 CPU 向量化）、`report`（彙總成跨日累加的 Excel，純 CPU）。

三個階段未來會由**不同平台／呼叫方式**負責，因此要在本 repo 內先拆成三個各自自成一體的
套件（各帶自己的 `pyproject`／`config`／`registry` 切片），讓依賴面收斂、各自可獨立安裝
與執行。

**本次範圍只到「在本 repo 內完成拆分」**；拆完之後往何處移植、如何移植，另行討論。

### 為什麼是「三包各自複製共用碼」而非「共用 lib」

此設計取決於未來移植目標的架構約束——該目標為「一堆各自獨立、自成一體專案」型
monorepo：每個專案各帶自己的 `pyproject`／`requirements`，**無共用 lib 目錄、無跨資料夾
import**。加上三個階段未來可能各自落到不同平台，一旦其中一個移走，共用 lib 立刻斷裂。故：

- **排除「共用 lib 資料夾」**：與目標架構慣例衝突，且會在跨平台移動時斷裂。
- **排除「維持單一套件」**：與「不同平台／呼叫方式各自負責」的初衷衝突。
- **採「三個獨立套件、各帶自己的 registry/config 切片」**，接受 `registry.py` 在
  zone-map／report 重複兩份。

## Scope

**包含**：把本 repo 重整為三個自成一體的 uv 專案，切分共用的 `core/config.py` 與
`core/registry.py`，改寫內部 import 與 config 載入路徑，並以 golden 對比驗證三者輸出
與拆分前一致。

**不包含**：
- 拆分後的移植目標與移植方式——另行討論，不在本計畫。
- 各階段接到目標平台的實際呼叫方式（env／args／排程）。
- **新增**測試。既有測試一律隨對應套件搬移、不擴充：`tests/analyze/test_fps_meter.py`
  與 `tests/io/test_video_reader.py` → `video-analyze`；`tests/report/*` → `flow-report`。
  `zone-mapping` 是唯一無既有測試的一包，本次不補。
- 模型權重 `yolo26m*.pt`、`bucket_name*/` 測試資料（非程式碼，不動）。

## 目標結構

```
video-analyze/     # GPU，重（含 io/、visualization/）；既有測試 2 份
zone-mapping/      # 純 CPU；唯一無既有測試
flow-report/       # 純 CPU；既有測試 2 份
```

每個資料夾自成一體：`pyproject.toml` + `src/<pkg>/` + `README.md` + `config.toml`，
另加 `tests/`（`zone-mapping` 無既有測試，本次不建）。Python 套件名用 snake_case：
`video_analyze`／`zone_mapping`／`flow_report`。

**既有測試歸屬**（`git ls-files tests` 為準，勿依賴 `CLAUDE.md` 的舊敘述）：

| 測試檔 | 歸屬 |
|---|---|
| `tests/analyze/test_fps_meter.py` | `video-analyze` |
| `tests/io/test_video_reader.py` | `video-analyze` |
| `tests/report/test_pipeline.py` | `flow-report` |
| `tests/report/test_stats.py` | `flow-report` |

四份**全部**要搬。任務 4 會刪掉舊 `tests/`，任一份漏搬即永久消失且無驗收條件抓得到。

### 共用程式碼分析（拆分的全部成本）

三者共用的程式碼**只有 `core/` 底下兩支**；`io/`、`visualization/` 是 analyze 專屬，
各自的 `stats.py` 本就獨立。

| 共用檔 | analyze | zone-map | report | 處理方式 |
|---|---|---|---|---|
| `core/config.py` | `tracker`/`model`/`output`/`input` | `input`/`zone` | `input`/`zone.bucket_minutes`/`report` | **好切**：各拿自己 `run_*` 實際讀到的區塊 |
| `core/registry.py` | `CameraRegistry`/`CameraEntry`/`StorageConfig`/`load_registry`/`resolve_cameras` | 完整（含 `Zone`/`parse_and_validate_zones`） | 完整（含 `load_registry_from_path`） | **複製兩種形狀**：analyze 精簡版；zone-map 與 report 各一份完整版（內容相同） |

`registry.py` 在 zone-map／report 重複兩份（約 200 行 × 2）是本次**唯一的實質重複**。

### `camera_registry.yaml`：一份資料、三份模型

```
      camera_registry.yaml   ← 只有一份，放在 bucket_dir（執行時參數傳入，不在 repo）
      ┌───────┼───────┐
 analyze   zone-map   report
 (精簡      (完整      (完整
  registry)  registry)  registry)
```

三支程式讀**同一份實體檔案**（`registry_path(bucket_dir)`），資料層面無重複、不需複製
yaml。但讀它的 Pydantic 模型會有三份，因此有個硬約束：

> 這份 yaml 內含 `zones` 欄位（格式上另有 `participates_in_zone_mapping`），而
> `CameraEntry` 用 `extra="forbid"`。所以**即使 analyze 用不到 zone，它的精簡 registry
> 也必須保留這兩個欄位**（`zones: list[Any]` 忽略即可），否則載入會直接解析失敗。

**測試覆蓋缺口（需明確處理）**：現有兩份 fixture（`bucket_name`／`bucket_name1`）的
registry **都有 `zones`、但都沒有 `participates_in_zone_mapping`**。因此「拿真實 yaml
載入不報錯」只驗得到 `zones`，驗不到 `participates_in_zone_mapping`——若精簡 registry
漏掉後者，golden 測試會**假性通過**，直到將來某份 registry 真的用到它才爆。相容性驗收
須額外用一份**含 `participates_in_zone_mapping` 的最小合成 yaml** 補上這個缺口。

### 階段間的檔案契約（非程式碼共用）

三階段透過 `outputs/{bucket}/{date}/` 檔案交棒：analyze 寫 `tracking_results.parquet`
→ zone-map 讀它、寫 `zone_counts.parquet` + `camera_registry_used.yaml` 快照 → report
讀這兩者。這是**檔案系統契約**，各套件各自保有 `OUTPUT_ROOT = Path("outputs")` 與路徑
慣例，並在各自 README 記錄上下游檔案，不需跨套件 import。

## 任務分解

**關鍵設計：先存 golden 基線，三個任務就變成真正獨立、可平行。** 三者都只是**新增**各自
資料夾（舊 `src/` 原封不動留到最後），彼此不碰同一個檔案（三個 PR 不會衝突）；驗證鏈所需
的上游 parquet 由 golden 基線提供，不必等前一個任務完成。

| # | 任務 | 產出 | 相依 |
|---|---|---|---|
| 0 | **前置：存 golden 基線**（非程式碼變更，無 PR） | 三份 golden 輸出 | 無 |
| 1 | `video-analyze` 套件 | sub-issue + PR | 任務 0 |
| 2 | `zone-mapping` 套件 | sub-issue + PR | 任務 0 |
| 3 | `flow-report` 套件 | sub-issue + PR | 任務 0 |
| 4 | **收尾：移除舊 monolith**（小 PR） | PR | 任務 1-3 全數合併 |

### 任務 0：存 golden 基線（前置，所有任務的驗證依據）

**fixture 用 `bucket_name1`（非 `bucket_name`）**：兩者同為 4 台攝影機（test_cam001-004）、
同一天（2026/05/01）、片段數相近（48 vs 44），但 `bucket_name1` 只有 5.6G，`bucket_name`
達 112G——**小 20 倍**，而 golden 對比要跑好幾輪 GPU 推理，用大的那份代價過高且無額外
覆蓋。`bucket_name1` 的 registry 有 5 個 zone 且**名稱跨攝影機唯一**，足以驗到
`parse_and_validate_zones` 的全域唯一規則。

#### 0a. 先探測可重現性（**必須先做，會決定任務 1 的 AC 怎麼寫**）

**現況 analyze 的輸出很可能不是 run-to-run 可重現的**，兩個獨立原因：

1. **列順序非決定性**：`analyze/inference.py` 的 `_collect_batch` 用 `get_nowait()` 輪詢
   各路、湊不滿目標批次就等 `_FILL_MAX_WAIT = 0.004` 秒。批次的跨串流交錯**取決於當下
   時序**，`TrackingResultCollector.add` 照批次順序 append，row group 切點也跟著漂。
2. **偵測數值本身可能變動**：`detector.predict` 直接餵一個 list，而 `bucket_name1`
   **是混解析度**（實測：cam001-003 為 1920×1080，**cam004 為 2880×1620**）。批次跨串流
   組成，一批可能同時含兩種尺寸，ultralytics letterbox 會 pad 到批內共同尺寸——**前處理
   隨批次組成變動 → 偵測 float 值變動 → track 座標變動**。

所以「逐值一致」若照字面寫，會一直紅燈且**分不清是拆壞了還是本來就不可重現**。

**做法**：用未改動的 monolith 對 `bucket_name1` **連跑兩次**，比對兩次的
`tracking_results.parquet`，記錄結論：

| 探測結果 | 任務 1 的 AC 應改成 |
|---|---|
| 完全一致 | 維持逐值一致 |
| 僅列順序不同 | 依 `(camera_id, frame_id, track_id)` **排序後**逐值比對 |
| 數值也有微小差異 | 座標改**容差比對**（定出 `abs_tol`），或把主驗收標的**下移到 `zone_counts.parquet`** |

**`zone_counts.parquet` 是更穩的比對點**：`zone_mapping/pipeline.py` 對它做了
`.sort("camera_id","zone","time_bucket")`，且經 time bucket 聚合後對微小座標抖動不敏感。
若 0a 顯示 analyze 輸出不可重現，就以它作為「拆分未破壞行為」的主要證據。

任務 2／3 不受此影響：它們的輸入是 golden 檔案而非重跑推理，輸出路徑本身也有排序。

#### 0b. 產生 golden

用**現況 monolith** 對 `bucket_name1` 跑完 `analyze → zone-map → report`，保留下列檔案
作為 golden（`{bucket}` = `bucket_name1`），三個任務各自比對：

- `outputs/bucket_name1/2026-05-01/tracking_results.parquet`（任務 1 的驗收標的；任務 2 的輸入）
- `outputs/bucket_name1/2026-05-01/zone_counts.parquet` + `camera_registry_used.yaml`
  （任務 2 的驗收標的；任務 3 的輸入）
- `outputs/bucket_name1/report.xlsx`（任務 3 的驗收標的）

跑之前需把 `config.toml` 的 `[input] bucket_dir` 由 `bucket_name` 改成 `bucket_name1`
（`date = 2026-05-01` 兩份 fixture 皆適用）。

### 任務 4：收尾（三個 PR 都合併後）

- 刪除 `src/video_flow_analytics/`、根目錄舊 `pyproject.toml`／`config.toml`／`uv.lock`、
  舊 `tests/`。
- 更新根 `README.md`／`CLAUDE.md`：改述為三包結構與各自進入點。

## Acceptance Criteria

- [ ] 資料輸入/輸出或 API 規格定義清楚：三包各自的進入點（`run_analyze`／`run_zone_map`／
      `run_report`）與 `outputs/{bucket}/{date}/` 檔案契約不變，見各細項計畫。
- [ ] 測試方式與驗收情境明確：三包各自對 golden 基線做輸出逐值比對，見各細項計畫。
- [ ] 觀測指標明確：`N/A`——本次為結構重整，不新增觀測指標；analyze 既有的處理 FPS log
      行為不變。
- [ ] 影響範圍已列出：見上方「共用程式碼分析」與「任務分解」。
- [ ] 三包各自 `uv sync` 可獨立安裝、`uv run ruff check .` 乾淨。
- [ ] **既有 4 份測試全部有著落**：`video-analyze` 與 `flow-report` 各自 `uv run pytest`
      全過（不得有 skip）；合計通過數 ≥ 拆分前 `uv run pytest` 的通過數（防止漏搬）。
- [ ] 依賴面收斂：zone-map／report 不含 torch／ultralytics／opencv；analyze 不含 openpyxl。
- [ ] **「一份 yaml、三份模型」相容性（兩層）**：
      (a) 同一份真實 `bucket_name1/camera_registry.yaml` 三包各自 `load_registry` 皆不
      報錯（驗 `zones` 欄位相容）；
      (b) 另用一份**含 `participates_in_zone_mapping` 的最小合成 yaml** 三包各自載入亦
      不報錯——現有 fixture 都沒有此欄位，不補這條會假性通過。
- [ ] 三包輸出與 golden 基線一致（fixture 為 `bucket_name1`）。**比對方式依任務 0a 的
      可重現性探測結果決定**，不預設「逐值一致」——`zone_counts.parquet` 與
      `report.xlsx` 的產生路徑有排序、可逐值比對；`tracking_results.parquet` 視 0a
      結果採排序後比對或容差比對。
- [ ] 舊 monolith 已移除，根 README／CLAUDE.md 已更新（任務 4）。

## Risk

- **registry.py 在 zone-map／report 重複兩份**：兩者未來可能各奔不同平台，重複可接受；
  真正的資料契約是 `camera_registry.yaml`（資料非程式碼）。短期由「三包載入同一份 yaml」
  的驗收條件把關。
- **三份 registry 模型欄位漂移**：某一支載入同一份 yaml 失敗（尤其 analyze 精簡版在
  `extra="forbid"` 下漏掉 `zones` 欄位）。以上方相容性驗收條件 (a) 把關。
- **相容性測試的覆蓋缺口**：現有 fixture 都沒有 `participates_in_zone_mapping`，漏掉該
  欄位不會被真實 yaml 測出來，而是等到將來某份 registry 用到它才爆。以驗收條件 (b) 的
  合成 yaml 補上。
- **config 全域 `settings` 單例 + `config.toml` 路徑**：現用 `parents[3]` 寫死深度定位
  repo 根。本計畫只把它修成各套件自足（`parents[2]`）；各平台真正的呼叫方式（不一定讀
  `config.toml`）屬 Scope 外。
- **輸出一致性假性 diff**：polars／openpyxl 版本需在三包 pin 一致，避免 parquet／xlsx
  細節差異造成非邏輯性的 diff。
- **analyze 輸出可能本來就不可重現**（時序相依湊批 + `bucket_name1` 混解析度導致
  letterbox 隨批次組成變動）：若不先探測就寫死「逐值一致」，AC 會恆為紅燈且無法分辨
  「拆壞了」與「本來就會變」。由任務 0a 先探測、再定 AC 化解。
- **既有測試漏搬**：`CLAUDE.md` 曾寫「僅 report 有測試」（已於本次一併修正），實際有 4
  份。任務 4 會刪掉舊 `tests/`，漏搬即永久消失。由「合計通過數 ≥ 拆分前」的 AC 把關。
- **成本／權限／模型準確率**：`N/A`——純結構重整，不動推理邏輯與模型權重。

## Related Links

- 細項計畫（sub-issue）：
  - [任務 1：video-analyze](2026-07-15-split-video-analyze.md)
  - [任務 2：zone-mapping](2026-07-15-split-zone-mapping.md)
  - [任務 3：flow-report](2026-07-15-split-flow-report.md)
- 前一輪設計：[zone-report design](../specs/2026-07-06-zone-report-design.md)
- 移植目標與方式：另行討論，不在本計畫。
