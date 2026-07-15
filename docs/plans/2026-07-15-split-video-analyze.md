# 任務 1：建立 `video-analyze` 獨立套件（細項計畫）

> 對應 Issue: #19 — https://github.com/ivansu-eslite/video-flow-analytics/issues/19

> 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)。
> 對應 sub-issue 與單一 PR。

## Context

`analyze` 階段（YOLO 偵測 + ByteTrack 多路追蹤）是三個階段中最重的一支：GPU、多進程、
共享記憶體環形緩衝。它未來會由**自己的平台／呼叫方式**負責，因此需從 monolith
`src/video_flow_analytics/` 抽成自成一體的套件。

`io/`（影格讀寫、環形緩衝）與 `visualization/`（追蹤框標註）**只有 analyze 使用**，
直接隨此套件走，不涉及與其他兩包的共用問題。真正要處理的只有 `core/config.py` 與
`core/registry.py` 的切片。

## Scope

**包含**：建立 `video-analyze/` 資料夾（自成一體的 uv 專案），搬入 analyze 專屬模組、
切出自己的 config 與精簡 registry、改寫 import、驗證輸出與 golden 基線一致。

**不包含**：舊 `src/video_flow_analytics/` 的刪除（父計畫任務 4）；拆分後的移植目標與
移植方式（另行討論）；**新增**測試（既有 2 份原樣搬入、僅改 import，不擴充）；模型權重與
`bucket_name*/` 測試資料。

## 目標結構

```
video-analyze/
├── pyproject.toml
├── config.toml                 # 只含 [tracker] [model] [output] [input]
├── README.md                   # 用途、進入點、上下游檔案契約
├── tests/                      # 既有測試，原樣搬入僅改 import
│   ├── analyze/test_fps_meter.py     # 由 tests/analyze/ 搬入
│   └── io/test_video_reader.py       # 由 tests/io/ 搬入
└── src/video_analyze/
    ├── __init__.py
    ├── config.py               # 由 core/config.py 切出
    ├── registry.py             # 由 core/registry.py 切出（精簡版）
    ├── pipeline.py             # analyze_daily / run_analyze
    ├── detector.py
    ├── fps_meter.py
    ├── inference.py
    ├── tracker.py
    ├── tracking_results.py
    ├── io/                     # analyze 專屬，原樣搬入
    │   ├── __init__.py
    │   ├── frame_ring.py
    │   ├── video_reader.py
    │   └── video_writer.py
    └── visualization/          # analyze 專屬，原樣搬入
        ├── __init__.py
        └── visualizer.py
```

## 實作細節

### 1. 檔案搬移（原樣，僅改 import）

`analyze/*`（detector, fps_meter, inference, pipeline, tracker, tracking_results）、
`io/*`（frame_ring, video_reader, video_writer）、`visualization/visualizer.py`。

**既有測試 2 份也要搬**（依 `git ls-files tests`；`CLAUDE.md` 舊敘述曾誤稱 analyze 無
測試，已修正）——任務 4 會刪掉舊 `tests/`，漏搬即永久消失：

| 測試檔 | 搬到 | import 改寫 |
|---|---|---|
| `tests/analyze/test_fps_meter.py` | `video-analyze/tests/analyze/` | `video_flow_analytics.analyze.fps_meter` → `video_analyze.fps_meter` |
| `tests/io/test_video_reader.py` | `video-analyze/tests/io/` | `video_flow_analytics.io.video_reader` → `video_analyze.io.video_reader`（測的是私有函式 `_parse_segment_start`） |

### 2. `config.py` 切片

保留 analyze 實際讀到的四個區塊（依 `grep settings.` 界定）：

| Model | 保留原因（實際讀取點） |
|---|---|
| `TrackerConfig` | `tracker.py`：`BYTETracker(args=settings.tracker)` 整包傳入 |
| `ModelConfig` | `detector.py`：`settings.model.model_path`／`settings.model.batch`；`inference.py`：`settings.model.batch` |
| `OutputConfig` | `pipeline.py`／`io/video_writer.py`：`settings.output.save_video` |
| `InputConfig` | `pipeline.py`：`settings.input.date`／`bucket_dir`／`camera_ids` |

- 刪掉 `ZoneConfig`、`ReportConfig` 及 `AppConfig` 對應欄位。
- **修路徑 hack**：`load_config()` 現用 `parents[3]` 定位 repo 根的 `config.toml`。本檔
  移到 `src/video_analyze/config.py` 後，改成 `parents[2]` 對到 `video-analyze/config.toml`，
  並同步更新該處說明註解的層數描述。
- 保留模組載入時建立全域 `settings` 單例的既有慣例（不改成依賴注入）。

### 3. `registry.py` 切片（**精簡版**）

analyze 只用 `load_registry` 與 `resolve_cameras`，完全不碰 zone 幾何。

- **保留**：`CameraRegistry`（含 `_unique_camera_identity` 驗證、`resolve_cameras`）、
  `CameraEntry`（含 `stream_dirname`）、`StorageConfig`、`_find_duplicates`、
  `registry_path`、`load_registry`。
- **移除**：`Zone`、`CameraEntry.parsed_zones()`、`parse_and_validate_zones`
  （analyze 皆未使用）。
- **`load_registry_from_path` 移除的是「公開 API」，不是它的程式碼**：analyze 不需要讀
  快照檔的入口，但 `load_registry` 現在就是靠它實作的——
  `return load_registry_from_path(registry_path(bucket_dir))`。直接刪掉函式會讓
  `load_registry` 在呼叫時 `NameError`。正確做法是把它的主體（存在性檢查 →
  `yaml.safe_load` → `CameraRegistry(**data)`）**inline 進 `load_registry`**。
- **必須保留欄位**：`CameraEntry.zones: list[Any]` 與 `participates_in_zone_mapping`。
  `CameraEntry` 用 `extra="forbid"`，registry 有這兩個欄位而模型沒有，載入會直接失敗。
  `zones` 保持 `list[Any]`（本就刻意不驗證幾何），此處單純接受並忽略。
  - `zones`：兩份 fixture 的 registry 都有，**真實 yaml 即可驗到**。
  - `participates_in_zone_mapping`：**兩份 fixture 都沒有**，真實 yaml 驗不到。它仍是
    registry 格式的正式訊號，漏掉會等到將來某份 registry 用到它才爆，故仍須保留，並用
    合成 yaml 驗（見下方驗收條件）。

### 4. import 改寫

`video_flow_analytics.analyze.X` → `video_analyze.X`；
`video_flow_analytics.io.X` → `video_analyze.io.X`；
`video_flow_analytics.visualization.visualizer` → `video_analyze.visualization.visualizer`；
`video_flow_analytics.core.config` → `video_analyze.config`；
`video_flow_analytics.core.registry` → `video_analyze.registry`。

### 5. 進入點

拆掉共用 `cli.py`（原三子命令 lazy import），本套件單一進入點：

```toml
[project.scripts]
video-analyze = "video_analyze.pipeline:run_analyze"
```

原 `cli.py` 為了讓 zone-map／report 不必載入 torch/ultralytics 而做的 lazy import，在拆包
後由「套件本身不含那些依賴」天然達成，不需保留。

### 6. `pyproject.toml` 依賴

`opencv-python`、`ultralytics`、`numpy`、`polars`、`pyarrow`、`lap`、`pydantic`、`pyyaml`。
**不需** `openpyxl`（那是 report 專屬）。
**dev group 需 `pytest`**（本包有 2 份既有測試），並沿用
`[tool.pytest.ini_options] testpaths=["tests"]`。
ruff 設定沿用（`line-length=88`、`select=["E","F","I","W"]`、`target-version="py312"`）。

## Acceptance Criteria

- [ ] 資料輸入/輸出或 API 規格定義清楚：進入點 `analyze_daily(date, bucket_dir,
      camera_ids=None) -> AnalysisResult` 簽章不變；輸出 `outputs/{bucket}/{date}/
      tracking_results.parquet` 路徑與 schema 不變。
- [ ] 測試方式與驗收情境明確：對 **`bucket_name1`** 測試資料（5.6G，非 112G 的
      `bucket_name`）跑 analyze，`tracking_results.parquet` 與父計畫任務 0b 的 golden
      比對通過。**比對方式依任務 0a 的可重現性探測結果決定**——不預設「逐值一致」，
      因為現況輸出很可能本來就不可重現（時序相依湊批 + `bucket_name1` 混解析度使
      letterbox 隨批次組成變動，見父計畫 0a）。
- [ ] **既有 2 份測試已搬入且全過**：`uv run pytest` 涵蓋 `tests/analyze/test_fps_meter.py`
      與 `tests/io/test_video_reader.py`，全數通過、不得有 skip。
- [ ] 觀測指標明確：`N/A`——結構重整，既有處理 FPS log 行為不變。
- [ ] 影響範圍已列出：僅新增 `video-analyze/`；舊 `src/` 不動（任務 4 才移除），與任務
      2／3 無檔案重疊。
- [ ] `video-analyze/` 可獨立 `uv sync`，且不含 `openpyxl` 等非必要依賴。
- [ ] 精簡 registry 的 `extra="forbid"` 相容性（**兩層，缺一不可**）：
      (a) 能載入真實 `bucket_name1/camera_registry.yaml`（驗 `zones`）；
      (b) 能載入一份**含 `participates_in_zone_mapping` 的最小合成 yaml**——現有 fixture
      都沒有此欄位，只做 (a) 會假性通過。
- [ ] `uv run ruff check .` 乾淨。

## Risk

- **精簡 registry 漏欄位導致載入失敗**：拿掉 `zones`／`participates_in_zone_mapping` 會在
  `extra="forbid"` 下爆掉。以上方相容性驗收條件 (a)(b) 把關。
- **`participates_in_zone_mapping` 的覆蓋缺口**：兩份 fixture 都沒有此欄位，漏掉它不會被
  真實 yaml 測出來，而是等到將來某份 registry 用到它才爆——這是本任務最容易被漏掉的風險，
  故驗收條件 (b) 特別要求合成 yaml。
- **config 路徑層數**：`parents[3]` → `parents[2]` 若漏改，會靜默找不到 `config.toml`
  並**以預設值回退**（現行 `load_config` 只印警告不中止），造成用錯參數卻不易察覺。
  驗收時需確認 golden 一致即可涵蓋此風險。
- **多進程／共享記憶體行為**：`io/frame_ring.py` 與 `pipeline.py` 的 fork 多進程、關檔
  順序等既有講究不在本次改動範圍，僅搬移不改邏輯，以 golden 對比確認未破壞。
- **刪錯 `load_registry_from_path`**：它是 `load_registry` 的實作依賴，照「移除」字面
  直接刪會 `NameError`（見上方切片說明，須 inline 而非刪除）。
- **既有 2 份測試漏搬**：任務 4 會刪掉舊 `tests/`，漏搬即永久消失。`CLAUDE.md` 舊敘述
  曾誤稱 analyze 無測試（已修正），勿再據此判斷。
- **資料品質／權限／成本／模型準確率**：`N/A`——不動推理邏輯與模型權重。

## Related Links

- 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)
