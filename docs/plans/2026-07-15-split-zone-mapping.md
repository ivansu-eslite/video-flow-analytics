# 任務 2：建立 `zone-mapping` 獨立套件（細項計畫）

> 對應 Issue: #20 — https://github.com/ivansu-eslite/video-flow-analytics/issues/20

> 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)。
> 對應 sub-issue 與單一 PR。

## Context

`zone-map` 階段讀 `tracking_results.parquet`，套上人工維護在 `camera_registry.yaml` 各
攝影機底下的 zone 幾何，輸出每時段每區域的人流統計（`unique_visitors`／`entries`）。
純 CPU 向量化運算，**不需要 GPU、也不需要 torch/ultralytics/opencv**——這正是它該獨立
成包的主因：拆開後依賴面大幅縮小，可跑在完全不同的平台上。

它未來會由自己的平台／呼叫方式負責，需從 monolith 抽成自成一體的套件。

## Scope

**包含**：建立 `zone-mapping/` 資料夾（自成一體的 uv 專案），搬入 `zone_mapping/*`、切出
自己的 config 與**完整版** registry、改寫 import、驗證輸出與 golden 基線一致。

**不包含**：舊 `src/video_flow_analytics/` 的刪除（父計畫任務 4）；拆分後的移植目標與
移植方式（另行討論）；新增測試（zone-map 目前無測試，維持現狀）。

## 目標結構

```
zone-mapping/
├── pyproject.toml
├── config.toml                 # 只含 [input] [zone]
├── README.md                   # 用途、進入點、上下游檔案契約
└── src/zone_mapping/
    ├── __init__.py
    ├── config.py               # 由 core/config.py 切出
    ├── registry.py             # 由 core/registry.py 切出（完整版）
    ├── pipeline.py             # map_zones_daily / run_zone_map
    └── stats.py                # points_in_polygon / count_zone_visits / validate_zone_cameras
```

## 實作細節

### 1. 檔案搬移（原樣，僅改 import）

`zone_mapping/pipeline.py`、`zone_mapping/stats.py`。

### 2. `config.py` 切片

保留 zone-map 實際讀到的區塊（依 `grep settings.` 界定，全部在 `pipeline.py` 的
`run_zone_map`）：

| Model | 保留欄位 | 實際讀取點 |
|---|---|---|
| `InputConfig` | `date`、`bucket_dir` | `settings.input.date`／`settings.input.bucket_dir` |
| `ZoneConfig` | `bucket_minutes`、`entry_debounce_frames` | `settings.zone.bucket_minutes`／`settings.zone.entry_debounce_frames` |

- `InputConfig` 可移除 `camera_ids`（zone-map 未使用）。
- 刪掉 `TrackerConfig`、`ModelConfig`、`OutputConfig`、`ReportConfig` 及 `AppConfig`
  對應欄位。
- **`load_config()` 的回退分支會跟著壞，必須一起改**（與 `load_registry_from_path` 同型
  的陷阱）：現況 `core/config.py` 的
  `return AppConfig(tracker=TrackerConfig(), model=ModelConfig())` 以名稱引用了上面剛
  刪掉的兩個 model。照「刪掉」字面做完就會在 `config.toml` 不存在時 `NameError`，而
  `settings = load_config()` 是模組載入時執行的，整包 import 直接炸。改成 `AppConfig()`
  （切片後 `tracker`／`model` 欄位已不存在，其餘欄位皆有 `default_factory`）。
  - 同時修 `load_config` docstring 的「僅 tracker/model 用預設值」——切片後不再成立。
- **修路徑 hack**：`load_config()` 的 `parents[3]` 改成 `parents[2]`，對到
  `zone-mapping/config.toml`，並同步更新說明註解的層數描述。
- `config.toml` 的 `[input] bucket_dir` 設為 **`bucket_name1`**，不要沿用根
  `config.toml`／`InputConfig` model 預設的 `bucket_name`（那是 112G 的 fixture，與
  golden 不符，見父計畫任務 0）。
- 保留全域 `settings` 單例的既有慣例。
- **`bucket_dir` 是 cwd 相對路徑**：本包一律以 `uv run --project zone-mapping
  zone-mapping` 在 repo 根執行，勿 `cd zone-mapping` 後再跑（見父計畫「硬約束：三包一律
  從 repo 根目錄執行」）。此約束需寫進本包 README。

### 3. `registry.py` 切片（**完整版**）

zone-map 需要 zone 幾何，`core/registry.py` **全數保留**：`Zone`、`CameraEntry`
（含 `parsed_zones()`）、`StorageConfig`、`CameraRegistry`、`_find_duplicates`、
`parse_and_validate_zones`、`registry_path`、`load_registry`、`load_registry_from_path`。

實際使用點：`pipeline.py` 用 `load_registry`、`parse_and_validate_zones`、`registry_path`；
`stats.py` 用 `Zone`。

> **與 `flow-report` 的關係**：兩包的 `registry.py` 內容相同，是本次唯一的實質重複
> （見父計畫「共用程式碼分析」）。刻意接受：兩者未來可能各奔不同平台。**不要**為了消重
> 而做跨資料夾 import。

保留現有的關鍵**設計**，勿在搬移時順手「優化」：

- `CameraEntry.zones` 維持 `list[Any]`（刻意不在此驗證幾何）；驗證延後到 `parsed_zones()`。
- `parse_and_validate_zones` 是「zone 名稱跨攝影機全域唯一」這條規則的**唯一實作位置**，
  `flow-report` 也有一份相同實作；改動時兩邊需同步。
- `CameraRegistry._unique_camera_identity` 對 `camera_id` 與 `stream_dirname` 的唯一性
  驗證（fail-loud，避免查詢字典靜默覆蓋）。

**但「不可改動的是行為，不是說明文字」**：現有註解／docstring 帶有跨模組引用，拆包後
會指向本包不存在的東西，**必須改寫**（否則留下騙人的註解）：

| 位置 | 現有說法 | 改成 |
|---|---|---|
| `registry.py` 的 `zones` 欄位 docstring | 「也被較重的 `analyze_daily` 讀取，若在此驗證…」 | 本包沒有 `analyze_daily`，該理由不成立。改寫為新理由（見下） |
| `registry.py` 的 `parsed_zones()` docstring | 「避免拖累 analyze_daily」 | 同上，本包無 `analyze_daily`；改為描述驗證順序的理由 |
| `registry.py` 的 `Zone.name` docstring | 「`zone_mapping` 與 `report` 皆會驗證」 | 兩包各自只看得到自己，改為描述本包行為 |
| `config.py` 的 `InputConfig` docstring | 「`analyze_daily` 輸入參數」／「正式呼叫端可直接以參數呼叫 `analyze_daily`」 | 本包無 `analyze_daily`。改為描述 `map_zones_daily` 的輸入 |
| `config.py` 的 `AppConfig` docstring | 「input: `analyze_daily` 輸入參數」 | 同上 |
| `config.py` 的 `load_config` docstring | 「讀取 **repo 根目錄**的 `config.toml`」 | 切片後對到的是**套件根**（`zone-mapping/config.toml`），非 repo 根 |
| `pipeline.py` 的 `_ZONE_COUNTS_SCHEMA` 註解 | 「見 `io/video_reader.py` 的 `_LOCAL_TZ`」 | 本包無 `io/`。改為「上游 `tracking_results.parquet` 的 `timestamp` 已是台北在地時間，見 README 的檔案契約」 |

**`list[Any]` 在本包仍要保留，但理由變了**（務必寫進 docstring，否則未來讀者會發現舊
理由不成立而誤刪這個設計）：不再是「別拖累 analyze」，而是 `pipeline.py` 刻意的驗證
順序——先 `validate_zone_cameras` 再 `parse_and_validate_zones`，讓「camera 對不上當天
資料」這個更根本的錯誤先報，而不是被 zone 定義的打字錯誤蓋過。

### 4. import 改寫

`video_flow_analytics.zone_mapping.stats` → `zone_mapping.stats`；
`video_flow_analytics.core.config` → `zone_mapping.config`；
`video_flow_analytics.core.registry` → `zone_mapping.registry`。

### 5. 進入點

```toml
[project.scripts]
zone-mapping = "zone_mapping.pipeline:run_zone_map"
```

### 6. `pyproject.toml` 依賴

`polars`、`pyarrow`、`numpy`、`pydantic`、`pyyaml`。
**不需** torch／ultralytics／opencv／openpyxl——拆包後依賴面縮小是本任務的重點成果之一。
ruff 設定沿用（`line-length=88`、`select=["E","F","I","W"]`、`target-version="py312"`）。

## Acceptance Criteria

- [ ] 資料輸入/輸出或 API 規格定義清楚：進入點 `map_zones_daily(date, bucket_dir,
      bucket_minutes, ...) -> Path` 簽章不變；讀 `tracking_results.parquet`、寫
      `zone_counts.parquet` + `camera_registry_used.yaml` 快照的路徑與 schema 不變。
- [ ] 測試方式與驗收情境明確：以父計畫任務 0 golden（fixture 為 `bucket_name1`）的
      `tracking_results.parquet` 為輸入跑 zone-map，輸出 `zone_counts.parquet` 與
      `camera_registry_used.yaml` 與 golden **逐值一致**。
- [ ] 觀測指標明確：`N/A`——結構重整，不新增觀測指標。
- [ ] 影響範圍已列出：僅新增 `zone-mapping/`；舊 `src/` 不動（任務 4 才移除），與任務
      1／3 無檔案重疊。
- [ ] `zone-mapping/` 可獨立 `uv sync`，且**不含** torch／ultralytics／opencv／openpyxl。
- [ ] **以 `uv run --project zone-mapping zone-mapping` 在 repo 根執行**，輸出落在 repo
      根的 `outputs/bucket_name1/2026-05-01/`（與 golden 同一棵樹）；README 已記載此
      cwd 約束。
- [ ] **`config.toml` 的 `[input] bucket_dir` 為 `bucket_name1`**，且 `[zone]
      bucket_minutes`／`entry_debounce_frames` 與產生 golden 時的根 `config.toml` 一致。
- [ ] **`config.toml` 不存在時不會炸**：暫時移開 `zone-mapping/config.toml` 後
      `python -c "import zone_mapping.pipeline"` 仍可載入（只印警告、走 `AppConfig()`
      回退），確認回退分支未殘留已刪除的 `TrackerConfig`／`ModelConfig` 引用。
- [ ] 完整 registry 能載入真實 `bucket_name1/camera_registry.yaml`（5 個 zone、名稱跨
      攝影機唯一），且跨攝影機重複 zone 名稱仍會 fail-loud（`parse_and_validate_zones`
      行為不變）。
- [ ] `uv run ruff check .` 乾淨。

## Risk

- **與 `flow-report` 的 registry.py 漂移**：兩份相同實作，未來改動需同步；由父計畫的
  「三包載入同一份 yaml」驗收把關。
- **config 路徑層數**：`parents[3]` → `parents[2]` 若漏改，`load_config` 只印警告並**以
  預設值回退**（`bucket_minutes=60`／`entry_debounce_frames=1`），會靜默用錯粒度。以
  golden 一致性驗收涵蓋。
- **`load_config()` 回退分支引用已刪除的 model**：`AppConfig(tracker=TrackerConfig(),
  model=ModelConfig())` 未一併改成 `AppConfig()` 的話，`config.toml` 一旦不存在就
  `NameError`，且因 `settings = load_config()` 在模組載入時執行，是整包 import 失敗而非
  單一功能壞掉。ruff 的 F821 可攔（`select` 含 `F`），另有上方「config.toml 不存在時不
  會炸」的 AC 直接驗。
- **cwd 相對路徑**：`bucket_dir`／`OUTPUT_ROOT` 跟著 cwd 走而非跟著檔案走。若 `cd
  zone-mapping` 後執行，會去 `zone-mapping/outputs/` 找上游 parquet 而 `FileNotFoundError`。
  由「一律從 repo 根以 `--project` 執行」的約束與對應 AC 把關。
- **`config.toml` 沿用 `bucket_name`**：切片時從根 `config.toml` 複製會帶到 112G 的
  fixture，與 golden 的 `bucket_name1` 路徑不同層，比對只得到「檔案不存在」。由上方
  AC 明訂。
- **`InputConfig` 移除 `camera_ids`**：確認 `run_zone_map` 確實未使用（現況 `grep
  settings.` 僅見 `input.date`／`input.bucket_dir`）；若誤刪需回補。
- **資料品質**：zone 幾何為人工維護，本次不改驗證邏輯，既有 fail-loud 行為需原樣保留。
- **留下騙人的註解**：搬移後若照抄跨模組引用（`analyze_daily`、`io/video_reader.py`），
  註解會指向本包不存在的東西。尤以 `zones: list[Any]` 為甚——舊理由（別拖累 analyze）
  在本包不成立，未來讀者可能因此誤刪這個刻意設計。見上方改寫對照表。
- **權限／安全／成本／模型準確率**：`N/A`——純 CPU 結構重整，不涉外部資源。

## Related Links

- 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)
- 姊妹任務（registry.py 相同）：[任務 3：flow-report](2026-07-15-split-flow-report.md)
