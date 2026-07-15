# 任務 3：建立 `flow-report` 獨立套件（細項計畫）

> 對應 Issue: #21 — https://github.com/ivansu-eslite/video-flow-analytics/issues/21

> 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)。
> 對應 sub-issue 與單一 PR。

## Context

`report` 階段讀 `zone_counts.parquet`，做人流統計分析與跨期彙總，持續 **Append** 至單一
`outputs/{bucket}/report.xlsx`（對接 Looker Studio 等 BI 工具做長期觀測）。純 CPU 運算，
依賴面小但**多一個 `openpyxl`**，且是三包中**唯一有既有測試**的（`tests/report/`）。

它未來會由自己的平台／呼叫方式負責（很可能與 zone-map 不同——它是報表／BI 導向），需從
monolith 抽成自成一體的套件。

## Scope

**包含**：建立 `flow-report/` 資料夾（自成一體的 uv 專案），搬入 `report/*` 與既有測試、
切出自己的 config 與**完整版** registry、改寫 import、驗證輸出與 golden 基線一致且測試
全過。

**不包含**：舊 `src/video_flow_analytics/` 的刪除（父計畫任務 4）；拆分後的移植目標與
移植方式（另行討論）；新增測試（既有 `tests/report/` 原樣搬入，僅改 import）。

## 目標結構

```
flow-report/
├── pyproject.toml
├── config.toml                 # 只含 [input] [zone](僅 bucket_minutes) [report]
├── README.md                   # 用途、進入點、上下游檔案契約
├── src/flow_report/
│   ├── __init__.py
│   ├── config.py               # 由 core/config.py 切出
│   ├── registry.py             # 由 core/registry.py 切出（完整版）
│   ├── pipeline.py             # export_report_daily / run_report
│   └── stats.py                # to_taipei / rollup_by_period / peak_per_day
└── tests/                      # 由 tests/report/ 原樣搬入
```

## 實作細節

### 1. 檔案搬移（原樣，僅改 import）

`report/pipeline.py`、`report/stats.py`、`tests/report/*`。

### 2. `config.py` 切片

保留 report 實際讀到的區塊（依 `grep settings.` 界定，全部在 `pipeline.py` 的
`run_report`）：

| Model | 保留欄位 | 實際讀取點 |
|---|---|---|
| `InputConfig` | `date`、`bucket_dir` | `settings.input.date`／`settings.input.bucket_dir` |
| `ReportConfig` | `period_minutes`、`metric`、`on_duplicate_date` | `settings.report.*` |
| `ZoneConfig` | **僅** `bucket_minutes` | `settings.zone.bucket_minutes` |

- **`zone.bucket_minutes` 必須保留**：`export_report_daily` 需要它驗證
  `period_minutes % bucket_minutes == 0`（`_build_report_frames` 的 fail-loud 檢查）。
  保留最小 `zone` 區塊（僅此一欄）而非搬進 `report` 區塊，維持與 `zone-mapping` 的
  `config.toml` 語意對齊——兩者的 `bucket_minutes` 必須一致才合理。
- `InputConfig` 可移除 `camera_ids`（report 未使用）。
- 刪掉 `TrackerConfig`、`ModelConfig`、`OutputConfig` 及 `AppConfig` 對應欄位；
  `ZoneConfig` 移除 `entry_debounce_frames`（report 未使用）。
- **修路徑 hack**：`load_config()` 的 `parents[3]` 改成 `parents[2]`，對到
  `flow-report/config.toml`，並同步更新說明註解的層數描述。
- 保留全域 `settings` 單例的既有慣例。

### 3. `registry.py` 切片（**完整版**）

report 需要 zone 名稱與參與旗標，`core/registry.py` **全數保留**，內容與 `zone-mapping`
的那份相同。

實際使用點：`pipeline.py` 用 `load_registry_from_path`（讀
`camera_registry_used.yaml` **快照**）、`parse_and_validate_zones`、以及
`CameraEntry.participates_in_zone_mapping` 與 `stream_dirname`。

> **為何讀快照而非當下的 `camera_registry.yaml`**：驗證對象必須是產生該日
> `zone_counts.parquet` 當時的 `camera_registry_used.yaml`。若兩者之間改過 zone 名稱，
> 用即時檔案驗證會通過，但 parquet 裡其實是舊定義，可能讓不同攝影機的人流被靜默合併。
> 此行為在搬移時**不可更動**。

> **與 `zone-mapping` 的關係**：兩包的 `registry.py` 內容相同，是本次唯一的實質重複
> （見父計畫「共用程式碼分析」）。刻意接受。**不要**為了消重而做跨資料夾 import。

`parse_and_validate_zones` 驗證「zone 名稱跨攝影機全域唯一」——這是 report 依 zone 名稱
分組彙總（不含 `camera_id`）的前提，行為不可更動。

### 4. import 改寫

`video_flow_analytics.report.stats` → `flow_report.stats`；
`video_flow_analytics.core.config` → `flow_report.config`；
`video_flow_analytics.core.registry` → `flow_report.registry`；
`tests/report/*` 內的 import 同步改。

### 5. 進入點

```toml
[project.scripts]
flow-report = "flow_report.pipeline:run_report"
```

### 6. `pyproject.toml` 依賴

`polars`、`pyarrow`、`pydantic`、`pyyaml`、`openpyxl`。
**不需** torch／ultralytics／opencv／numpy（除非搬移後 `ruff`/測試顯示 `stats.py` 直接
用到 numpy，屆時再加）。
dev group 需 `pytest`（本包是三包中唯一有測試的）。
沿用 `[tool.pytest.ini_options] testpaths=["tests"]` 與 ruff 設定（`line-length=88`、
`select=["E","F","I","W"]`、`target-version="py312"`）。

## Acceptance Criteria

- [ ] 資料輸入/輸出或 API 規格定義清楚：進入點 `export_report_daily(date, bucket_dir,
      period_minutes, metric, on_duplicate_date, bucket_minutes) -> Path` 簽章不變；讀
      `zone_counts.parquet` + `camera_registry_used.yaml`、Append 至
      `outputs/{bucket}/report.xlsx`（三分頁：每小時人流／每日尖峰／活動事件）行為不變。
- [ ] 測試方式與驗收情境明確：以父計畫任務 0 golden（fixture 為 `bucket_name1`）的
      `zone_counts.parquet` + `camera_registry_used.yaml` 為輸入跑 report，輸出
      `outputs/bucket_name1/report.xlsx` 與 golden **逐值一致**；既有 `tests/` 全數通過
      （`uv run pytest`，不得有 skip）。
- [ ] 觀測指標明確：`N/A`——結構重整，不新增觀測指標。
- [ ] 影響範圍已列出：僅新增 `flow-report/`；舊 `src/` 不動（任務 4 才移除），與任務
      1／2 無檔案重疊。
- [ ] `flow-report/` 可獨立 `uv sync`，且**不含** torch／ultralytics／opencv。
- [ ] 完整 registry 能載入 golden 的 `camera_registry_used.yaml` 快照而不報錯；
      `on_duplicate_date` 三種模式（`overwrite`／`append`／`error`）行為不變。
- [ ] `uv run ruff check .` 乾淨。

## Risk

- **與 `zone-mapping` 的 registry.py 漂移**：兩份相同實作，未來改動需同步；由父計畫的
  「三包載入同一份 yaml」驗收把關。
- **`zone.bucket_minutes` 被漏掉或改語意**：它是 `period_minutes % bucket_minutes` 驗證的
  依據，漏掉會讓不合法的 `period_minutes` 靜默通過，產出錯誤彙總。
- **改讀「當下 registry」而非快照**：搬移時若順手把 `load_registry_from_path(快照)` 改成
  `load_registry(bucket_dir)`，會導致不同攝影機的人流被靜默合併。明確禁止。
- **`report.xlsx` 逐值比對**：openpyxl 版本差異可能造成非邏輯性 diff（樣式／中繼資料）。
  需在三包 pin 一致的 openpyxl 版本；比對以**儲存格值**為準，必要時忽略樣式。
- **Append 語意**：golden 比對前需確保起始 `report.xlsx` 狀態一致（`on_duplicate_date`
  預設 `append` 會直接加到尾端不檢查），避免重跑造成假性 diff。
- **`metric='unique_visitors'` 為近似值**：既有已知限制（跨 bucket 停留會被重複計入），
  本次不修，僅原樣搬移。
- **權限／安全／成本／模型準確率**：`N/A`。

## Related Links

- 父計畫：[拆成三個獨立套件（總體計畫）](2026-07-15-split-three-packages.md)
- 姊妹任務（registry.py 相同）：[任務 2：zone-mapping](2026-07-15-split-zone-mapping.md)
- 前一輪設計：[zone-report design](../specs/2026-07-06-zone-report-design.md)
