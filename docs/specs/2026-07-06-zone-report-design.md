# Zone 人流 Excel 報表 — 設計文件

日期：2026-07-06

## 背景與目標

`zone_mapping` 階段已將每日每攝影機／每 zone 的人流統計輸出成
`outputs/{bucket}/{date}/zone_counts.parquet`。這次要新增第三個下游階段，把
`zone_counts.parquet` 彙總成一份格式參考 `Sample Report.xlsx` 的 Excel 報表，
供現場營運人員直接檢視／後續人工補充活動事件。

## 範圍

- 每次呼叫只處理**單一天**（一個 `zone_counts.parquet`），輸出寫入
  **跨日累加更新**的單一報表檔案 `outputs/{bucket_name}/report.xlsx`。
- 報表分三個分頁，改用中文命名（Sample 原本用英文，使用者同意重新命名）：
  1. **每小時人流**：日期／星期／小時／區域／人流量
  2. **每日尖峰**：日期／星期／區域／尖峰時段／尖峰人流／每日提醒
  3. **活動事件**：日期／星期／開始時間／結束時間／區域／活動名稱／活動類型
     —— 只建標題列、不寫資料，之後由其他來源填入，本次不處理。
- 若 `report.xlsx` 已存在：只重建「每小時人流」「每日尖峰」兩個分頁的資料列，
  「活動事件」分頁完全不動。

## 架構

新增子套件 `src/video_flow_analytics/report/`，比照 `zone_mapping/` 的
stats／pipeline 分工，只依賴 `core`：

- **`report/stats.py`**（純運算，可單元測試）
  - `to_taipei`：UTC → 台北時區（+8）轉換
  - `rollup_by_period`：依 `period_minutes` 對 `time_bucket` 做期間彙總
    - `entries`：可安全跨 sub-bucket 加總（累計轉換次數，無重複計算問題）
    - `unique_visitors`：加總，接受跨 sub-bucket 可能重複計算 track_id 的近似值
      （真正去重需要原始 track 明細，非 `zone_counts.parquet` 已聚合過的資料所能
      提供，超出本次範圍）
  - `peak_per_day`：每日每區域找出彙總後人流量最高的期間；若有並列最高值，
    取時間較早的期間（確定性 tie-break，不隨機）
  - `meal_time_reminder(hour: int) -> str`：用餐時段提醒規則
    - 尖峰期間起始小時 ∈ [11, 14) → `"加強午餐動線"`
    - 尖峰期間起始小時 ∈ [17, 20) → `"加強晚餐動線"`
    - 其餘 → `"無"`
  - `weekday_zh(date) -> str`：星期一～星期日

- **`report/pipeline.py`**（I/O + orchestration）
  - 讀 `outputs/{bucket}/{date}/zone_counts.parquet`；不存在則 fail-loud
    （比照 `map_zones_daily` 的錯誤訊息風格）
  - 讀 `camera_registry.yaml`，檢查所有攝影機的 zone 名稱**全域唯一**
    （見下方「新增前提」）；違反時 fail-loud 中止
  - 呼叫 `stats.py` 算出本次的「每小時人流」「每日尖峰」資料
  - 讀取既有 `outputs/{bucket}/report.xlsx`（若存在），依 `on_duplicate_date`
    處理後寫回；不存在則建立新檔案（含三個分頁的標題列）
  - 寫檔採 `.tmp` + atomic rename，比照 `tracking_results.parquet` /
    `zone_counts.parquet` 的既有慣例
  - 對外進入點：
    ```python
    def export_report_daily(
        date: datetime.date,
        bucket_dir: str,
        period_minutes: int,
        metric: Literal["entries", "unique_visitors"],
        on_duplicate_date: Literal["overwrite", "append", "error"],
        bucket_minutes: int,
        output_root: Path = OUTPUT_ROOT,
    ) -> Path
    ```
  - `run_report()`：從 `config.toml` 的 `[report]` 與 `[zone]`（`bucket_minutes`）
    取參數後呼叫，供 CLI 使用

- **`cli.py`**：新增 `report` 子命令，lazy import
  `video_flow_analytics.report.pipeline.run_report`，比照 `analyze` / `zone-map`。

- **依賴**：`pyproject.toml` 新增 `openpyxl`（讀寫 xlsx）；
  `dependency-groups.dev` 新增 `pytest`。

## 新增前提：zone 名稱全域唯一

目前 `camera_registry.yaml` 只驗證「同一攝影機內」zone name 不可重複
（`CameraEntry.parsed_zones()`）。跨攝影機可以同名（例如 `cam001` 與 `cam004`
都有 `entrance`）。

報表的「區域」欄位以 zone 名稱為唯一鍵、不含 camera_id，因此需要新的前提：
**整份 `camera_registry.yaml` 裡，所有攝影機的 zone 名稱都必須全域唯一**。

- 此驗證只加在 `report/pipeline.py`（產報表時才檢查），不動
  `CameraRegistry` / `CameraEntry`，避免拖累 `analyze_daily` 或
  `zone_mapping` 既有路徑。
- 目前 `bucket_name1/camera_registry.yaml` 有違反此前提的同名 zone
  （`entrance`、`checkout` 重複於 cam001／cam004），使用者會自行修改該檔案，
  本次實作不處理既有資料的相容性。
- 未來若有 UI 維護 `camera_registry.yaml`，會在該處擋掉重複命名；目前先靠
  `report/pipeline.py` 的 fail-loud 檢查。
- 此規則會補充進 `CLAUDE.md` 的 Zone Mapping 章節，作為長期文件。

## Config 設定

`config.toml` 新增 `[report]`：

```toml
[report]
period_minutes = 60              # 報表人流彙總的時段粒度（分鐘），需為 zone.bucket_minutes 的倍數
metric = "entries"                # "entries" 或 "unique_visitors"，決定人流量/尖峰人流用哪個統計量
on_duplicate_date = "overwrite"   # 同一天資料已存在時："overwrite"（預設）/ "append" / "error"
```

`core/config.py` 新增 `ReportConfig`（Pydantic model）與 `AppConfig.report` 欄位。

`period_minutes % zone.bucket_minutes != 0` 時，`export_report_daily` 在讀資料前
就 fail-loud 報錯（`bucket_minutes` 由呼叫端傳入，比照 `map_zones_daily` 的
`bucket_minutes` 參數模式，不在函式內部讀全域 `settings`）。

## Excel 輸出格式細節

- 日期／星期／小時等欄位維持**字串**型別（與 Sample 一致：`"2026-05-01"`、
  `"星期五"`、`"11:00"`），人流量／尖峰人流維持數值型別。
- 標題列套用粗體樣式，欄寬依內容概略設定；不建立 openpyxl 的 Table 物件
  （Sample 本身也只是純資料格線，無 Excel Table）。
- 時間顯示一律轉換成台北時區（+8），比照 `CLAUDE.md` 既有慣例「本地時段報表
  需自行轉時區」。

### 既有檔案合併（`on_duplicate_date`）

以本次算出的（轉時區後的）本地日期集合，比對「每小時人流」「每日尖峰」兩個
分頁既有資料列的日期欄位：

- **`overwrite`**（預設）：刪除兩個分頁中日期屬於本次日期集合的既有列，插入
  本次新算出的列，並依（日期／小時／區域）（每日尖峰依日期／區域）重新排序。
- **`append`**：直接把本次新列加到分頁尾端，不檢查、不刪除既有列（重跑同一
  天會產生重複資料，需使用者自行注意）。
- **`error`**：若本次日期集合與既有資料有重疊，整個中止、不寫入任何內容
  （檢查完兩個分頁都無重疊才動筆，避免只改一半）。

「活動事件」分頁在以上三種模式下都完全不動；檔案不存在時只建立標題列。

## 已知限制

`zone_counts.parquet` 以 UTC 曆日切分（`outputs/{bucket}/{date}/...`），但報表
顯示的是台北時區（+8）的本地小時／日期。單次執行涵蓋的 UTC 一天資料，轉換後
會落在「本地當天 08:00–23:59」與「本地隔天 00:00–07:59」兩個本地曆日。

- 若現場凌晨無營業（預期無偵測資料），此邊界不影響報表正確性。
- 若有 24 小時營運資料，本地隔天凌晨時段的資料只會出現在**當次**執行的輸出
  中，不會等到隔天執行時自動合併成完整一天；這是 UTC-by-day 儲存架構下的既有
  限制，這次不特別處理，未來若需要可再設計合併邏輯。

## 測試

- 新增 `pytest`，針對 `report/stats.py` 的純函式寫單元測試：
  - `to_taipei` 時區轉換
  - `rollup_by_period`（含 `entries` / `unique_visitors` 兩種 metric）
  - `peak_per_day`
  - `meal_time_reminder`（含邊界值：10:59 / 11:00 / 13:59 / 14:00 / 16:59 /
    17:00 / 19:59 / 20:00）
  - `weekday_zh`
- `report/pipeline.py`（讀寫檔案／Excel／CLI 串接）不寫測試，符合專案目前
  「沒有測試框架、按需新增」的現況。

## 不在本次範圍

- 「活動事件」分頁的資料寫入（由其他來源／流程負責）。
- 跨執行合併本地曆日邊界資料的完整性保證。
- `camera_registry.yaml` 的 UI 維護與重複 zone 名稱的即時擋錯（留給未來 UI）。
