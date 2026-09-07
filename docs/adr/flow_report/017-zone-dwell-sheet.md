# ADR-017: 報表的停留人次分頁

## Status

Accepted

## Context

`zone_mapping` 在 `zone_counts.parquet` 新增了第三個指標 `dwell_events`（該時段內有幾
人次在該區域連續停留達門檻秒數，見 [ADR-016](../zone_mapping/016-zone-dwell-threshold.md)）。
那個改動到此為止：報表上看不到它。

`flow_report` 現況有兩個原因讀不到這一欄：

- `rollup_by_period(df, period_minutes, metric)` 只取 `metric` 指定的那一欄，而
  `[report].metric` 的型別是 `Literal["entries", "unique_visitors"]`，**只能選一個**。
  新欄位對現行程式是相容的（照欄名取值、不驗 schema），但不動任何東西它永遠不會被讀到。
- 就算放寬那個 `Literal`，也只能「用停留取代人流量」，兩者不能並陳。`zone_mapping` 一次
  跑就把三個指標都算出來了，報表卻只能挑一個看。

同時，升級後會出現一種新的輸入：**舊版 `zone_mapping` 產生的 `zone_counts.parquet` 沒有
`dwell_events` 欄**。這不是假想題——手上的驗收素材就有一份 2026-08-13 產的舊檔。這種檔
怎麼處理，決定了報表能不能被信任。

## Options Considered

### 呈現方式

#### Option A：放寬 `[report].metric` 的 `Literal`，讓它可以選 `dwell_events`

Description：`metric` 多一個可選值，「各區域人流」頁的「人流量」欄改成停留人次。

- Advantages：程式改動最小，一個型別標註加三處文件。
- Disadvantages：三種口徑（進入次數、不重複人數、停留人次）共用同一個中文表頭「人流量」，
  換設定就換口徑、報表本身沒有任何訊號。這正是這條 repo 一直在擋的靜默換口徑。而且三個
  指標是同一次跑就都算出來的，只能挑一個看沒有道理。

#### Option B：在「各區域人流」頁多一欄「停留人次」

Description：`ZONE_HOURLY_COLUMNS` 加第六個序對，同一頁並陳兩個指標。

- Advantages：不新增分頁，BI 端接的分頁數不變；同一列就能對照兩個口徑。
- Disadvantages：**在升級既有 `report.xlsx` 這條路上會靜默壞掉**。`_write_report` 只在
  分頁**不存在**時才呼叫 `_init_sheet` 寫表頭（既有檔的舊分頁不做遷移，見 `flow_report`
  README），於是 6 欄的資料列會被 append 到 5 欄的表頭底下，多出來的那一格沒有表頭、
  而且不會有任何錯誤——正是這條 repo 要擋的那種無聲資料錯位。`_append_rows` 只在資料側
  **缺**欄時由 polars 拋錯，多一欄不擋。

#### Option C：新增「各區域停留人次」分頁

Description：加一個與「各區域人流」對稱的分頁，逐時段逐區域寫 `dwell_events` 的期間彙總。

- Advantages：走的是既有已驗證的升級路徑——`_SHEET_LAYOUT` 缺哪個分頁就補建哪個，出入口
  三頁當初就是這樣加進來的，`test_write_report_adds_missing_sheets_to_legacy_workbook`
  已釘住這條路徑。既有五頁的欄位、表頭、列內容完全不動，BI 端不必重接。三個指標並存，
  不必挑一個。
- Disadvantages：BI 端要多接一個分頁；升級之前寫進報表的日期在新分頁上是空的。

### 舊檔（缺 `dwell_events` 欄）怎麼辦

#### Option D：缺欄就跳過停留分頁，其餘照產

- Advantages：補跑歷史日期不會被擋，操作上最省事。
- Disadvantages：跨日累加的報表會出現「有些日期有停留、有些沒有」，而看報表的人**無從
  分辨是「那天沒人停留」還是「那天沒算」**——0 與缺資料在 Excel 裡長得一樣。

#### Option E：缺欄即 fail loud，訊息指出重跑 `zone_mapping`

- Advantages：不會產出口徑不明的報表；`zone_mapping` 是純 CPU 階段，重跑的代價只是幾秒
  的向量化運算，不必重跑偵測。
- Disadvantages：中止的不只停留那一頁，出入口三頁也一起沒有（與 ADR-005 的「缺
  `line_counts.parquet` 即整份中止」同型）。補跑歷史日期的人一定會撞到這件事。

## Decision

**Option C ＋ Option E。**

1. 新增分頁「各區域停留人次」，欄位為日期／星期／小時／區域／停留人次，排在區域兩頁之後、
   出入口三頁之前（該順序只影響全新建立的檔案；既有檔的新分頁一律接在最後）。
2. 資料側直接複用 `rollup_by_period(df, period_minutes, "dwell_events")`，不新增彙總
   函式、不改函式簽名。`dwell_events` 與 `entries` 同型（事件型、上游只在事件發生的那個
   bucket 記一次），跨 bucket `sum()` 不像 `unique_visitors` 有重複計入的近似問題。
3. 三個 zone frame 共用同一次 `read_parquet`／`to_taipei`／`_reject_unknown_pairs`：
   新分頁的 (camera_id, zone) 維度與既有兩頁完全相同，沒有第二次驗證的理由。
4. **不放寬 `[report].metric`**。停留有自己的分頁，`metric` 仍只作用於「人流量」與
   「尖峰人流」兩頁；新分頁固定用 `dwell_events`。
5. 缺 `dwell_events` 欄時在 `_zone_frames` 讀完 parquet、`_reject_unknown_pairs` 之前
   拋 `ValueError`（schema 不對就沒必要再驗內容）。**不靠 polars 自己拋
   `ColumnNotFoundError`**——那個訊息只說找不到欄位，讀的人無從分辨是「上游是舊版」還是
   「報表程式寫錯欄名」，更不會知道重跑 `zone_mapping` 就能解決。訊息比照既有的缺檔
   訊息，帶檔案路徑、指出該檔由舊版 `zone_mapping` 產生、指出重跑該階段即可（純 CPU）。
   只檢查欄位存在、不檢查型別：既有兩欄也不驗型別，加一道不對稱的檢查沒有理由。
6. **不加「各區域每日停留尖峰」分頁。** 停留的母體小到每個 zone 每個時段常是個位數
   （規劃期在驗收素材上量到 8010 條軌跡只有 910 條活過 20 秒，那還是「在整個畫面裡」），
   在個位數上取「尖峰時段」解讀價值很低。`peak_per_day` 已經參數化，真要對稱補上的成本
   是常數一組、`_DATA_SHEETS` 一列、一行呼叫。
7. **報表不記錄停留門檻秒數。** `zone_counts.parquet` 沒有記門檻（ADR-016 已否決寫進欄位
   名與 parquet metadata，理由與 [ADR-008](../shared/008-config-section-namespace.md)
   同型），`flow_report` 唯一能做的是自己開一個純標籤用的設定——那會製造「報表標 20 秒、
   上游其實跑 30 秒」這種沒有訊號的假標籤，比不標更糟。改成寫進 README 的已知限制。

表頭用「停留人次」而非「停留人數」是刻意的：同一人在同一時段內兩段都達標會計 2。

## Consequences

- **「停留人次」不是人數，也不是「人流量」的子集。** 同一個 zone 同一個時段，停留人次
  可能大於 `metric = "unique_visitors"` 的人流量。任何下游（含 BI 端）都不可以寫
  `停留人次 <= 人流量` 這種 sanity check。
- **`flow_report` 從此綁死 `zone_mapping` 的版本**：舊版產的 `zone_counts.parquet` 會讓
  整份報表產不出來，含出入口三頁。這是 ADR-005 那句話的第二個版本，差別只在這次判的是
  欄位而不是檔案。解法是用新版 `zone_mapping` 重跑該日（純 CPU）。根 `CLAUDE.md` 的
  跨套件硬性契約已記入這條。
- **升級前寫進報表的日期，在新分頁上是空的。** `report.xlsx` 是跨日累加的，本階段不回填
  歷史日期；BI 端若把空值當 0 解讀，會看成「那幾天沒人停留」。這是加欄位到累加報表的
  必然結果，只能寫進已知限制。要補回來只能把那幾天逐日重跑，且 `on_duplicate_date` 設成
  `overwrite`。
- **跨日混口徑不會有訊號。** 中途改 `dwell_threshold_seconds` 又沒重跑歷史日期的話，
  同一欄底下會混兩種定義的數字——與 [ADR-006](../zone_mapping/006-zone-boundary-band.md)
  記的「跨日報表混口徑」同型，只是這次混的是指標定義本身。
- **兩個 zone 逐時段分頁的資料欄名都是 `value`**（`rollup_by_period` 的輸出），前四欄的
  表頭也完全相同。兩者是不同的 DataFrame，欄名相同不會互相污染，但這是本次最容易靜默
  錯位的地方，測試裡兩頁刻意給不同的值。
- `on_duplicate_date` 的三種模式、`_frame_dates` 的日期聯集、結束時的 `rows_by_sheet`
  日誌都是遍歷 `_DATA_SHEETS` 寫的，新分頁加進那個 tuple 就一併生效。

## Related Links

- [ADR-016](../zone_mapping/016-zone-dwell-threshold.md)：`dwell_events` 的語義正本
  （判定基礎、兩個參數、容忍窗的上界）
- [ADR-005](005-report-input-requirement-from-snapshot.md)：哪些輸入是必要的由 registry
  決定；本次的缺欄 fail loud 與它同型
- [ADR-006](../zone_mapping/006-zone-boundary-band.md)：三個指標為何用不同判定；
  「跨日報表混口徑」的先例
- [ADR-008](../shared/008-config-section-namespace.md)：把參數寫進 parquet metadata
  這條路已被否決（本次「不記門檻」沿用同一個結論）
