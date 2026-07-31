# ADR-005: 報表的輸入必要性由 registry 快照決定

## Status

Proposed

## Context

`flow_report` 從只讀 `zone_counts.parquet` 變成同時讀 `zone_counts.parquet` 與
`line_counts.parquet`（issue #69）。有了第二個上游輸入之後，「當日缺某一份 parquet 時
該怎麼辦」才第一次成為需要決定的問題——只有一個輸入時，缺檔就是缺檔，直接報錯即可。

兩種情況會產生一模一樣的檔案系統狀態（`line_counts.parquet` 不存在）：

1. 這個 bucket 的攝影機根本沒有定義計數線，`line_counting` 沒有東西可算。
2. 有定義計數線，但當天忘了跑 `line_counting`（或它失敗了）。

第 1 種要跳過三個出入口分頁、照樣產出區域兩頁；第 2 種是操作錯誤，產出的報表會少掉當天
全部的進出人數，而且沒有任何訊號。

根 `README.md` 目前明列的階段設計原則是「下游以『上游輸出檔是否存在』判定相依是否滿足」。
這條原則在單一輸入時足夠，在這裡不夠：它只看得到檔案，看不到「本來應該有幾份輸入」。

`outputs/{bucket}/{date}/camera_registry_used.yaml` 快照裡有這個資訊——它是產生當日資料
時的攝影機與幾何定義，`flow_report` 本來就要讀它來驗證 zone／line 名稱（見 `flow_report`
README「驗證對象是快照、不是當下的 `camera_registry.yaml`」）。

## Options Considered

### Option A：維持「檔案在不在」

Description：`line_counts.parquet` 存在就讀、不存在就跳過三個分頁。

- Advantages：與現行的階段相依原則一致，不必新增概念；兩包分開排程時互不影響。
- Disadvantages：上面第 2 種情況靜默少三頁，報表看起來完全正常。BI 端要自己發現「今天
  出入口沒有資料」，而這正是報表要回答的問題之一。

### Option B：由快照的定義決定（本次選擇）

Description：快照中有任一攝影機定義了非空 `lines` → 必須有 `line_counts.parquet`，缺檔
即 `FileNotFoundError` 並指出要先跑 `line_counting`；快照中沒有任何 `lines` → 跳過三個
出入口分頁，不算錯誤。zone 同理（判準是「有任一攝影機 `participates_in_zone_mapping`
且 `zones` 非空」）。兩邊都沒有定義 → `ValueError`。

- Advantages：兩種情況被區分開，操作錯誤會被擋下。判斷依據是產生當日資料時的定義，與
  下游其他驗證（zone／line 名稱唯一、`(camera_id, zone)` 組合須存在於快照）同一個來源。
  與 `line_counting` 現行的「定義了計數線的攝影機在當日追蹤明細中必須有資料」是同型判斷。
- Disadvantages：`flow_report` 從此被 `line_counting` 綁住——快照只要有任一 `lines`，
  沒跑 `line_counting` 就連區域兩頁都產不出來。對「兩包分開排程」或「`line_counting`
  尚未上線」的 bucket，這是新增的失敗模式。也讓 `flow_report` 與根 README 明列的階段
  相依原則產生局部例外。

### Option C：用 `config.toml` 的旗標開關

Description：加一個 `[report] include_lines = true/false`，由執行者宣告這次要不要出入口
三頁。

- Advantages：不必動相依原則，逃生口明確。
- Disadvantages：把「這個 bucket 有沒有計數線」這件事實變成人工維護的設定，而它已經寫在
  registry 裡了；兩者不同步時（新增了計數線但沒改旗標）又回到靜默少三頁。等於用一份會
  漂移的副本取代權威來源。

## Decision

採 Option B：**輸入的必要性由 `camera_registry_used.yaml` 快照的定義決定，不由檔案是否
存在決定**。

**不留旗標逃生口**（不接受「有 `lines` 但允許跳過」的設定）：留了就等於把 Option A 的
靜默少三頁重新放回來，只是多繞一層設定。要跳過的正當做法是把該攝影機的 `lines` 從
registry 拿掉——那本來就是「這個 bucket 沒有計數線」的正確表達方式。

不論該 bucket 有沒有計數線，6 個分頁的表頭一律建立，讓 BI 端接到的 schema 穩定。

本次只針對 `flow_report` 修正根 README 的階段相依原則，`zone_mapping`／`line_counting`
維持「檔案在不在」的原樣——它們的上游只有一個 `tracking_results.parquet`，沒有本 ADR
要處理的歧義。根 README 該段一併改寫成「原則上看檔案，`flow_report` 因有兩個上游輸入而
改看快照」。

## Consequences

Positive:

- 「定義了計數線卻沒跑 `line_counting`」會在報表產出時就被擋下，訊息直接指出要跑哪個
  階段，而不是等 BI 端發現數字不見。
- 沒有計數線的 bucket 不需要任何額外設定就能照常產報表；反之亦然（只有計數線、沒有區域
  的 bucket 也能跑）。
- 判斷與驗證用的是同一份快照，不會出現「用當下 registry 判斷、卻驗證舊資料」的錯位。

Negative:

- `flow_report` 對 `line_counting` 產生硬相依。快照只要有任一 `lines`，沒跑
  `line_counting` 就整個 `export_report_daily` 中止，區域兩頁也產不出來。排程上要把
  `zone_mapping`／`line_counting` 都排在 `flow_report` 之前。
- **快照被覆蓋的既有問題會傳導到這個判斷**：`zone_mapping` 與 `line_counting` 都把
  `camera_registry_used.yaml` 寫到同一路徑，後跑的覆蓋先跑的。正常流程下兩者是同一份
  `camera_registry.yaml` 的複製、內容相同；但若兩階段之間改過 registry，`flow_report`
  只看得到後寫的那份，判斷與驗證的對象會與其中一份 parquet 的實際定義不一致。這屬上游
  行為，本 ADR 不處理，僅記錄。
- 根 README 的階段相依原則從此有一個例外，讀者需要知道哪一段適用哪條規則。
