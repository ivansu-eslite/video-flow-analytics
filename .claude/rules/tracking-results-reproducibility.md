---
paths:
  - "video_analyze/**"
  - "zone_mapping/**"
  - "line_counting/**"
---
# `tracking_results.parquet` 的重現性與正確性判準（非拆分相關，屬既有特性）

**偵測結果本身可重現，`track_id` 與列順序不可重現。** 以 `(camera_id, timestamp)` 對齊
同一格畫面之後，該格內的框座標集合逐值相同——2026-08-28 實測（issue #130 第二階段驗收，
九路全開）同一份程式碼自身重跑、以及只改推論批次大小（16→8）兩組，偏差 p50／p99／p99.9／
最大值全部是 0.0、配對率 100%、逐格偵測數 0 差；`foot_x`／`foot_y` 也一併驗過，
146,124 列全部逐值相同。但 **ByteTrack 的 `track_id` 指派在重跑間會改變**（同一組 1018
個 id 只有 730 個在兩次跑批中重複出現），多進程落盤的列順序也不固定，所以檔案不會
byte 級相同。

- **`(camera_id, timestamp)` 是「同一格畫面」的識別，不是唯一鍵**（一格內每個目標一列）。
  比對要分兩步：先用它把兩份的同一格取出來，再在該格內以框的距離做配對。直接拿它當
  SQL join key 會笛卡兒展開，算出假的大幅偏差——這正是下一句要避開 `frame_id` 的同一種
  錯誤。
- **不可用 `frame_id`**（片段內幀序、跨片段重複，同一個值會對到不同片段的畫面），
  **也不可拿 `track_id` 當 key**（重跑就變）。
- **`track_id` 的唯一範圍只到「同一路之內」**（issue #140 之後）：分片讓各追蹤進程各自
  持有 ByteTrack 的計數器，同一個 id 值會出現在分屬不同片的兩台攝影機。zone／line 都先
  依 `camera_id` 過濾再分窗，所以兩包不受影響——但任何 `group_by("track_id")` 沒帶
  `camera_id` 都會把兩個人併成一個，而輸出檔完全正常。
- 逐 byte／逐列比對對這份檔案沒有意義，但那不代表它不可重現——是 key 的選法問題。
- `zone_counts.parquet`／`line_counts.parquet` 經 `time_bucket` 聚合後**比逐格穩定得多**，
  是**交付期／大重構做 golden 回歸比對**時更省事的標的（vfa 日常改動的把關是各包
  pytest、不依賴 golden；golden 產在交付期、存放於 argus GCS）。⚠ 但**不是 byte 級一致
  的保證**：`unique_visitors` 是 `n_unique(track_id)`，entries、zone 停留人次
  （`dwell_events`，見 [ADR-016](../../docs/adr/zone_mapping/016-zone-dwell-threshold.md)）與計數線
  進出都以 `.over("track_id")` 分窗，仍吃 track 的分群結果，
  [ADR-006](../../docs/adr/zone_mapping/006-zone-boundary-band.md)
  就記過同設定重跑訪客數 55→57 的案例。`dwell_events` 對 track 分群比另外兩者更敏感：
  它量的是同一個 `track_id` 連續在區內多久，一次斷軌就把一段達標的停留切成兩段不達標的。
  2026-08-28 那批的下游輸出確實 byte 級相同，但那是該批的實測值，不能當成通則。

**列順序不在契約內，跑到一半時輸出目錄會多一個 `tracking_results.parts/`。** 追蹤進程
依攝影機分片之後（`[tracker].shards`，預設 2，見 ADR-012），各片先寫自己的
`tracking_results.parts/shard<k>.parquet`，主進程在全部到齊後合併成正式檔名；正常跑完
那個目錄就不存在。因此：

- **列順序是「逐片相接、片內才交錯」**，改分片數就會變。下游 zone／line 都走 `group_by`
  向量化、不依賴列順序，比對兩份輸出也一律先用 `(camera_id, timestamp)` 對齊同一格
  （見上一段），所以這不影響任何判準——但拿列順序當穩定性訊號會誤判。
- **那一天的鎖在 `tracking_results.parts/.lock`**，由**主進程**持有、子進程靠 `fork`
  繼承（改成 spawn 會靜默失去「孤兒進程仍守著鎖」那道保護）。同一個 bucket 的同一天
  被兩個執行同時跑會在認領時擋下。
- 中途崩掉留下的 parts 目錄由**下一次執行認領時清掉**，不必人工處理；改版前的
  `tracking_results.parquet.tmp` 殘檔則不再有人清，認領時只記一行 warning。

**改動會改變送進模型的畫面時（換解碼器、改縮放或色彩轉換路徑），判準用「相當於多大的
輸入擾動」，不要用「不大於控制組自身重跑」。** 後者曾經是本檔寫的判準，但既然自身重跑
的偏差是 0，那條件等於要求逐位元相同，而這類改動不可能逐位元相同——issue #130 第二階段
就是照它誤判成「沒通過」的。改用同型的輸入擾動當參照組：在解出的畫面上加 ±k 灰階的均勻
整數抖動（其餘完全不動）跑幾組，看待測改動的偏差落在哪個 k 上。第二階段（NVDEC 硬解）
量出來相當於 ±2 灰階抖動，±5 與 ±10 明顯拉開，據此判定它是偵測對像素微擾的固有敏感度、
不是解碼錯誤。**逐格指標要與下游聚合層一起看**：逐格偵測數不同的比例可以到 23%，而總
偵測數只差 0.03%、下游 30 分鐘 bucket 的計數逐列只差 ±1。作法與完整數字見
`outputs/vfa_perf/docs/report.md` 5.2「第二階段」（該檔不進版控）。
