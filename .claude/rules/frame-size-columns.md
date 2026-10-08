---
paths:
  - "video_analyze/**"
  - "zone_mapping/**"
  - "line_counting/**"
---
# `tracking_results.parquet` 的影像尺寸欄位（跨套件硬性契約）

`tracking_results.parquet` 帶 `frame_width`／`frame_height`（issue #63）：值來自
`video_analyze` 以 `probe_frame_shape` 探測首格得到的 `frame_shapes`，逐列重複寫入。
**這兩欄不是給 `video_analyze` 自己用的**——`line_counting` 與 `zone_mapping` 都是純 CPU
套件、部署時不掛載影片，只能從這裡取得尺寸，把設定檔的 1080p 基準像素
（`crossing_band_px_1080p`／`boundary_band_px_1080p`）換算成各攝影機的實際像素。因此：

- 缺這兩欄的舊 parquet 會被 `line_counting` 與 `zone_mapping` 兩包都 **fail loud 擋下**
  （不給 fallback：「找不到尺寸就當 1080p」會讓 4K 攝影機靜默套用只有一半寬的判定區域，
  正是要消除的錯誤本身）。ADR-004 當時記的「`zone_mapping` 照跑」不對稱，在 `zone_mapping`
  也改吃 1080p 基準參數後（issue #68）**已不再成立**，見 ADR-006。
- `video_analyze` 日後改 `TRACKING_RESULTS_SCHEMA` 要一併考慮 `line_counting` 與
  `zone_mapping` 會不會直接崩；換算與檢查的位置、以及為何不改用「人形肩寬百分比」當尺規，
  見 ADR-004。
- 影格縮放移到讀取端之後（issue #108），`frame_shapes` 不再用來配置環形緩衝（緩衝改照
  推論尺寸 640×384 配置），但**仍必須是原始解析度**：除了寫進這兩欄，它也是把框與落腳點
  映射回原始解析度的參數來源。傳成推論尺寸會讓兩者一起靜默出錯（座標停在推論尺度、
  `frame_width` 寫成 640），故 `run_track_worker` 直接擋下這個值——追蹤與落盤移出推論
  進程後（issue #109），這兩個消費端都在追蹤進程，該檢查也跟著搬過去。
