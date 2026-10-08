---
paths:
  - "video_analyze/**"
  - "zone_mapping/**"
  - "line_counting/**"
---
# 落腳點是資料欄位，不是各包各自算的公式（跨套件硬性契約）

`tracking_results.parquet` 帶 `foot_x`／`foot_y`（issue #72）：人站在地面的位置，由
`video_analyze` 用 head 框對 fbody 框中心做點反射推算（`foot = 2 × C_fbody − H`，
`H` 為 head 框頂邊中點），推算不出來才退回舊定義 `((x1+x2)/2, y2)`。改動前這條公式由各
消費端從 bbox 現算，散在 `line_map.py`／`zone_map.py` 與 overlay 五個模組共十餘處。

- **只能在上游算**：推算需要 head 框，而 head **不進 tracker**——送進去的話同一個人會多
  出一條頭部軌跡，`track_id` 的語義從「一個人」變成「一個偵測目標」，下游的不重複訪客與
  進出人數直接翻倍，而輸出檔本身完全正常。偵測 `classes` 因此是 `[0, 2]`，但
  `services/track_worker.py` 只把 fbody 子集餵給 ByteTracker（issue #109 之前在
  `services/inference.py`）。
- 缺這兩欄的舊 parquet 被 `line_counting` 與 `zone_mapping` 兩包 fail loud 擋下，與影像
  尺寸欄位同一道檢查。`[foot_point].method = "bbox_bottom"` 可切回舊定義；此時
  `classes` 不必含 head，但 `method = "head"` 卻少了 head 會直接拋錯（否則每列都退回
  框底邊中點，改動靜默失效）。
- 配對條件、多候選 head 的選法（實測推翻了規劃階段的直覺判準）與被否決的替代方案
  （OBB／pose／ground plane）見 [ADR-009](../../docs/adr/shared/009-head-based-foot-point.md)。
