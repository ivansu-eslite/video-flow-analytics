---
paths:
  - "zone_mapping/**"
  - "line_counting/**"
  - "flow_report/**"
  - "libs/vfa_registry/**"
---
# zone／line 名稱全域唯一

`zone_mapping` 與 `flow_report` 的報表都以 zone 名稱（不含 `camera_id`）分組彙總，因此
`camera_registry.yaml` 的 zone 名稱**跨攝影機也不可重複**（非僅同一攝影機內）。此驗證
的實作是共用 lib `vfa_registry` 的 `parse_and_validate_zones`——`zone_mapping` 與
`flow_report` 都會呼叫（`video_analyze` 不呼叫），**即使當天不會產生報表，`zone_mapping`
本身也會擋下跨攝影機重複的 zone 命名**。`flow_report` 驗證的對象是 `bucket_dir` 下當下的
`camera_registry.yaml`（ADR-007 之前是產生該日 parquet 時的快照）——產生 parquet 之後
改過 zone 名稱的話，`_reject_unknown_pairs` 的 (camera, zone) 組合驗證會擋下；只改幾何
座標則不會有訊號，這是移除快照時接受的代價。

`line_counting` 的計數線名稱有**同樣**的約束：下游同樣以 line 名稱（不含 `camera_id`）
分組彙總，故 line 名稱跨攝影機也不可重複，由 `vfa_registry` 的 `parse_and_validate_lines`
擋下——**即使當天不會產生報表，`line_counting` 本身也會擋下跨攝影機重複的 line 命名**。
`flow_report` 於 issue #69 串接 `line_counts.parquet` 後，也對同一份 registry 呼叫
`parse_and_validate_lines`，與 zone 那側是同型驗證。

`Line` 另帶一個 `line_group` 欄位（issue #59），標示一條計數線屬於哪個範圍（例如同一
賣場的數個出入口）。**`line_group` 是與上述規則刻意相反的例外**：跨攝影機同名不但不
擋，還正是分組的用途——一個範圍的出入口本來就可能分屬不同攝影機。`line` 名稱本身仍全域
唯一，故 `(line_group, line)` 組合天然唯一。取捨與「為何不能順手補上同型驗證」見
[ADR-002](../../docs/adr/line_counting/002-line-group-semantics.md)。
