---
paths:
  - "flow_report/**"
  - "zone_mapping/**"
  - "line_counting/**"
---
# `flow_report` 的輸入必要性看 registry，不看檔案（跨套件契約）

`flow_report` 有兩個上游輸入，**該有哪幾份由 `bucket_dir/camera_registry.yaml` 的定義
決定**：registry 裡有攝影機定義了非空 `lines` 就必須有 `line_counts.parquet`，缺檔即
fail loud；沒有任何 `lines` 定義則整批跳過出入口三個分頁、不算錯誤（zone 那側同理，判準是
`participates_in_zone_mapping` 且 `zones` 非空）。這是**這條 repo 唯一不照「下游看上游輸出
檔是否存在」原則的階段**，根 README 的階段相依原則那段已寫明這個例外；`zone_mapping`／
`line_counting` 只有單一上游、沒有這個歧義，維持看檔案。取捨與「為何刻意不留跳過用的
旗標」見 [ADR-005](../../docs/adr/flow_report/005-report-input-requirement-from-snapshot.md)；資料來源
為何由快照改成當下的檔案見 [ADR-007](../../docs/adr/shared/007-remove-registry-snapshot.md)。

後果是 `flow_report` 從此被 `line_counting` 綁住：registry 只要有任一 `lines`，沒跑
`line_counting` 就連 zone 三頁都產不出來（整個 `export_report_daily` 中止）。排程上
`zone_mapping` 與 `line_counting` 都要排在 `flow_report` 之前。

**同一個判準也作用在欄位上（跨套件硬性契約）**：registry 有區域定義時，
`zone_counts.parquet` 必須帶 `dwell_events` 欄，缺這一欄 `flow_report` 整份報表中止
（含出入口三頁），不是只跳過「各區域停留人次」那一頁。靜默跳過的話，跨日累加的報表會
出現「有些日期有停留、有些沒有」，而看報表的人無從分辨是「那天沒人停留」還是「那天沒
算」。解法是用新版 `zone_mapping` 重跑該日（純 CPU，不必重跑偵測）。**加欄的
`zone_mapping`（issue #163）沒有造成這個耦合**——加完 `flow_report` 照樣跑得動；把它變成
硬性要求的是報表側接上這一欄的改動（issue #166），見
[ADR-017](../../docs/adr/flow_report/017-zone-dwell-sheet.md)。停留人次是**人次不是人數**、也
不是「人流量」的子集，下游不可寫 `dwell_events <= unique_visitors` 這種 sanity check。
