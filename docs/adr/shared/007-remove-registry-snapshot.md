# ADR-007: 移除 registry 快照，改以 bucket 當下的 camera_registry.yaml 為準

## Status

Accepted

修訂 [ADR-005](../flow_report/005-report-input-requirement-from-snapshot.md)：判準（輸入必要性由 registry
的定義決定，不由檔案是否存在決定）維持不變，**只換資料來源**——由
`outputs/{bucket}/{date}/camera_registry_used.yaml` 快照改為 `bucket_dir/camera_registry.yaml`。

## Context

`zone_mapping` 與 `line_counting` 各自在寫完當日 parquet 後，把當次套用的
`camera_registry.yaml` 複製一份到 `outputs/{bucket}/{date}/camera_registry_used.yaml`；
`flow_report` 讀這份快照，用它決定該有哪幾份上游輸入，並驗證 zone／line 名稱。設計動機是
「用產生該日資料當時的定義做驗證」，避免事後改過 registry 後，不同攝影機的人流被靜默合併。

部署端（argus）已把同一套機制拿掉：產生端不再複製，各階段改為以 `--registry-root` 指向
共用的 registry 目錄直接讀取，golden sample 的 `expected/` 只剩 parquet。兩邊因此反向
漂移——部署端移除快照的同時，本 repo 反而加深了對它的依賴（ADR-005 把它寫成 Accepted
決策），輸出目錄結構也不再一致。

一個必須先講清楚的差異：**部署端的 registry 帶 `date_YYYY-MM-DD/` 分層**，「當日凍結」
並沒有消失，只是責任從 pipeline 的複製動作移到存放結構本身。本 repo 的
`bucket_dir/camera_registry.yaml` 是單一檔案、不分層，因此移除快照等於真的失去回溯能力，
不是把責任換個地方放。

## Options Considered

### Option A：維持快照

輸出結構與部署端不一致的狀態延續下去。回溯能力保住，但代價是：這份複製沒有其他讀者
（只有 `flow_report` 讀）、兩包寫到同一路徑後跑者覆蓋先跑的（ADR-005 已記錄的既有問題），
且兩邊 repo 的差異會在每次同步交付副本時重新浮現。

### Option B：移除快照，並補回溯替代方案

例如把 registry 的 sha256 記進 log、或寫進 parquet 的 metadata。輸出結構對齊了，回溯也
還在。但這是為了一個沒有實際發生過的需求（至今沒有查過「當時的幾何是什麼」）新增機制，
而且 sha256 只能告訴你「不一樣」，不能告訴你「當時長什麼樣」——要能還原就得存下整份內容，
等於換個位置放快照。

### Option C：移除快照，接受回溯損失（採用）

`flow_report` 改讀 `bucket_dir` 當下的 `camera_registry.yaml`，當日輸出目錄只剩 parquet。

## Decision

採 Option C。

**判準不變、來源改變**：ADR-005 的結論（輸入必要性由 registry 的定義決定，不由檔案是否
存在決定）原樣保留，`_build_report_frames` 只是把 `load_registry_from_path(快照路徑)`
換成 `load_registry(bucket_path)`。`camera_registry.yaml` 缺檔時 fail loud（`load_registry`
既有的 `FileNotFoundError`），不提供任何 fallback。

**不為本 repo 的 registry 加日期分層**：那是部署端因應多租戶與雲端儲存演化出來的結構，
本 repo 是單機開發與驗證用途，加了只是把一個沒有需求的機制搬過來。

**不清理既有輸出目錄下已存在的快照檔**：不讀、不刪。留著的舊檔不影響彙總結果（測試已
釘住這點），使用者要不要刪是他的事。

## Consequences

Positive:

- 當日輸出目錄只剩 parquet，與部署端的輸出結構一致；交付副本同步時不再需要處理這處差異。
- ADR-005 記錄的「快照被上游覆蓋」問題直接消失——沒有快照就沒有覆蓋。
- `zone_mapping`／`line_counting` 各少一個 `shutil.copyfile` 與兩個 import；
  `flow_report` 少一個常數。
- 舊的 `outputs/{date}/` 目錄在新版下照樣能彙總（舊版缺快照會直接報錯），相容性方向是
  放寬的，不需要遷移步驟。

Negative:

- **「幾何座標被調整、名稱未變」會變成靜默的錯位**：報表數字仍是舊幾何算出來的，但用來
  驗證的是新的 registry，沒有任何訊號。這是本次接受的主要代價。zone／line **名稱**的錯位
  （改名或移除後拿舊 parquet 彙總）仍會 fail loud，由 `_reject_unknown_pairs` 與
  `parse_and_validate_zones`／`parse_and_validate_lines` 的全域唯一驗證擋下。
- **`line_group` 的改名同樣是靜默漂移**：`_reject_unknown_pairs` 比對的是
  `(camera_id, line)`，而報表「群組」欄的值取自 `line_counts.parquet`。registry 只改
  `line_group`、不改 `line` 名稱時，報表會沿用舊的群組名而沒有訊號。刻意不把
  `line_group` 納入組合驗證：它跨攝影機同名是正常用途（見 [ADR-002](../line_counting/002-line-group-semantics.md)），
  納入驗證等於要求「分組不得調整」，代價高於它能擋下的錯誤——`line_group` 是報表的維度
  欄位，錯了會讓分組看起來怪，不會讓進出人數算錯。
- **「整側定義被清空」需要一道額外的 fail loud**：`_reject_unknown_pairs` 靠的是「該側
  仍有定義、拿 parquet 的組合去比對」，整側清空時 `_build_report_frames` 直接跳過該側，
  那條路走不到，該類統計會無聲從報表消失（`on_duplicate_date = "overwrite"` 還會清掉
  既有的舊列）。因此改讀當下 registry 時一併補上 `_reject_orphan_counts`：registry 已無
  該側定義、當日 parquet 卻有資料就報錯。0 列的 parquet 不算——那是上游對「這個 bucket
  沒有該側定義」的正常產物。舊設計不需要這道檢查，快照與 parquet 是同時寫出的。
- **`flow_report` 從此需要讀得到 `bucket_dir`**：改動前它只用 `Path(bucket_dir).name` 組
  輸出路徑，完全不碰 bucket 目錄；現在必須讀 `bucket_dir/camera_registry.yaml`。只掛載
  `outputs/` 的容器或排程會因此失敗。這也正是部署端改用 `--registry-root` 指向共用
  registry 的原因——它那邊的 registry 本來就不在 bucket 底下。
- 舊 parquet 無從得知當時的幾何依據。要重建只能靠 registry 自身的版控歷史（它不進版控，
  所以實務上是靠人記得）。
- `load_registry_from_path` 在本 repo 只剩 `load_registry` 內部與 lib 自己的測試在用。
  函式與 `__all__` 匯出保留——它是公開 API，且部署端那條線需要吃任意路徑／`gs://` URI。

## Related Links

- [ADR-005](../flow_report/005-report-input-requirement-from-snapshot.md)（本 ADR 修訂的對象）
- [ADR-006](../zone_mapping/006-zone-boundary-band.md)（「新 ADR 修訂舊 ADR」的既有慣例）
- 部署端對應變更：argus PR #33
