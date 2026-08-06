# ADR-006: zone 的 entry 判定改用區域邊界緩衝帶

## Status

Accepted

## Context

`zone_mapping` 的 `entries` 原本用**時間**去抖：`entry_debounce_frames` 要求連續 N 格
腳底點都落在多邊形內才算一次進入。這擋不住「人在區域邊界附近徘徊」——腳底點每次跨過
邊界就重新計一次進入。

實測 `bucket_20260728_small` / 2026-07-28，cam003／cam004 共 4 個 zone，皆 4K
3840x2160（重現腳本 `overlay/analysis/zone_newscale.py`，開發用工具、不進版控）。
格式為 `entries`／貢獻過 entry 的不同人數，band 為 4K 實際像素：

| zone | 訪客 | 現行 d=1 | band 30 | band 50 | band 60 | band 115 |
|---|---:|---|---|---|---|---|
| 平擺桌1_外側 | 47 | 223／47 | 64／34 | 42／27 | 36／25 | 10／10 |
| 平擺桌1_內側 | 103 | 424／103 | 120／79 | 88／69 | 81／67 | 10／10 |
| 平擺桌2_外側 | 57 | 145／57 | 80／46 | 64／40 | 54／36 | 27／20 |
| 平擺桌2_內側 | 92 | 385／92 | 128／87 | 96／74 | 78／65 | 17／16 |

47 個訪客貢獻 223 次進入、103 個訪客貢獻 424 次——每人平均被計 4 次以上，這個數字不能
當「進入次數」用。單一 track 的佐證：track 1646 肉眼只該算 1 次，時間去抖下算 38 次，
4K band 30 是 3 次、band 60 是 1 次（`overlay/analysis/trace_1646.py`）。

> 上表的訪客數與更早的實驗記錄略有出入（例如平擺桌2_外側 55 → 57）：
> `tracking_results.parquet` 本身不可重現（ByteTrack 的 `track_id` 指派每次重跑都會變），
> 見 CLAUDE.md。此表是本次改動當下重跑的版本，是這些數字的版控出處。

四個 zone 的內切半徑（用本次實作的格點取樣算法重算）為 141.1／172.1／181.8／217.8 px
（4K），是選 band 上限的實質限制：4K band 115 已讓三個代表性 track 完全不再被計入，
`entries` 掉到 10／10。

## Options Considered

### Option A：維持時間去抖，調高 `entry_debounce_frames`

擋不掉本文要處理的那類重複——徘徊的人「連續 N 格都在區域內」是常態，只是邊界抖動讓他
反覆離開又回來。調高只會延後進入事件的歸戶時段，並漏掉停留短於 N 格的真實訪客。

### Option B：空間緩衝帶＋時間去抖並行

4K band 20 再疊「連續 3 格」，`entries` 只少 9 次卻掉 5 個人；同樣的人數損失，單純加寬
band 能多砍三倍以上的重複計數。四個 zone 有三個同結論（`overlay/analysis/zone_combo.py`）。
原因是邊界抖動已被緩衝帶解決，再疊一層時間條件擋掉的變成「停留很短的真實訪客」，會把
停留時長的語義混進「進入次數」。排除。

### Option C：空間緩衝帶＋Schmitt-trigger（採用）

多邊形邊界加寬成 `±band` 的帶：帶號距離 `> band` 確認在區內、`< -band` 確認在區外、
落在帶內時沿用前一個已確認狀態。已確認狀態由外翻內才算一次進入。邊界附近的抖動整段
維持同一個已確認狀態，只計一次；真的離開（越過帶的外側）再回來仍算兩次。

## Decision

1. **`entries` 用緩衝帶＋Schmitt-trigger，`unique_visitors` 維持原判定。** 兩欄刻意用
   不同判定：`entries` 是事件型指標，黏著狀態正好濾掉邊界抖動的重複計數；
   `unique_visitors` 是佔用型指標，吃了黏著狀態會讓「走出區域後停在邊界外緩衝帶內」的人
   在其後每個 `time_bucket` 都被算成區內訪客。`unique_visitors` 沿用同一個
   `points_in_polygon` 布林值，數值與改動前**完全一致**（不是近似）。
2. **`entry_debounce_frames` 移除**，沿用舊參數名直接報錯並說明語義變更（單位由「連續
   格數」變成「1080p 基準像素」，數值不可直接搬）。
3. **帶號距離的符號由 `points_in_polygon` 給**，絕對值取點到多邊形各邊線段的最小距離；
   多邊形自動閉合（`Zone.polygon` 不重複首點，收尾邊也要算）。函式一併回傳 `inside`
   布林值供 `unique_visitors` 使用，避免每個 zone 對全天列數多跑一次逐邊迴圈；**不可
   改用 `d > 0` 反推 `inside`**——邊界點 `d == 0` 會與 `points_in_polygon` 的
   implementation-defined 結果分岔。
4. **首格語義維持現行、與 `line_counting` 相反**：track 首次出現即在區內算一次進入
   （人可能從畫面外直接走進區內）。狀態機的守衛寫成「前一個已確認狀態為 null 或 -1」，
   不可寫成 `!= 1`（polars 的 `null != 1` 得 `null`，該列會被 `filter` 丟掉，首格即在
   區內的那次進入會靜默消失）。ADR-001 已描述此語義差異。
5. **band 的尺規沿用 ADR-004**：設定值 `boundary_band_px_1080p` 以 1080p（寬 1920）為
   基準，逐攝影機依 `frame_width` 換算成實際像素。**採用預設值 25**（4K 50 px），比實測
   建議的 30（4K 60 px）保守。缺 `frame_width`／`frame_height` 的舊 parquet fail loud。
6. **窄區域 fail loud**：多邊形內切半徑小於換算後的 band 時直接報錯。這種 zone 內部沒有
   任何點滿足「帶號距離 > band」，`entries` 會恆為 0——靜默恆為 0 正是 fail-loud 要擋的
   那類錯誤（與 ADR-004 拒絕「找不到尺寸就當 1080p」同型），寫在 README 攔不住日後改
   zone 幾何的人。內切半徑用格點取樣求：範圍取多邊形自身的 bounding box（不需要影像
   尺寸），步長 `band/8`、長短邊各至少 32 點。

### 對 ADR-004 的修訂

ADR-004 的 Consequences 記了一條「同一份舊 parquet 會出現 `zone_mapping` 照跑、
`line_counting` 報錯的不對稱狀態」。本 ADR 起**該條不再成立**：`zone_mapping` 同樣要
換算像素參數，因此也要求 `frame_width`／`frame_height`，缺欄位的舊 parquet 兩包都擋。
ADR-004 的其餘內容（尺規的選擇、尺寸來源的取捨）不變。

### zone 與 line 的幾何刻意不同，不要「統一」

`line_counting` 用「最近段無限直線＋`inside_point` 錨點」定側別，其實作註解明確禁止改用
全域 signed-side——前提是 polyline 開放、且要處理包住 `inside_point` 的ㄇ形 barrier。
封閉多邊形的內外由 `points_in_polygon` 全域決定，不受此限；同理 zone 不需要
`segment_crosses_polyline` 那道有限線段閘門（多邊形封閉，沒有「繞過端點」的問題）。
兩份實作看起來相似，但前提不同，合併會讓其中一邊失去正確性。

### band 換算各包一份，不抽共用

`line_counting` 與 `zone_mapping` 各有一份三行的換算，兩邊函式本體幾乎逐字相同，差別只在
zone 這邊多回一個 `scale`（窄區域的錯誤訊息要把建議上限換算回 1080p 基準值）。為這點差異
抽一個殼，可讀性沒有變好。**第三個消費者出現時抽進 `libs/`**。（CLAUDE.md 記的 registry 防呆補丁漂移前科
風險型態不同：那是同一份會演化的邏輯，這裡尺規正本已由 ADR-004 固定、
`BASELINE_FRAME_WIDTH` 是常數。）

## Consequences

Positive

- `entries` 不再被邊界抖動灌水：實測四個 zone 由 223／424／145／385 降到 42／88／64／96
  （降 56–81%），`unique_visitors` 逐值不變。
- `boundary_band_px_1080p = 0` 時退化成純內外判定，與改動前 `entry_debounce_frames = 1`
  的輸出逐值一致（實測整表 `equals` 為 True），可用來對照舊資料。理論上的已知例外是腳底
  點恰在邊界（`d == 0`）：`points_in_polygon` 對邊界點的結果是 implementation-defined，
  band=0 時這種點會落在帶內走 forward_fill。實測資料沒有出現這種點。
- 同一個設定值在 1080p 與 4K 上代表同樣的實際距離，混解析度的 bucket 不必各調一套。
- zone 幾何畫得太窄時直接報錯，不會產出一份靜默恆為 0 的 `entries`。

Negative

- **加寬 band 會同時砍掉真實訪客**：基準值 25 下，貢獻過 entry 的不同人數降到原本的
  57–80%（47→27、103→69、57→40、92→74）。只在區域邊緣淺淺待過的人不再被計為進入。
  這是本判定的取捨，不是可以靠調參消掉的誤差。
- **預設值 25 未經直接實測**，是從實測的 4K 30／60 之間取的保守值；所有證據只來自兩台
  4K 攝影機、4 個 zone，**1080p 攝影機的 zone 從未掃過**。
- **內切半徑檢查是必要條件、不是充分條件**：四個 zone 半徑 141–218 px，但 4K band 115
  仍讓三個代表性 track 歸零。通過檢查不代表 band 可用。
- **內切半徑是格點取樣的下界**，會略微低估、可能誤擋邊緣案例；誤擋時整天所有攝影機的
  `zone_mapping` 都不產出，因此錯誤訊息要同時給出兩條操作路徑（調小 band、改 zone 幾何）。
- **舊 parquet 全面失效**：`zone_mapping` 從此擋下缺影像尺寸欄位的 parquet，原本能跑的
  舊資料會停跑，需以現行 `video_analyze` 重跑（GPU）。
- **`zone_mapping` 擋下的那一天，整份報表都產不出來**：ADR-005 起 `flow_report` 的輸入
  必要性看 registry 的定義（[ADR-007](007-remove-registry-snapshot.md) 起是
  `bucket_dir/camera_registry.yaml`，原為當日輸出目錄下的快照）——裡面有 `zones` 定義就
  必須有 `zone_counts.parquet`，缺檔即中止整個 `export_report_daily`，出入口三個分頁也不會產出。
  因此本 ADR 新增的兩道 fail loud（缺影像尺寸欄位、窄區域）不只影響區域兩頁，誤擋的代價
  是該日沒有報表。
- **跨日報表會混到兩種口徑**：`flow_report` 的 `on_duplicate_date = "append"` 會把新舊
  語義的 `entries` 疊進同一份 `report.xlsx`。要口徑一致需重跑歷史日期，或在報表註明改動
  日期。`unique_visitors` 不受影響。
- **既有 zone golden 全部失效**（交付期比對標的，存於 argus GCS），需重新產生。
- `entries` 的語義與 `line_counting` 相反（首格即在區內算一次），兩包的 docstring 互相
  註明，但這仍是讀 code 的人要自己記住的事。

## Related Links

- [ADR-001](001-line-crossing-detection.md)：計數線的跨越判定，末段描述與 zone 相反的首格語義
- [ADR-004](004-band-resolution-scaling.md)：像素參數的尺規與影像尺寸來源（本 ADR 修訂其
  「zone_mapping 照跑」的不對稱條款）
- [ADR-005](005-report-input-requirement-from-snapshot.md)：報表的輸入必要性由快照決定
  （本 ADR 的 fail loud 因此會擋掉當日整份報表）
- `overlay/zone_band.py`：本判定的可運作原型（開發用工具，不進版控）
- `overlay/analysis/`：`zone_newscale.py`（band 掃描）、`zone_combo.py`（空間＋時間並行的
  排除證據）、`trace_1646.py`（單 track 驗證）
