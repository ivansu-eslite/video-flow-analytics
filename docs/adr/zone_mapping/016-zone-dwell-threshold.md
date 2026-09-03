# ADR-016: zone 的停留人次判定

## Status

Accepted

## Context

`zone_mapping` 原本只有兩個指標，兩者都是**瞬時**判定——看的是「這一格他在不在區內」：
`unique_visitors`（該時段在區域內出現過的不重複 `track_id` 數）與 `entries`（區外→區內
的轉換次數）。停在展示桌前看商品的人，跟從旁邊走過去的人，在這兩個指標裡分不出來。

[ADR-006](006-zone-boundary-band.md) 的實測正好說明差多少：47 個訪客貢獻了 223 次
`entries`，每人平均被計 4 次以上——那個數字既不是人數也不是停留時間。

新增第三個指標 `dwell_events`：該時段內有幾**人次**在該區域連續停留達到門檻。事件型
指標，各 `time_bucket` 相加不重複，`flow_report` 可直接 `sum`。

第三個指標的判定基礎又與前兩個不同，所以需要這份 ADR——ADR-006 已經替前兩個寫過一份
同型的說明。

### 本次的實測

素材 `bucket_20260801_small` / 2026-08-01（1,238,817 列，9 路中 4 路帶 zone 幾何、共
10 個 zone；`test_cam002`／`003`／`009` 為 3840 寬、`test_cam007` 為 1920 寬）。各表的
重現方式見 Related Links 末條——不是同一支腳本產的。

同一 `track_id` 內相鄰兩列的間隔，分兩種口徑量：

| 口徑 | 樣本數 | p50 | p99 | max |
|---|---:|---:|---:|---:|
| **所有列**（純漏偵測的空洞） | 1,230,807 | 66.7 ms | 333.3 ms | **2.133 s** |
| **只看區內列**（漏偵測＋真的踏出區域） | 199,472 | 50 ms | 450 ms | 417.9 s |

帶 zone 幾何的四路，取樣間隔中位數為 50.0 ms（20 fps，三台 4K）與 66.67 ms
（15 fps，`test_cam007`）；另外五路未參與判定，其中一路是 33.3 ms（30 fps）。

## Options Considered

### 判定基礎 A：生的 `in_zone`（採用）

`count_zone_visits` 算出兩組狀態：`in_zone`（`points_in_polygon` 的生布林值，
`unique_visitors` 用）與 `_committed`（線段區域的 Schmitt-trigger 黏著狀態，`entries`
用）。停留判定接在 `in_zone` 上。

### 判定基礎 B：沿用 `_committed` 的黏著狀態

排除。ADR-006 Decision 1 已經替 `unique_visitors` 論證過同一件事：黏著會把「走出區域後
停在帶外線段區域內」的人算成還在區內，而停留與 `unique_visitors` 同為**佔用型**的量，
繼承同一個理由。更糟的是站在區域邊界上的人可能整段 `_committed` 都是 null（從未確認在
區內），停留會完全算不到。

### 判定基礎 C：`in_zone` 再加「該段至少要有一列真的深入區內」

A 的已知風險是**邊界徘徊的人會被算成完整停留**：他們只要腳底點每隔幾格跨進來一次、
間隔小於容忍窗，就會被串成一次停留。這個族群不是假想的——ADR-006 量到四個 zone 的
`entries` 被線段區域從 223／424／145／385 砍到 42／88／64／96，被砍掉的 60–80% 就是
他們。

C 是在 A 上加一行條件：該段至少要有一列 `signed_d > boundary_band_px`（腳底點越過線段
區域的內緣）。實測 A 與 C 的差距（同一份素材，容忍窗 3.0 秒）：

| 門檻 X | A 人次 | C 人次 | C／A | 只有 A 算到 |
|---:|---:|---:|---:|---:|
| 10 s | 237 | 221 | 93.2% | 16 |
| **20 s** | **140** | **135** | **96.4%** | **5** |
| 30 s | 98 | 95 | 96.9% | 3 |
| 60 s | 39 | 39 | 100.0% | 0 |

差距隨門檻上升而收斂，在拍板的 X = 20 秒只有 5 段（3.6%），且集中在兩個 zone
（`平擺桌1_內側` 19→16、`平擺桌2_內側` 26→24），其餘八個 zone 逐 zone 完全相同。

更關鍵的是那 5 段的性質：它們段內最大深入距離是 27.2–41.8 px（平均 35.0），而該路的
線段區域是 50 px。這些人確實走進區域內數十像素，只是沒有越過為 `entries` 調出來的
那條線——C 排除掉的不是「貼著邊界抖動的人」，而是「淺淺待了 20 秒的人」。**排除 C。**

### 一段只計一次 vs 每個達標的格點都計

排除「每格都計」。那量的是「達標後又待了幾格」，是停留**時長**的另一種寫法，會讓
`dwell_events` 隨 fps 變動，也不能跨 `time_bucket` 相加。

### 多門檻（同時輸出 20 秒與 60 秒兩欄）

不做（使用者拍板先做一條）。要換門檻改設定重跑本階段即可，不必重跑 GPU 偵測。

## Decision

1. **停留判定用生的 `in_zone`，不吃 `_committed`**（上節判定基礎 A）。三個指標的判定
   基礎因此是：`unique_visitors` 與 `dwell_events` 用生的 `in_zone`，`entries` 用線段
   區域的黏著狀態。`dwell_events` 與 `unique_visitors` **同源**（判定基礎），與 `entries`
   **同型**（事件型、跨時段相加不重複）。

2. **段的切法是單一容忍窗 `dwell_gap_seconds`（T）**：同一 `track_id` 相鄰兩次「在區內」
   的間隔超過 T 就開新段。段內 `最後一格 − 第一格` **>=** 門檻 `dwell_threshold_seconds`
   （X）即計一次人次，歸戶到該段**首次跨過門檻**那一格的 `time_bucket`。一段只計一次；
   同一人的兩段各自達標則計兩次。

   比較用 `>=` 不是 `>`：寫成 `>` 會讓「剛好待滿 X 秒」整批消失，而從輸出檔看不出來。

3. **T 的有效上界是漏偵測空洞的上界，超過之後 T 不再是在補漏偵測。** ByteTrack 的
   `max_frames_lost` 就是 `[tracker].track_buffer`（30 格，**不隨 fps 縮放**），所以同一
   `track_id` 內「完全沒有列」的空洞上界是 `track_buffer / fps` ≈ 1.0–2.0 秒——本次實測
   全列間隔的 max 是 **2.133 s**，與這個推導吻合。超過這條線，T 純粹是在容忍「人真的走
   出區域再走回來」。

   實測佐證（X = 20 s 固定，掃 T）：

   | T | 2.2 s | 3.0 s | 5.0 s | 10.0 s |
   |---|---:|---:|---:|---:|
   | `dwell_events` 總計 | 140 | 140 | 149 | 161 |

   T 從 2.2 拉到 3.0 一段都沒多——漏偵測的空洞在 2.2 秒就吃完了；再往上才開始增加
   （+6.4%、+15.0%），那些是被橋接起來的真實離區。**採用預設 T = 3.0 秒**：涵蓋各路
   fps 差異（15 fps 的 30 格 = 2.0 秒，加上餘裕），同時停在增長開始之前。

4. **fail loud 的統計量是「同一 `track_id` 內相鄰兩列的間隔」，不是該攝影機的
   `timestamp` diff。** 容忍窗小於取樣間隔時每一格都會被切成獨立的段，段長恆為 0，
   這台攝影機的 `dwell_events` 整天是 0，而輸出檔與其餘兩個指標完全正常——與 ADR-006
   的窄區域檢查同型，靜默恆為 0 正是 fail-loud 要擋的那類錯誤。

   camera-wide 的 `timestamp` diff 有兩個致命問題：`(camera_id, timestamp)` 不是唯一鍵
   （一格畫面每個目標一列，見 CLAUDE.md），同一格多人時 diff = 0，**忙碌的攝影機中位數
   直接變 0、檢查永遠不觸發**；而沒有偵測的時段完全沒有列，**冷清的攝影機中位數會膨脹
   到幾十秒而誤擋**——規劃期實測 camera-wide 的間隔 max 有 61 秒，per-track 只有 2.1 秒。

   per-track 序列另有一個結構性的好性質：由 Decision 3 的上界，它的中位數必落在
   `[1/fps, track_buffer/fps]`，最壞情況也只高估到約 1–2 秒，誤擋風險被限制住。

   檢查是 camera 級（估的是該路的取樣間隔），與 zone 幾何無關，在 zone 迴圈之前做一次。
   `gaps` 樣本少於 8 個（每個 track 都只有一兩列）時記 warning 並放行——證據不足時放行，
   與 `_validate_zone_fits_band` 對「該攝影機沒有 zone 就不檢查」同一個立場，誤擋的代價
   是該日整份報表都產不出來。

5. **兩個參數是絕對時間，不隨解析度換算**，刻意不走 [ADR-004](../shared/004-band-resolution-scaling.md)
   的尺規那條路。名稱在 `config.toml`／`ZoneConfig`／`map_zones_daily` 三處完全相同：
   ADR-006 之所以有 `boundary_band_px_1080p` vs `boundary_band_px` 兩個名字，是因為單位
   在中途換了（基準像素 → 實際像素），X 與 T 全程都是秒。

6. **兩個參數用 `gt=0`，不是 band 的 `ge=0`。** `boundary_band_px_1080p = 0` 有明確的
   退化語義（純內外判定），X 與 T 都沒有：X = 0 會讓每個在區內的段都達標，量到的就不再
   是停留。`map_zones_daily` 本身也擋一次——它是 README 明列的正式進入點，直接呼叫時不
   經 pydantic。

7. **`count_zone_visits` 的兩個新參數 keyword-only 且不給預設值**，不納入「預設值三處
   一致」的檢查。`boundary_band_px = 0` 有中性的退化值所以可以有預設，X／T 沒有；沒有
   預設值就不可能有人拿到錯的預設值。

8. **時間一律用整數微秒比較。** `timestamp` 是 `Datetime("us")`，由
   `segment.start + frame_index / fps` 算出，30 fps 的相鄰間隔在 33333／33334 µs 之間
   跳動，用秒的浮點數比會在除不盡的間隔上分岔。

9. **欄位名固定為 `dwell_events`，不帶門檻數字。** 見 Consequences 的「改 X 等於換一個
   指標」；否決過的兩招是欄名帶門檻（`dwell_events_30s` 讓 schema 變動態，`flow_report`
   的固定 schema 接不住）與寫進 parquet metadata（[ADR-008](../shared/008-config-section-namespace.md)
   已為 `bucket_minutes` 否決過同一招）。

## Consequences

Positive

- 三個指標第一次能區分「路過」與「停留」。實測 X = 20 s 時十個 zone 共 140 人次，
  對照同一批的 `unique_visitors` 合計 1,724、`entries` 合計 1,774。
- **既有兩個指標逐列不變**：同一份 `tracking_results.parquet` 跑改動前後的程式碼，
  `unique_visitors` 與 `entries` 十列逐值相同（`Series.equals` 為 True）。
- 門檻對輸出是單調的：X = 10／20／30／60／120 秒得 237／140／98／39／15 人次，且**逐列**
  單調不增，可當作粗回歸檢查。
- 容忍窗設錯（小於取樣間隔）會直接報錯，不會產出一份靜默恆為 0 的 `dwell_events`。
- `flow_report` 對新舊 `zone_counts.parquet` 都跑得動（照欄名讀、不驗 schema），新欄位
  不影響既有兩個 metric——但在 `flow_report` 放寬 `metric` 的 `Literal` 之前，這一欄不會
  出現在 `report.xlsx`。

Negative

- **邊界徘徊與「淺淺待著」的人被算成完整停留**，沒有任何防護。上節量到 X = 20 s 時有
  5 段（3.6%）從未深入到線段區域內緣以外；判定基礎 C 可以擋掉它們，但同時擋掉的是真的
  在區域邊緣待了 20 秒的人，所以沒有採用。X 越小這個比例越大（X = 10 s 時是 6.8%）。
- **T 會把真實的離區時間一起算進停留**：`elapsed` 是 `last − first`，含所有被橋接的
  空洞。實測 199,472 個區內列間隔中，有 **136 個（0.068%）落在 2.133 秒以上、3.0 秒
  以內**——超過漏偵測空洞的上界、卻仍在 T 之內，那些必然是被橋接起來的真實離區。
  T 小時可忽略，T 大時就是實質灌水——這是「單一容忍參數」這個拍板的必然代價。
- **軌跡斷裂造成的低估無法從輸出檔看出來。** 容忍窗只吃得掉 ≤ 約 2.1 秒的空檔
  （Decision 3 的上界），超過就換新 `track_id`，永遠接不回來（跨 `track_id` 縫合已排除：
  誤合會往灌水方向錯，而且要標註素材才驗得了）。X = 20 秒對這件事相當敏感，低估幅度隨
  場景擁擠度變動，輸出檔本身完全正常。
- **母體小是預期值，不是 bug。** 規劃期實測（`bucket_20260728_small`，八路 3001 條軌跡）
  活得夠久（≥ 20 秒）的軌跡只佔 9.5%，而那是「在整個畫面裡活多久」，在單一 zone 內停留
  20 秒的只會更少。本次十個 zone 一小時共 140 人次，每個 zone 是 9–26。
- **`dwell_events` 不是 `unique_visitors` 的子集**：同一人同一 bucket 內兩段都達標會計 2，
  而 `unique_visitors` 對他只算 1。下游任何 sanity check 都不可以寫
  `dwell_events <= unique_visitors`。
- **改 X 等於換一個指標，而輸出檔沒有任何地方記得產生時用的 X。** `flow_report` 的
  `on_duplicate_date = "append"` 會把不同 X 產生的列疊進同一份 `report.xlsx`，與 ADR-006
  記的「跨日報表混口徑」同型，只是這次混的是指標定義本身。改 X 要重跑歷史日期。
- **X 設太大導致全天 0 沒有任何訊號可擋**，Decision 4 那道檢查只看 T。這是 ADR-006
  「內切半徑檢查是必要條件、不是充分條件」在這裡的對應版本。
- **T 的預設 3.0 秒沒有針對 zone 內的行為實測過**——推導依據是漏偵測空洞的上界，不含
  「人真的踏出區域又回來」那一類。實跑後若發現同一人被切成多段，這是第一個要調的參數。
- **既有 zone golden 全部失效**（交付期比對標的，存於 argus GCS）：加欄位讓 schema 比對
  不成立，需重新產生。ADR-006 也記過同一件事。
- 新增的 fail loud 與 ADR-006 的兩道一樣，擋下的那一天**整份報表都產不出來**
  （[ADR-005](../flow_report/005-report-input-requirement-from-snapshot.md)／
  [ADR-007](../shared/007-remove-registry-snapshot.md)：registry 有 `zones` 就必須有
  `zone_counts.parquet`），出入口三個分頁也不會產出。

## Related Links

- [ADR-006](006-zone-boundary-band.md)：既有兩個指標為何用不同判定，本 ADR 與它同型
  （`dwell_events` 與 `unique_visitors` 同源、與 `entries` 同型）
- [ADR-004](../shared/004-band-resolution-scaling.md)：像素參數的尺規（本次兩個參數是
  時間單位，刻意不走這條路）
- [ADR-008](../shared/008-config-section-namespace.md)：否決「把參數寫進 parquet
  metadata」的先例
- [ADR-012](../video_analyze/012-track-worker-sharding.md)：`track_id` 的唯一範圍只到
  「同一路之內」，本判定先依 `camera_id` 過濾再分段，不受影響
- `board/C26/zone_dwell_band.py`：判定基礎 A vs C 的比較腳本（開發用工具，不進版控），
  Options Considered 那張 A／C 對照表、逐 zone 的差異與「只有 A 算到的 5 段深入 27–42 px」
  出自它的輸出 `board/C26/zone_dwell_band.out`。**其餘數字不在該腳本裡**：Context 的間隔
  分布表是直接對 `tracking_results.parquet` 依 `(camera_id, track_id)` 分組算的；
  Decision 3 的 T 掃描表與 Consequences 的 X 掃描是逐次改環境變數
  （`ZONE__DWELL_GAP_SECONDS`／`ZONE__DWELL_THRESHOLD_SECONDS`）重跑
  `zone_mapping` 後讀 `zone_counts.parquet` 得到的；「既有兩欄逐列不變」是把 70bb3ab 的
  `zone_mapping/src` 取出、以 `PYTHONPATH` 蓋過工作樹版本跑同一份輸入比對的
