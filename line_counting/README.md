# line_counting

方向性計數線進出人數統計：把一整天的追蹤明細對映到各攝影機的計數線（line）幾何，統計
每個 track 跨越計數線的次數與方向，轉換成「每時段、每計數線」的進出人數。

`line_counting` 與 [`zone_mapping`](../zone_mapping) 的輸入相同
（`tracking_results.parquet` ＋ `camera_registry.yaml`），都以腳底點做純 CPU 向量化判定，
差別在幾何：zone 判「腳底是否落在多邊形內」（區域佔用），line 判「腳底是否跨越計數線
及其方向」（方向性進出）。

## 概述

輸入是追蹤明細 `tracking_results.parquet` 與 `camera_registry.yaml` 的計數線定義；以每個
track 的腳底中心點 `((x1 + x2) / 2, y2)` 對計數線算帶號垂直距離，用「帶線段區域的
Schmitt-trigger」偵測側別翻轉（跨越）與方向，再依 `time_bucket` 聚合出兩項指標，輸出
`line_counts.parquet`：

| 指標 | 定義 |
| --- | --- |
| `in_count` | 該時段內 track 由外側跨越到內側（往 `inside_point` 那一側）的次數 |
| `out_count` | 該時段內 track 由內側跨越到外側的次數 |

本階段只做「事件轉化」，不做跨期間彙總或分析。純 CPU 向量化運算，不需 GPU。只調整計數線
幾何時僅需重跑本階段。

**進入點是函式呼叫，CLI 只是外殼**：核心是
`count_lines_daily(date, bucket_dir, bucket_minutes, crossing_band_px_1080p=25,
output_root=OUTPUT_ROOT) -> Path`（在 `services/line_map.py`），CLI 進入點
`line_counting.main:main` 只是從 `config.toml` 組出參數後呼叫它。

原始碼採 DDD 分層（`src/line_counting/`）：

| 目錄 | 內容 |
| --- | --- |
| `main.py` | CLI 外殼：讀 `settings` → 組參數 → 呼叫 `count_lines_daily` |
| `config/constants.py` | 非 Pydantic 靜態常數（輸出根目錄、輸入輸出檔名、parquet schema） |
| `models/config.py` | pydantic-settings 設定模型與全域單例 `settings` |
| `services/line_map.py` | 讀檔、逐攝影機/逐計數線套用演算法、寫檔與 registry 快照 |
| `services/stats.py` | 計數線跨越判定與進出人數聚合等純函式 |

`camera_registry.yaml` 的模型與計數線驗證（`vfa_registry` 的 `Line`／
`parse_and_validate_lines`）、單行 JSON 的 `StructuredLogger`（`vfa_observability`）由四包
共用的 lib 提供，為 uv workspace 成員，不在本包內：
[libs/vfa_registry](../libs/vfa_registry)、[libs/vfa_observability](../libs/vfa_observability)。

## 演算法

1. **帶號距離**：`signed_distance_to_polyline` 對 polyline 的每一段算「點到線段」的有限
   距離（端點外夾到端點），取最近的一段，輸出該最近段**無限直線**的帶號垂直距離；符號
   以 `inside_point` 相對同一段無限直線的側別定為正（正 = 與 `inside_point` 同側 = 內側）。
2. **帶線段區域的 Schmitt-trigger**：帶號距離 `d > band` 判內側（`+1`）、`d < -band` 判外側
   （`-1`）、落在 `[-band, band]` 之內即在線段區域內（`null`，沿用前一個已確認側別，hysteresis）。
   已確認側別翻轉即一次跨越——翻到內側計 `in`、翻到外側計 `out`。
3. **有限線段閘門**：側別是用最近段的**無限直線**定的，繞過線的兩端走過去同樣會讓側別
   翻轉。因此翻轉還要再過一關——自上一次確認側別以來，該 track 的逐格位移線段必須真的
   與計數線的**有限線段**相交過（`segment_crosses_polyline`），否則不計。**線的長度因此
   是實質判準**：只算真的走過所畫線段的人。閘門是事件級而非逐格判定，門口線段區域內駐留後
   進場、斜穿門、掉幀都不受影響，取捨見
   [ADR-003](../docs/adr/003-finite-line-segment.md)。
4. **起始側不計**：track 起始就在某側（前一格為 `null`）不算跨越——計數線只認「側別
   翻轉」，起始側不構成翻轉。此點與 `zone_mapping` **相反**（`zone_mapping` 首次即在
   區內會算一次 entry）。
5. **線段區域寬度依解析度換算**：設定的 `crossing_band_px_1080p` 是以 1080p（寬 1920）為基準
   的像素值，每台攝影機依自己的 `frame_width` 換算成實際像素（`基準值 × frame_width /
   1920`，只用寬度、線性）後才進判定。同一個設定值在 1080p 與 4K 上代表同樣的實際距離，
   不必為混解析度的 bucket 各調一套。尺寸來自 `tracking_results.parquet` 的
   `frame_width`／`frame_height` 欄位，取捨見
   [ADR-004](../docs/adr/004-band-resolution-scaling.md)。
6. `crossing_band_px_1080p = 0` 時線段區域退化為單點，等同幾何零交越（每次穿過線段的幾何跨越
   都計）；`0` 換算後仍是 `0`，不受解析度影響。**預設值是 `25`**，由實測挑出（見
   「已知限制」的最後一條）。

### 已知限制

- **`inside_point` 的側別只錨定被穿越的最近段（局部直線），不建全域 signed-side**。這讓
  包住 `inside_point` 的ㄇ形（凸向 inside）barrier 也能正確判向；代價是**凹角**（reflex
  頂點朝 `inside_point`）在 medial-axis 附近理論上有幽靈翻轉風險。門口計數線的凸／直線
  barrier 不受影響。**勿改成全域 signed-side**——那會在包住 inside 的ㄇ形 barrier 上算錯
  （見 `services/stats.py` 的註解）。
- **線的長度就是能被算到的門寬**：標註時線要**拉滿可通行寬度**。線畫得比實際門框窄，
  從沒被線蓋到的那段門進出的人不會被計入（改動前是相反的問題：線再短也會把牆邊走過的人
  算進去）。
- **`in` 與 `out` 不平衡是正常的**：從門進場、繞牆邊出去的人是 `in=1 / out=0`。有限線段
  閘門只算真的走過線段的那一次，不保證每個人的進與出成對出現。
- **位移線段是直線近似**：閘門逐格檢查「前一格到這一格」的直線，不是實際走過的路徑。
  正常影格間隔下差異可忽略，但 track 中間掉偵測時，那一格的直線可能與實際路徑不同。
- **`band > 0` 時有殘留的假陽性**：人在線段區域內反覆穿過線段累積了相交，之後繞端點外出
  造成側別翻轉，閘門會放行。他確實穿過線段，只是那次翻轉的方向被歸因於繞端點那一次。
- **`crossing_band_px_1080p` 仍需依場景調，解析度那一項已由換算消除**：band 對「走出畫面」
  友善（腳底點消失前的抖動被線段區域吃掉），但幅度大於 band 的大抖動仍可能誤計。門口駐留抖動
  明顯時調高，代價是幅度小於 band 的真實貼線跨越會被濾掉。**預設值 `25` 由實測挑出，
  且已接近單一全域值的上限**：八條計數線掃 0／10／15／20／25／30 基準值，多數線從 10 起
  就不再變動；由 20 調到 25 的全部代價是 `出入口103` 少一個人（44 → 43），換到
  `出入口152` 的每人平均被計次數由 1.67 降到 1.17 且沒有掉人。再往上就會碰到懸崖——把
  各線的穿越深度換算回基準單位後，`出入口152` 的 p25 只有 31，基準值 40 會讓那條線
  12 個人裡 11 個整個消失。注意換算後 4K 攝影機的實際線段區域寬度是 50 px（該台的人形也大一倍，
  佔肩寬比例反而是所有攝影機中最低的）。
- **有些重複計數 band 解不掉**：`出入口119` 的每人平均被計次數 1.48，但基準值從 10 掃到
  30 幾乎不影響它（84 → 78）——那條線的穿越深度中位數 181 px，線段區域根本碰不到。它的重複
  來自別的原因（真實往返或 track 延續），調 band 不是解法。
  **換算只消除解析度造成的差異**：
  同一個值在各攝影機上代表的「佔人形多少比例」仍會因安裝高度與視角而不同（實測同為 1080p
  的六台攝影機，肩寬中位數差 2.77 倍），同一畫面內的透視變異也還在，所以單一全域參數不會
  在各攝影機上有一致的效果。
- **掉幀誤算**：track 中間掉偵測時，大跨距仍會正確判為一次跨越；但極大 gap 下腳底點跳躍
  可能誤穿線。

## 環境需求

| 類別 | 需求 |
| --- | --- |
| 執行環境 | Python `>= 3.12` |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（安裝與執行皆透過 uv，使用倉庫根目錄單一 `uv.lock`） |
| GPU | 不需要，純 CPU |

執行期依賴（由 `uv sync` 安裝，各套件用途）：

| 套件 | 用途 |
| --- | --- |
| `numpy` | 向量化幾何運算（點到 polyline 的帶號距離） |
| `polars` / `pyarrow` | parquet 讀寫與聚合 |
| `pydantic` / `pydantic-settings` | 設定與 registry 的資料模型與驗證，config 從 `config.toml`／環境變數載入 |
| `pyyaml` | 讀取 `camera_registry.yaml` |
| `vfa_registry` | 共用 lib：`camera_registry.yaml` 的模型與計數線驗證 |
| `vfa_observability` | 共用 lib：單行 JSON 的 `StructuredLogger` |

## 安裝與快速開始

```bash
uv sync --package line_counting
uv run --package line_counting line_counting
```

CLI 不接受任何旗標，所有參數都讀自 `config.toml`。執行前需備妥：

1. 當日的 `outputs/{bucket}/{date}/tracking_results.parquet`。
2. 一份本機的 `bucket_dir/`，內含 `camera_registry.yaml`（格式見根 README 的
   [設定](../README.md#設定)）。
3. `line_counting/config.toml`（指定本次要跑哪個 bucket、哪一天與統計參數）。

### 執行位置（cwd 約束）

**一律在倉庫根目錄執行，以 `--package` 指定本套件**（`uv run` 不改變 cwd）。下列兩者皆為
**cwd 相對路徑**，並非相對於本資料夾：

| 路徑 | 來源 | 在 `line_counting/` 內執行時 |
| --- | --- | --- |
| `OUTPUT_ROOT = outputs/` | `config/constants.py` 常數 | 去 `line_counting/outputs/` 找輸入而 `FileNotFoundError`，產出也落在錯的樹 |
| `settings.input.bucket_dir` | `config.toml` `[input]` | 對到不存在的 `line_counting/bucket_name1`；實務上不會走到，上一列的輸入檢查會先失敗 |

本套件自己的 `config.toml` 以 `find_project_root`（往上找 `pyproject.toml`）定位，不受
cwd 影響。

## 設定

`config.toml` 置於本套件根目錄（`line_counting/config.toml`），只含 `[input]` 與 `[line]`
兩個區塊，透過 pydantic-settings 載入；**找不到此檔**時會印出警告並以各項預設值啟動，
**此檔存在但參數不合法**則直接報錯（不靜默套用預設值）。同理，出現**未知的頂層區塊**
（例如把 `[line]` 拼成 `[lines]`）也會直接報錯，而非被靜默忽略。各欄位亦可用環境變數覆寫
（巢狀分隔符 `__`，例如 `LINE__CROSSING_BAND_PX_1080P=5`）。注意欄位名未加前綴，
`LINE`／`INPUT` 這兩個名稱本身也是有效的覆寫來源（設成 JSON 會整段取代該區塊），
在共用的執行環境中留意不要與其他程式的環境變數撞名。範例：

```toml
[input]
bucket_dir = "bucket_name1"
date = 2026-05-01

[line]
bucket_minutes = 60          # 事件統計時間粒度（分鐘）
crossing_band_px_1080p = 25  # 跨越去抖的線段區域寬度（1080p 基準）；0 = 細線純零交越
```

| 區塊 | 欄位 | 預設 | 約束 / 說明 |
| --- | --- | --- | --- |
| `[input]` | `bucket_dir` | `"bucket_name"` | 本機模擬 GCS bucket 的根目錄（cwd 相對） |
| | `date` | — | 統計日期；未設定時報錯 |
| `[line]` | `bucket_minutes` | `60` | 事件統計時間粒度（分鐘），`>= 1` |
| | `crossing_band_px_1080p` | `25` | 跨越去抖的線段區域寬度，以 1080p（寬 1920）為基準的像素值，`>= 0`；逐攝影機依 `frame_width` 換算成實際像素；`0` = 細線純零交越 |

舊參數名 `crossing_band_px` 已不存在：沿用它會直接報錯並說明新語義，而不是被當成未知欄位
（值的意義也變了——同一個數字在 4K 攝影機上換算後是兩倍）。

`camera_registry.yaml`（攝影機清單與計數線定義，放在 `bucket_dir` 根目錄、不進版控）與本
階段相關的計數線定義格式如下（與各攝影機的 `zones:` 平行）：

```yaml
lines:
  - name: "front_door"                          # 全域唯一（比照 zone）
    points: [[100, 400], [300, 380], [500, 400]]  # polyline，>= 2 頂點；固定解析度像素座標
    inside_point: [300, 200]                    # 場內一點；跨越往這側 = in
    line_group: "mall_main_entrance"            # 這條線所屬的範圍（可跨攝影機同名）
```

與本階段相關的使用限制（皆為 fail-loud，違反時直接報錯）：

- **`camera_id` 與 `location_camera_id` 皆須唯一**：兩者都是查詢字典的鍵，重複會靜默
  覆蓋其中一筆攝影機，載入 registry 時即擋下。
- **`line` 名稱須全域唯一**：不只同一攝影機內不可重複，跨攝影機也不可重複（下游報表以
  計數線名稱、不含 `camera_id` 分組彙總，同名計數線會被合併）。
- **`line_group` 必填，但刻意不驗證跨攝影機唯一**——與 `name` 的規則相反，見下方
  「`line_group`：一個範圍的所有出入口」。
- **`points` 至少需要 2 個頂點**（polyline），且不可有零長度段（連續重複頂點）；座標為
  該攝影機固定解析度下的像素座標。**線的長度是實質判準**：只有真的穿過所畫線段的軌跡
  才計入進出，故標註時要拉滿整個可通行寬度，畫太短會漏算（見「已知限制」）。
- **`inside_point` 不可落在任一段的無限延伸線上**：否則該段的側別無法定號、方向判定失效。
- **參與判定以 `lines` 非空為準**：`lines` 空（或未定義）的攝影機直接跳過；要停用某台就
  移除其 `lines`（不另設參與旗標）。
- **定義了計數線的攝影機在當日追蹤明細中必須有資料**：攝影機改名或 key 打錯時直接報錯，
  而非靜默算出漏掉出入口的進出人數。

### `line_group`：一個範圍的所有出入口

要統計的對象常是「一個範圍的進出」（例如一個賣場、一個樓層），而這種範圍通常不只一個
出入口，還可能分屬不同攝影機。`line_group` 標示一條計數線屬於哪個範圍，讓
`line_counts.parquet` 的每一列都帶著這個維度，供後續彙總依據——**本階段只輸出這個
標示，不在 `line_counting` 產生範圍層級的加總數字**，那是後續任務的功能。

與 `line` 名稱相反，**`line_group` 允許、也預期會跨攝影機同名**：一個賣場的數個門本來
就分屬不同攝影機，要能歸成同一組（見 [ADR-002](../docs/adr/002-line-group-semantics.md)）。

**使用前提（程式不驗證，需配置時自行保證）**：一個範圍的進出是把該範圍各出入口的
`in`／`out` 相加，這只有在**每個出入口的 `inside_point` 都指向該範圍的內側**時才成立。
某個門的 `inside_point` 指反了，從那個門進來的人會被記成 `out`，加總後同時少算進、
多算出，且不會有任何錯誤訊息——`vfa_registry` 驗不出這件事，它只知道 `inside_point`
不能落在線段的延伸線上，無從得知哪一側才是「這個範圍的內側」。標註頁會在每條線上畫出
「進」方向的箭頭，配置同一群組的線時可用它確認箭頭是否都朝向範圍內側。

計數線幾何在載入 registry 時刻意不驗證，而是先確認攝影機對得上當日資料、再解析幾何，讓
「攝影機對不上」這個更根本的錯誤先報出來，不被計數線定義的筆誤蓋過。

## 輸入 / 輸出檔案

`{bucket}` = `bucket_dir` 的目錄名，皆位於倉庫根目錄的 `outputs/` 下：

| 路徑 | 讀 / 寫 | 內容 |
| --- | --- | --- |
| `outputs/{bucket}/{date}/tracking_results.parquet` | 讀 | 追蹤明細；缺少時報錯。須含 `frame_width`／`frame_height`（線段區域寬度的解析度換算靠它），2026-07 之前產出的舊檔沒有這兩欄，會直接報錯要求重跑 `video_analyze` |
| `{bucket_dir}/camera_registry.yaml` | 讀 | 攝影機清單與計數線幾何 |
| `outputs/{bucket}/{date}/line_counts.parquet` | 寫 | 每時段每計數線進出人數，欄位 `line_group` / `camera_id` / `line` / `time_bucket` / `in_count` / `out_count` |
| `outputs/{bucket}/{date}/camera_registry_used.yaml` | 寫 | 本次套用的 `camera_registry.yaml` 快照，供下游以「產生此份資料時的定義」為準做驗證 |

**時區**：`tracking_results.parquet` 的 `timestamp` 已是台北在地時間（`Asia/Taipei`），
`time_bucket` 沿用之，本階段不做任何時區位移。

**重跑冪等**：`line_counts.parquet` 先寫入 `.tmp` 再 `rename` 成正式檔名，藉由 `rename`
的原子性確保中斷時不會在正式檔名下留下半成品。

## 開發

```bash
uv run --directory line_counting ruff check .   # lint（line-length = 100，select = ["E", "F", "I", "W"]）
uv run --directory line_counting pytest         # 執行測試
```

> 測試的 cwd 要求與執行 CLI 相反：這裡用 `--directory`（會 chdir 進 `line_counting/`），
> 讓 pytest 的 rootdir 解析到本套件；測試本身不碰 `bucket_dir` 與 `outputs/`。
> 等價寫法：`uv run --package line_counting pytest line_counting/tests`。
