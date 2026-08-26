# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

`video-flow-analytics`（vfa）是**單一 uv workspace**（repo 根為 workspace root），由四個
成員套件組成：`video_analyze/`（YOLO+ByteTrack 偵測與多路追蹤，GPU、多進程、重）、
`zone_mapping/`（zone 區域佔用人流統計，純 CPU 向量化）、`line_counting/`（方向性計數線
進出人數統計，純 CPU 向量化）、`flow_report/`（彙總成跨日累加的 Excel，純 CPU）。每個成員
各帶自己的 `pyproject.toml`／`config.toml`／`src/`／`tests/`，彼此無跨資料夾 import；共用碼
放在 `libs/`（見「四包共用碼的處理方式」），以 `{ workspace = true }` 引用；單一 root
`uv.lock`／`.venv`。

`zone_mapping` 與 `line_counting` 的輸入相同（`tracking_results.parquet` ＋
`camera_registry.yaml`），都以落腳點做純 CPU 向量化判定，差別在幾何——zone 判「落腳點是否
落在多邊形內」（區域佔用），line 判「落腳點是否跨越計數線及其方向」（方向性進出）。

本 repo 原為單一套件 `src/video_flow_analytics/`，2026-07 拆成 `video_analyze`／`zone_mapping`／
`flow_report` 三包（issue #18），同月再收斂成 uv workspace（issue #56），其後新增
`line_counting`（issue #41）；**各套件的完整實作細節（模組結構、多進程 pipeline、fail-loud
處理、演算法、`config.toml` 完整欄位、函式介面）以各自 README 為準**，本檔只記錄跨套件、
不易從單一套件程式碼本身看出的設計決策：

- [video_analyze/README.md](video_analyze/README.md)
- [zone_mapping/README.md](zone_mapping/README.md)
- [line_counting/README.md](line_counting/README.md)
- [flow_report/README.md](flow_report/README.md)

## 常用指令

```bash
uv sync --all-packages                             # 全量同步（含 torch）

uv run --package video_analyze video_analyze       # 偵測/追蹤 → tracking_results.parquet
uv run --package zone_mapping  zone_mapping        # zone 區域佔用統計 → zone_counts.parquet
uv run --package line_counting line_counting       # 計數線進出人數統計 → line_counts.parquet
uv run --package flow_report   flow_report         # 報表彙總 → report.xlsx

uv run --directory <pkg> ruff check .              # lint；<pkg> = video_analyze / zone_mapping / line_counting / flow_report
uv run --directory <pkg> pytest                    # 測試（四包各 15／3／3／3 支測試檔）

uv run --directory libs/vfa_registry pytest        # 共用 lib 的測試（4 支）自成一套，不在四包底下
uv run --directory libs/vfa_registry ruff check .
uv run --directory libs/vfa_observability pytest   # （1 支）
uv run --directory libs/vfa_observability ruff check .
uv run --directory libs/vfa_config pytest          # （1 支）
uv run --directory libs/vfa_config ruff check .

uv run pytest                                      # 文件契約測試（根層 tests/，釘住文件裡可機械驗證的斷言）
```

**torch 隔離**：workspace 為單一 `.venv`，但 `uv sync --package <pkg>` 只裝該包依賴子樹
（`flow_report`／`zone_mapping`／`line_counting` 不含 torch）；`uv sync --all-packages` 才裝含
torch 的完整環境。部署時各容器以 `uv sync --package <pkg>` 維持 CPU-only 隔離。

**pytest／ruff 用 `--directory`（切換 cwd）而非 `--package`**：`--package` 不改變 cwd，
`pytest` 會從 repo 根遞迴收集到所有套件的測試而撞名衝突（`tests/test_config.py` 等檔名
四包重複）；`--directory` 切進該套件資料夾，才會只解析到該套件自己的 `tests/`。

**執行 cwd 約束**：`bucket_dir` 與 `OUTPUT_ROOT = Path("outputs")` 是**cwd 相對路徑**，
與各套件 `config.toml` 的檔案定位（四包 DDD 重構後皆用 `find_project_root` 往上找
`pyproject.toml`）是兩套機制。四包一律以 `--package`／`--directory` 指定套件、**在 repo
根目錄執行**（`uv run` 不改變 cwd）；若改在套件資料夾內執行，`bucket_dir` 會對到不存在
的路徑，`outputs/` 也會裂成四棵互不相通的樹，讓階段間的檔案契約失效。

## 架構

技術決策記在 [docs/adr/](docs/adr/)，依影響的模組分子目錄：只動一個套件的放
`docs/adr/<套件名>/`，跨套件的放 `docs/adr/shared/`；編號是全域流水號、與子目錄無關，
新增一律取下一號。ADR 的清單、影響範圍與各自主題見
[README.md 的「架構決策紀錄」](README.md#架構決策紀錄)（唯一索引，本檔不另列一份）。

### `tracking_results.parquet` 的影像尺寸欄位（跨套件硬性契約）

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

### 落腳點是資料欄位，不是各包各自算的公式（跨套件硬性契約）

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
  （OBB／pose／ground plane）見 [ADR-009](docs/adr/shared/009-head-based-foot-point.md)。

### 四包共用碼的處理方式

- **`registry.py`、`structured_logging.py` → 抽成 `libs/` 共用 lib（issue #48）。**
  `libs/vfa_registry`（`camera_registry.yaml` 的 Pydantic 模型與 zone／line 驗證，四包都吃）
  與 `libs/vfa_observability`（`StructuredLogger`，四包都吃——`video_analyze` 於 issue #50
  一併改用）。四包在自己的 `pyproject.toml` 以 `[tool.uv.sources]` 的
  `{ workspace = true }` 引用（issue #56 起）。line 支援（`Line` 模型、
  `CameraEntry.lines` 欄位、`parse_and_validate_lines` 跨攝影機全域唯一驗證）於 issue #41
  加在此 lib，registry 只改這裡——三包經 workspace 依賴自動吃到 `lines` 忽略欄位相容，
  本身無需改碼。
- **`config.py`：私有區塊各包分開，`[input]` 抽成 `libs/vfa_config`（issue #79）。**
  各包只保留自己 `run_*` 實際讀到的**私有**區塊（`video_analyze` 的
  `tracker`/`model`/`foot_point`/`output`；`zone_mapping` 的 `zone`；`line_counting` 的
  `line`；`flow_report` 的 `report`）；四包都有的 `[input]` 則由 `libs/vfa_config` 提供單一
  `InputConfig`，欄位取四包需求的**聯集**（`bucket_dir`/`date`/`camera_ids`/`bucket_minutes`），
  `find_project_root`／`get_toml_path` 一併抽在此 lib。四包皆已 DDD 重構（`flow_report`
  issue #42、`zone_mapping` issue #46、`video_analyze` issue #50、`line_counting` issue #41
  沿用 `zone_mapping` 的結構建立），config 都在 `models/config.py` 並改用 pydantic-settings
  （`config.toml`＋環境變數覆寫）、以 `get_toml_path(__file__)` 定位設定檔。

  **`[input]` 為何不能比照私有區塊各包裁剪**：`env_nested_delimiter="__"` 讓頂層區塊名
  等於一段**全域**的環境變數命名空間——`INPUT__CAMERA_IDS` 對四包都是「`[input]` 的
  `camera_ids`」，而四包共用一份環境設定執行是常態、`models/config.py` 又在模組層就
  `load_config()`。裁剪的後果是別包連 import 都以 `extra_forbidden` 崩潰（抽出前
  `INPUT__CAMERA_IDS` 打死另外三包、`INPUT__BUCKET_MINUTES` 打死另外三包）。因此
  `InputConfig` 刻意含各包用不到的欄位（`video_analyze` 不讀 `bucket_minutes`、
  `flow_report` 不讀 `camera_ids`），四包各有一支區塊契約測試釘住這件事。新增頂層區塊前
  要確認該名稱在其他三包沒被用過。規則、否決過的替代方案與已知缺口見
  [ADR-008](docs/adr/shared/008-config-section-namespace.md)。

  `bucket_minutes` 一併從 `zone_mapping` 的 `[zone]`、`line_counting` 的 `[line]` 移進
  `[input]`（issue #79），三包共用單一環境變數 `INPUT__BUCKET_MINUTES`。**三份
  `config.toml` 仍各填一次，填不一致沒有訊號**——這是 ADR-008 明列已接受的缺口，不是待辦。

**共用 lib 存在的理由**：抽出前，`load_registry_from_path` 的 yaml 型別防呆補丁三包各自
維護、版本各自漂移——flow_report 先補（PR #45），zone_mapping 隔一個工作單元才補
（issue #46），video_analyze 直到抽 lib 前**從未補上**，空檔或純註解的 registry 在該包
會以沒有檔名線索的 `TypeError` 崩潰。改為單一 lib 後同一份實作四包共用，不再需要人工
同步；`line_counting` 直接用這份 lib，沒有再各自複製一份。

`camera_registry.yaml` 本身**只有一份**（放在 `bucket_dir`，執行時參數傳入，不進版控），
四包讀的是同一份實體檔案。此檔含 `zones`／`lines`／`participates_in_zone_mapping` 三個欄位，
即使 `video_analyze` 用不到 zone 與 line，模型也必須保留這些欄位，否則在 `extra="forbid"`
下會直接解析失敗；`video_analyze` 不呼叫 `parse_and_validate_zones`／`parse_and_validate_lines`，
因此吃完整版 lib 後 zone／line 幾何仍不會被驗證。

### zone／line 名稱全域唯一

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
[ADR-002](docs/adr/line_counting/002-line-group-semantics.md)。

### `flow_report` 的輸入必要性看 registry，不看檔案（跨套件契約）

`flow_report` 有兩個上游輸入，**該有哪幾份由 `bucket_dir/camera_registry.yaml` 的定義
決定**：registry 裡有攝影機定義了非空 `lines` 就必須有 `line_counts.parquet`，缺檔即
fail loud；沒有任何 `lines` 定義則整批跳過出入口三個分頁、不算錯誤（zone 那側同理，判準是
`participates_in_zone_mapping` 且 `zones` 非空）。這是**這條 repo 唯一不照「下游看上游輸出
檔是否存在」原則的階段**，根 README 的階段相依原則那段已寫明這個例外；`zone_mapping`／
`line_counting` 只有單一上游、沒有這個歧義，維持看檔案。取捨與「為何刻意不留跳過用的
旗標」見 [ADR-005](docs/adr/flow_report/005-report-input-requirement-from-snapshot.md)；資料來源
為何由快照改成當下的檔案見 [ADR-007](docs/adr/shared/007-remove-registry-snapshot.md)。

後果是 `flow_report` 從此被 `line_counting` 綁住：registry 只要有任一 `lines`，沒跑
`line_counting` 就連 zone 兩頁都產不出來（整個 `export_report_daily` 中止）。排程上
`zone_mapping` 與 `line_counting` 都要排在 `flow_report` 之前。

### 時區不變量（貫穿四包）

檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`video_analyze` 解析時即轉換成台北在地時間
（`Asia/Taipei`，UTC+8）。此後 `tracking_results.parquet` 的 `timestamp`、
`zone_counts.parquet`／`line_counts.parquet` 的 `time_bucket`、`report.xlsx` 的日期／小時
欄位皆為台北在地時間，下游（`zone_mapping`／`line_counting`／`flow_report`）不需要、也不
應該再對它們做任何 UTC→+8 位移。

### `tracking_results.parquet` 不可重現（非拆分相關，屬既有特性）

`video_analyze` 的批次跨串流組成受時序影響（非阻塞輪詢湊批），加上 **ByteTrack 的
`track_id` 指派本身在重跑間會改變**（同一輸入重跑兩次即可能有數千列 key 對不上、座標差可達數百 px）。因此
`tracking_results.parquet` 逐值比對對「邏輯是否正確」沒有驗收力；`zone_counts.parquet`
經 `time_bucket` 聚合後穩定（同輸入重跑可逐值/byte 級一致），是**交付期／大重構做 golden
回歸比對**時更可靠的標的（vfa 日常改動的把關是各包 pytest、不依賴 golden；golden 產在
交付期、存放於 argus GCS）。若需驗證 `video_analyze` 的推理邏輯未被改壞，不可用固定容差比對
`tracking_results.parquet`——未改動的程式碼自身重跑就可能差上千 px；改用**控制組相對
條件**：改動後對 golden 的偏離，須不大於未改動程式碼自身重跑對同一份 golden 的偏離。
比對時 join key 用 `(camera_id, timestamp, track_id)`，不可用 `frame_id`（片段內幀序、
跨片段重複，會笛卡兒展開而算出假的大幅座標差）。

## 其他注意事項

- `*.pt`（模型權重）、`bucket_*/`、`outputs`（刻意不帶尾斜線，讓它是 symlink 時也擋得住）
  皆在 `.gitignore`，不進版控
  （`camera_registry.yaml` 含 zone／line 定義，隨 `bucket_*/` 一起不進版控）。
  `bucket_*/` 涵蓋所有 `bucket_` 開頭的目錄，`bucket_name1` 與
  `bucket_<日期>_<變體>` 兩種命名都擋得住。
- 四包版本 pin 成彼此一致（`torch`/`ultralytics`/`numpy`/`opencv` 等推理堆疊、
  `polars`/`pyarrow`/`openpyxl` 等輸出格式相關套件），避免函式庫版本漂移造成非邏輯性的
  輸出差異；`line_counting` 的 `numpy`/`polars`/`pyarrow`/`pydantic`/`pyyaml` 與 `zone_mapping`
  pin 成同版，`libs/` 底下三個 lib 的 `pydantic`／`pyyaml` 也在此範圍內。單一 root `uv.lock`
  下版本一致由 `uv lock` 自動把關——`==` pin 彼此衝突會直接讓 `uv lock` 解析失敗；新增或
  升級依賴時留意是否需要四包同步。
