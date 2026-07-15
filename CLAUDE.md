# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

`video-flow-analytics` 是三個**各自獨立的 uv 專案**（非單一套件）：`video-analyze/`
（YOLO+ByteTrack 偵測與多路追蹤，GPU、多進程、重）、`zone-mapping/`（zone 人流統計，純
CPU 向量化）、`flow-report/`（彙總成跨日累加的 Excel，純 CPU）。每包各帶自己的
`pyproject.toml`／`config.toml`／`uv.lock`／`src/`／`tests/`，彼此無跨資料夾 import。

本 repo 原為單一套件 `src/video_flow_analytics/`，2026-07 拆成上述三包，拆分背景、共用碼
處理方式與設計取捨見
[docs/plans/2026-07-15-split-three-packages.md](docs/plans/2026-07-15-split-three-packages.md)
（issue #18）；**各套件的完整實作細節（模組結構、多進程 pipeline、fail-loud 處理、
演算法、`config.toml` 完整欄位、函式介面）以各自 README 為準**，本檔只記錄跨套件、不易
從單一套件程式碼本身看出的設計決策：

- [video-analyze/README.md](video-analyze/README.md)
- [zone-mapping/README.md](zone-mapping/README.md)
- [flow-report/README.md](flow-report/README.md)

## 常用指令

```bash
uv sync --project video-analyze && uv sync --project zone-mapping && uv sync --project flow-report

uv run --project video-analyze video-analyze   # 偵測/追蹤 → tracking_results.parquet
uv run --project zone-mapping  zone-mapping    # zone 事件統計 → zone_counts.parquet
uv run --project flow-report   flow-report     # 報表彙總 → report.xlsx

uv run --directory <pkg> ruff check .          # lint；<pkg> = video-analyze / zone-mapping / flow-report
uv run --directory <pkg> pytest                # 測試（zone-mapping 目前尚無既有測試）
```

**執行 cwd 約束**：`bucket_dir` 與 `OUTPUT_ROOT = Path("outputs")` 是**cwd 相對路徑**，
與各套件 `config.toml` 的 `__file__` 定位是兩套機制。三包一律以 `--project`／
`--directory` 指定套件、**在 repo 根目錄執行**（`uv run` 不改變 cwd）；若改在套件資料夾
內執行，`bucket_dir` 會對到不存在的路徑，`outputs/` 也會裂成三棵互不相通的樹，讓階段間
的檔案契約失效。

## 架構

### 三包共用碼的處理方式

三者共用的程式碼只有原本 `core/` 底下兩支：

- **`config.py`**：好切，各包只保留自己 `run_*` 實際讀到的區塊（`video-analyze` 保留
  `tracker`/`model`/`output`/`input`；`zone-mapping` 保留 `input`/`zone`；`flow-report`
  保留 `input`/`zone.bucket_minutes`/`report`）。
- **`registry.py`**：複製兩種形狀後各自維護——`video-analyze` 是精簡版（無 zone 幾何，
  但仍需保留 `zones: list[Any]` 忽略欄位，因為 `CameraEntry` 用 `extra="forbid"`）；
  `zone-mapping`／`flow-report` 各一份完整版（含 `Zone`／`parse_and_validate_zones`），
  內容相同、刻意重複而非共用 lib——三個階段未來可能各奔不同平台，共用 lib 會在其中一個
  移走時斷裂。**改動任一份 `registry.py` 的驗證邏輯時，需同步檢查另外兩份是否也該同步
  改，三份目前並無自動同步機制。**

`camera_registry.yaml` 本身**只有一份**（放在 `bucket_dir`，執行時參數傳入，不進版控），
三包讀的是同一份實體檔案，資料層面無重複；上述重複只發生在讀它的 Pydantic 模型層。此檔
含 `zones`／`participates_in_zone_mapping` 兩個欄位，即使 `video-analyze` 用不到 zone，
精簡 registry 也必須保留這兩個欄位（忽略其值即可），否則在 `extra="forbid"` 下會直接
解析失敗。

### zone 名稱全域唯一

`zone-mapping` 與 `flow-report` 的報表都以 zone 名稱（不含 `camera_id`）分組彙總，因此
`camera_registry.yaml` 的 zone 名稱**跨攝影機也不可重複**（非僅同一攝影機內）。此驗證
的實作是各包自己那份 `registry.py` 裡的 `parse_and_validate_zones`——`zone-mapping` 與
`flow-report` 都會呼叫，**即使當天不會產生報表，`zone-mapping` 本身也會擋下跨攝影機重複
的 zone 命名**。`flow-report` 驗證的對象是產生該日 `zone_counts.parquet` 當時的
`camera_registry_used.yaml` **快照**，而非「當下」的 `camera_registry.yaml`——若兩者之間
改過 zone 名稱，用即時檔案驗證會通過，但 parquet 裡的 zone 名稱其實是舊定義，可能讓不同
攝影機的人流被靜默合併。

### 時區不變量（貫穿三包）

檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`video-analyze` 解析時即轉換成台北在地時間
（`Asia/Taipei`，UTC+8）。此後 `tracking_results.parquet` 的 `timestamp`、
`zone_counts.parquet` 的 `time_bucket`、`report.xlsx` 的日期／小時欄位皆為台北在地時間，
下游（`zone-mapping`／`flow-report`）不需要、也不應該再對它們做任何 UTC→+8 位移。

### `tracking_results.parquet` 不可重現（非拆分相關，屬既有特性）

`video-analyze` 的批次跨串流組成受時序影響（非阻塞輪詢湊批）、`bucket_name1` fixture 為
混解析度會讓 letterbox 隨批次組成變動，加上 **ByteTrack 的 `track_id` 指派本身在重跑間
會改變**（同一輸入重跑兩次即可能有數千列 key 對不上、座標差可達數百 px）。因此
`tracking_results.parquet` 逐值比對對「邏輯是否正確」沒有驗收力；`zone_counts.parquet`
經 `time_bucket` 聚合後穩定（同輸入重跑可逐值/byte 級一致），是更可靠的驗收與回歸比對
標的。詳細實測見上方拆分計畫文件的「0a 先探測可重現性」一節。

## 其他注意事項

- `yolo26m*.pt`、`bucket_name*/`、`outputs/` 皆在各包 `.gitignore`，不進版控
  （`camera_registry.yaml` 含 zone 定義，隨 `bucket_name*/` 一起不進版控）。
- 三包版本 pin 成彼此一致（`torch`/`ultralytics`/`numpy`/`opencv` 等推理堆疊、
  `polars`/`pyarrow`/`openpyxl` 等輸出格式相關套件），避免函式庫版本漂移造成非邏輯性的
  輸出差異；新增或升級依賴時留意是否需要三包同步。
