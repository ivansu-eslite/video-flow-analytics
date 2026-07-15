# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

多路離線影片流分析系統：以「一天」為單位，從本機模擬 GCS bucket 的目錄結構（`bucket_name/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv`）讀取各攝影機的影片片段，用 YOLO（僅偵測 `person`）做偵測、ByteTrack 做多路追蹤。**檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，`io/video_reader.py` 解析時即把它轉換成台北在地時間（`Asia/Taipei`，UTC+8）**（見該檔 `_FILENAME_TZ` / `_LOCAL_TZ`）；此後 `tracking_results.parquet` 的 `timestamp` 與 `zone_counts.parquet` 的 `time_bucket` 皆為台北在地時間，下游不需再位移。主要輸出為追蹤明細 Parquet（`outputs/{bucket_name}/{date}/tracking_results.parquet`）；逐片段標註影片（路徑鏡射輸入、根目錄換成 `outputs/{bucket_name}/`）為開發／偵錯輔助，由 `output.save_video` 控制。攝影機清單與各攝影機的 zone 定義統一寫在 `bucket_dir/camera_registry.yaml`（不進版控）。

進入點是函式呼叫，CLI 只是從 `config.toml` 組參數再呼叫：
- `analyze.pipeline.analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult`（YOLO 偵測 + ByteTrack 多路追蹤，輸出 `tracking_results.parquet`，另可選輸出偵錯用標註影片；GPU、多進程，執行成本高）
- `zone_mapping.pipeline.map_zones_daily(date, bucket_dir, bucket_minutes, ...) -> Path`（讀 `tracking_results.parquet`，把追蹤結果與 zone 結合、轉成每時段每區域的事件統計（`unique_visitors`／`entries`），輸出 `zone_counts.parquet`；純 CPU 向量化運算，不必重跑 GPU 偵測）
- `report.pipeline.export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date, bucket_minutes) -> Path`（讀 `zone_counts.parquet` 做人流統計分析與跨期彙總，持續 Append 至單一 `outputs/{bucket}/report.xlsx`（對接 Looker Studio 等 BI 工具做長期觀測）；純 CPU 運算，不需重跑偵測或 zone mapping）

三個階段刻意獨立：調 `camera_registry.yaml` 內的 zone 定義只需重跑 `zone-map`，不必重跑昂貴的 GPU 偵測；調報表參數後也只重跑 `report`。

## 常用指令

```bash
uv sync                              # 安裝依賴
uv run video-flow-analytics analyze  # 偵測/追蹤，參數讀 config.toml 的 [input]
uv run video-flow-analytics zone-map # zone 事件統計，參數讀 config.toml 的 [zone]
uv run video-flow-analytics report   # 彙總 zone 人流成 Excel 報表，參數讀 config.toml 的 [report]
uv run ruff check .                  # lint（line-length=88, select=["E","F","I","W"]）
```

已設定 pytest（`[tool.pytest.ini_options] testpaths=["tests"]`）；目前有測試的模組為 `report`（`tests/report/`）、`analyze`（`tests/analyze/test_fps_meter.py`）、`io`（`tests/io/test_video_reader.py`），`zone_mapping`／`core`／`visualization` 尚無。新增測試前先確認是否為任務需求。`YOLODetector` 用 `torch.cuda.is_available()` 判斷，不可用時 fallback CPU（明顯變慢）。

## 架構

### 套件結構

`src/video_flow_analytics/` 依職責分六個子套件，依賴方向單向、無循環：

- **`core/`**：`config.py`（Pydantic 設定模型 + 全域 `settings` 單例）、`registry.py`（`CameraRegistry`/`CameraEntry`/`Zone`，讀 `camera_registry.yaml`）。不依賴其他子套件。
- **`io/`**：`video_reader.py`（逐日掃描片段、讀影格）、`video_writer.py`（標註影片輸出）、`frame_ring.py`（共享記憶體環形緩衝）。只依賴 `core`。
- **`visualization/`**：`visualizer.py`（`TrackAnnotator`，畫追蹤框）。無內部依賴。
- **`analyze/`**：`detector.py`、`tracker.py`、`inference.py`、`pipeline.py`、`tracking_results.py`。依賴 `core`、`io`、`visualization`。
- **`zone_mapping/`**：`stats.py`、`pipeline.py`，獨立下游功能。只依賴 `core`。
- **`report/`**：`stats.py`（時區轉換、期間彙總、尖峰計算、用餐時段規則）、
  `pipeline.py`（讀 `zone_counts.parquet`、驗證、寫 Excel），獨立下游功能。
  只依賴 `core`。

`cli.py` 是唯一進入點，三個子命令 `analyze`／`zone-map`／`report` 各自 lazy import 對應模組（讓 `zone-map` 與 `report` 不必載入 torch/ultralytics）。

### 多進程 pipeline（`analyze/pipeline.py`）

`analyze_daily` 用參數傳入的 `bucket_dir`（而非全域 `settings`，讓它可重複以不同 bucket 呼叫）：讀 registry 解析攝影機清單 → 主進程 `discover_segments` 掃出當天片段清單、`probe_frame_shape` 讀首格解析度 → `multiprocessing`（fork）拆成 N 個讀取進程 + 1 個推理進程。

- **影格走共享記憶體、不走 pickle**（`io/frame_ring.py`）：每路一塊 `RING_SLOTS` 格的環形緩衝（`mp.RawArray`），queue 只傳 slot 索引，避免每格 6MB 影格逐格 pickle 進 `mp.Queue` 的高成本。**假設同一攝影機整天解析度固定**（尺寸不符會在 `write_slot` 拋 `ValueError`）。
- **讀取進程**（`DailyStreamVideoReader`）：無空 slot 時阻塞，形成對推理進程的天然背壓。**時間戳 = 該片段檔名時間 + 片段內幀序/fps**（逐段算，不能用全日累計幀數推）。整天讀完送 `None`；中途例外送 `READER_FAILED`（見下方錯誤處理）。
- **推理進程**（`InferencePipeline.start_loop`）：`_collect_batch` 非阻塞輪詢各路 `data_queue` 湊批（目標 `model.batch × 2`，不足等 `_FILL_MAX_WAIT` 再送），維持 GPU 批次效率。每個 packet 依序：`detector.predict` → `MultiStreamByteTracker.update`（每路各自獨立 `BYTETracker` 實例，`track_id` 跨片段延續）→ `TrackingResultCollector.add` → `TrackAnnotator.draw_bboxes` → `MultiStreamVideoWriter.write`。
- **mp4v 編碼在背景執行緒**：`write()` 只入列、不 inline 編碼，與下一批 GPU 推理重疊。**關檔順序有講究**：某路讀到 `None` 常與該路尾端幾格同批出現，必須等這批全部 `write()` 完才呼叫 `close_stream`，否則背景緒會先收尾關檔、之後補寫的影格重開檔案把它截斷。

`TrackingResultCollector` 累積到 20 萬列就 flush 成一個 parquet row group，全部串流結束後 `save()` 把 `.tmp` 原子性 `rename` 成正式檔名；中途例外改 `discard()` 刪 `.tmp`，正式檔名下不會出現不完整的 parquet。

### 錯誤處理（fail-loud）

- 檔名格式錯誤 → `discover_segments` 在主進程直接拋 `ValueError`（子進程尚未啟動）。片段開檔/讀 FPS 失敗 → 讀取子進程拋 `ValueError`、以非零 exitcode 結束。`MultiStreamVideoWriter._open_writer` 開檔失敗（`cv2.VideoWriter.isOpened()` 為 False）→ 背景緒記錄例外，主緒在下次 `write()`/`close_all()` 時重拋。
- `analyze_daily` 以 0.5 秒輪詢所有子進程存活狀態；任一進程非零結束 → `_terminate_all` 後拋 `RuntimeError`；`KeyboardInterrupt` → `_terminate_all` 後重拋、`main()` 以 exit code 130 收斂。`_terminate_all`：先 `terminate()`，5 秒後仍存活再 `kill()`。

### Zone Mapping

- **zone 定義**（`core/registry.py` → `CameraEntry.zones: list[Any]`）：與攝影機身份一起寫在 `camera_registry.yaml`（**不進版控**，人工維護），以 `CameraEntry.stream_dirname`（`<location>_<camera_id>`）對齊 parquet 的 `camera_id`。刻意不拆成獨立檔案，因為此專案沒有真的雲端同步流程會覆蓋這份人工維護的檔案。**`zones` 刻意留在未經驗證的原始形式，不在 `CameraEntry` 上驗證幾何**：`CameraEntry` 也被 `analyze_daily`（重、GPU 路徑）透過 `load_registry` 讀取，若在此驗證 zone 內容，zone 定義打錯字（如頂點數 <3、整段結構寫錯）會連帶讓不需要 zone 的 `analyze_daily` 也失敗。`CameraEntry.parsed_zones()` 才把原始資料解析、驗證成 `Zone` model（含同攝影機內 zone name 不可重複），只在 `zone_mapping` 真正需要 zone 幾何時呼叫。`CameraRegistry` 另外對 `camera_id` 與 `stream_dirname` 都做唯一性驗證（fail-loud，避免重複登錄的攝影機讓 `resolve_cameras`／zone mapping 的查詢字典靜默覆蓋其中一筆）。**`CameraEntry.participates_in_zone_mapping: bool`（預設 `True`）**是該攝影機是否參與 zone mapping 的正式訊號；`False` 時 `map_zones_daily` 直接跳過，不受 `zones` 內容影響——取代舊版「`zones` 空清單代表不參與」的隱含推斷。
- **`map_zones_daily`**：讀 parquet → `load_registry` 篩出 `participates_in_zone_mapping=True` 的攝影機 → `validate_zone_cameras` fail-loud（先比對 camera 對不上當天資料，不必等 zone 幾何解析完才報錯）→ 通過後才呼叫 `core/registry.py.parse_and_validate_zones` 逐台解析 zone 幾何、同時驗證跨攝影機 zone 名稱唯一（見下方 Report 段落，`report` 也呼叫同一函式）→ 逐 camera/zone 呼叫 `stats.py` 演算法 → 寫 `zone_counts.parquet`（先寫 `.tmp`，完成後 `rename`；`rename` 具原子性）→ 快照 `camera_registry.yaml` 成 `camera_registry_used.yaml`。
- **`stats.py`**（純運算）：`points_in_polygon`（numpy 向量化 ray casting）判定腳底中心點 `((x1+x2)/2, y2)`；`count_zone_visits` 依 `time_bucket` 聚合 `unique_visitors`（不重複 track_id 數）與 `entries`（out→in 轉換次數，`entry_debounce_frames` 控制去抖，預設 1 = 不去抖）。

### Report（Excel 人流報表）

- **輸出**：`outputs/{bucket}/report.xlsx`，是跨日累加更新的單一檔案（不像
  `zone_counts.parquet` 逐日各一份）——刻意持續 **Append** 至同一檔，以對接
  **Looker Studio 等 BI 工具**做長期資料觀測與視覺化。含三個分頁：「每小時人流」
  「每日尖峰」「活動事件」。「活動事件」目前只建標題列、由其他來源填入，
  `export_report_daily` 不會動這個分頁。
- **zone 名稱全域唯一（新前提）**：報表的「區域」欄位以 zone 名稱分組、不含
  camera_id，因此要求整份 `camera_registry.yaml` 的 zone 名稱**跨攝影機也不可
  重複**（原本 `parsed_zones()` 只驗證同一攝影機內不重複）。此驗證的唯一實作
  是 `core/registry.py.parse_and_validate_zones`，`zone_mapping.map_zones_daily`
  與 `report.export_report_daily` 都會呼叫它，因此**也會影響 `zone_mapping`**：
  即使當天不會產生報表，`zone-map` 本身也會擋下跨攝影機重複的 zone 命名。
  未來若有 UI 維護 `camera_registry.yaml`，會在該處即時擋下重複命名。**驗證對象
  是產生該日 `zone_counts.parquet` 當時的
  `camera_registry_used.yaml` 快照，而非「當下」的 `camera_registry.yaml`**：
  若兩者之間改過 zone 名稱，用即時檔案驗證會通過，但 parquet 裡的 zone 名稱其實
  是舊定義，可能讓不同攝影機的人流被靜默合併。
- **時區**：檔名為真正的 UTC，已於 `io/video_reader.py` 解析時轉換成台北在地
  時間（見專案概述段落與該檔 `_LOCAL_TZ`），`tracking_results.parquet` 的
  `timestamp` 與 `zone_counts.parquet` 的 `time_bucket` 皆為台北在地時間，
  報表不需要、也不應該再對它們做任何 UTC→+8 的時區位移。**因錄影時段固定為
  台北 11:00–22:00（即 UTC 03:00–14:00），一個 UTC 日資料夾偏移後仍落在同一
  台北曆日**，報表算出來的日期／小時會與 `zone_counts.parquet` 所在的日期資料夾
  一致，不會跨曆日。
- **`on_duplicate_date`**（`config.toml` 的 `[report]`）：重跑同一天時的處理
  方式，`overwrite` 刪除既有相同日期的列後插入新列，`append`（預設）直接加到
  尾端不檢查，`error` 發現重複日期就整個中止不寫入。
- **`metric='unique_visitors'` 的彙總是近似值**：`rollup_by_period` 對
  `unique_visitors` 一律用 `sum()` 把多個 `bucket_minutes` 彙總成
  `period_minutes`，但 `unique_visitors` 本身是「該 bucket 內不重複人數」，
  若同一人跨越多個相鄰 bucket 停留，會在彙總後被重複計入。`zone_counts.parquet`
  未保留原始 `track_id`，`report` 這層無法在彙總時消除重複計數。`metric='entries'`
  不受影響（本身即為可疊加的事件次數）。

### 設定

- `core/config.py`：Pydantic 模型（`TrackerConfig`/`ModelConfig`/`OutputConfig`/`InputConfig`/`ZoneConfig`）組成 `AppConfig`，模組載入時建立全域單例 `settings`，各模組直接 import 使用（非依賴注入）。`load_config()` 找不到 repo 根目錄的 `config.toml` 就印警告、回退預設值。
- `camera_registry.yaml`（描述「資料長什麼樣＋zone 定義」）與 `config.toml`（描述「這次要怎麼跑」）分工分開。

### 其他注意事項

- `yolo26m.pt`、`bucket_name*/`、`outputs/` 皆在 `.gitignore`，不進版控（`camera_registry.yaml` 含 zone 定義，隨 `bucket_name*/` 一起不進版控）。
- `tracking_results.parquet` 的 `timestamp`／`zone_counts.parquet` 的 `time_bucket`
  皆為台北在地時間（`Asia/Taipei`；檔名的 UTC 已於解析時轉換，見專案概述段落），
  下游不需要、也不應該再對它們做 UTC→台北 +8 的時區位移。
