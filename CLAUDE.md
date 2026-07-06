# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

多路離線影片流分析系統：以「一天」為單位，從本機模擬 GCS bucket 的目錄結構（`bucket_name/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv`，檔名時間為 RFC 3339 UTC）讀取各攝影機的影片片段，用 YOLO（僅偵測 `person`）做偵測、ByteTrack 做多路追蹤。輸出兩種結果：追蹤明細 Parquet（`outputs/{bucket_name}/{date}/tracking_results.parquet`）與逐片段標註影片（路徑鏡射輸入、根目錄換成 `outputs/{bucket_name}/`）。攝影機清單定義在 `bucket_dir/camera_registry.yaml`（雲端輸出，不進版控）。

進入點是函式呼叫，CLI 只是從 `config.toml` 組參數再呼叫：
- `analyze.pipeline.analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult`（偵測/追蹤，重、GPU、多進程）
- `zone_mapping.pipeline.map_zones_daily(date, bucket_dir, zones_path, bucket_minutes, ...) -> Path`（zone 人流統計，輕、純運算，讀上一步的 parquet）

兩階段刻意獨立：調 `zones.yaml` 只需重跑 `zone-map`，不必重跑昂貴的 GPU 偵測。

## 常用指令

```bash
uv sync                              # 安裝依賴
uv run video-flow-analytics analyze  # 偵測/追蹤，參數讀 config.toml 的 [input]
uv run video-flow-analytics zone-map # zone 人流統計，參數讀 config.toml 的 [zone]
uv run ruff check .                  # lint（line-length=88, select=["E","F","I","W"]）
```

目前沒有測試框架（`tests/` 為空）。新增測試前先確認是否為任務需求。`YOLODetector` 用 `torch.cuda.is_available()` 判斷，不可用時 fallback CPU（明顯變慢）。

## 架構

### 套件結構

`src/video_flow_analytics/` 依職責分五個子套件，依賴方向單向、無循環：

- **`core/`**：`config.py`，Pydantic 設定模型 + 全域 `settings` 單例。不依賴其他子套件。
- **`io/`**：`video_reader.py`（逐日掃描片段、讀影格）、`video_writer.py`（標註影片輸出）、`frame_ring.py`（共享記憶體環形緩衝）。只依賴 `core`。
- **`visualization/`**：`visualizer.py`（`TrackAnnotator`，畫追蹤框）。無內部依賴。
- **`analyze/`**：`detector.py`、`tracker.py`、`inference.py`、`pipeline.py`、`tracking_results.py`、`registry.py`。依賴 `core`、`io`、`visualization`。
- **`zone_mapping/`**：`zones.py`、`stats.py`、`pipeline.py`，獨立下游功能。只依賴 `core`。

`cli.py` 是唯一進入點，依子命令 lazy import `analyze` 或 `zone_mapping`（讓 `zone-map` 不必載入 torch/ultralytics）。

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

- **zone 定義**（`zone_mapping/zones.py` → `ZoneRegistry`）：`zones.yaml`（**不進版控**，人工維護），key 為 `<location>_<camera_id>`（需對齊 parquet 的 `camera_id`）。與雲端輸出的 `camera_registry.yaml` 分開，避免人工 zone 被覆蓋。
- **`map_zones_daily`**：讀 parquet → `validate_zone_cameras` fail-loud（camera 對不上當天資料就報錯）→ 逐 camera/zone 呼叫 `stats.py` 演算法 → 寫 `zone_counts.parquet`（`.tmp` + 原子 rename）→ 快照 `zones.yaml` 成 `zones_used.yaml`。
- **`stats.py`**（純運算）：`points_in_polygon`（numpy 向量化 ray casting）判定腳底中心點 `((x1+x2)/2, y2)`；`count_zone_visits` 依 `time_bucket` 聚合 `unique_visitors`（不重複 track_id 數）與 `entries`（out→in 轉換次數，`entry_debounce_frames` 控制去抖，預設 1 = 不去抖）。

### 設定

- `core/config.py`：Pydantic 模型（`TrackerConfig`/`ModelConfig`/`OutputConfig`/`InputConfig`/`ZoneConfig`）組成 `AppConfig`，模組載入時建立全域單例 `settings`，各模組直接 import 使用（非依賴注入）。`load_config()` 找不到 repo 根目錄的 `config.toml` 就印警告、回退預設值。
- `camera_registry.yaml`（雲端輸出，描述「資料長什麼樣」）與 `config.toml`（描述「這次要怎麼跑」）分工分開。

### 其他注意事項

- `yolo26m.pt`、`bucket_name*/`、`outputs/`、`zones.yaml` 皆在 `.gitignore`，不進版控。
- 下游任務讀 `tracking_results.parquet` 的時間戳為 UTC，本地時段報表需自行轉時區（台北 +8）。
