# video-analyze

多路離線影片流的偵測與追蹤階段：以 YOLO（僅偵測 `person`）做偵測、ByteTrack
做多路追蹤。GPU、多進程、共享記憶體環形緩衝，是三個階段中最重的一支。

## 進入點

```
video_analyze.pipeline.analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult
```

CLI 進入點 `video-analyze` 從 `config.toml` 的 `[input]` 組參數再呼叫 `analyze_daily`。

## 執行方式（cwd 硬約束）

`bucket_dir` 與輸出根目錄 `OUTPUT_ROOT = outputs/` 都是 **cwd 相對路徑**（非
`__file__` 相對）。本套件一律**在 repo 根目錄執行**，以 `--project` 指定套件，
`uv run` 不改變 cwd：

```bash
uv run --project video-analyze video-analyze
```

**切勿 `cd video-analyze` 後再跑**：那樣 cwd 會變成 `video-analyze/`，`bucket_name1`
會對到不存在的 `video-analyze/bucket_name1`，且輸出落在 `video-analyze/outputs/`，
下游 zone-map 讀不到。套件自己的 `config.toml` 因是 `__file__` 定位（`parents[2]`），
不受 cwd 影響。

## 檔案契約

- **讀**：`bucket_dir/<location>_<camera_id>/{YYYY}/{MM}/{DD}/{HHmmss}.{SSS}Z.mkv`
  影片片段，以及 `bucket_dir/camera_registry.yaml`（攝影機清單；本套件只讀身份，
  不解析 zone 幾何）。檔名的 `Z` 尾綴依 RFC 3339 為真正的 UTC，解析時即轉成台北
  在地時間（`Asia/Taipei`）。
- **寫**（下游 zone-map 的輸入）：
  `outputs/{bucket}/{date}/tracking_results.parquet`；`timestamp` 為台北在地時間。
  另可由 `[output] save_video` 控制輸出逐片段標註影片（開發／偵錯輔助）。

## 設定

`config.toml` 只含 `[tracker]`／`[model]`／`[output]`／`[input]` 四個區塊
（zone／report 屬其他兩個套件）。`[input] bucket_dir` 預設為 `bucket_name1`
（5.6G 的測試 fixture），非 112G 的 `bucket_name`。

## 測試

```bash
uv run --project video-analyze pytest
```
