# video-analyze

多路離線影片流的偵測與追蹤階段：以 YOLO（僅偵測 `person`）做偵測、ByteTrack
做多路追蹤。GPU、多進程、共享記憶體環形緩衝，是三個階段中最重的一支。

## 進入點

```
video_analyze.pipeline.analyze_daily(date, bucket_dir, camera_ids=None) -> AnalysisResult
```

CLI 進入點 `video-analyze` 從 `config.toml` 的 `[input]` 組參數再呼叫 `analyze_daily`。

## 執行方式（cwd 硬約束）

下列三者都是 **cwd 相對路徑**（非 `__file__` 相對）：

| 路徑 | 來源 | 違反 cwd 約束時 |
|---|---|---|
| `settings.input.bucket_dir` | `config.toml` `[input]` | fail-loud（找不到片段／registry） |
| `OUTPUT_ROOT = outputs/` | `pipeline.py` 常數 | 產出落在錯的樹，下游讀不到 |
| `settings.model.model_path` | `config.toml` `[model]` | **不 fail-loud**：ultralytics 找不到權重會**靜默從網路下載**一份，權重可能與預期不同卻無任何錯誤訊息 |

故本套件一律**在 repo 根目錄執行**，以 `--project` 指定套件（`uv run` 不改變 cwd）：

```bash
uv run --project video-analyze video-analyze
```

**切勿 `cd video-analyze` 後再跑**：那樣 cwd 會變成 `video-analyze/`，`bucket_name1`
會對到不存在的 `video-analyze/bucket_name1`，輸出落在 `video-analyze/outputs/`，
下游 zone-map 讀不到，且 `yolo26m.pt` 會被靜默重新下載。套件自己的 `config.toml`
因是 `__file__` 定位（`parents[2]`），不受 cwd 影響。

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
uv run --directory video-analyze pytest
```

**測試的 cwd 要求與跑 analyze 相反**：這裡要用 `--directory`（會 chdir 進
`video-analyze/`），**不能**用跑 analyze 那個 `--project`。因為 `--project` 不改變
cwd，pytest 的 rootdir 會解析到 repo 根、讀到根 `pyproject.toml` 的
`testpaths = ["tests"]`，於是去收集舊 monolith 的 `tests/` 而 collection error。
測試本身不碰 `bucket_dir`／`outputs/`，故不受 cwd 約束影響。
（等價寫法：`uv run --project video-analyze pytest video-analyze/tests`。）
