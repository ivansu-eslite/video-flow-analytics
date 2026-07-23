# vfa-observability

輸出單行 JSON log 的 `StructuredLogger`，由 `video-analyze`／`zone-mapping`／`flow-report`
三包共用（`video-analyze` 於 issue #50 的 DDD 重構一併改用）。

## 內容

`StructuredLogger(*, component)`，方法 `.info/.warning/.error(msg, **fields)` 與
`.exception(msg, *, error)`。每筆記錄序列化成一行 JSON 印到 `stdout`（`ERROR` 以上走
`stderr`），欄位值經 `_normalize_log_value` 轉成 JSON-safe 值（Pydantic model 會
`model_dump(mode="json")`）。

`ensure_ascii=False` 是**刻意的地端差異**：本專案 log 訊息為中文，逸出成 `\uXXXX`
會讓終端機直接閱讀的維運性變差；輸出仍是合法 UTF-8 JSON。argus serverless 版用
`True` 對接 Cloud Logging。

## 使用方式

```toml
dependencies = ["vfa-observability"]

[tool.uv.sources]
vfa-observability = { path = "../libs/vfa-observability", editable = true }
```

```python
from vfa_observability import StructuredLogger

logger = StructuredLogger(component="report_builder")
```

## 已知限制

多進程下多個子進程寫同一個 fd，短行實務上不易交錯（Linux pipe `PIPE_BUF` 為 4096），
但 `.exception()` 帶完整 stacktrace 的長行可能超過而被切斷交錯。這是**用法**問題而非
本 lib 的介面問題（類別本身無狀態，沒有 handler／file handle 繼承問題）。
