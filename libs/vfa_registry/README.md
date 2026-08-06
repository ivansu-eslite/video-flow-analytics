# vfa_registry

`camera_registry.yaml` 的 Pydantic 模型與 zone／line 驗證，由 `video_analyze`／
`zone_mapping`／`line_counting`／`flow_report` 四包共用。

## 內容

| 名稱 | 用途 |
| --- | --- |
| `CameraRegistry` | `camera_registry.yaml` 全檔模型；載入時驗證 `camera_id`／`stream_dirname` 不重複 |
| `CameraEntry` | 單一攝影機；`stream_dirname` 屬性對應 bucket 目錄命名 `<location>_<camera_id>` |
| `Zone` / `parsed_zones()` | 多邊形區域模型；幾何驗證刻意延後到呼叫端明確要求時 |
| `parse_and_validate_zones` | 驗證 zone 名稱**跨攝影機全域唯一**（下游報表依 zone 名稱分組、不含 camera_id）|
| `Line` / `parsed_lines()` | 計數線模型（`points` polyline ＋ `inside_point` 方向參考點 ＋ `line_group`）；幾何驗證同樣延後到呼叫端 |
| `parse_and_validate_lines` | 驗證 line 名稱**跨攝影機全域唯一**，理由同 zone；**`line_group` 刻意不驗**，跨攝影機同名正是分組用途（見 ADR-002）|
| `StorageConfig` | bucket 內影片片段的儲存格式參數 |
| `load_registry` / `load_registry_from_path` / `registry_path` | 讀檔；四包一律用 `load_registry(bucket_dir)`，`load_registry_from_path` 吃任意路徑，供 registry 不在 `bucket_dir` 底下的呼叫端（例如部署端的共用 registry 目錄）|

## 使用方式

各消費套件以 workspace 成員引用：

```toml
dependencies = ["vfa_registry"]

[tool.uv.sources]
vfa_registry = { workspace = true }
```

```python
from vfa_registry import (
    CameraRegistry,
    load_registry,
    parse_and_validate_lines,
    parse_and_validate_zones,
)
```

`zone_mapping` 與 `flow_report` 呼叫 `parse_and_validate_zones`，`line_counting` 與
`flow_report` 呼叫 `parse_and_validate_lines`；`video_analyze` 兩者都不呼叫，故在該包
zone／line 幾何不會被驗證（模型仍須保留這些欄位，否則 `extra="forbid"` 下會解析失敗）。

依賴版本（`pydantic`／`pyyaml`）與各消費套件 pin 成一致，避免函式庫版本漂移造成非邏輯性
的輸出差異；改版時留意是否需要四包同步。

## 測試

在本 lib 目錄下執行：

```bash
uv run pytest
uv run ruff check .
```
