# vfa_registry

`camera_registry.yaml` 的 Pydantic 模型與 zone 驗證，由 `video_analyze`／`zone_mapping`／
`flow_report` 三包共用。

## 內容

| 名稱 | 用途 |
| --- | --- |
| `CameraRegistry` | `camera_registry.yaml` 全檔模型；載入時驗證 `camera_id`／`stream_dirname` 不重複 |
| `CameraEntry` | 單一攝影機；`stream_dirname` 屬性對應 bucket 目錄命名 `<location>_<camera_id>` |
| `Zone` / `parsed_zones()` | 多邊形區域模型；幾何驗證刻意延後到呼叫端明確要求時 |
| `parse_and_validate_zones` | 驗證 zone 名稱**跨攝影機全域唯一**（下游報表依 zone 名稱分組、不含 camera_id）|
| `StorageConfig` | bucket 內影片片段的儲存格式參數 |
| `load_registry` / `load_registry_from_path` / `registry_path` | 讀檔；後者吃任意路徑，供讀取 `camera_registry_used.yaml` 快照 |

## 使用方式

各消費套件以 workspace 成員引用：

```toml
dependencies = ["vfa_registry"]

[tool.uv.sources]
vfa_registry = { workspace = true }
```

```python
from vfa_registry import CameraRegistry, load_registry, parse_and_validate_zones
```

依賴版本（`pydantic`／`pyyaml`）與各消費套件 pin 成一致，避免函式庫版本漂移造成非邏輯性
的輸出差異；改版時留意是否需要三包同步。

## 測試

在本 lib 目錄下執行：

```bash
uv run pytest
uv run ruff check .
```
