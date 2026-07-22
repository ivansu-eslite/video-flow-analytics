# vfa-registry

`camera_registry.yaml` 的 Pydantic 模型與 zone 驗證，由 `video-analyze`／`zone-mapping`／
`flow-report` 三包共用。

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

三包以 path 依賴（editable）引用，不經 uv workspace（理由見 repo 根 `CLAUDE.md`
「三包共用碼的處理方式」）：

```toml
dependencies = ["vfa-registry"]

[tool.uv.sources]
vfa-registry = { path = "../libs/vfa-registry", editable = true }
```

```python
from vfa_registry import CameraRegistry, load_registry, parse_and_validate_zones
```

依賴版本（`pydantic`／`pyyaml`）與三個消費套件 pin 成一致；改版時三處要一起改，
否則消費端解析 lock 會撞版本衝突。

## 測試

```bash
uv run --directory libs/vfa-registry pytest
uv run --directory libs/vfa-registry ruff check .
```
