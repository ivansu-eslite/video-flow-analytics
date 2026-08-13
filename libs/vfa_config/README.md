# vfa_config

四包共用的 `[input]` 設定區塊與 `config.toml` 定位，由 `video_analyze`／`zone_mapping`／
`line_counting`／`flow_report` 四包共用。

## 內容

| 名稱 | 用途 |
| --- | --- |
| `InputConfig` | `[input]` 區塊的唯一定義；欄位是四包需求的**聯集**（`bucket_dir`／`date`／`camera_ids`／`bucket_minutes`），各包一律吃完整版 |
| `find_project_root` | 從起始路徑向上找含 `pyproject.toml` 的目錄，取代寫死的 `parents[N]` |
| `get_toml_path` | 由呼叫端傳入自己的 `__file__`，定位該套件根的 `config.toml`；找不到套件根時退回 cwd |

## 為什麼 `[input]` 不能各包裁剪

`pydantic-settings` 的 `env_nested_delimiter="__"` 讓**頂層區塊名等於一段全域的環境變數
命名空間**：`INPUT__CAMERA_IDS` 對四包都是「`[input]` 底下的 `camera_ids`」。四包共用一份
環境設定執行是常態，而各包的 `models/config.py` 在模組層就 `load_config()`，所以某一包的
`[input]` 少列一個別包有的欄位，設了該變數的環境會讓這包連 `import` 都以
`extra_forbidden` 失敗。抽出前四包各自裁剪，`INPUT__CAMERA_IDS` 會打死另外三包、
`INPUT__BUCKET_MINUTES` 會打死另外三包（`flow_report` 自己的錯誤訊息正好叫維運人員去設
這個變數）。取捨與命名空間規則見 [ADR-008](../../docs/adr/shared/008-config-section-namespace.md)。

各包私有的 `[tracker]`／`[model]`／`[zone]`／`[line]`／`[report]` **不放進本
lib**：那些區塊名本來就只有一包用，沒有撞名面。

## 使用方式

各消費套件以 workspace 成員引用：

```toml
dependencies = ["vfa_config"]

[tool.uv.sources]
vfa_config = { workspace = true }
```

```python
from vfa_config import InputConfig, find_project_root, get_toml_path


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=get_toml_path(__file__),
        env_nested_delimiter="__",
        extra="forbid",
    )

    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)   # 各包私有區塊照舊
```

`get_toml_path` 必須傳呼叫端的 `__file__`；傳本 lib 的檔案會定位到 `libs/vfa_config/`。

依賴版本（`pydantic`）與各消費套件 pin 成一致，避免函式庫版本漂移造成非邏輯性的行為差異；
改版時留意是否需要四包同步。

## 測試

在本 lib 目錄下執行：

```bash
uv run pytest
uv run ruff check .
```
