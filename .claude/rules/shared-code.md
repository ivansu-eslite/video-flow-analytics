---
paths:
  - "libs/**"
  - "pyproject.toml"
  - "*/pyproject.toml"
  - "*/config.toml"
  - "*/src/*/models/config.py"
  - "*/tests/test_config.py"
---
# 四包共用碼的處理方式

- **`registry.py`、`structured_logging.py` → 抽成 `libs/` 共用 lib（issue #48）。**
  `libs/vfa_registry`（`camera_registry.yaml` 的 Pydantic 模型與 zone／line 驗證，四包都吃）
  與 `libs/vfa_observability`（`StructuredLogger`，四包都吃——`video_analyze` 於 issue #50
  一併改用）。四包在自己的 `pyproject.toml` 以 `[tool.uv.sources]` 的
  `{ workspace = true }` 引用（issue #56 起）。line 支援（`Line` 模型、
  `CameraEntry.lines` 欄位、`parse_and_validate_lines` 跨攝影機全域唯一驗證）於 issue #41
  加在此 lib，registry 只改這裡——三包經 workspace 依賴自動吃到 `lines` 忽略欄位相容，
  本身無需改碼。
- **`config.py`：私有區塊各包分開，`[input]` 抽成 `libs/vfa_config`（issue #79）。**
  各包只保留自己 `run_*` 實際讀到的**私有**區塊（`video_analyze` 的
  `tracker`/`model`/`foot_point`/`output`；`zone_mapping` 的 `zone`；`line_counting` 的
  `line`；`flow_report` 的 `report`）；四包都有的 `[input]` 則由 `libs/vfa_config` 提供單一
  `InputConfig`，欄位取四包需求的**聯集**（`bucket_dir`/`date`/`camera_ids`/`bucket_minutes`），
  `find_project_root`／`get_toml_path` 一併抽在此 lib。四包皆已 DDD 重構（`flow_report`
  issue #42、`zone_mapping` issue #46、`video_analyze` issue #50、`line_counting` issue #41
  沿用 `zone_mapping` 的結構建立），config 都在 `models/config.py` 並改用 pydantic-settings
  （`config.toml`＋環境變數覆寫）、以 `get_toml_path(__file__)` 定位設定檔。

  **`[input]` 為何不能比照私有區塊各包裁剪**：`env_nested_delimiter="__"` 讓頂層區塊名
  等於一段**全域**的環境變數命名空間——`INPUT__CAMERA_IDS` 對四包都是「`[input]` 的
  `camera_ids`」，而四包共用一份環境設定執行是常態、`models/config.py` 又在模組層就
  `load_config()`。裁剪的後果是別包連 import 都以 `extra_forbidden` 崩潰（抽出前
  `INPUT__CAMERA_IDS` 打死另外三包、`INPUT__BUCKET_MINUTES` 打死另外三包）。因此
  `InputConfig` 刻意含各包用不到的欄位（`video_analyze` 不讀 `bucket_minutes`、
  `flow_report` 不讀 `camera_ids`），四包各有一支區塊契約測試釘住這件事。新增頂層區塊前
  要確認該名稱在其他三包沒被用過。規則、否決過的替代方案與已知缺口見
  [ADR-008](../../docs/adr/shared/008-config-section-namespace.md)。

  `bucket_minutes` 一併從 `zone_mapping` 的 `[zone]`、`line_counting` 的 `[line]` 移進
  `[input]`（issue #79），三包共用單一環境變數 `INPUT__BUCKET_MINUTES`。**三份
  `config.toml` 仍各填一次，填不一致沒有訊號**——這是 ADR-008 明列已接受的缺口，不是待辦。

**共用 lib 存在的理由**：抽出前，`load_registry_from_path` 的 yaml 型別防呆補丁三包各自
維護、版本各自漂移——flow_report 先補（PR #45），zone_mapping 隔一個工作單元才補
（issue #46），video_analyze 直到抽 lib 前**從未補上**，空檔或純註解的 registry 在該包
會以沒有檔名線索的 `TypeError` 崩潰。改為單一 lib 後同一份實作四包共用，不再需要人工
同步；`line_counting` 直接用這份 lib，沒有再各自複製一份。

`camera_registry.yaml` 本身**只有一份**（放在 `bucket_dir`，執行時參數傳入，不進版控），
四包讀的是同一份實體檔案。此檔含 `zones`／`lines`／`participates_in_zone_mapping` 三個欄位，
即使 `video_analyze` 用不到 zone 與 line，模型也必須保留這些欄位，否則在 `extra="forbid"`
下會直接解析失敗；`video_analyze` 不呼叫 `parse_and_validate_zones`／`parse_and_validate_lines`，
因此吃完整版 lib 後 zone／line 幾何仍不會被驗證。
