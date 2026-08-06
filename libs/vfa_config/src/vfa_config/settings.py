"""四包共用的 `[input]` 區塊與 `config.toml` 定位。"""

import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class InputConfig(BaseModel):
    """四包共用的 `[input]` 區塊，欄位是四包需求的**聯集**。

    `pydantic-settings` 的 `env_nested_delimiter="__"` 讓頂層區塊名等於一段全域的
    環境變數命名空間：`INPUT__CAMERA_IDS` 對四包都是「`[input]` 底下的 `camera_ids`」，
    而四包共用一份環境設定執行是常態。因此**刻意包含各包用不到的欄位**——`camera_ids`
    只有 `video_analyze` 讀、`bucket_minutes` 只有另外三包讀，但少列任何一個，設了該
    變數的環境就會讓其他包在 import 期以 `extra_forbidden` 崩潰（模組層即
    `load_config()`）。裁剪成各包實際讀到的欄位曾是四包各自的做法，代價就是這個撞名。

    各包私有的 `[tracker]`／`[model]`／`[output]`／`[zone]`／`[line]`／`[report]` 不受此
    限制：那些區塊名本來就只有一包用。取捨與命名空間規則見 ADR-008。

    Attributes:
        bucket_dir: 本機模擬 GCS bucket 的根目錄（內含 `camera_registry.yaml`）。
            `flow_report` 只取其目錄名來組出 `outputs/{bucket}/` 下的路徑。
        date: 開發時由 config 指定處理日期；正式呼叫端可直接以參數呼叫各包的
            `*_daily` 進入點。
        camera_ids: 要分析的攝影機清單，空清單代表 registry 內全部攝影機。
            **只有 `video_analyze` 讀**。
        bucket_minutes: 統計的時段粒度（分鐘）。`zone_mapping`／`line_counting` 依此
            在台北時間上向下取整 `time_bucket`，`flow_report` 用它驗證
            `report.period_minutes` 是它的倍數，故三包須填一致的值。
            **`video_analyze` 不讀**。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None
    camera_ids: list[str] = Field(default_factory=list)
    bucket_minutes: int = Field(default=60, ge=1)


def find_project_root(start_path: Path) -> Path | None:
    """從起始路徑向上搜尋，直到找到包含 `pyproject.toml` 的專案根目錄。"""
    for parent in start_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def get_toml_path(caller_file: str | Path) -> str | None:
    """由呼叫端所在檔案往上定位它自己套件根的 `config.toml`。

    各包的 `models/config.py` 位於套件根底下三層（`src/<pkg>/models/`），但改用
    `find_project_root` 往上找 `pyproject.toml`，避免寫死 `parents[N]` 在搬移後定位錯。
    抽成共用函式後起點不能再用本檔的 `__file__`（那會定位到本 lib），故由呼叫端傳入
    自己的 `__file__`。

    Args:
        caller_file: 呼叫端模組的 `__file__`。

    Returns:
        `config.toml` 的絕對路徑；連 cwd 都找不到時為 `None`（呼叫端據此警告並以
        預設值啟動）。回傳的路徑不保證存在——`find_project_root` 命中時只組出路徑，
        存在與否由呼叫端的 `load_config` 判斷並警告。
    """
    root = find_project_root(Path(caller_file).resolve())
    if root:
        return str(root / "config.toml")
    # 容器環境下 pyproject.toml 可能未一併複製，退回以 cwd 尋找 config.toml。
    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return str(cwd_config)
    return None
