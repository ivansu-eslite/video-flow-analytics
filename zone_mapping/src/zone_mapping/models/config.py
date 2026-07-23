import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_observability import StructuredLogger

logger = StructuredLogger(component="config")


class ZoneConfig(BaseModel):
    """Zone 人流統計參數。

    Attributes:
        bucket_minutes: 人流統計的時段粒度（分鐘），time_bucket 依此在台北
            時間上向下取整。
        entry_debounce_frames: 連續幾格都在區域內才算一次「進入」，用來濾除
            邊界抖動；預設 1 = 不去抖。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_minutes: int = Field(default=60, ge=1)
    entry_debounce_frames: int = Field(default=1, ge=1)


class InputConfig(BaseModel):
    """`map_zones_daily` 輸入參數。

    Attributes:
        bucket_dir: 本機模擬 GCS bucket 的根目錄（內含 camera_registry.yaml，
            zone 幾何定義在各攝影機底下）。
        date: 開發時由 config 指定統計日期；正式呼叫端可直接以參數呼叫
            `map_zones_daily`。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None


def find_project_root(start_path: Path) -> Path | None:
    """從起始路徑向上搜尋，直到找到包含 `pyproject.toml` 的專案根目錄。"""
    for parent in start_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _get_toml_path() -> str | None:
    # 本檔位於 zone_mapping/models/config.py，比套件根多兩層目錄；改用
    # find_project_root 往上找 pyproject.toml，避免寫死 parents[N] 在搬移後定位錯。
    root = find_project_root(Path(__file__).resolve())
    if root:
        return str(root / "config.toml")
    # 容器環境下 pyproject.toml 可能未一併複製，退回以 cwd 尋找 config.toml。
    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return str(cwd_config)
    return None


class AppConfig(BaseSettings):
    """`config.toml` 與環境變數對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        input: `map_zones_daily` 輸入參數。
        zone: Zone 人流統計參數。
    """

    # extra="forbid" 是 BaseSettings 的預設值，這裡明寫出來讓行為可見：`config.toml`
    # 出現未知的頂層區塊會直接報錯（拼錯的區塊名不會被靜默忽略）。
    model_config = SettingsConfigDict(
        toml_file=_get_toml_path(),
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)


def load_config() -> AppConfig:
    """載入 `config.toml`（並支援環境變數覆寫）組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數啟動；設定檔存在但內容不合法時直接拋出
    `ValidationError`，不吞掉錯誤——參數錯了卻靜默套用預設值，會讓統計以非預期
    的口徑產出而無人察覺。

    Returns:
        解析後的 `AppConfig`；找不到 `config.toml` 時為預設值版本。

    Raises:
        ValidationError: `config.toml` 或環境變數提供的值不合法。
    """
    toml_path = _get_toml_path()
    if toml_path is None or not Path(toml_path).exists():
        logger.warning("找不到 config.toml，將使用預設參數啟動", path=toml_path)
    return AppConfig()


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
