import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_observability import StructuredLogger

from zone_mapping.config.constants import DEFAULT_BOUNDARY_BAND_PX_1080P

logger = StructuredLogger(component="config")


class ZoneConfig(BaseModel):
    """Zone 人流統計參數。

    Attributes:
        bucket_minutes: 人流統計的時段粒度（分鐘），time_bucket 依此在台北
            時間上向下取整。
        boundary_band_px_1080p: entry 判定的區域邊界緩衝帶寬度，以 1080p
            （寬 1920）為基準的像素值；執行時依各攝影機的 `frame_width` 換算成
            實際像素。`0` = 純內外判定（每次跨越邊界都計），且 0 換算後仍是 0。
            預設 25 取自實測（見 README「已知限制」）。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_minutes: int = Field(default=60, ge=1)
    boundary_band_px_1080p: float = Field(
        default=DEFAULT_BOUNDARY_BAND_PX_1080P, ge=0
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_entry_debounce_frames(cls, data: Any) -> Any:
        """舊參數名 `entry_debounce_frames` 要報出「判定方式已換」，而非通用的未知欄位。

        `extra="forbid"` 本來就會擋下舊名，但訊息只說不允許額外欄位，看不出時間去抖
        已整個被空間緩衝帶取代；沿用舊設定的人需要知道原本的「連續格數」不能直接當成
        新參數的值（單位由格數變成像素，且是 1080p 基準值）。`mode="before"` 先於
        `extra="forbid"` 觸發，toml 與環境變數兩條路徑都會走到這裡。
        """
        if isinstance(data, dict) and "entry_debounce_frames" in data:
            raise ValueError(
                "[zone] 的 entry_debounce_frames 已移除：entry 判定改用區域邊界"
                "緩衝帶（boundary_band_px_1080p），不再用「連續 N 格都在區內」的"
                "時間去抖。新參數的單位是以 1080p（寬 1920）為基準的像素，執行時依"
                "各攝影機的影像寬度換算（1920 → ×1、3840 → ×2），原本的格數不可"
                "直接沿用。見 ADR-006。"
            )
        return data


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
