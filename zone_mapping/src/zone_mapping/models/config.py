from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_config import InputConfig, get_toml_path
from vfa_observability import StructuredLogger

from zone_mapping.config.constants import DEFAULT_BOUNDARY_BAND_PX_1080P

logger = StructuredLogger(component="config")


class ZoneConfig(BaseModel):
    """Zone 人流統計參數。

    Attributes:
        boundary_band_px_1080p: entry 判定的線段區域寬度，以 1080p
            （寬 1920）為基準的像素值；執行時依各攝影機的 `frame_width` 換算成
            實際像素。`0` = 純內外判定（每次跨越邊界都計），且 0 換算後仍是 0。
            預設 25 取自實測（見 README「已知限制」）。
    """

    model_config = ConfigDict(extra="forbid")

    boundary_band_px_1080p: float = Field(
        default=DEFAULT_BOUNDARY_BAND_PX_1080P, ge=0
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_moved_bucket_minutes(cls, data: Any) -> Any:
        """`bucket_minutes` 已從 `[zone]` 移到 `[input]`，要報出新位置。

        `extra="forbid"` 本來就會擋下，但訊息只說不允許額外欄位，看不出該搬到哪個
        區塊；而環境變數形式 `ZONE__BUCKET_MINUTES` 過去是本包的合法設定，沿用舊設定
        的人需要知道改用 `INPUT__BUCKET_MINUTES`。`mode="before"` 先於
        `extra="forbid"` 觸發，toml 與環境變數兩條路徑都會走到這裡。
        """
        if isinstance(data, dict) and "bucket_minutes" in data:
            raise ValueError(
                "[zone] 的 bucket_minutes 已移到 [input]：時段粒度是四包共用的輸入"
                "參數，`flow_report` 也要對得上同一個值，故收斂成單一區塊。請把它移到"
                " [input] 底下（環境變數由 ZONE__BUCKET_MINUTES 改為 "
                "INPUT__BUCKET_MINUTES）。見 ADR-008。"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_entry_debounce_frames(cls, data: Any) -> Any:
        """舊參數名 `entry_debounce_frames` 要報出「判定方式已換」，而非通用的未知欄位。

        `extra="forbid"` 本來就會擋下舊名，但訊息只說不允許額外欄位，看不出時間去抖
        已整個被線段區域取代；沿用舊設定的人需要知道原本的「連續格數」不能直接當成
        新參數的值（單位由格數變成像素，且是 1080p 基準值）。`mode="before"` 先於
        `extra="forbid"` 觸發，toml 與環境變數兩條路徑都會走到這裡。
        """
        if isinstance(data, dict) and "entry_debounce_frames" in data:
            raise ValueError(
                "[zone] 的 entry_debounce_frames 已移除：entry 判定改用"
                "線段區域（boundary_band_px_1080p），不再用「連續 N 格都在區內」的"
                "時間去抖。新參數的單位是以 1080p（寬 1920）為基準的像素，執行時依"
                "各攝影機的影像寬度換算（1920 → ×1、3840 → ×2），原本的格數不可"
                "直接沿用。見 ADR-006。"
            )
        return data


class AppConfig(BaseSettings):
    """`config.toml` 與環境變數對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        input: 共用的 `[input]` 輸入參數（本包讀 `bucket_dir`／`date`／
            `bucket_minutes`）。
        zone: Zone 人流統計參數。
    """

    # extra="forbid" 是 BaseSettings 的預設值，這裡明寫出來讓行為可見：`config.toml`
    # 出現未知的頂層區塊會直接報錯（拼錯的區塊名不會被靜默忽略）。
    model_config = SettingsConfigDict(
        toml_file=get_toml_path(__file__),
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
    toml_path = get_toml_path(__file__)
    if toml_path is None or not Path(toml_path).exists():
        logger.warning("找不到 config.toml，將使用預設參數啟動", path=toml_path)
    return AppConfig()


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
