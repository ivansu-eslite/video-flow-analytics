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

from line_counting.config.constants import DEFAULT_CROSSING_BAND_PX_1080P

logger = StructuredLogger(component="config")


class LineConfig(BaseModel):
    """計數線進出人數統計參數。

    Attributes:
        bucket_minutes: 進出人數統計的時段粒度（分鐘），time_bucket 依此在台北
            時間上向下取整。
        crossing_band_px_1080p: 跨越去抖的線段區域寬度，以 1080p（寬 1920）為
            基準的像素值；執行時依各攝影機的 `frame_width` 換算成實際像素。
            `0` = 細線純零交越（每次幾何跨越都計），且 0 換算後仍是 0。
            預設 25 取自實測（見 README「已知限制」）。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_minutes: int = Field(default=60, ge=1)
    crossing_band_px_1080p: float = Field(
        default=DEFAULT_CROSSING_BAND_PX_1080P, ge=0
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_crossing_band_px(cls, data: Any) -> Any:
        """舊參數名 `crossing_band_px` 要報出「已改名且語義變了」，而非通用的未知欄位。

        `extra="forbid"` 本來就會擋下舊名，但訊息只說不允許額外欄位，看不出這個值
        現在是 1080p 基準值（4K 攝影機上實際的線段區域寬度會是兩倍）；沿用舊設定的人需要知道
        該不該改數值，不是只知道欄位名不對。`mode="before"` 先於 `extra="forbid"`
        觸發，toml 與環境變數兩條路徑都會走到這裡。
        """
        if isinstance(data, dict) and "crossing_band_px" in data:
            raise ValueError(
                "[line] 的 crossing_band_px 已更名為 crossing_band_px_1080p，"
                "語義同時改為「以 1080p（寬 1920）為基準的像素值」，執行時依各攝影機的"
                "影像寬度換算（1920 → ×1、3840 → ×2）。請改用新欄位名，並確認原本的"
                "數值是針對哪個解析度調出來的。"
            )
        return data


class InputConfig(BaseModel):
    """`count_lines_daily` 輸入參數。

    Attributes:
        bucket_dir: 本機模擬 GCS bucket 的根目錄（內含 camera_registry.yaml，
            計數線幾何定義在各攝影機底下）。
        date: 開發時由 config 指定統計日期；正式呼叫端可直接以參數呼叫
            `count_lines_daily`。
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
    # 本檔位於 line_counting/models/config.py，比套件根多兩層目錄；改用
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
        input: `count_lines_daily` 輸入參數。
        line: 計數線進出人數統計參數。
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
    line: LineConfig = Field(default_factory=LineConfig)


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
