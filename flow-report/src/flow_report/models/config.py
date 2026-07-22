import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from flow_report.observability import StructuredLogger

logger = StructuredLogger(component="config")


class ZoneConfig(BaseModel):
    """上游 zone 人流統計的參數，本包只需要其中的時段粒度。

    Attributes:
        bucket_minutes: 上游 `zone_counts.parquet` 的時段粒度（分鐘）。
            `export_report_daily` 用它驗證 `report.period_minutes` 是它的倍數，
            故須與產生該份 parquet 時的設定一致。
    """

    bucket_minutes: int = Field(default=60, ge=1)


class ReportConfig(BaseModel):
    """Excel 人流報表參數。

    Attributes:
        period_minutes: 報表人流彙總的時段粒度（分鐘），需為
            zone.bucket_minutes 的倍數。
        metric: 決定「人流量」「尖峰人流」用哪個統計量。
        on_duplicate_date: 同一天資料已存在時的處理方式。
    """

    period_minutes: int = Field(default=60, ge=1)
    metric: Literal["entries", "unique_visitors"] = "entries"
    on_duplicate_date: Literal["overwrite", "append", "error"] = "append"


class InputConfig(BaseModel):
    """`export_report_daily` 輸入參數。

    Attributes:
        bucket_dir: 本機模擬 GCS bucket 的根目錄；本包只取其目錄名來組出
            `outputs/{bucket}/` 下的輸入與輸出路徑。
        date: 開發時由 config 指定彙總日期；正式呼叫端可直接以參數呼叫
            `export_report_daily`。
    """

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None


def find_project_root(start_path: Path) -> Path | None:
    """從起始路徑向上搜尋，直到找到包含 `pyproject.toml` 的專案根目錄。"""
    for parent in start_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _get_toml_path() -> str | None:
    # 本檔位於 flow_report/models/config.py，比套件根多兩層目錄；改用
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
        input: `export_report_daily` 輸入參數。
        zone: 上游 zone 人流統計參數。
        report: Excel 報表參數。
    """

    model_config = SettingsConfigDict(
        toml_file=_get_toml_path(),
        env_nested_delimiter="__",
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
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config() -> AppConfig:
    """載入 `config.toml`（並支援環境變數覆寫）組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數啟動；設定檔存在但內容不合法時直接拋出
    `ValidationError`，不吞掉錯誤——參數錯了卻靜默套用預設值，會讓報表以非預期
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
