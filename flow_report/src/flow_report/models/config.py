import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_observability import StructuredLogger

logger = StructuredLogger(component="config")


class ReportConfig(BaseModel):
    """Excel 人流報表參數。

    Attributes:
        period_minutes: 報表人流彙總的時段粒度（分鐘），需為
            input.bucket_minutes 的倍數。
        metric: 決定「人流量」「尖峰人流」用哪個統計量；只作用於區域統計，
            計數線固定用 `in_count`／`out_count`。
        on_duplicate_date: 同一天資料已存在時的處理方式。
    """

    model_config = ConfigDict(extra="forbid")

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
        bucket_minutes: 上游 `zone_counts.parquet`／`line_counts.parquet` 的
            時段粒度（分鐘）。`export_report_daily` 用它驗證
            `report.period_minutes` 是它的倍數，故須與產生這兩份 parquet 時
            的設定一致。它描述的是**輸入資料**的粒度（上游兩包寫檔時用的
            值），不是報表的呈現粒度（後者是 `report.period_minutes`），
            故放在 `[input]` 而非 `[report]`。
    """

    model_config = ConfigDict(extra="forbid")

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None
    bucket_minutes: int = Field(default=60, ge=1)


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
        report: Excel 報表參數。
    """

    # extra="forbid" 是 BaseSettings 的預設值，這裡明寫出來讓行為可見：`config.toml`
    # 出現未知的頂層區塊會直接報錯（拼錯的區塊名不會被靜默忽略）。
    model_config = SettingsConfigDict(
        toml_file=_get_toml_path(),
        env_nested_delimiter="__",
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_moved_zone_section(cls, data: Any) -> Any:
        """舊的 `[zone]` 區塊要報出「已移到哪裡」，而非通用的未知區塊訊息。

        `extra="forbid"` 本來就會擋下 `[zone]`，但訊息只說不允許額外欄位，看不出
        `bucket_minutes` 該搬到哪個區塊；沿用舊設定的人需要知道下一步怎麼改。
        `mode="before"` 先於 `extra="forbid"` 觸發，toml 與環境變數兩條路徑都會
        走到這裡。
        """
        if isinstance(data, dict) and "zone" in data:
            raise ValueError(
                "[zone] 區塊已移除：bucket_minutes 改放 [input]，由區域統計與"
                "計數線統計共用同一個上游時段粒度。請把 [zone] 底下的 "
                "bucket_minutes 移到 [input] 底下。"
            )
        return data

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
