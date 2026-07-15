import datetime
import logging
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class AppConfig(BaseModel):
    """`config.toml` 對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        input: `export_report_daily` 輸入參數。
        zone: 上游 zone 人流統計參數。
        report: Excel 報表參數。
    """

    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config() -> AppConfig:
    """讀取套件根目錄（`flow-report/`）的 `config.toml` 組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數回退，讓程式仍可啟動而非直接中止。

    Returns:
        解析後的 `AppConfig`；`config.toml` 不存在時為預設值版本。
    """
    # 本檔案位於 flow-report/src/flow_report/config.py，即套件根目錄下第 2 層，
    # 故 parents[2] 對應到 flow-report/；搬動此檔案的目錄深度時需同步調整這個數字。
    config_path = Path(__file__).resolve().parents[2] / "config.toml"

    if not config_path.exists():
        logger.warning("找不到 %s，將使用預設參數啟動。", config_path)
        return AppConfig()

    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    return AppConfig(**config_data)


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
