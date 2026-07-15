import datetime
import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ZoneConfig(BaseModel):
    """Zone 人流統計參數。

    Attributes:
        bucket_minutes: 人流統計的時段粒度（分鐘），time_bucket 依此在台北
            時間上向下取整。
        entry_debounce_frames: 連續幾格都在區域內才算一次「進入」，用來濾除
            邊界抖動；預設 1 = 不去抖。
    """

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

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None


class AppConfig(BaseModel):
    """`config.toml` 對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        input: `map_zones_daily` 輸入參數。
        zone: Zone 人流統計參數。
    """

    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)


def load_config() -> AppConfig:
    """讀取套件根目錄（`zone-mapping/`）的 `config.toml` 組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數回退，讓程式仍可啟動而非直接中止。

    Returns:
        解析後的 `AppConfig`；`config.toml` 不存在時為預設值版本。
    """
    # 本檔案位於 zone-mapping/src/zone_mapping/config.py，即套件根目錄下第 2 層，
    # 故 parents[2] 對應到 zone-mapping/；搬動此檔案的目錄深度時需同步調整這個數字。
    config_path = Path(__file__).resolve().parents[2] / "config.toml"

    if not config_path.exists():
        logger.warning("找不到 %s，將使用預設參數啟動。", config_path)
        return AppConfig()

    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    return AppConfig(**config_data)


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
