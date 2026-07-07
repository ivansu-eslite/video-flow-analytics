import datetime
import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrackerConfig(BaseModel):
    tracker_type: str = "bytetrack"
    track_high_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    gnum: int = 1
    cnum: int = 1
    fuse_score: bool = True
    gmc_method: str = "none"


class ModelConfig(BaseModel):
    model_path: str = "yolo26m.pt"
    batch: int = Field(default=1, ge=1)


class OutputConfig(BaseModel):
    save_video: bool = True


class ZoneConfig(BaseModel):
    # 人流統計的時段粒度（分鐘），time_bucket 依此在台北時間上向下取整
    bucket_minutes: int = Field(default=15, ge=1)
    # 連續幾格都在區域內才算一次「進入」，用來過濾邊界抖動造成的假進入；
    # 預設 1 = 不去抖（一格在內就算一次進入）
    entry_debounce_frames: int = Field(default=1, ge=1)


class InputConfig(BaseModel):
    # 本機模擬 GCS bucket 的根目錄（內含 camera_registry.yaml 與各攝影機片段）
    bucket_dir: str = "bucket_name"
    # 開發時由 config 指定分析日期；正式呼叫端可直接以參數呼叫 analyze_daily
    date: datetime.date | None = None
    # 空清單 = registry 內全部攝影機
    camera_ids: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    tracker: TrackerConfig
    model: ModelConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)


def load_config() -> AppConfig:
    # 此檔案為 src/video_flow_analytics/core/config.py，parents[3] 對應到 repo 根目錄
    config_path = Path(__file__).resolve().parents[3] / "config.toml"

    if not config_path.exists():
        logger.warning("找不到 %s，將使用預設參數啟動。", config_path)
        return AppConfig(tracker=TrackerConfig(), model=ModelConfig())

    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    return AppConfig(**config_data)


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
