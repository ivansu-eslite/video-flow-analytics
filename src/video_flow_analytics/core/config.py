import datetime
import logging
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrackerConfig(BaseModel):
    """ByteTrack 多路追蹤器參數。

    Attributes:
        tracker_type: 傳給 ultralytics `BYTETracker` 的演算法名稱欄位；本專案
            `analyze/tracker.py` 目前寫死只建構 `BYTETracker`，並未依此值切換
            實作，改動此欄位不會改變實際使用的追蹤演算法。
        track_high_thresh: 高信心度偵測框的關聯門檻。
        track_low_thresh: 低信心度偵測框的關聯門檻。
        new_track_thresh: 建立新軌跡所需的最低偵測信心度。
        track_buffer: 軌跡遺失後可保留等待重新關聯的幀數。
        match_thresh: 偵測框與既有軌跡配對的 IoU 門檻。
        gnum: 傳給 `BYTETracker` 的欄位；已於原始碼確認 `gnum` 未出現在目前安裝的
            ultralytics 套件任何地方，改動它不會影響實際追蹤行為。
        cnum: 同 `gnum`，傳入但目前未被 ultralytics 讀取。
        fuse_score: 是否將信心度分數融入 IoU 距離計算。
        gmc_method: 全域運動補償方法。
    """

    tracker_type: Literal["bytetrack"] = "bytetrack"
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
    """YOLO 偵測模型參數。

    Attributes:
        model_path: 模型權重檔路徑。
        batch: 推理批次大小。
    """

    model_path: str = "yolo26m.pt"
    batch: int = Field(default=1, ge=1)


class OutputConfig(BaseModel):
    """輸出行為參數。

    Attributes:
        save_video: 是否輸出逐片段標註影片。
    """

    save_video: bool = True


class ZoneConfig(BaseModel):
    """Zone 人流統計參數。

    Attributes:
        bucket_minutes: 人流統計的時段粒度（分鐘），time_bucket 依此在台北
            時間上向下取整。
        entry_debounce_frames: 連續幾格都在區域內才算一次「進入」，用來濾除
            邊界抖動；預設 1 = 不去抖。
    """

    bucket_minutes: int = Field(default=15, ge=1)
    entry_debounce_frames: int = Field(default=1, ge=1)


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
    on_duplicate_date: Literal["overwrite", "append", "error"] = "overwrite"


class InputConfig(BaseModel):
    """`analyze_daily` 輸入參數。

    Attributes:
        bucket_dir: 本機模擬 GCS bucket 的根目錄（內含 camera_registry.yaml
            與各攝影機片段）。
        date: 開發時由 config 指定分析日期；正式呼叫端可直接以參數呼叫
            `analyze_daily`。
        camera_ids: 要分析的攝影機清單；空清單代表 registry 內全部攝影機。
    """

    bucket_dir: str = "bucket_name"
    date: datetime.date | None = None
    camera_ids: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """`config.toml` 對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        tracker: ByteTrack 追蹤器參數。
        model: YOLO 模型參數。
        output: 輸出行為參數。
        input: `analyze_daily` 輸入參數。
        zone: Zone 人流統計參數。
        report: Excel 報表參數。
    """

    tracker: TrackerConfig
    model: ModelConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config() -> AppConfig:
    """讀取 repo 根目錄的 `config.toml` 組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數（僅 tracker/model 用預設值）回退，
    讓程式仍可啟動而非直接中止。

    Returns:
        解析後的 `AppConfig`；`config.toml` 不存在時為預設值版本。
    """
    # 本檔案位於 src/video_flow_analytics/core/config.py，即 repo 根目錄下第 3 層，
    # 故 parents[3] 對應到 repo 根目錄；搬動此檔案的目錄深度時需同步調整這個數字。
    config_path = Path(__file__).resolve().parents[3] / "config.toml"

    if not config_path.exists():
        logger.warning("找不到 %s，將使用預設參數啟動。", config_path)
        return AppConfig(tracker=TrackerConfig(), model=ModelConfig())

    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    return AppConfig(**config_data)


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
