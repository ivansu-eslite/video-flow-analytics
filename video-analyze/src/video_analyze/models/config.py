import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_observability import StructuredLogger

logger = StructuredLogger(component="config")


class TrackerConfig(BaseModel):
    """ByteTrack 多路追蹤器參數。

    Attributes:
        track_high_thresh: 高信心度偵測框的關聯門檻。
        track_low_thresh: 低信心度偵測框的關聯門檻。
        new_track_thresh: 建立新軌跡所需的最低偵測信心度。
        track_buffer: 軌跡遺失後可保留等待重新關聯的幀數。
        match_thresh: 偵測框與既有軌跡配對的 IoU 門檻。
        fuse_score: 是否將信心度分數融入 IoU 距離計算。
        gmc_method: 全域運動補償方法。
    """

    track_high_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    fuse_score: bool = True
    gmc_method: str = "none"


class ModelConfig(BaseModel):
    """YOLO 偵測模型參數。

    Attributes:
        model_path: 模型權重檔路徑。
        batch: 推理批次大小。
        classes: 要保留的偵測類別 id 清單，對應權重的類別定義。
    """

    model_path: str = "20260714-153811_yolo26m_baseline.pt"
    batch: int = Field(default=1, ge=1)
    classes: list[int] = Field(default_factory=lambda: [2], min_length=1)


class OutputConfig(BaseModel):
    """輸出行為參數。

    Attributes:
        save_video: 是否輸出逐片段標註影片。
    """

    save_video: bool = False


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


def find_project_root(start_path: Path) -> Path | None:
    """從起始路徑向上搜尋，直到找到包含 `pyproject.toml` 的專案根目錄。"""
    for parent in start_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _get_toml_path() -> str | None:
    # 本檔位於 video_analyze/models/config.py，比套件根多兩層目錄；改用
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
        tracker: ByteTrack 追蹤器參數。
        model: YOLO 模型參數。
        output: 輸出行為參數。
        input: `analyze_daily` 輸入參數。
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

    # tracker/model 給 default_factory，讓找不到 config.toml 時 `AppConfig()` 仍能以
    # 全預設值啟動（兩者本就無必填參數）；與另兩包的 load_config 一致。
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)


def load_config() -> AppConfig:
    """載入 `config.toml`（並支援環境變數覆寫）組成 `AppConfig`。

    找不到設定檔時記錄警告並以預設參數啟動；設定檔存在但內容不合法時直接拋出
    `ValidationError`，不吞掉錯誤——參數錯了卻靜默套用預設值，會讓推理以非預期
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
