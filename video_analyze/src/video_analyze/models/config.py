from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from vfa_config import InputConfig, get_toml_path
from vfa_observability import StructuredLogger

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID

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

    model_config = ConfigDict(extra="forbid")

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
        classes: 要保留的偵測類別 id 清單，對應權重的類別定義。預設同時保留
            head 與 fbody：fbody 是追蹤目標，head 只供落腳點推算用（不進
            tracker，見 `services/inference.py`）。
    """

    model_config = ConfigDict(extra="forbid")

    model_path: str = "20260714-153811_yolo26m_baseline.pt"
    batch: int = Field(default=1, ge=1)
    classes: list[int] = Field(
        default_factory=lambda: [HEAD_CLASS_ID, FBODY_CLASS_ID], min_length=1
    )


class FootPointConfig(BaseModel):
    """落腳點（人站在地面的位置）的推算方式。

    Attributes:
        method: `"head"` 由 head 框推算，修正斜向視角下 axis-aligned 框底邊中點
            落在人體外的偏移；`"bbox_bottom"` 為改用 head 推算之前的做法（框底邊
            中點），保留供對照與回退。兩者的取捨見 ADR-009。
    """

    model_config = ConfigDict(extra="forbid")

    # 這裡的字面值與 services/foot_point.py 的 FOOT_POINT_METHODS 是同一組；
    # Literal 只吃字面值，無法引用常數，新增算法時兩處要一起改。
    method: Literal["head", "bbox_bottom"] = "head"


class OutputConfig(BaseModel):
    """輸出行為參數。

    Attributes:
        save_video: 是否輸出逐片段標註影片。
    """

    model_config = ConfigDict(extra="forbid")

    save_video: bool = False


class AppConfig(BaseSettings):
    """`config.toml` 與環境變數對應的完整設定，模組載入時組成全域單例 `settings`。

    Attributes:
        tracker: ByteTrack 追蹤器參數。
        model: YOLO 模型參數。
        foot_point: 落腳點推算方式。
        output: 輸出行為參數。
        input: 共用的 `[input]` 輸入參數（本包讀 `bucket_dir`／`date`／`camera_ids`）。
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

    # tracker/model 給 default_factory，讓找不到 config.toml 時 `AppConfig()` 仍能以
    # 全預設值啟動（兩者本就無必填參數）；與另兩包的 load_config 一致。
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    foot_point: FootPointConfig = Field(default_factory=FootPointConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)

    @model_validator(mode="after")
    def _check_classes_cover_pipeline(self) -> "AppConfig":
        """`classes` 要涵蓋 pipeline 實際需要的類別，缺了就 fail loud。

        兩者都是「設定看起來合法、跑起來也不會爆，但結果靜默劣化」的組合：少了
        fbody 就沒有東西可追蹤，整天的 parquet 會是空的；`method="head"` 卻少了
        head，則每一列都配不到頭而退回框底邊中點——輸出仍完整，只是改動等於沒生效。
        """
        if FBODY_CLASS_ID not in self.model.classes:
            raise ValueError(
                f"[model].classes 必須包含 fbody（id={FBODY_CLASS_ID}）："
                "它是追蹤目標，少了它整天不會有任何追蹤結果。"
                f"目前為 {self.model.classes}。"
            )
        if self.foot_point.method == "head" and HEAD_CLASS_ID not in self.model.classes:
            raise ValueError(
                f"[foot_point].method = \"head\" 需要偵測 head（id={HEAD_CLASS_ID}），"
                f"但 [model].classes 為 {self.model.classes}。請把 {HEAD_CLASS_ID} "
                "加進 classes，或把 method 改成 \"bbox_bottom\"——否則每一列都會配不到"
                "頭而退回框底邊中點，改動靜默失效。"
            )
        return self


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
    toml_path = get_toml_path(__file__)
    if toml_path is None or not Path(toml_path).exists():
        logger.warning("找不到 config.toml，將使用預設參數啟動", path=toml_path)
    return AppConfig()


# 模組載入時建立全域單例，其他模組直接 import 使用（而非依賴注入）
settings = load_config()
