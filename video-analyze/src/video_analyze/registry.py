from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _find_duplicates(items: list[str]) -> set[str]:
    return {item for item, count in Counter(items).items() if count > 1}


class CameraEntry(BaseModel):
    """單一攝影機的身份與 zone 定義。

    Attributes:
        camera_id: 攝影機代號；在 `CameraRegistry` 內必須唯一（見
            `CameraRegistry._unique_camera_identity`），重複會在載入
            registry 時 fail-loud。
        location: 攝影機所在位置名稱；與 `camera_id` 組成的 `stream_dirname`
            同樣必須在 `CameraRegistry` 內唯一。
        ip: 攝影機 IP。
        participates_in_zone_mapping: 是否參與 zone mapping；`False` 代表下游
            應跳過這台攝影機（即使 `zones` 有內容也不處理）。analyze 不使用此
            欄位，語義由 zone-mapping 那包實作（見該包 `map_zones_daily`），
            此處僅為相容 registry 格式而保留。
        zones: 原始 zone 定義（未經驗證的 dict 清單）。analyze 不使用 zone
            幾何，此處僅為相容 `camera_registry.yaml` 的欄位而保留並忽略
            （`CameraEntry` 用 `extra="forbid"`，缺這個欄位會載入失敗）。
            zone 幾何的解析與驗證在 zone-mapping／flow-report 兩包各自進行。
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    location: str
    ip: str
    participates_in_zone_mapping: bool = Field(default=True)
    zones: list[Any] = Field(default_factory=list)

    @property
    def stream_dirname(self) -> str:
        """對應 bucket 內的目錄命名規則 `<location>_<camera_id>`。"""
        return f"{self.location}_{self.camera_id}"


class StorageConfig(BaseModel):
    """bucket 內影片片段的儲存格式參數。

    Attributes:
        file_ext: 片段檔案副檔名。
        target_codec: 原始錄影的編碼格式。
        segment_strategy: 分段策略。
        segment_seconds: 每段影片的秒數。
    """

    file_ext: str = "mkv"
    target_codec: str = "h265"
    segment_strategy: str = "time"
    segment_seconds: int = Field(default=1800, ge=1)


class CameraRegistry(BaseModel):
    """`camera_registry.yaml` 對應的完整設備登錄資料。

    Attributes:
        bucket_name: bucket 名稱。
        storage: 影片片段儲存格式參數。
        cameras: 攝影機清單。

    Raises:
        ValueError: `cameras` 中有重複的 `camera_id` 或 `stream_dirname`。
    """

    bucket_name: str
    storage: StorageConfig
    cameras: list[CameraEntry]

    @model_validator(mode="after")
    def _unique_camera_identity(self) -> "CameraRegistry":
        # camera_id 與 stream_dirname 是兩處查詢字典的鍵，重複會讓攝影機被靜默覆蓋
        dupes = _find_duplicates(
            [cam.camera_id for cam in self.cameras]
        ) | _find_duplicates([cam.stream_dirname for cam in self.cameras])
        if dupes:
            raise ValueError(
                f"camera_registry.yaml 中有重複的攝影機（camera_id 或 "
                f"location_camera_id 相同）: {sorted(dupes)}"
            )
        return self

    def resolve_cameras(self, camera_ids: list[str] | None) -> list[CameraEntry]:
        """依 `camera_ids` 過濾出對應的攝影機。

        Args:
            camera_ids: 要保留的 camera_id 清單；`None` 或空清單代表全部。

        Returns:
            過濾後的 `CameraEntry` 清單，順序依 `camera_ids` 指定順序。

        Raises:
            ValueError: `camera_ids` 中有查無對應設備登錄的 ID。
        """
        if not camera_ids:
            return list(self.cameras)
        by_id = {cam.camera_id: cam for cam in self.cameras}
        unknown = [cid for cid in camera_ids if cid not in by_id]
        if unknown:
            raise ValueError(
                f"camera_registry.yaml 中找不到這些 camera_id: {unknown}"
            )
        return [by_id[cid] for cid in camera_ids]


def registry_path(bucket_dir: Path) -> Path:
    """組出 `bucket_dir` 內 `camera_registry.yaml` 的路徑。

    Args:
        bucket_dir: bucket 根目錄。

    Returns:
        `camera_registry.yaml` 的完整路徑。
    """
    return bucket_dir / "camera_registry.yaml"


def load_registry(bucket_dir: Path) -> CameraRegistry:
    """讀取 `bucket_dir` 下的 `camera_registry.yaml`。

    Args:
        bucket_dir: bucket 根目錄。

    Returns:
        解析驗證後的 `CameraRegistry`。

    Raises:
        FileNotFoundError: `camera_registry.yaml` 不存在。
    """
    path = registry_path(bucket_dir)
    if not path.exists():
        raise FileNotFoundError(f"找不到設備登錄檔: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CameraRegistry(**data)
