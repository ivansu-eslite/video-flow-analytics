from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class Zone(BaseModel):
    """單一攝影機畫面內的一個多邊形區域。polygon 為 pixel 座標的頂點清單，
    對應該攝影機整天固定的解析度。"""

    name: str
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _need_three_vertices(cls, value: list[tuple[float, float]]):
        if len(value) < 3:
            raise ValueError("polygon 至少需要 3 個頂點才能構成區域")
        return value


class CameraEntry(BaseModel):
    camera_id: str
    location: str
    ip: str
    # 該攝影機的 zone 定義（人工維護）；空清單代表這台攝影機不參與 zone mapping
    zones: list[Zone] = Field(default_factory=list)

    @field_validator("zones")
    @classmethod
    def _unique_zone_names(cls, value: list[Zone]):
        names = [z.name for z in value]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"同一攝影機的 zone name 不可重複: {sorted(dupes)}")
        return value

    @property
    def stream_dirname(self) -> str:
        # 對應 bucket 內的目錄命名規則 <location>_<camera_id>
        return f"{self.location}_{self.camera_id}"


class StorageConfig(BaseModel):
    file_ext: str = "mkv"
    target_codec: str = "h265"
    segment_strategy: str = "time"
    segment_seconds: int = Field(default=1800, ge=1)


class CameraRegistry(BaseModel):
    bucket_name: str
    storage: StorageConfig
    cameras: list[CameraEntry]

    def resolve_cameras(self, camera_ids: list[str] | None) -> list[CameraEntry]:
        """依 camera_ids 過濾；None 或空清單代表全部。查無對應 ID 時直接報錯。"""
        if not camera_ids:
            return list(self.cameras)
        by_id = {cam.camera_id: cam for cam in self.cameras}
        unknown = [cid for cid in camera_ids if cid not in by_id]
        if unknown:
            raise ValueError(
                f"camera_registry.yaml 中找不到這些 camera_id: {unknown}"
            )
        return [by_id[cid] for cid in camera_ids]


def load_registry(bucket_dir: Path) -> CameraRegistry:
    registry_path = bucket_dir / "camera_registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"找不到設備登錄檔: {registry_path}")
    with open(registry_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CameraRegistry(**data)
