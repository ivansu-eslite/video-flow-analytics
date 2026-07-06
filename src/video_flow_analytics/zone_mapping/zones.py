from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


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


class CameraZones(BaseModel):
    zones: list[Zone]

    @field_validator("zones")
    @classmethod
    def _unique_zone_names(cls, value: list[Zone]):
        names = [z.name for z in value]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"同一攝影機的 zone name 不可重複: {sorted(dupes)}")
        return value


class ZoneRegistry(BaseModel):
    # key 為 stream_dirname（<location>_<camera_id>），直接對齊 tracking_results
    # 的 camera_id 欄位，供 zone mapping 逐攝影機套用其區域定義。
    cameras: dict[str, CameraZones]


def load_zones(zones_path: Path) -> ZoneRegistry:
    if not zones_path.exists():
        raise FileNotFoundError(f"找不到 zone 定義檔: {zones_path}")
    with open(zones_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ZoneRegistry(**data)
