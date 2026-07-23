"""`camera_registry.yaml` 的 Pydantic 模型與 zone 驗證（三包共用）。"""

from .registry import (
    CameraEntry,
    CameraRegistry,
    StorageConfig,
    Zone,
    load_registry,
    load_registry_from_path,
    parse_and_validate_zones,
    registry_path,
)

__all__ = [
    "CameraEntry",
    "CameraRegistry",
    "StorageConfig",
    "Zone",
    "load_registry",
    "load_registry_from_path",
    "parse_and_validate_zones",
    "registry_path",
]
