"""`camera_registry.yaml` 的 Pydantic 模型與 zone／line 驗證（四包共用）。"""

from .registry import (
    CameraEntry,
    CameraRegistry,
    Line,
    StorageConfig,
    Zone,
    load_registry,
    load_registry_from_path,
    parse_and_validate_lines,
    parse_and_validate_zones,
    registry_path,
)

__all__ = [
    "CameraEntry",
    "CameraRegistry",
    "Line",
    "StorageConfig",
    "Zone",
    "load_registry",
    "load_registry_from_path",
    "parse_and_validate_lines",
    "parse_and_validate_zones",
    "registry_path",
]
