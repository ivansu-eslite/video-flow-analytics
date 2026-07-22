from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _find_duplicates(items: list[str]) -> set[str]:
    return {item for item, count in Counter(items).items() if count > 1}


class Zone(BaseModel):
    """單一攝影機畫面內的一個多邊形區域。

    polygon 為 pixel 座標的頂點清單，對應該攝影機整天固定的解析度。

    Attributes:
        name: 區域名稱。`parsed_zones()` 只驗證同一攝影機內不可重複；
            `parse_and_validate_zones` 另驗證跨攝影機也不可重複——本包的報表正是
            那個依 zone 名稱分組彙總、不含 camera_id 的下游，同名區域會被合併成
            同一列。
        polygon: 多邊形頂點清單，至少 3 個 `(x, y)` pixel 座標。

    Raises:
        ValueError: `polygon` 頂點數少於 3。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _need_three_vertices(cls, value: list[tuple[float, float]]):
        if len(value) < 3:
            raise ValueError("polygon 至少需要 3 個頂點才能構成區域")
        return value


class CameraEntry(BaseModel):
    """單一攝影機的身份與 zone 定義。

    Attributes:
        camera_id: 攝影機代號；在 `CameraRegistry` 內必須唯一（見
            `CameraRegistry._unique_camera_identity`），重複會在載入
            registry 時 fail-loud。
        location: 攝影機所在位置名稱；與 `camera_id` 組成的 `stream_dirname`
            同樣必須在 `CameraRegistry` 內唯一。
        ip: 攝影機 IP。
        participates_in_zone_mapping: 該攝影機是否參與 zone mapping；`False`
            時其 zone 不會出現在上游 `zone_counts.parquet`，`_build_report_frames`
            也據此把它排除在 zone 名稱唯一性驗證之外（即使 `zones` 有內容）。
        zones: 原始 zone 定義（未經驗證的 dict 清單）。刻意用 `list[Any]`
            （非 `list[Zone]`）：本包只對「有參與 zone mapping」的攝影機解析
            幾何（見 `_build_report_frames` 呼叫 `parse_and_validate_zones`）。
            若在載入 registry 時就驗證成 `list[Zone]`，快照中某台**不參與**的
            攝影機的 zone 筆誤也會讓報表載不進來，而那台攝影機根本不在報表
            範圍內；驗證因此延後到 `parsed_zones()`。
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

    def parsed_zones(self) -> list[Zone]:
        """把原始 zone 定義解析並驗證成 `Zone` model。

        由 `parse_and_validate_zones` 對已篩出「有參與 zone mapping」的攝影機
        呼叫，因此不參與的攝影機即使 zone 定義有誤也不影響報表。

        Returns:
            解析驗證後的 `Zone` 清單。

        Raises:
            ValueError: 任一 zone 定義不合法，或同一攝影機內 zone name 重複。
        """
        zones = [Zone(**z) for z in self.zones]
        dupes = _find_duplicates([z.name for z in zones])
        if dupes:
            raise ValueError(f"同一攝影機的 zone name 不可重複: {sorted(dupes)}")
        return zones


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
            ValueError: `camera_ids` 中有查無對應設備登錄的 ID，或有重複的
                camera_id。
        """
        if not camera_ids:
            return list(self.cameras)
        by_id = {cam.camera_id: cam for cam in self.cameras}
        unknown = [cid for cid in camera_ids if cid not in by_id]
        if unknown:
            raise ValueError(
                f"camera_registry.yaml 中找不到這些 camera_id: {unknown}"
            )
        dupes = _find_duplicates(camera_ids)
        if dupes:
            raise ValueError(f"camera_ids 有重複的 camera_id: {sorted(dupes)}")
        return [by_id[cid] for cid in camera_ids]


def parse_and_validate_zones(entries: dict[str, CameraEntry]) -> dict[str, list[Zone]]:
    """解析已篩選過攝影機的 zone 定義，並驗證跨攝影機 zone 名稱全域唯一。

    「zone 名稱跨攝影機也不可重複」這條規則的需求方就是本包：報表依 zone 名稱
    分組彙總、不含 camera_id，同名區域會讓不同攝影機的人流被靜默合併。
    `zone-mapping` 套件有一份相同實作（兩包各自獨立、刻意不跨資料夾 import），
    改動時兩邊需同步。

    Args:
        entries: 已依需求篩選過（例如 participates_in_zone_mapping）的
            stream_dirname -> CameraEntry 對照表。

    Returns:
        stream_dirname -> 解析驗證後的 Zone 清單。

    Raises:
        ValueError: 任一攝影機的 zone 定義不合法、同一攝影機內 zone name
            重複，或跨攝影機有重複的 zone 名稱。
    """
    zone_cameras = {
        camera_id: entry.parsed_zones() for camera_id, entry in entries.items()
    }
    dupes = sorted(
        _find_duplicates(
            [zone.name for zones in zone_cameras.values() for zone in zones]
        )
    )
    if dupes:
        raise ValueError(
            f"camera_registry.yaml 中有跨攝影機重複的 zone 名稱，zone 名稱須"
            f"全域唯一（不只同一攝影機內唯一）: {dupes}"
        )
    return zone_cameras


def registry_path(bucket_dir: Path) -> Path:
    """組出 `bucket_dir` 內 `camera_registry.yaml` 的路徑。

    Args:
        bucket_dir: bucket 根目錄。

    Returns:
        `camera_registry.yaml` 的完整路徑。
    """
    return bucket_dir / "camera_registry.yaml"


def load_registry_from_path(path: Path) -> CameraRegistry:
    """讀指定路徑的 registry yaml。

    本包用它讀該日輸出目錄下的 `camera_registry_used.yaml` **快照**（見
    `_build_report_frames`），而非 `bucket_dir` 下當下的 `camera_registry.yaml`
    （後者由 `load_registry` 負責）。驗證對象必須是產生該份 `zone_counts.parquet`
    當時的定義：若兩者之間改過 zone 名稱，拿當下的檔案驗證會通過，但 parquet 裡
    其實是舊定義，不同攝影機的人流會被靜默合併。

    Args:
        path: registry yaml 檔案路徑。

    Returns:
        解析驗證後的 `CameraRegistry`。

    Raises:
        FileNotFoundError: `path` 不存在。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到設備登錄檔: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CameraRegistry(**data)


def load_registry(bucket_dir: Path) -> CameraRegistry:
    """讀取 `bucket_dir` 下的 `camera_registry.yaml`。

    Args:
        bucket_dir: bucket 根目錄。

    Returns:
        解析驗證後的 `CameraRegistry`。

    Raises:
        FileNotFoundError: `camera_registry.yaml` 不存在。
    """
    return load_registry_from_path(registry_path(bucket_dir))
