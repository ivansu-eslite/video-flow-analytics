import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _find_duplicates(items: list[str]) -> set[str]:
    return {item for item, count in Counter(items).items() if count > 1}


# inside_point 到某段無限直線的垂直距離小於此值即視為共線——side 無法定號，fail-loud。
# 以像素為單位取一個遠小於任何合理標註誤差的門檻。
_COLLINEAR_EPS_PX = 1e-6


class Zone(BaseModel):
    """單一攝影機畫面內的一個多邊形區域。

    polygon 為 pixel 座標的頂點清單，對應該攝影機整天固定的解析度。

    Attributes:
        name: 區域名稱。`parsed_zones()` 只驗證同一攝影機內不可重複；
            `parse_and_validate_zones` 另驗證跨攝影機也不可重複（原始需求來自
            下游報表依 zone 名稱分組彙總、不含 camera_id）。
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


class Line(BaseModel):
    """單一攝影機畫面內的一條方向性計數線。

    points 為 pixel 座標的 polyline 頂點清單（可彎折），對應該攝影機整天固定的
    解析度。`inside_point` 是場內一參考點，決定跨越的方向：往 `inside_point` 那一側
    跨為「進」（in），反向為「出」（out）。

    Attributes:
        name: 計數線名稱。`parsed_lines()` 只驗證同一攝影機內不可重複；
            `parse_and_validate_lines` 另驗證跨攝影機也不可重複（原始需求來自
            下游報表依計數線名稱分組彙總、不含 camera_id）。
        points: polyline 頂點清單，至少 2 個 `(x, y)` pixel 座標。
        inside_point: 場內一參考點 `(x, y)`；不可落在任一段的無限延伸線上（否則該段
            的側別無法定號）。

    Raises:
        ValueError: `points` 頂點數少於 2、有零長度段（連續重複頂點），或
            `inside_point` 落在任一段的無限延伸線上。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    points: list[tuple[float, float]]
    inside_point: tuple[float, float]

    @field_validator("points")
    @classmethod
    def _need_two_vertices(cls, value: list[tuple[float, float]]):
        if len(value) < 2:
            raise ValueError("計數線的 points 至少需要 2 個頂點才能構成折線")
        return value

    @model_validator(mode="after")
    def _inside_point_defines_side(self) -> "Line":
        # 帶號距離的正負以「inside_point 相對最近段無限直線的側別」定義；任一段的無限
        # 直線通過 inside_point（外積≈0）時側別無法定號，方向判定失效，故 fail-loud。
        px, py = self.inside_point
        for (ax, ay), (bx, by) in zip(self.points, self.points[1:]):
            dx, dy = bx - ax, by - ay
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                raise ValueError(
                    f"計數線 {self.name!r} 有零長度線段（連續重複頂點）: {(ax, ay)}"
                )
            # inside_point 到該段無限直線的垂直距離 = |cross| / |段向量|
            perp = abs(dx * (py - ay) - dy * (px - ax)) / seg_len
            if perp < _COLLINEAR_EPS_PX:
                raise ValueError(
                    f"計數線 {self.name!r} 的 inside_point {self.inside_point} 落在"
                    f"線段 {(ax, ay)}->{(bx, by)} 的延伸線上，無法判定進出方向"
                )
        return self


class CameraEntry(BaseModel):
    """單一攝影機的身份與 zone 定義。

    Attributes:
        camera_id: 攝影機代號；在 `CameraRegistry` 內必須唯一（見
            `CameraRegistry._unique_camera_identity`），重複會在載入
            registry 時 fail-loud。
        location: 攝影機所在位置名稱；與 `camera_id` 組成的 `stream_dirname`
            同樣必須在 `CameraRegistry` 內唯一。
        ip: 攝影機 IP。
        participates_in_zone_mapping: 是否參與 zone mapping；`False` 代表
            zone 相關處理應跳過這台攝影機（即使 `zones` 有內容也不處理）。
            正式訊號，取代舊版「`zones` 空清單代表不參與」的隱含推斷。
        zones: 原始 zone 定義（未經驗證的 dict 清單）。刻意用 `list[Any]`
            （非 `list[Zone]`）：載入 registry 時就驗證幾何，會讓某台攝影機的
            zone 筆誤蓋過更根本、也更該先報出來的錯誤（例如攝影機對不上當天
            資料、或該攝影機根本不在本次處理範圍內）；幾何驗證因此延後到
            呼叫端明確呼叫 `parsed_zones()` 的時候。
        lines: 原始計數線定義（未經驗證的 dict 清單）。理由與 `zones` 相同：
            刻意用 `list[Any]` 把幾何驗證延後到呼叫端 `parsed_lines()`。
            `line_counting` 以「`lines` 是否非空」決定攝影機是否參與（不另設
            參與旗標）；`video_analyze` 用不到 line，但共用模型仍須保留此欄位，
            否則含 `lines:` 的 yaml 會在 `extra="forbid"` 下解析失敗。
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    location: str
    ip: str
    participates_in_zone_mapping: bool = Field(default=True)
    zones: list[Any] = Field(default_factory=list)
    lines: list[Any] = Field(default_factory=list)

    @property
    def stream_dirname(self) -> str:
        """對應 bucket 內的目錄命名規則 `<location>_<camera_id>`。"""
        return f"{self.location}_{self.camera_id}"

    def parsed_zones(self) -> list[Zone]:
        """把原始 zone 定義解析並驗證成 `Zone` model。

        刻意不在載入 registry 時自動執行：呼叫端通常要先確認攝影機篩選與資料
        對應無誤，才輪到幾何是否合法（見 `zones` 欄位說明）。

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

    def parsed_lines(self) -> list["Line"]:
        """把原始計數線定義解析並驗證成 `Line` model。

        與 `parsed_zones()` 同理，刻意不在載入 registry 時自動執行：呼叫端
        （`count_lines_daily`）通常要先確認攝影機篩選與資料對應無誤，才輪到
        計數線幾何是否合法（見 `lines` 欄位說明）。

        Returns:
            解析驗證後的 `Line` 清單。

        Raises:
            ValueError: 任一計數線定義不合法，或同一攝影機內 line name 重複。
        """
        lines = [Line(**ln) for ln in self.lines]
        dupes = _find_duplicates([ln.name for ln in lines])
        if dupes:
            raise ValueError(f"同一攝影機的計數線 name 不可重複: {sorted(dupes)}")
        return lines


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

    「zone 名稱跨攝影機也不可重複」這條規則來自下游報表依 zone 名稱分組彙總、
    不含 camera_id：同名區域會讓不同攝影機的人流被靜默合併成同一列。

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


def parse_and_validate_lines(entries: dict[str, CameraEntry]) -> dict[str, list[Line]]:
    """解析已篩選過攝影機的計數線定義，並驗證跨攝影機 line 名稱全域唯一。

    「計數線名稱跨攝影機也不可重複」這條規則來自下游報表依 line 名稱分組彙總、
    不含 camera_id：同名計數線會讓不同攝影機的進出人數被靜默合併成同一列。與
    `parse_and_validate_zones` 是同型驗證。

    Args:
        entries: 已依需求篩選過（例如只留 `lines` 非空）的
            stream_dirname -> CameraEntry 對照表。

    Returns:
        stream_dirname -> 解析驗證後的 Line 清單。

    Raises:
        ValueError: 任一攝影機的計數線定義不合法、同一攝影機內 line name
            重複，或跨攝影機有重複的 line 名稱。
    """
    line_cameras = {
        camera_id: entry.parsed_lines() for camera_id, entry in entries.items()
    }
    dupes = sorted(
        _find_duplicates(
            [line.name for lines in line_cameras.values() for line in lines]
        )
    )
    if dupes:
        raise ValueError(
            f"camera_registry.yaml 中有跨攝影機重複的計數線名稱，line 名稱須"
            f"全域唯一（不只同一攝影機內唯一）: {dupes}"
        )
    return line_cameras


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

    刻意吃任意路徑而非只吃 `bucket_dir`：呼叫端自行決定要讀 bucket 下當下的
    `camera_registry.yaml`（見 `load_registry`），或是某日輸出目錄下的
    `camera_registry_used.yaml` 快照——後者才能反映產生該日資料當時的定義。

    Args:
        path: registry yaml 檔案路徑。

    Returns:
        解析驗證後的 `CameraRegistry`。

    Raises:
        FileNotFoundError: `path` 不存在。
        ValueError: 檔案內容不是 YAML mapping（空檔或只有註解時
            `yaml.safe_load` 會回傳 `None`）。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到設備登錄檔: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"設備登錄檔格式不正確或內容為空: {path}")
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
