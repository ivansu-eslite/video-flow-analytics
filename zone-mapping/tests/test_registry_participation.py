"""registry 的 `extra="forbid"` 相容性測試（父計畫 0c 的 (b) 條款）。

`CameraEntry` 用 `extra="forbid"`，模型缺任一欄位都會在載入
`camera_registry.yaml` 時直接失敗。現有真實 fixture 都沒有
`participates_in_zone_mapping`，只驗真實 yaml 會假性通過，故用三包共用的同一份
合成 yaml 補這個缺口——本包正是靠這個欄位決定哪些攝影機要參與 zone mapping。
"""

from pathlib import Path

import yaml

from zone_mapping.registry import CameraRegistry

_FIXTURE = Path(__file__).parent / "fixtures" / "registry_with_participation.yaml"


def test_registry_accepts_participation_field():
    """含 `participates_in_zone_mapping` 的合成 yaml 能在 extra="forbid" 下載入，
    且欄位值被正確解析（True/False 兩種都涵蓋）。"""
    data = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    registry = CameraRegistry(**data)

    by_id = {cam.camera_id: cam for cam in registry.cameras}
    assert by_id["cam_a"].participates_in_zone_mapping is True
    assert by_id["cam_b"].participates_in_zone_mapping is False
    # zones 以原始 list[Any] 保留、不解析幾何（幾何驗證延後到 parsed_zones()）
    assert by_id["cam_a"].zones == [
        {"name": "zone_a", "polygon": [[0, 0], [10, 0], [10, 10]]}
    ]
