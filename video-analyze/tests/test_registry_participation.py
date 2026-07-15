"""精簡 registry 的 `extra="forbid"` 相容性測試（父計畫 0c 的 (b) 條款）。

精簡 registry 拿掉了 zone 幾何，但 `camera_registry.yaml` 仍帶
`participates_in_zone_mapping` 與 `zones` 兩個欄位。`CameraEntry` 用
`extra="forbid"`，模型缺任一欄位都會在載入時直接失敗。現有真實 fixture 都沒有
`participates_in_zone_mapping`，只驗真實 yaml 會假性通過，故用合成 yaml 補這個缺口。
"""

from pathlib import Path

import yaml

from video_analyze.registry import CameraRegistry

_FIXTURE = Path(__file__).parent / "fixtures" / "registry_with_participation.yaml"


def test_slim_registry_accepts_participation_field():
    """含 `participates_in_zone_mapping` 的合成 yaml 能在 extra="forbid" 下載入，
    且欄位值被正確解析（True/False 兩種都涵蓋）。"""
    data = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    registry = CameraRegistry(**data)

    by_id = {cam.camera_id: cam for cam in registry.cameras}
    assert by_id["cam_a"].participates_in_zone_mapping is True
    assert by_id["cam_b"].participates_in_zone_mapping is False
    # zones 以原始 list[Any] 保留、不解析幾何
    assert by_id["cam_a"].zones == [
        {"name": "zone_a", "polygon": [[0, 0], [10, 0], [10, 10]]}
    ]
