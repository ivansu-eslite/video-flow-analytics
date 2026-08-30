"""偵測結果拆分的契約測試：head 不可流進 tracker。

偵測改成同時保留 head 與 fbody 之後，「餵給 tracker 的是哪些框」變成一條看不見的
契約——head 若混進去，同一個人會多出一條頭部軌跡，`track_id` 從「一個人」變成
「一個偵測目標」，下游的不重複訪客與進出人數會直接翻倍，而輸出檔本身完全正常。
這裡用合成的 N×6 偵測陣列（payload 的佈局）把這條契約釘住。

`split_detections` 隨追蹤一起搬進 `services/track_worker.py`（issue #109）：拆分的
下一步就是 `tracker.update`，兩者同在追蹤進程。主迴圈實際把拆出來的哪一份交給
tracker，由 test_track_worker.py 釘住。
"""

import numpy as np

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.track_worker import split_detections

_VBODY_CLASS_ID = 1


def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> np.ndarray:
    """`rows` 為 `(x1, y1, x2, y2, conf, cls)`，與 ultralytics 的 data 佈局一致。"""
    return np.array(rows, dtype=np.float32)


def test_only_fbody_reaches_the_tracker():
    boxes = _boxes(
        [
            (10.0, 10.0, 20.0, 20.0, 0.9, HEAD_CLASS_ID),
            (0.0, 0.0, 100.0, 200.0, 0.8, FBODY_CLASS_ID),
            (5.0, 5.0, 90.0, 120.0, 0.7, _VBODY_CLASS_ID),
            (300.0, 0.0, 400.0, 200.0, 0.6, FBODY_CLASS_ID),
        ]
    )

    fbody, heads = split_detections(boxes)

    assert fbody.cls.tolist() == [FBODY_CLASS_ID, FBODY_CLASS_ID]
    np.testing.assert_allclose(
        np.asarray(fbody.xyxy), [[0.0, 0.0, 100.0, 200.0], [300.0, 0.0, 400.0, 200.0]]
    )


def test_heads_are_returned_separately_for_foot_point_estimation():
    boxes = _boxes(
        [
            (10.0, 10.0, 20.0, 20.0, 0.9, HEAD_CLASS_ID),
            (0.0, 0.0, 100.0, 200.0, 0.8, FBODY_CLASS_ID),
            (310.0, 10.0, 330.0, 30.0, 0.5, HEAD_CLASS_ID),
        ]
    )

    _fbody, heads = split_detections(boxes)

    np.testing.assert_allclose(
        heads, [[10.0, 10.0, 20.0, 20.0], [310.0, 10.0, 330.0, 30.0]]
    )


def test_frame_without_fbody_yields_empty_tracker_input():
    """整格只有頭：tracker 收到空輸入，而不是拿 head 去建軌跡。"""
    boxes = _boxes([(10.0, 10.0, 20.0, 20.0, 0.9, HEAD_CLASS_ID)])

    fbody, heads = split_detections(boxes)

    assert len(fbody) == 0
    assert heads.shape == (1, 4)


def test_frame_without_head_yields_empty_head_array():
    """沒有頭時回傳 (0, 4)，讓推算端無條件退回框底邊中點。"""
    boxes = _boxes([(0.0, 0.0, 100.0, 200.0, 0.8, FBODY_CLASS_ID)])

    fbody, heads = split_detections(boxes)

    assert len(fbody) == 1
    assert heads.shape == (0, 4)


def test_tracker_input_carries_the_inference_shape():
    """包出來的 `Boxes` 其 `orig_shape` 是推論尺寸，不是原始解析度。

    座標一路都停在推論尺度上（影格在讀取端就縮好了），這裡給錯只會讓 `xyxyn`／`xywhn`
    這類正規化屬性靜默算錯：`BYTETracker` 只讀 `.conf`／`.cls`／`.xywh`，追蹤結果與
    parquet 都完全正常，要到有人開始讀正規化座標才會發現。
    """
    boxes = _boxes([(0.0, 0.0, 100.0, 200.0, 0.8, FBODY_CLASS_ID)])

    fbody, _heads = split_detections(boxes)

    assert fbody.orig_shape == (INFER_HEIGHT, INFER_WIDTH)
