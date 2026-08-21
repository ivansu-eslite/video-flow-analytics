"""縮放前移的契約測試。

這一項的失敗模式不是崩潰，是**座標整批偏掉而輸出完全正常**：parquet 照樣產出、筆數照樣
對，只是每個框都平移或縮放錯了，要到 zone／line 統計出現莫名其妙的數字才會被發現。所以
這裡釘的是「正向縮放與反向映射互為逆運算」，不是個別函式的表面行為。

反算插在主迴圈哪一步（那條會讓落腳點靜默退回框底邊中點的靜默路徑）由
`test_inference_loop.py` 釘住，不在這裡。
"""

import numpy as np
import pytest

from video_analyze.services.frame_ring import FrameRing, create_ring_buffer
from video_analyze.services.letterbox import (
    INFER_HEIGHT,
    INFER_WIDTH,
    clip_to_content_inplace,
    content_box,
    letterbox,
    letterbox_params,
    unscale_boxes_inplace,
    unscale_points_inplace,
)

# 正式環境的兩種來源規格；兩者都是 16:9，故縮完的填充量相同、只有 scale 不同
_1080P = (1080, 1920)
_4K = (2160, 3840)


@pytest.mark.parametrize(("shape", "expected_scale"), [(_1080P, 1 / 3), (_4K, 1 / 6)])
def test_params_match_ultralytics_rect_mode(shape, expected_scale):
    """縮放比例與填充量必須與 ultralytics rect 模式一致。

    不一致的話，改動前後的偵測結果會有系統性差異，而那個差異看起來會像是「把縮放
    搬到讀取端造成的」，實際上只是兩邊用了不同的縮放參數。
    """
    scale, pad_x, pad_y = letterbox_params(*shape)
    assert scale == pytest.approx(expected_scale)
    # 16:9 縮到 640 寬會是 640×360，上下各填 12 湊到 stride 32 的倍數 384
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(12.0)


@pytest.mark.parametrize("shape", [_1080P, _4K])
def test_letterbox_output_shape(shape):
    """輸出必須剛好是推論尺寸——環形緩衝是照這個尺寸配置的，不符會寫爆 slot。"""
    frame = np.zeros((*shape, 3), dtype=np.uint8)

    assert letterbox(frame).shape == (INFER_HEIGHT, INFER_WIDTH, 3)


def test_letterbox_is_noop_when_already_infer_size():
    """已是推論尺寸時不再處理，且不複製——避免白做一次 memcpy。"""
    frame = np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)

    assert letterbox(frame) is frame


def test_letterboxed_frame_fits_the_ring_configured_for_inference_size():
    """縮放後的尺寸與環形緩衝的配置尺寸是同一個約定，必須對得上。

    緩衝改吃推論尺寸、讀取端卻漏叫 `letterbox()`（或反過來）是本次改動的組合失誤，
    這裡用真實的 `FrameRing` 把兩邊釘在一起：縮過的寫得進去、沒縮的當場拋錯。
    """
    num_slots = 2
    ring = FrameRing(
        create_ring_buffer(num_slots, INFER_HEIGHT, INFER_WIDTH),
        num_slots,
        INFER_HEIGHT,
        INFER_WIDTH,
    )
    frame = np.zeros((*_1080P, 3), dtype=np.uint8)

    ring.write_slot(0, letterbox(frame))  # 不應拋出

    with pytest.raises(ValueError, match="letterbox"):
        ring.write_slot(0, frame)


@pytest.mark.parametrize("shape", [_1080P, _4K])
def test_scale_then_unscale_round_trips(shape):
    """**核心契約**：原始座標 → 推論尺度 → 反算，必須回到原值。

    容差取 0.5 px：反算是除法，浮點誤差遠小於此；真正會讓它超標的是 scale 或 pad
    其中一邊被改掉，那正是要擋的情況。
    """
    height, width = shape
    scale, pad_x, pad_y = letterbox_params(height, width)
    # 涵蓋畫面四角與中央，含貼齊邊界的框
    original = np.array(
        [
            [0.0, 0.0, 100.0, 200.0, 1, 0.9, 0, 0],
            [width - 100.0, height - 200.0, float(width), float(height), 2, 0.8, 0, 1],
            [width / 2, height / 2, width / 2 + 50, height / 2 + 80, 3, 0.7, 0, 2],
        ]
    )
    # 正向：等比縮放後置中填充
    scaled = original.copy()
    scaled[:, [0, 2]] = scaled[:, [0, 2]] * scale + pad_x
    scaled[:, [1, 3]] = scaled[:, [1, 3]] * scale + pad_y

    unscale_boxes_inplace(scaled, scale, pad_x, pad_y)

    np.testing.assert_allclose(scaled[:, :4], original[:, :4], atol=0.5)
    # 非座標欄（track_id、score、cls、idx）不可被動到
    np.testing.assert_array_equal(scaled[:, 4:], original[:, 4:])


@pytest.mark.parametrize("shape", [_1080P, _4K])
def test_points_round_trip_with_the_same_params_as_boxes(shape):
    """落腳點與框必須共用同一組參數，否則同一列的兩者會落在不同尺度上。

    落腳點是由框推算出來的（`foot = 2 × C_fbody − H`），兩者不同步時 parquet 每一列
    的 bbox 與落腳點互不自洽，而下游 zone／line 只讀落腳點——bbox 看起來完全正常。
    """
    height, width = shape
    scale, pad_x, pad_y = letterbox_params(height, width)
    original = np.array([[0.0, 0.0], [width / 2, height / 2], [float(width), 0.0]])
    scaled = original * scale + np.array([pad_x, pad_y])

    unscale_points_inplace(scaled, scale, pad_x, pad_y)

    np.testing.assert_allclose(scaled, original, atol=0.5)


def test_unscale_tolerates_empty_tracks():
    """沒有存活軌跡是常態（空畫面），兩種空陣列表示法都不能炸。

    `MultiStreamByteTracker` 在 stream_id 不存在時回傳 1D 的 `np.array([])`，
    無偵測時回傳 (0, 8)；前者沒有第二個維度，直接切欄會 IndexError。落腳點那側
    `FootPointEstimator.estimate` 對空軌跡回傳 (0, 2)。
    """
    for empty in (np.array([]), np.empty((0, 8), dtype=float)):
        unscale_boxes_inplace(empty, 1 / 3, 0.0, 12.0)  # 不應拋出
    unscale_points_inplace(np.empty((0, 2)), 1 / 3, 0.0, 12.0)  # 不應拋出


def test_unscale_undoes_padding_not_just_scaling():
    """填充量必須被扣掉，只除以 scale 是不夠的。

    這支測試專門擋「忘了扣 pad」：4K 的 pad_y=12 在反算後會放大成 72 px 的垂直偏移，
    足以讓落腳點跨越計數線，而偵測數與軌跡數完全正常。
    """
    scale, pad_x, pad_y = letterbox_params(*_4K)
    boxes = np.array([[0.0, pad_y, 10.0, pad_y + 10.0, 1, 0.9, 0, 0]])
    points = np.array([[0.0, pad_y]])

    unscale_boxes_inplace(boxes, scale, pad_x, pad_y)
    unscale_points_inplace(points, scale, pad_x, pad_y)

    # 推論尺度上位於 y=pad_y 的點，就是原圖的 y=0
    assert boxes[0, 1] == pytest.approx(0.0, abs=0.5)
    assert points[0, 1] == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("shape", [_1080P, _4K])
def test_clip_keeps_boxes_inside_the_source_frame_after_unscaling(shape):
    """裁切的口徑是「反算後落在畫面內」，不是「落在 640×384 內」。

    改動前 ultralytics 反算座標時就把框裁進原圖了；只裁到推論尺寸邊界的話，上下各
    12 px 的填充帶會被當成合法範圍，反算時再放大回去——4K 一個推論像素等於 6 個原始
    像素，實測有 15% 的框因此外擴、最大 8.3 px，而框數與軌跡數完全正常。
    """
    height, width = shape
    scale, pad_x, pad_y = letterbox_params(height, width)
    # 兩個都伸進填充帶：第一個超出上緣，第二個超出下緣與右緣
    boxes = np.array(
        [
            [-3.0, 0.0, 50.0, 80.0, 1, 0.9, 0, 0],
            [600.0, 300.0, INFER_WIDTH + 5.0, float(INFER_HEIGHT), 2, 0.8, 0, 1],
        ]
    )

    clip_to_content_inplace(boxes, content_box(height, width))
    unscale_boxes_inplace(boxes, scale, pad_x, pad_y)

    assert boxes[:, [0, 2]].min() >= -0.5
    assert boxes[:, [0, 2]].max() <= width + 0.5
    assert boxes[:, [1, 3]].min() >= -0.5
    assert boxes[:, [1, 3]].max() <= height + 0.5
    # 沒裁的話這格會變成 -18（4K）／-9（1080p），而不是 0
    assert boxes[0, 1] == pytest.approx(0.0, abs=0.5)


def test_clip_tolerates_empty_detections():
    """空畫面沒有偵測是常態，`(0, 6)` 與 1D 空陣列都不能炸。"""
    content = content_box(*_1080P)
    clip_to_content_inplace(np.empty((0, 6)), content)
    clip_to_content_inplace(np.array([]), content)


def test_padding_offset_matches_the_params_for_an_odd_aspect_ratio():
    """填充量必須是**內容實際起點**的整數格，不是 `.5` 的分數。

    ultralytics 的 `ops.scale_boxes` 用 `round(pad - 0.1)` 反算，置中除不盡時內容是從
    整數那一格開始的。回傳分數的話每個座標都會偏約 0.75 個原始像素，而 16:9 剛好整除
    （pad 0 與 12）看不出來——這裡用 4:3 來源把兩者釘在一起：影格填滿 255，letterbox
    之後第一個非填充值的列／欄就該等於 `letterbox_params` 給的 pad。
    """
    height, width = 576, 704  # 4:3，縮完 469×384，左右除不盡（171 / 2）
    _scale, pad_x, pad_y = letterbox_params(height, width)
    assert (pad_x, pad_y) == (85, 0)

    out = letterbox(np.full((height, width, 3), 255, dtype=np.uint8))

    columns = np.flatnonzero((out != 114).any(axis=(0, 2)))
    rows = np.flatnonzero((out != 114).any(axis=(1, 2)))
    assert columns[0] == pad_x
    assert rows[0] == pad_y
    # 內容區的右／下界同樣要對得上（除不盡的那半格算在右／下）
    assert (columns[-1] + 1, rows[-1] + 1) == content_box(height, width)[2:]
