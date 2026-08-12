"""落腳點推算的行為測試（合成 bbox，不需影片、不需模型）。

守住四件事：

1. **退化性質**——人站直時推算結果精確等於改動前的框底邊中點。這條是整個改動可以
   安全上線的前提：站直的人（多數）不受影響，只有傾斜的人被修正。反射基準若誤用
   head 框中心而非頂邊中點，站直的人會系統性上偏半顆頭，這裡會抓到。
2. **傾斜時往正確方向移動**——head 偏左上時落腳點要往右下，這正是要修的偏移。
3. **配不到 head 時退回底邊中點**，且不產生空值或例外（下游沒有處理 null 的路徑）。
4. **多候選的選法**——取主軸最接近垂直的那顆。實測顯示「離框頂最近」會系統性挑到
   後方那個人的頭（透視下他在畫面中更高），故那個直覺的判準刻意不採用，見 ADR-009。
"""

import numpy as np
import numpy.testing as npt
import pytest

from video_analyze.services import foot_point as fp
from video_analyze.services.foot_point import (
    FootPointEstimator,
    bbox_bottom_center,
    compute_foot_points,
    estimate_from_heads,
)

# 一個站直的人：框 200 寬、800 高，頭在框頂正中央、佔框高 1/8
_UPRIGHT_BODY = np.array([[100.0, 0.0, 300.0, 800.0]])
_UPRIGHT_HEAD = np.array([[160.0, 0.0, 240.0, 100.0]])


def test_upright_person_reproduces_bbox_bottom_center_exactly():
    """站直時推算結果與改動前逐值相同——多數人不受這次改動影響。"""
    got = estimate_from_heads(_UPRIGHT_BODY, _UPRIGHT_HEAD)

    npt.assert_allclose(got, bbox_bottom_center(_UPRIGHT_BODY))
    npt.assert_allclose(got, [[200.0, 800.0]])


def test_reflection_uses_head_top_edge_not_head_center():
    """反射基準誤用 head 中心的話，站直的人會上偏半顆頭（此處為 50px）。

    這條鎖住上面的退化性質不被「看起來更自然」的 head 中心改寫掉。
    """
    got = estimate_from_heads(_UPRIGHT_BODY, _UPRIGHT_HEAD)
    head_center_variant = 800.0 - (100.0 - 0.0) / 2  # y2 − h_head/2

    assert got[0][1] == pytest.approx(800.0)
    assert got[0][1] != pytest.approx(head_center_variant)


def test_tilted_person_shifts_foot_point_toward_the_opposite_corner():
    """head 在框左上角 → 落腳點往右下，正是斜向視角要修的偏移方向。"""
    body = np.array([[0.0, 0.0, 400.0, 800.0]])
    head = np.array([[40.0, 0.0, 120.0, 80.0]])  # 頂邊中點 (80, 0)

    got = estimate_from_heads(body, head)

    # foot = 2*(200, 400) − (80, 0)
    npt.assert_allclose(got, [[320.0, 800.0]])
    assert got[0][0] > bbox_bottom_center(body)[0][0]  # 比底邊中點更靠右


def test_no_head_falls_back_to_bbox_bottom_center():
    got = estimate_from_heads(_UPRIGHT_BODY, np.empty((0, 4)))

    npt.assert_allclose(got, bbox_bottom_center(_UPRIGHT_BODY))
    assert np.isfinite(got).all()


def test_head_outside_body_box_is_not_matched():
    """框外的頭不算候選——否則相鄰的人會互相汙染。"""
    outside = np.array([[600.0, 0.0, 680.0, 80.0]])

    got = estimate_from_heads(_UPRIGHT_BODY, outside)

    npt.assert_allclose(got, bbox_bottom_center(_UPRIGHT_BODY))


def test_oversized_head_is_rejected():
    """head 面積超過框的 25%：多半是把整個人或上半身當成頭，採用它會嚴重偏移。"""
    body = np.array([[0.0, 0.0, 200.0, 400.0]])  # 面積 80000
    huge = np.array([[10.0, 10.0, 190.0, 160.0]])  # 面積 27000 > 20000

    got = estimate_from_heads(body, huge)

    npt.assert_allclose(got, bbox_bottom_center(body))


def test_head_below_body_center_is_rejected():
    """head 落在框中心以下時主軸方向會反轉，推出來的點會跑到人的頭上方。"""
    body = np.array([[0.0, 0.0, 200.0, 400.0]])
    low = np.array([[80.0, 300.0, 120.0, 340.0]])

    got = estimate_from_heads(body, low)

    npt.assert_allclose(got, bbox_bottom_center(body))


def test_head_beyond_tilt_limit_is_rejected():
    """主軸傾角超過 50°：多半是配到鄰座的頭，不是這個框裡的人。"""
    # 框中心 (100, 200)，head 中心 (190, 190)：dx=90、dy=10 → 約 83°
    body = np.array([[0.0, 0.0, 200.0, 400.0]])
    far = np.array([[170.0, 170.0, 210.0, 210.0]])

    got = estimate_from_heads(body, far)

    npt.assert_allclose(got, bbox_bottom_center(body))


def test_two_candidates_picks_the_most_vertical_axis():
    """兩顆候選：取主軸最接近垂直的（框中心正上方那顆），而非離框頂最近的那顆。

    這裡刻意讓「離框頂最近」與「主軸最垂直」指向不同的頭——前者是後方那個人的頭
    （更高、更偏），實測中它會被系統性挑中，故本專案不採用（ADR-009）。
    """
    body = np.array([[0.0, 0.0, 400.0, 800.0]])
    heads = np.array(
        [
            [20.0, 0.0, 80.0, 60.0],  # 更靠近框頂，但水平偏離中線 150px
            [170.0, 100.0, 230.0, 160.0],  # 稍低，但在框中心正上方
        ]
    )

    got = estimate_from_heads(body, heads)

    # 選中第二顆：foot = 2*(200, 400) − (200, 100)
    npt.assert_allclose(got, [[200.0, 700.0]])


def test_rows_are_matched_independently():
    """多列一起算時，每列各配各的頭，不會互相錯位。"""
    bodies = np.concatenate([_UPRIGHT_BODY, np.array([[500.0, 0.0, 900.0, 800.0]])])
    heads = np.concatenate([_UPRIGHT_HEAD, np.array([[540.0, 0.0, 620.0, 80.0]])])

    got = estimate_from_heads(bodies, heads)

    assert got[0] == pytest.approx([200.0, 800.0])
    assert got[1] == pytest.approx([2 * 700.0 - 580.0, 800.0])


def test_empty_input_returns_empty_two_column_array():
    """整格沒有軌跡時要回傳 (0, 2)，呼叫端才能無條件與 tracks 逐列對應。"""
    got = estimate_from_heads(np.empty((0, 4)), _UPRIGHT_HEAD)

    assert got.shape == (0, 2)


def test_compute_foot_points_bbox_bottom_ignores_heads():
    """切回舊算法時，即使 head 存在也必須逐值等於框底邊中點。"""
    body = np.array([[0.0, 0.0, 400.0, 800.0]])
    head = np.array([[40.0, 0.0, 120.0, 80.0]])

    got = compute_foot_points(body, head, "bbox_bottom")

    npt.assert_allclose(got, bbox_bottom_center(body))


def test_compute_foot_points_rejects_unknown_method():
    with pytest.raises(ValueError, match="未知的落腳點算法"):
        compute_foot_points(_UPRIGHT_BODY, _UPRIGHT_HEAD, "pose")


# --- 跨幀延續（FootPointEstimator）-------------------------------------------
#
# 逐幀獨立推算時，同一條軌跡在「配到頭」與「配不到頭」之間切換，落腳點會在推算點與
# 框底邊中點之間彈跳；實測那個幅度足以製造假 entry／假跨越（見 ADR-009）。以下鎖住
# 「配不到頭時沿用上次偏移」這個行為，以及它該在什麼時候放棄沿用。

_TILTED_BODY = np.array([[0.0, 0.0, 400.0, 800.0, 7]])  # 第 5 欄為 track_id
_TILTED_HEAD = np.array([[40.0, 0.0, 120.0, 80.0]])  # 頂邊中點 (80, 0) → foot (320, 800)


def test_missing_head_reuses_last_offset_instead_of_snapping_back():
    """配不到頭的那一格要沿用上次偏移，不可彈回框底邊中點。"""
    est = FootPointEstimator("head")

    first = est.estimate(0, _TILTED_BODY, _TILTED_HEAD)
    second = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))

    npt.assert_allclose(first, [[320.0, 800.0]])
    npt.assert_allclose(second, first)  # 沒有跳回 (200, 800)


def test_reused_offset_scales_with_box_size():
    """偏移量存成相對框尺寸的比例：人走遠、框變小時，沿用的偏移要等比縮小。"""
    est = FootPointEstimator("head")
    est.estimate(0, _TILTED_BODY, _TILTED_HEAD)  # 偏移 +120px = 框寬 400 的 0.3

    half = np.array([[0.0, 0.0, 200.0, 400.0, 7]])
    got = est.estimate(0, half, np.empty((0, 4)))

    npt.assert_allclose(got, [[100.0 + 0.3 * 200.0, 400.0]])


def test_track_that_never_matched_falls_back_to_bbox_bottom():
    """從沒配到過頭的軌跡沒有偏移可沿用，維持舊定義。"""
    est = FootPointEstimator("head")

    got = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))

    npt.assert_allclose(got, bbox_bottom_center(_TILTED_BODY[:, :4]))


def test_offset_expires_after_ttl():
    """太久沒配到頭就放棄沿用——姿勢早就變了，舊偏移不再有代表性。"""
    est = FootPointEstimator("head")
    est.estimate(0, _TILTED_BODY, _TILTED_HEAD)

    for _ in range(fp._OFFSET_TTL_FRAMES):
        got = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))
    npt.assert_allclose(got, [[320.0, 800.0]])  # 剛好在 TTL 內，仍沿用

    got = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))
    npt.assert_allclose(got, bbox_bottom_center(_TILTED_BODY[:, :4]))


def test_same_track_id_on_different_streams_does_not_leak():
    """不同路的同號軌跡不可共用偏移量。

    ultralytics 目前的 `track_id` 是全域計數器、跨路不會撞號，所以這個情境在產線上
    不會發生；測的是「改成 per-tracker 計數也不會壞」——那是別人的實作細節，而它一旦
    改變，這裡沒有 stream_id 就會靜默共用偏移量。
    """
    est = FootPointEstimator("head")
    est.estimate(0, _TILTED_BODY, _TILTED_HEAD)

    other = est.estimate(1, _TILTED_BODY, np.empty((0, 4)))

    npt.assert_allclose(other, bbox_bottom_center(_TILTED_BODY[:, :4]))


def test_bbox_bottom_method_never_reuses_offsets():
    """切回舊算法時，即使有 head 也不推算、不延續，逐列等於框底邊中點。"""
    est = FootPointEstimator("bbox_bottom")

    first = est.estimate(0, _TILTED_BODY, _TILTED_HEAD)
    second = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))

    npt.assert_allclose(first, bbox_bottom_center(_TILTED_BODY[:, :4]))
    npt.assert_allclose(second, bbox_bottom_center(_TILTED_BODY[:, :4]))


def test_head_shared_within_a_frame_is_not_written_to_cache():
    """一顆 head 被同一幀兩個 fbody 配到時，兩邊都不寫快取。

    配對沒有全域指派，共用時至少有一邊是錯配、且分不出是哪一邊（ADR-009）。寫進快取
    的話這一幀的錯配會被沿用最多 60 格，等於把單幀誤差放大成約兩秒的持續偏移——而輸出
    的 parquet 完全正常，沒有任何訊號。當格照樣用推算結果，只是不留下來。
    """
    est = FootPointEstimator("head")
    shared_head = np.array([[170.0, 0.0, 230.0, 60.0]])  # 頂邊中點 (200, 0)
    two_bodies = np.array(
        [
            [0.0, 0.0, 400.0, 800.0, 7],  # 中心 (200, 400)
            [100.0, 0.0, 500.0, 800.0, 8],  # 中心 (300, 400)
        ]
    )

    first = est.estimate(0, two_bodies, shared_head)

    npt.assert_allclose(first, [[200.0, 800.0], [400.0, 800.0]])  # 當格仍用推算結果
    assert est._offsets == {}

    # 下一格配不到頭：沒有偏移可沿用，兩條都退回框底邊中點
    second = est.estimate(0, two_bodies, np.empty((0, 4)))

    npt.assert_allclose(second, bbox_bottom_center(two_bodies[:, :4]))


def test_separate_heads_in_one_frame_are_both_cached():
    """同一幀各配各的頭時兩筆都要進快取——共用的判定不可波及正常路徑。"""
    est = FootPointEstimator("head")
    heads = np.array([[40.0, 0.0, 120.0, 80.0], [540.0, 0.0, 620.0, 80.0]])
    two_bodies = np.array(
        [
            [0.0, 0.0, 400.0, 800.0, 7],
            [500.0, 0.0, 900.0, 800.0, 8],
        ]
    )

    est.estimate(0, two_bodies, heads)

    assert set(est._offsets) == {(0, 7), (0, 8)}


def test_estimator_rejects_unknown_method():
    with pytest.raises(ValueError, match="未知的落腳點算法"):
        FootPointEstimator("pose")


def test_estimator_returns_empty_for_frame_without_tracks():
    est = FootPointEstimator("head")

    assert est.estimate(0, np.empty((0, 5)), _UPRIGHT_HEAD).shape == (0, 2)


def test_expired_state_is_pruned():
    """整天數萬條軌跡：過期狀態要被清掉，不能無限累積。"""
    est = FootPointEstimator("head")
    est.estimate(0, _TILTED_BODY, _TILTED_HEAD)
    assert len(est._offsets) == 1

    empty = np.empty((0, 4))
    other = np.array([[0.0, 0.0, 400.0, 800.0, 99]])
    for _ in range(fp._PRUNE_EVERY_FRAMES):
        est.estimate(0, other, empty)

    assert (0, 7) not in est._offsets


def test_prune_keeps_live_state_on_this_and_other_streams():
    """清理只能掃掉過期的那些。

    清過頭沒有任何訊號：被誤刪的軌跡下一格靜默彈回框底邊中點，parquet 完全正常，
    正是這個機制要防的那種跳動。這裡讓清理觸發時同時存在三種狀態，斷言只有過期的
    那一筆消失。
    """
    est = FootPointEstimator("head")
    empty = np.empty((0, 4))
    stale = np.array([[0.0, 0.0, 400.0, 800.0, 1]])
    fresh_other_stream = np.array([[0.0, 0.0, 400.0, 800.0, 2]])
    fresh_same_stream = np.array([[0.0, 0.0, 400.0, 800.0, 3]])

    est.estimate(0, stale, _TILTED_HEAD)  # 這一筆之後不再更新，會過期
    for _ in range(fp._PRUNE_EVERY_FRAMES - 2):
        est.estimate(0, stale, empty)
    # 清理觸發的前一格，讓另一路與本路各存一筆新鮮的偏移量
    est.estimate(1, fresh_other_stream, _TILTED_HEAD)
    est.estimate(0, fresh_same_stream, _TILTED_HEAD)  # 這一格觸發 prune

    assert (0, 1) not in est._offsets  # 過期，該刪
    assert (1, 2) in est._offsets  # 另一路的新鮮狀態不可被掃到
    assert (0, 3) in est._offsets  # 本路的新鮮狀態也不可被掃到
