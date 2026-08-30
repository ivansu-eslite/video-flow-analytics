"""落腳點推算的行為測試（合成 bbox，不需影片、不需模型）。

守住四件事：

1. **退化性質**——人站直、且 head 框頂邊與 fbody 框頂邊重合時，推算結果精確等於改動
   前的框底邊中點。反射基準若誤用 head 框中心而非頂邊中點，這種人會系統性上偏半顆頭，
   這裡會抓到。兩個框頂邊差 δ 時推算結果上偏 δ，也一併釘住——真實偵測下 δ 不會是 0，
   「站直的人不受影響」只在合成 fixture 的 δ = 0 下成立，不是產線上的保證。
2. **傾斜時往正確方向移動**——head 偏左上時落腳點要往右下，這正是要修的偏移。
3. **配不到 head 時退回底邊中點**，且不產生空值或例外（下游沒有處理 null 的路徑）。
4. **多候選的選法**——取主軸最接近垂直的那顆。實測顯示「離框頂最近」會系統性挑到
   後方那個人的頭（透視下他在畫面中更高），故那個直覺的判準刻意不採用，見 ADR-009。
"""

from collections import Counter

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

# 一個站直的人：框 200 寬、800 高，頭在框頂正中央、佔框高 1/8。兩個框的頂邊刻意齊平
# （`y1` 都是 0.0），退化性質才成立——這是合成 fixture 才有的條件，見下面的 δ≠0 那支。
_UPRIGHT_BODY = np.array([[100.0, 0.0, 300.0, 800.0]])
_UPRIGHT_HEAD = np.array([[160.0, 0.0, 240.0, 100.0]])


def test_flush_head_top_reproduces_bbox_bottom_center_exactly():
    """head 框頂邊與 fbody 框頂邊齊平的站直人，推算結果與改動前逐值相同。"""
    got = estimate_from_heads(_UPRIGHT_BODY, _UPRIGHT_HEAD)

    npt.assert_allclose(got, bbox_bottom_center(_UPRIGHT_BODY))
    npt.assert_allclose(got, [[200.0, 800.0]])


def test_head_top_below_body_top_shifts_foot_point_up_by_that_gap():
    """兩個框頂邊差 δ 時，站直的人推算結果上偏 δ——退化性質的實際適用範圍。

    head 與 fbody 由模型各自迴歸，真實偵測下頂邊不會剛好對齊，所以上面那條「與改動前
    逐值相同」不能讀成「站直的人在產線上不受影響」。ADR-009 的實測（82.3% 的列與框底
    邊中點不同、位移中位 29 px）已反映這件事，這裡把 δ≠0 的行為本身釘住。
    """
    delta = 12.0
    head = _UPRIGHT_HEAD + np.array([0.0, delta, 0.0, delta])

    got = estimate_from_heads(_UPRIGHT_BODY, head)

    npt.assert_allclose(got, [[200.0, 800.0 - delta]])


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


def test_shared_head_frame_leaves_earlier_offset_intact():
    """共用的那一格只是不寫快取，不能連既有的偏移一起清掉。

    清掉的話，這條軌跡下一次配不到頭就靜默彈回框底邊中點——正是偏移沿用要防的跳動，
    而 parquet 完全正常。
    """
    est = FootPointEstimator("head")
    est.estimate(0, _TILTED_BODY, _TILTED_HEAD)  # 單獨配到，偏移 +120px = 框寬的 0.3

    shared_head = np.array([[170.0, 0.0, 230.0, 60.0]])
    with_neighbour = np.array(
        [
            [0.0, 0.0, 400.0, 800.0, 7],
            [100.0, 0.0, 500.0, 800.0, 8],
        ]
    )
    est.estimate(0, with_neighbour, shared_head)  # 這一格兩條共用同一顆頭
    after = est.estimate(0, _TILTED_BODY, np.empty((0, 4)))

    npt.assert_allclose(after, [[320.0, 800.0]])  # 沿用第一格的 0.3，非共用那格的 0


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


def test_prune_clears_expired_state_of_a_stream_that_stopped_calling():
    """已經不再送幀的那一路，留下的過期狀態要被別路的清理掃掉。

    每路只清自己的話，先讀完的攝影機其狀態會留到進程結束。過期與否要用**該路自己的**
    tick 判斷：tick 是 per-stream 的，拿觸發清理那一路的 tick 去比別路，只要兩路進度
    有差就會誤刪還活著的軌跡（上一支測的就是這件事）。
    """
    est = FootPointEstimator("head")
    empty = np.empty((0, 4))
    stopped = np.array([[0.0, 0.0, 400.0, 800.0, 1]])
    ongoing = np.array([[0.0, 0.0, 400.0, 800.0, 2]])

    est.estimate(1, stopped, _TILTED_HEAD)
    for _ in range(fp._OFFSET_TTL_FRAMES + 1):  # 跑過 TTL 之後就不再呼叫
        est.estimate(1, stopped, empty)
    assert (1, 1) in est._offsets  # 該路自己的清理週期還沒到

    for _ in range(fp._PRUNE_EVERY_FRAMES):  # 由還在跑的那一路觸發清理
        est.estimate(0, ongoing, empty)

    assert (1, 1) not in est._offsets


# --- 向量化與逐框參考實作等價 -------------------------------------------------
#
# 配對、反射與偏移量在 issue #146 改成整格一次算完（原本是每個框各跑一次
# `_match_head`、再逐列建小陣列）。改的是計算佈局，輸出必須逐值不變——而「逐值」在這裡
# 不是修辭：落腳點會進 parquet，下游 line／zone 的跨越判定吃的就是這兩欄。
#
# 參考實作是改動前那一份的最小重寫，**刻意抄在測試檔裡**：從產線程式碼 import 一份
# 「舊版」的話，兩邊會一起被改掉，這支測試就退化成自己跟自己比。


def _reference_match_head(body, heads):
    """改動前的逐框配對：替單一 fbody 框挑一顆 head，挑不到回 `None`。"""
    if len(heads) == 0:
        return None
    x1, y1, x2, y2 = body
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    body_area = max((x2 - x1) * (y2 - y1), 1e-6)

    head_centers = np.stack(
        [(heads[:, 0] + heads[:, 2]) / 2, (heads[:, 1] + heads[:, 3]) / 2], axis=1
    )
    head_areas = (heads[:, 2] - heads[:, 0]) * (heads[:, 3] - heads[:, 1])
    dx = head_centers[:, 0] - center[0]
    dy = center[1] - head_centers[:, 1]
    tilt = np.degrees(np.arctan2(np.abs(dx), np.maximum(dy, 1e-6)))

    ok = (
        (head_centers[:, 0] >= x1)
        & (head_centers[:, 0] <= x2)
        & (head_centers[:, 1] >= y1)
        & (head_centers[:, 1] <= y2)
        & (head_areas <= fp._MAX_HEAD_AREA_RATIO * body_area)
        & (head_centers[:, 1] < center[1])
        & (tilt <= fp._MAX_AXIS_TILT_DEG)
    )
    if not ok.any():
        return None
    return int(np.argmin(np.where(ok, tilt, np.inf)))


def _reference_reflect(body, head):
    """改動前的逐列反射。"""
    center = np.array([(body[0] + body[2]) / 2, (body[1] + body[3]) / 2])
    head_top_mid = np.array([(head[0] + head[2]) / 2, head[1]])
    return 2 * center - head_top_mid


def _reference_estimate_from_heads(boxes, heads):
    """改動前的 `estimate_from_heads`。"""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    heads = np.asarray(heads, dtype=float).reshape(-1, 4)
    points = bbox_bottom_center(boxes)
    for i, body in enumerate(boxes):
        j = _reference_match_head(body, heads)
        if j is not None:
            points[i] = _reference_reflect(body, heads[j])
    return points


class _ReferenceEstimator:
    """改動前的 `FootPointEstimator`：逐框配對、逐列建 size 與偏移量。

    `_prune` 與 TTL／tick 的語義不重寫，直接沿用產線那一份——這支測試要比的是計算
    佈局，不是跨幀狀態的生命週期規則（那些由上面既有的測試各自釘住）。
    """

    def __init__(self, method):
        self._method = method
        self._offsets = {}
        self._ticks = {}

    def estimate(self, stream_id, tracks, heads):
        tracks = np.asarray(tracks, dtype=float)
        if tracks.size == 0:
            return np.empty((0, 2))
        boxes = tracks[:, :4]
        points = bbox_bottom_center(boxes)
        if self._method == "bbox_bottom":
            return points

        tick = self._ticks.get(stream_id, 0) + 1
        self._ticks[stream_id] = tick
        heads = np.asarray(heads, dtype=float).reshape(-1, 4)

        matched = [_reference_match_head(body, heads) for body in boxes]
        shared = {
            j
            for j, n in Counter(j for j in matched if j is not None).items()
            if n > 1
        }

        for i, body in enumerate(boxes):
            key = (stream_id, int(tracks[i][4]))
            size = np.array(
                [max(body[2] - body[0], 1e-6), max(body[3] - body[1], 1e-6)]
            )
            bottom = points[i].copy()
            j = matched[i]
            if j is not None:
                points[i] = _reference_reflect(body, heads[j])
                if j not in shared:
                    self._offsets[key] = ((points[i] - bottom) / size, tick)
                continue
            remembered = self._offsets.get(key)
            if (
                remembered is not None
                and tick - remembered[1] <= fp._OFFSET_TTL_FRAMES
            ):
                points[i] = bottom + remembered[0] * size

        fp.FootPointEstimator._prune(self, tick)
        return points


def _random_frame(rng, *, n_bodies, n_heads):
    """一格合成資料：框在推論尺度的畫布上，head 刻意含各種邊界情形。

    三種頭混在一起，`_match_heads` 七個判準各自可能翻轉的區域才都抽得到：

    - **一般**（機率 0.45）：位置與大小都落在判準內側，配得到——沒有這種的話整批都
      配不到頭，新舊實作都只回框底邊中點，比的是兩份空跑。
    - **邊界**（機率 0.3）：面積比壓在 25% 上下、主軸傾角壓在 50° 上下
      （`tan 50° ≈ 1.19`，所以 dx/dy 取在 1.19 附近），落在門檻兩側。
    - **無關**（機率 0.25）：畫布上隨機一顆，多半在所有框外。

    前兩種都從既有的框長出來，所以同一顆頭常常同時落在兩個重疊的框裡，共用路徑
    也走得到。
    """
    x1 = rng.uniform(0, 500, n_bodies)
    y1 = rng.uniform(0, 250, n_bodies)
    bodies = np.stack(
        [x1, y1, x1 + rng.uniform(20, 140, n_bodies), y1 + rng.uniform(40, 300, n_bodies)],
        axis=1,
    )

    heads = []
    for _ in range(n_heads):
        if len(bodies) and rng.random() < 0.75:
            # 從某個框身上長出來的頭
            body = bodies[rng.integers(len(bodies))]
            width = body[2] - body[0]
            height = body[3] - body[1]
            centre_y = (body[1] + body[3]) / 2
            boundary = rng.random() < 0.4
            # 傾角：dy 先定，dx 由目標傾角回推
            dy = rng.uniform(0.05, 0.45) * height
            tilt_deg = rng.uniform(45.0, 55.0) if boundary else rng.uniform(0.0, 35.0)
            dx = np.tan(np.radians(tilt_deg)) * dy * rng.choice([-1.0, 1.0])
            head_cx = (body[0] + body[2]) / 2 + dx
            head_cy = centre_y - dy
            ratio = rng.uniform(0.2, 0.32) if boundary else rng.uniform(0.02, 0.15)
            side = np.sqrt(ratio * width * height)
            heads.append(
                [
                    head_cx - side / 2,
                    head_cy - side / 2,
                    head_cx + side / 2,
                    head_cy + side / 2,
                ]
            )
        else:
            hx = rng.uniform(0, 640)
            hy = rng.uniform(0, 384)
            side = rng.uniform(8, 40)
            heads.append([hx, hy, hx + side, hy + side])
    return bodies, np.asarray(heads, dtype=float).reshape(-1, 4)


@pytest.mark.parametrize("seed", range(8))
def test_vectorised_matching_matches_the_per_box_reference(seed):
    """無狀態路徑：整格向量化與逐框參考實作逐值相同（含 N=0、M=0）。"""
    rng = np.random.default_rng(seed)

    for n_bodies, n_heads in [(0, 3), (3, 0), (0, 0), (1, 1), (5, 4), (9, 12)]:
        bodies, heads = _random_frame(rng, n_bodies=n_bodies, n_heads=n_heads)

        got = estimate_from_heads(bodies, heads)
        want = _reference_estimate_from_heads(bodies, heads)

        assert got.shape == want.shape
        assert np.array_equal(got, want), (n_bodies, n_heads)


def test_random_frames_actually_exercise_matching_and_sharing():
    """上面那支的素材真的走得到配對與共用兩條路徑——不然它比的是兩份空跑。

    亂數 fixture 最容易失效的方式是「全都配不到頭」：那時新舊實作都只回框底邊中點，
    測試照樣綠，而向量化的配對一行都沒被執行到。
    """
    rng = np.random.default_rng(0)
    matched_rows = 0
    shared_frames = 0

    for _ in range(60):
        bodies, heads = _random_frame(rng, n_bodies=6, n_heads=5)
        matched = fp._match_heads(bodies, heads)
        found = matched[matched >= 0]
        matched_rows += len(found)
        _, counts = np.unique(found, return_counts=True)
        shared_frames += int((counts > 1).any())

    assert matched_rows > 100  # 實測 137/360 列配到頭
    assert shared_frames >= 5  # 實測 60 格裡有 12 格出現共用


def test_vectorised_estimator_matches_the_per_row_reference_frame_by_frame():
    """有狀態路徑：逐格輸出與逐格的 `_offsets`（key 集合與值）都要一致。

    軌跡刻意有延續、有中斷、有跨 stream，偏移量的寫入／沿用／過期三條路徑才都走得到；
    比對放在每一格之後，跨幀狀態一旦開始漂移就當場紅，不會被後面的格數稀釋掉。
    """
    rng = np.random.default_rng(20260829)
    new = FootPointEstimator("head")
    old = _ReferenceEstimator("head")

    for frame in range(400):
        stream_id = int(rng.integers(2))
        n_bodies = int(rng.integers(0, 7))
        bodies, heads = _random_frame(
            rng, n_bodies=n_bodies, n_heads=int(rng.integers(0, 6))
        )
        # 軌跡編號從小池子抽：同一個 id 會在數格後再出現（中斷後接回），也會有從沒
        # 配到過頭的 id
        track_ids = rng.choice(np.arange(1, 12), size=n_bodies, replace=False)
        tracks = np.concatenate(
            [bodies, np.asarray(track_ids, dtype=float).reshape(-1, 1)], axis=1
        ).reshape(-1, 5)

        got = new.estimate(stream_id, tracks, heads)
        want = old.estimate(stream_id, tracks, heads)

        assert np.array_equal(got, want), f"第 {frame} 格的落腳點不同"
        assert set(new._offsets) == set(old._offsets), f"第 {frame} 格的快取鍵不同"
        for key, (offset, tick) in new._offsets.items():
            assert np.array_equal(offset, old._offsets[key][0]), f"第 {frame} 格 {key}"
            assert tick == old._offsets[key][1], f"第 {frame} 格 {key} 的 tick"
