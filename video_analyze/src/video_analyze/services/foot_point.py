"""落腳點推算：由 head 框推算「人站在地面的位置」，取代 fbody 框的底邊中點。

斜向俯視、廣角的店內攝影機下人體在畫面中是傾斜的，axis-aligned 的 fbody 框其底邊
中點經常落在人體外的地板上。人體主軸的兩端是頭頂與腳底，框中心是主軸中點，所以把
頭頂中點對框中心做點反射就能推回腳底：

    foot = 2 × C_fbody − H,  H = head 框的頂邊中點

H 取**頂邊中點**而非 head 框中心，是為了退化性質：人站直、且 head 框頂邊與 fbody 框
頂邊重合時，fbody 的垂直範圍就是 [頭頂, 腳底]、框中心是兩者中點，`2 × C_y − 頭頂`
精確等於 `y2`，推算結果與改動前一致。若改用 head 中心，同樣的人會系統性上偏半顆頭。

兩個框由模型各自迴歸，真實偵測下頂邊不會剛好對齊：head 框頂邊比 fbody 框頂邊低 δ 時，
推算結果就上偏 δ（站直與否無關）。所以退化性質是「δ = 0 時精確成立」，不是「站直的人
一定不受影響」——ADR-009 實測 82.3% 的列與框底邊中點不同、位移中位 29 px。

配對與選法的取捨（含被實測推翻的直覺）見 docs/adr/shared/009-head-based-foot-point.md。

配對、反射與偏移量都是**整格一次算完**的 numpy 運算，不逐框迴圈：一格 N 個框、M 顆頭
時，判準算在 `[N, M]` 的矩陣上。逐框版本每個框都要重算同一份 head 中心與面積，再做十
來個長度 M 的小陣列運算，那些固定開銷乘上 N 之後是追蹤進程 tracker 以外的最大一段
（issue #146）。改的只是計算佈局：兩條會影響輸出值的算式（`foot = 2 × C − H`、偏移沿用
的 `bottom + offset × size`）逐元素運算順序不變，配對本身則是離散決策。
"""

import numpy as np

# head 框面積佔 fbody 框的比例上限：超過就不是「框裡的一顆頭」（多半是把整個人
# 或上半身當成頭），採用它會讓反射基準嚴重偏移。
_MAX_HEAD_AREA_RATIO = 0.25
# 主軸（fbody 中心 → head 中心）與垂直線的夾角上限。魚眼邊緣的人體傾斜可達 30–40°，
# 上限放到 50° 才不會把要修的傾斜案例本身擋掉；再大就多半是配到鄰座的頭。
_MAX_AXIS_TILT_DEG = 50.0

FOOT_POINT_METHODS = ("head", "bbox_bottom")

# 配不到頭時最多沿用幾格前算出的偏移量。取 60 格：ByteTrack 的 `track_buffer` 是 30
# 格，軌跡中斷再接回也在這個範圍內；再久姿勢早就變了，舊偏移不再有代表性，寧可退回
# 框底邊中點。
#
# 這裡的「格」是**該路有軌跡的幀**，不是該路讀過的幀——`estimate` 對空幀在 tick 遞增
# 之前就早退，整路無人的時段不消耗 TTL。因此它涵蓋的實際時間跨度只會比「30fps 下
# 2 秒」長，長多少取決於該路的人流密度。
_OFFSET_TTL_FRAMES = 60
# 每這麼多格清一次過期狀態（整天數萬條軌跡，不清會一直累積）。「格」的口徑同上。
_PRUNE_EVERY_FRAMES = 300


def bbox_bottom_center(boxes: np.ndarray) -> np.ndarray:
    """框底邊中點——改用 head 推算之前的落腳點定義。

    Args:
        boxes: `[N, 4]` 的 xyxy 框；`N` 為 0 時回傳 `(0, 2)` 空陣列。

    Returns:
        `[N, 2]` 的 `(x, y)`。
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def _axis_tilt_deg(
    head_centers: np.ndarray, body_centers: np.ndarray
) -> np.ndarray:
    """主軸（body 中心 → head 中心）與垂直線的夾角（度），每個框對每顆頭各一個。

    head 在 body 中心正上方時為 0；水平偏移越大角度越大。head 落在中心以下（`dy<=0`）
    的情形由呼叫端另行排除，這裡只保證不除以 0。

    Args:
        head_centers: `[M, 2]` 的 head 中心。
        body_centers: `[N, 2]` 的 fbody 框中心。

    Returns:
        `[N, M]`，第 `i` 列第 `j` 行是第 `i` 個框配第 `j` 顆頭的傾角。
    """
    dx = head_centers[None, :, 0] - body_centers[:, None, 0]
    dy = body_centers[:, None, 1] - head_centers[None, :, 1]
    return np.degrees(np.arctan2(np.abs(dx), np.maximum(dy, 1e-6)))


def _match_heads(boxes: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """替每個 fbody 框各挑一顆 head，回傳其在 `heads` 中的索引；挑不到為 `-1`。

    四項條件全部滿足才算候選：head 中心在框內、head 面積不超過框的
    `_MAX_HEAD_AREA_RATIO`、head 中心在框中心之上（否則主軸方向反轉）、主軸傾角
    不超過 `_MAX_AXIS_TILT_DEG`。

    多顆候選時取**主軸最接近垂直**的那顆。一個 fbody 框內出現兩顆以上候選的比例約
    12–16%（cam003／cam008 實測），最常見的成因是框同時罩住前後兩個人；透視下後方
    的人在畫面中更高、更靠近框頂邊，因此「離框頂最近」這個直覺的判準會系統性挑中
    後方那個人，主軸判準則因為後方的人多半偏在框的左右上角而能排除他。實測比較與
    被否決的判準見 ADR-009。

    各框彼此獨立——同一顆 head 可以被多個框挑中（沒有全域指派，見 ADR-009 的已知
    限制），呼叫端要不要因此改變行為由它自己判斷。

    Args:
        boxes: `[N, 4]` 的 fbody 框。
        heads: `[M, 4]` 的 head 框。

    Returns:
        `[N]` 的整數索引，配不到的列為 `-1`；`N` 或 `M` 為 0 時整條回 `-1`
        （`argmin` 對長度 0 的軸會拋錯，也沒有候選可挑）。
    """
    if len(boxes) == 0 or len(heads) == 0:
        return np.full(len(boxes), -1, dtype=np.int64)

    body_centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
    )
    body_areas = np.maximum(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6
    )
    head_centers = np.stack(
        [(heads[:, 0] + heads[:, 2]) / 2, (heads[:, 1] + heads[:, 3]) / 2], axis=1
    )
    head_areas = (heads[:, 2] - heads[:, 0]) * (heads[:, 3] - heads[:, 1])
    tilt = _axis_tilt_deg(head_centers, body_centers)

    head_x = head_centers[None, :, 0]
    head_y = head_centers[None, :, 1]
    ok = (
        (head_x >= boxes[:, None, 0])
        & (head_x <= boxes[:, None, 2])
        & (head_y >= boxes[:, None, 1])
        & (head_y <= boxes[:, None, 3])
        & (head_areas[None, :] <= _MAX_HEAD_AREA_RATIO * body_areas[:, None])
        & (head_y < body_centers[:, None, 1])
        & (tilt <= _MAX_AXIS_TILT_DEG)
    )
    # 未通過的候選排到最後，argmin 才不會挑中它們；整列都沒通過的再標成 -1
    best = np.argmin(np.where(ok, tilt, np.inf), axis=1)
    return np.where(ok.any(axis=1), best, -1)


def _reflect(bodies: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """`foot = 2 × C_body − H`，`H` 取 head 框的頂邊中點，逐列對應。

    公式只寫這一份：無狀態的 `estimate_from_heads` 與帶跨幀延續的 `FootPointEstimator`
    都呼叫它，兩條路徑才不可能各自漂移（例如其中一邊被改成用 head 中心當基準）。

    Args:
        bodies: `[N, 4]` 的 fbody 框。
        heads: `[N, 4]` 的 head 框，第 `i` 列是配給第 `i` 個框的那一顆。

    Returns:
        `[N, 2]` 的落腳點。
    """
    centers = np.stack(
        [(bodies[:, 0] + bodies[:, 2]) / 2, (bodies[:, 1] + bodies[:, 3]) / 2], axis=1
    )
    head_top_mid = np.stack([(heads[:, 0] + heads[:, 2]) / 2, heads[:, 1]], axis=1)
    return 2 * centers - head_top_mid


def estimate_from_heads(boxes: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """對每個框配一顆 head 並推算落腳點，配不到的退回框底邊中點。

    Args:
        boxes: `[N, 4]` 的 xyxy 框（產線上是 tracker 輸出的 Kalman 平滑框，
            落腳點才與同列寫進 parquet 的 bbox 自洽）。
        heads: `[M, 4]` 的同一幀 head 框。

    Returns:
        `[N, 2]` 的落腳點；每一列都有值，不會是 NaN 或空值。
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    heads = np.asarray(heads, dtype=float).reshape(-1, 4)
    points = bbox_bottom_center(boxes)
    matched = _match_heads(boxes, heads)
    found = matched >= 0
    if found.any():
        points[found] = _reflect(boxes[found], heads[matched[found]])
    return points


def compute_foot_points(
    boxes: np.ndarray, heads: np.ndarray, method: str
) -> np.ndarray:
    """依設定的算法計算落腳點（無狀態版本，逐幀獨立）。

    產線走的是 `FootPointEstimator`——它在這之上多一層跨幀延續，避免同一條軌跡在
    「配到頭」與「配不到頭」之間跳動。此函式保留給不需要延續的呼叫端與測試。

    Args:
        boxes: `[N, 4]` 的 xyxy 框。
        heads: `[M, 4]` 的同一幀 head 框；`method` 為 `"bbox_bottom"` 時不使用。
        method: `"head"` 由頭部位置推算，`"bbox_bottom"` 為框底邊中點。

    Returns:
        `[N, 2]` 的落腳點。

    Raises:
        ValueError: `method` 不在 `FOOT_POINT_METHODS` 內。
    """
    if method == "head":
        return estimate_from_heads(boxes, heads)
    if method == "bbox_bottom":
        return bbox_bottom_center(boxes)
    raise ValueError(
        f"未知的落腳點算法 {method!r}，可用值為 {list(FOOT_POINT_METHODS)}。"
    )


class FootPointEstimator:
    """逐幀推算落腳點，並記住每條軌跡最近一次成功推算的偏移量。

    直接逐幀獨立推算會有一個嚴重的副作用：同一條軌跡在「這格配到頭、下一格配不到」
    之間切換時，落腳點會在推算點與框底邊中點之間彈跳。實測（`bucket_20260801_small`
    九台攝影機、126 萬列）跳動幅度 p90 達 144 px，遠超線段區域（1080p 基準 25 px），
    直接變成假 entry／假跨越：配不到頭的比例最高的 cam009（43.7%）entries 由 74 漲到
    282，而比例最低的 cam007（1.4%）幾乎不變。

    因此配不到頭時**不退回框底邊中點，改為沿用該軌跡上一次成功推算的偏移量**。偏移量
    存成相對框尺寸的比例（人走遠時框變小，偏移也該等比縮小），沿用時再乘回當前框的
    寬高。距上次成功推算超過 `_OFFSET_TTL_FRAMES` 格才放棄沿用、退回框底邊中點——姿勢
    早就變了，舊偏移不再有代表性。

    寫入快取的條件比「這格推算成功」更嚴：同一幀內若一顆 head 被兩個以上 fbody 配到，
    這些框的推算結果照樣使用，但都不寫入快取。配對沒有全域指派（ADR-009 的已知限制），
    共用時至少有一邊是錯配且分不出是哪一邊，寫進去等於把單幀的錯配放大成最多 60 個有
    軌跡的幀的持續偏移。

    狀態以 `(stream_id, track_id)` 為鍵。實測（ultralytics 8.4.75）`track_id` 由
    `BaseTrack._count` 這個 class 變數發放，同一進程內所有 `BYTETracker` 實例共用，
    跨路不會撞號——所以 `stream_id` 目前是多餘的。**刻意保留**：那是 ultralytics 的
    實作細節，若改成 per-tracker 計數，不同攝影機的軌跡會共用同一份偏移量，而且不會
    有任何錯誤訊息。
    """

    def __init__(self, method: str):
        """
        Args:
            method: 同 `compute_foot_points`；`"bbox_bottom"` 時完全不做配對與延續。

        Raises:
            ValueError: `method` 不在 `FOOT_POINT_METHODS` 內。
        """
        if method not in FOOT_POINT_METHODS:
            raise ValueError(
                f"未知的落腳點算法 {method!r}，可用值為 {list(FOOT_POINT_METHODS)}。"
            )
        self._method = method
        # (stream_id, track_id) -> (偏移比例 [2], 該路最後一次成功推算時的幀序)
        self._offsets: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self._ticks: dict[int, int] = {}

    def estimate(
        self, stream_id: int, tracks: np.ndarray, heads: np.ndarray
    ) -> np.ndarray:
        """算出這一格該路每條軌跡的落腳點。

        Args:
            stream_id: 該路的編號（`track_id` 的命名空間）。
            tracks: `MultiStreamByteTracker.update` 的輸出，需含 `[x1,y1,x2,y2,track_id]`
                前五欄；空陣列時回傳 `(0, 2)`。
            heads: `[M, 4]` 的同一幀 head 框。

        Returns:
            `[N, 2]` 的落腳點，逐列對應 `tracks`；每列都有值，不會是空值。

        Note:
            **同一格內 `track_id` 唯一是前提**。快取的寫入整格一次做完、才輪到配不到頭
            的列去讀，所以同一格出現兩列相同 `track_id` 時，配不到頭的那列會讀到同格
            另一列剛寫進去的偏移量（逐列版讀不到，會退回框底邊中點）。ByteTrack 的
            `joint_stracks` 依 `track_id` 去重，`_format_output` 不會輸出重複 id，故
            產線走不到；換掉 tracker 時要重新確認這件事。
        """
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

        matched = _match_heads(boxes, heads)
        found = matched >= 0
        sizes = np.stack(
            [
                np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6),
                np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6),
            ],
            axis=1,
        )
        bottom = points.copy()  # 覆寫前先留著，偏移量是相對它算的
        if found.any():
            points[found] = _reflect(boxes[found], heads[matched[found]])
        track_ids = tracks[:, 4].astype(np.int64).tolist()

        # 同一幀被兩個以上 fbody 配到的 head：其中至少一邊是錯配，而沒有全域指派就
        # 分不出是哪一邊。這種推算結果只用在這一格、不進快取，錯配才不會被沿用成最多
        # 60 格的持續偏移。
        used, counts = np.unique(matched[found], return_counts=True)
        shared = np.isin(matched, used[counts > 1])

        # 整格算好偏移量，只有「配到頭且沒被共用」的列寫進快取。配不到頭的列此刻
        # points 仍等於 bottom，算出來是 0、不會被寫進去
        offsets = (points - bottom) / sizes
        for i in np.nonzero(found & ~shared)[0]:
            self._offsets[(stream_id, track_ids[i])] = (offsets[i].copy(), tick)

        rows = []
        remembered = []
        for i in np.nonzero(~found)[0]:
            entry = self._offsets.get((stream_id, track_ids[i]))
            if entry is not None and tick - entry[1] <= _OFFSET_TTL_FRAMES:
                rows.append(i)
                remembered.append(entry[0])
        if rows:
            index = np.asarray(rows)
            points[index] = bottom[index] + np.asarray(remembered) * sizes[index]

        self._prune(tick)
        return points

    def _prune(self, tick: int) -> None:
        """丟掉已過期的偏移量，避免整天累積數萬條死軌跡的狀態。

        清理由跑到 `_PRUNE_EVERY_FRAMES` 整數格的那一路觸發，但掃的是**所有路**的
        條目，過期與否各自用該路自己的 tick 判斷——tick 是 per-stream 的，拿觸發那
        一路的 tick 去比別路會誤刪還活著的軌跡。

        殘留：某一路讀完後就不再呼叫 `estimate`，它的 tick 停在最後一格，最後
        `_OFFSET_TTL_FRAMES` 格內建立的條目相對這個停住的 tick 永遠算新鮮，會留到
        進程結束。上限是該路收尾那 60 格內仍活著的軌跡數，量級遠小於整天的軌跡數，
        接受這個殘留而不另外做「該路已結束」的通知。

        Args:
            tick: 觸發這次呼叫的那一路的當前幀序，只用來決定要不要清。
        """
        if tick % _PRUNE_EVERY_FRAMES:
            return
        self._offsets = {
            k: v
            for k, v in self._offsets.items()
            if self._ticks[k[0]] - v[1] <= _OFFSET_TTL_FRAMES
        }
