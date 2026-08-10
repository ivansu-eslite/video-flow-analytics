"""落腳點推算：由 head 框推算「人站在地面的位置」，取代 fbody 框的底邊中點。

斜向俯視、廣角的店內攝影機下人體在畫面中是傾斜的，axis-aligned 的 fbody 框其底邊
中點經常落在人體外的地板上。人體主軸的兩端是頭頂與腳底，框中心是主軸中點，所以把
頭頂中點對框中心做點反射就能推回腳底：

    foot = 2 × C_fbody − H,  H = head 框的頂邊中點

H 取**頂邊中點**而非 head 框中心，是為了退化性質：人站直時 fbody 的垂直範圍就是
[頭頂, 腳底]、框中心是兩者中點，`2 × C_y − 頭頂` 精確等於 `y2`，推算結果與改動前
一致。若改用 head 中心，站直的人會系統性上偏半顆頭。

配對與選法的取捨（含被實測推翻的直覺）見 docs/adr/009-head-based-foot-point.md。
"""

import numpy as np

# head 框面積佔 fbody 框的比例上限：超過就不是「框裡的一顆頭」（多半是把整個人
# 或上半身當成頭），採用它會讓反射基準嚴重偏移。
_MAX_HEAD_AREA_RATIO = 0.25
# 主軸（fbody 中心 → head 中心）與垂直線的夾角上限。魚眼邊緣的人體傾斜可達 30–40°，
# 上限放到 50° 才不會把要修的傾斜案例本身擋掉；再大就多半是配到鄰座的頭。
_MAX_AXIS_TILT_DEG = 50.0

FOOT_POINT_METHODS = ("head", "bbox_bottom")

# 配不到頭時最多沿用幾格前算出的偏移量。取 60 格（30fps 約 2 秒）：ByteTrack 的
# `track_buffer` 是 30 格，軌跡中斷再接回也在這個範圍內；再久姿勢早就變了，舊偏移
# 不再有代表性，寧可退回框底邊中點。
_OFFSET_TTL_FRAMES = 60
# 每這麼多格清一次過期狀態（整天數萬條軌跡，不清會一直累積）。
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
    head_centers: np.ndarray, body_center: np.ndarray
) -> np.ndarray:
    """主軸（body 中心 → head 中心）與垂直線的夾角（度）。

    head 在 body 中心正上方時為 0；水平偏移越大角度越大。head 落在中心以下（`dy<=0`）
    的情形由呼叫端另行排除，這裡只保證不除以 0。
    """
    dx = head_centers[:, 0] - body_center[0]
    dy = body_center[1] - head_centers[:, 1]
    return np.degrees(np.arctan2(np.abs(dx), np.maximum(dy, 1e-6)))


def _match_head(body: np.ndarray, heads: np.ndarray) -> int | None:
    """替單一 fbody 框挑一顆 head，回傳其在 `heads` 中的索引；挑不到回傳 `None`。

    四項條件全部滿足才算候選：head 中心在框內、head 面積不超過框的
    `_MAX_HEAD_AREA_RATIO`、head 中心在框中心之上（否則主軸方向反轉）、主軸傾角
    不超過 `_MAX_AXIS_TILT_DEG`。

    多顆候選時取**主軸最接近垂直**的那顆。一個 fbody 框內出現兩顆以上候選的比例約
    12–16%（cam003／cam008 實測），最常見的成因是框同時罩住前後兩個人；透視下後方
    的人在畫面中更高、更靠近框頂邊，因此「離框頂最近」這個直覺的判準會系統性挑中
    後方那個人，主軸判準則因為後方的人多半偏在框的左右上角而能排除他。實測比較與
    被否決的判準見 ADR-009。

    Args:
        body: 單一 fbody 框 `[x1, y1, x2, y2]`。
        heads: `[M, 4]` 的 head 框；`M` 為 0 時直接回傳 `None`。
    """
    if len(heads) == 0:
        return None
    x1, y1, x2, y2 = body
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    body_area = max((x2 - x1) * (y2 - y1), 1e-6)

    head_centers = np.stack(
        [(heads[:, 0] + heads[:, 2]) / 2, (heads[:, 1] + heads[:, 3]) / 2], axis=1
    )
    head_areas = (heads[:, 2] - heads[:, 0]) * (heads[:, 3] - heads[:, 1])
    tilt = _axis_tilt_deg(head_centers, center)

    ok = (
        (head_centers[:, 0] >= x1)
        & (head_centers[:, 0] <= x2)
        & (head_centers[:, 1] >= y1)
        & (head_centers[:, 1] <= y2)
        & (head_areas <= _MAX_HEAD_AREA_RATIO * body_area)
        & (head_centers[:, 1] < center[1])
        & (tilt <= _MAX_AXIS_TILT_DEG)
    )
    if not ok.any():
        return None
    # 未通過的候選排到最後，argmin 才不會挑中它們
    return int(np.argmin(np.where(ok, tilt, np.inf)))


def _reflect(body: np.ndarray, head: np.ndarray) -> np.ndarray:
    """`foot = 2 × C_body − H`，`H` 取 head 框的頂邊中點。

    公式只寫這一份：無狀態的 `estimate_from_heads` 與帶跨幀延續的 `FootPointEstimator`
    都呼叫它，兩條路徑才不可能各自漂移（例如其中一邊被改成用 head 中心當基準）。
    """
    center = np.array([(body[0] + body[2]) / 2, (body[1] + body[3]) / 2])
    head_top_mid = np.array([(head[0] + head[2]) / 2, head[1]])
    return 2 * center - head_top_mid


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
    for i, body in enumerate(boxes):
        j = _match_head(body, heads)
        if j is not None:
            points[i] = _reflect(body, heads[j])
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
    九台攝影機、126 萬列）跳動幅度 p90 達 144 px，遠超判定緩衝帶（1080p 基準 25 px），
    直接變成假 entry／假跨越：配不到頭的比例最高的 cam009（43.7%）entries 由 74 漲到
    282，而比例最低的 cam007（1.4%）幾乎不變。

    因此配不到頭時**不退回框底邊中點，改為沿用該軌跡上一次成功推算的偏移量**。偏移量
    存成相對框尺寸的比例（人走遠時框變小，偏移也該等比縮小），沿用時再乘回當前框的
    寬高。距上次成功推算超過 `_OFFSET_TTL_FRAMES` 格才放棄沿用、退回框底邊中點——姿勢
    早就變了，舊偏移不再有代表性。

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

        for i, body in enumerate(boxes):
            key = (stream_id, int(tracks[i][4]))
            size = np.array(
                [max(body[2] - body[0], 1e-6), max(body[3] - body[1], 1e-6)]
            )
            bottom = points[i].copy()  # 覆寫前先留著，偏移量是相對它算的
            j = _match_head(body, heads)
            if j is not None:
                points[i] = _reflect(body, heads[j])
                self._offsets[key] = ((points[i] - bottom) / size, tick)
                continue
            remembered = self._offsets.get(key)
            if remembered is not None and tick - remembered[1] <= _OFFSET_TTL_FRAMES:
                points[i] = bottom + remembered[0] * size

        self._prune(stream_id, tick)
        return points

    def _prune(self, stream_id: int, tick: int) -> None:
        """丟掉該路已過期的偏移量，避免整天累積數萬條死軌跡的狀態。"""
        if tick % _PRUNE_EVERY_FRAMES:
            return
        self._offsets = {
            k: v
            for k, v in self._offsets.items()
            if k[0] != stream_id or tick - v[1] <= _OFFSET_TTL_FRAMES
        }
