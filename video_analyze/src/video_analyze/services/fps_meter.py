from collections import defaultdict
from dataclasses import dataclass


@dataclass
class FpsSummary:
    """一次 analyze 執行的 FPS 統計結果（純資料，供呼叫端 log 用）。"""

    per_camera_frames: dict[str, int]
    per_camera_fps: dict[str, float]
    total_frames: int
    overall_fps: float
    detection_fps: float
    tracking_fps: float
    elapsed_seconds: float


class FpsMeter:
    """累計 analyze 推理迴圈的處理格數與各階段耗時，最後換算平均 FPS。

    純運算、不做 logging、不自取時間來源（耗時由呼叫端量好傳入），方便單元測試。
    逐路 FPS 與整體 FPS 共用同一段 wall-clock（`summary` 的 `elapsed_seconds`）當
    分母，因此逐路 FPS 相加即整體 FPS，語意為「該路對整體吞吐的貢獻」。
    """

    def __init__(self) -> None:
        self._frames_per_camera: dict[str, int] = defaultdict(int)
        self._total_detect_seconds = 0.0
        self._total_track_seconds = 0.0

    def record(self, camera_id: str) -> None:
        """記錄某路攝影機完整處理了一格。"""
        self._frames_per_camera[camera_id] += 1

    def add_detection_time(self, seconds: float) -> None:
        """累計偵測側的耗時。

        呼叫端決定口徑：推論進程傳的是 `detector.submit` 與 `detector.collect` 兩段
        wall time 的合計（ADR-014）。**這與流水化之前的口徑不同**——`collect` 內等 GPU
        的時間仍算在內，但被 host 工作蓋掉的那段 GPU 時間不算，所以 `detection_fps`
        在流水化之後代表「host 為偵測付出的時間」而不是「一批從送出到拿回的時間」，
        不可與改動前的數字直接相減。端到端的比較看 `overall_fps`。
        """
        self._total_detect_seconds += seconds

    def add_tracking_time(self, seconds: float) -> None:
        """累計一格追蹤側的耗時。

        呼叫端決定口徑：追蹤進程傳的是**該格的全部工作**（裁切、拆分、`tracker.update`、
        落腳點、反算、累積結果），不只 `tracker.update`——`tracking_fps` 是該進程餘裕
        評估的分子，只算其中一段會高估餘裕。
        """
        self._total_track_seconds += seconds

    def summary(self, elapsed_seconds: float) -> FpsSummary:
        """依整段 wall-clock 與各階段累計耗時換算平均 FPS。

        任一分母 <= 0 時，對應的 FPS 回 0.0（避免除零，讓總結仍印得出來）。
        """
        per_camera_frames = dict(self._frames_per_camera)
        total_frames = sum(per_camera_frames.values())
        per_camera_fps = {
            camera_id: _safe_div(frames, elapsed_seconds)
            for camera_id, frames in per_camera_frames.items()
        }
        return FpsSummary(
            per_camera_frames=per_camera_frames,
            per_camera_fps=per_camera_fps,
            total_frames=total_frames,
            overall_fps=_safe_div(total_frames, elapsed_seconds),
            detection_fps=_safe_div(total_frames, self._total_detect_seconds),
            tracking_fps=_safe_div(total_frames, self._total_track_seconds),
            elapsed_seconds=elapsed_seconds,
        )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0
