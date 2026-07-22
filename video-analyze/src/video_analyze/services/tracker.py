import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from video_analyze.models.config import settings


class MultiStreamByteTracker:
    """多路 ByteTrack 狀態管理器：每路各自持有獨立的 `BYTETracker` 實例，
    讓 `track_id` 能跨片段延續（同一路內連續呼叫 `update` 才會累積軌跡）。
    """

    def __init__(self, num_streams: int):
        """建立每一路各自獨立的 `BYTETracker` 實例。

        Args:
            num_streams: 要追蹤的攝影機路數，依此建立對應數量的獨立
                `BYTETracker` 實例（stream_id 為 0 ~ num_streams - 1）。
        """
        self.trackers = {
            i: BYTETracker(args=settings.tracker) for i in range(num_streams)
        }

    def update(self, stream_id: int, yolo_boxes: Boxes) -> np.ndarray:
        """把某一路當前這批 YOLO 偵測框餵給對應的追蹤器，取得更新後的軌跡。

        Args:
            stream_id: 要更新的攝影機編號。
            yolo_boxes: YOLO 偵測結果的 boxes（可能位於 CUDA，內部會轉回 CPU
                供 `BYTETracker` 使用）。

        Returns:
            numpy 陣列，每列格式由 ultralytics `BYTETracker.update` 決定，
            目前為 `[x1, y1, x2, y2, track_id, score, cls, idx]`（本專案唯一
            定義此格式之處，`TrackingResultCollector.add` 與
            `TrackAnnotator.draw_bboxes` 皆沿用此處的說明，ultralytics
            版本升級改變欄位時需一併檢查這兩處）；`stream_id` 不存在或當批
            無存活軌跡時回傳空陣列。
        """
        tracker = self.trackers.get(stream_id)
        if tracker is None:
            return np.array([])
        # YOLO 推論可能在 CUDA 上執行，BYTETracker 需要 CPU tensor
        return tracker.update(yolo_boxes.cpu())
