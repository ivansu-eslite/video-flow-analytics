import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from video_flow_analytics.core.config import settings


class MultiStreamByteTracker:
    """多路 ByteTrack 狀態管理器"""

    def __init__(self, num_streams: int):
        self.trackers = {
            i: BYTETracker(args=settings.tracker) for i in range(num_streams)
        }

    def update(self, stream_id: int, yolo_boxes: Boxes) -> np.ndarray:
        tracker = self.trackers.get(stream_id)
        if tracker is None:
            return np.array([])
        # YOLO 推論可能在 CUDA 上執行，BYTETracker 需要 CPU tensor
        return tracker.update(yolo_boxes.cpu())
