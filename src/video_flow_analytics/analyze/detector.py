import logging

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_flow_analytics.core.config import settings

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO 偵測器包裝層，隔離強耦合"""

    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning(
                "未偵測到可用的 CUDA 裝置，改用 CPU 執行（推論速度會明顯變慢）。"
            )
            device = "cpu"
        self.model = YOLO(settings.model.model_path).to(device)

    def predict(self, batch_frames: list[np.ndarray]) -> list[Results]:
        """執行物件偵測並回傳原始 Result 列表"""
        if not batch_frames:
            return []
        return self.model.predict(
            batch_frames,
            verbose=False,
            batch=settings.model.batch,
            classes=[0],  # COCO：0 = person，僅偵測人
        )
