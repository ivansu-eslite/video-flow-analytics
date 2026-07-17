import logging

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_analyze.config import settings

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO 偵測器包裝層，隔離強耦合"""

    def __init__(self):
        """載入 `settings.model.model_path` 指定的權重；有 CUDA 就用 GPU，
        否則 fallback CPU（並記錄警告，因為 CPU 推論明顯變慢）。
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning(
                "未偵測到可用的 CUDA 裝置，改用 CPU 執行（推論速度會明顯變慢）。"
            )
            device = "cpu"
        self.model = YOLO(settings.model.model_path).to(device)

    def predict(self, batch_frames: list[np.ndarray]) -> list[Results]:
        """對一批影格執行物件偵測，僅偵測 `person`（COCO 類別 0）。

        Args:
            batch_frames: 要偵測的影格清單（BGR）；空清單直接回傳空結果，
                不呼叫模型。

        Returns:
            與 `batch_frames` 一一對應的 ultralytics `Results` 列表。
        """
        if not batch_frames:
            return []
        return self.model.predict(
            batch_frames,
            verbose=False,
            classes=[0],  # COCO：0 = person，僅偵測人
        )
