import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_analyze.config import settings

logger = logging.getLogger(__name__)


def _basename(value: object) -> object:
    """把（可能是）訓練機器上的絕對路徑只留檔名，避免外洩訓練環境路徑。

    非字串或空字串原樣回傳。
    """
    if not isinstance(value, str) or not value:
        return value
    return Path(value).name


def _log_model_metadata(model: YOLO) -> None:
    """記錄已載入權重的 metadata（class 名稱、base 架構、訓練版本、日期、指標），
    方便日後追溯實際跑的是哪個訓練版本。

    以 `getattr`／`.get` 防護讀取，缺欄位只記拿得到的部分；任何例外都只
    `warning`，不讓模型載入失敗。
    """
    try:
        names = getattr(model, "names", None)
        if names:
            logger.info("模型類別: %s", names)

        ckpt = getattr(model, "ckpt", None)
        if not isinstance(ckpt, dict):
            logger.warning("權重無 ckpt metadata，略過記錄。")
            return

        train_args = ckpt.get("train_args")
        if isinstance(train_args, dict):
            logger.info(
                "訓練參數: base 模型=%s, 訓練資料集=%s",
                _basename(train_args.get("model")),
                _basename(train_args.get("data")),
            )
        logger.info("訓練 ultralytics 版本: %s", ckpt.get("version"))
        logger.info("訓練日期: %s", ckpt.get("date"))
        logger.info("驗證指標: %s", ckpt.get("train_metrics"))
    except Exception:
        logger.warning("記錄模型 metadata 時發生例外，略過。", exc_info=True)


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
        _log_model_metadata(self.model)

    def predict(self, batch_frames: list[np.ndarray]) -> list[Results]:
        """對一批影格執行物件偵測，僅保留 `settings.model.classes` 指定的類別
        （預設為 fbody）。

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
            classes=settings.model.classes,
        )
