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

    Args:
        value: 待處理的值，通常來自 ckpt 的 `train_args`。

    Returns:
        `value` 為非空字串時回傳其檔名（`Path(value).name`）；
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

    Args:
        model: 已載入權重的 `ultralytics.YOLO` 實例。
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


def _validate_classes(model: YOLO) -> None:
    """驗證 `settings.model.classes` 指定的類別 id 皆存在於已載入模型的類別定義。

    避免權重與 `classes` 設定不一致卻靜默通過——常見於指定的權重檔案遺失、
    ultralytics 轉而 fallback 下載到別的預設模型（如 COCO 版），此時類別 id
    雖然存在，語義卻已經不同（如 CrowdHuman 的 `2=fbody` 對到 COCO 的 `2=car`）。

    Args:
        model: 已載入權重的 `ultralytics.YOLO` 實例。

    Raises:
        ValueError: `settings.model.classes` 內有不存在於 `model.names` 的類別 id。
    """
    names = getattr(model, "names", None)
    if not names:
        return
    missing = [c for c in settings.model.classes if c not in names]
    if missing:
        raise ValueError(
            f"settings.model.classes 指定的類別 id {missing} 不存在於已載入模型的"
            f"類別定義 {names}；權重可能與預期不符（例如指定的權重檔案遺失，"
            "ultralytics 已 fallback 下載到不同的預設模型）。"
        )


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
        _validate_classes(self.model)

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
