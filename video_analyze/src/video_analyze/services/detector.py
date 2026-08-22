from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from vfa_observability import StructuredLogger

from video_analyze.models.config import settings
from video_analyze.services.engine_metadata import (
    current_gpu_environment,
    read_engine_metadata,
    validate_engine_metadata,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH

logger = StructuredLogger(component="detector")

# 正式推論路徑只吃這一種副檔名。Torch 權重的推論路徑不在套件內（ADR-011）——它活在
# `tools/build_engine.py`／`tools/compare_backend.py`，那兩支不隨 wheel 出貨。
ENGINE_SUFFIX = ".engine"


def _require_engine_file(model_path: Path) -> None:
    """`model_path` 必須是**存在的** `.engine` 檔，兩條各自 fail loud。

    副檔名這條擋的是「還指著 `.pt`」：那條路載得起來、跑得出結果，只是慢一個量級而
    完全沒有訊號。

    檔案存在性這條擋的是 ultralytics `check_file` 的三種替代行為——**它們都會讓
    「跑的不是你以為的那顆模型」而輸出檔本身完全正常**：

    - `check_model_file_from_stem` 會把 `yolo26m` 這種沒有副檔名的值補成 `yolo26m.pt`
      並靜默下載官方權重；官方權重的類別 id 剛好也存在，`_validate_classes` 只驗 id
      存在，擋不下「id 存在但語義不同」。
    - 檔案不存在時 `check_file` 會 `glob.glob(ROOT/"**"/file, recursive=True)`
      **遞迴搜尋整個 ultralytics 套件目錄**同名檔，找到就用。
    - `gs://` 開頭的值會被改寫成 `https://storage.googleapis.com/...` 公開下載
      （`REMOTE_FILE_PREFIXES`）——連 bucket 是不是我們的都不確定。

    載入引擎的路徑會走 `check_file`（`Model._load` 對非 `.pt` 一律呼叫它），所以這道
    前置檢查在改成引擎之後**沒有變得多餘，只是擋的對象從 `.pt` 換成 `.engine`**。

    Args:
        model_path: `settings.model.model_path` 指定的路徑。

    Raises:
        ValueError: 副檔名不是 `.engine`。
        FileNotFoundError: 檔案不存在。
    """
    if model_path.suffix != ENGINE_SUFFIX:
        raise ValueError(
            f"[model].model_path 指到 {model_path}，但正式推論路徑只載入 "
            f"{ENGINE_SUFFIX}（TensorRT FP16 引擎，見 ADR-011）。引擎要用 "
            "`tools/build_engine.py` 在**目標 GPU 上**建——引擎不可跨架構重用。"
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到引擎檔 {model_path}。不讓 ultralytics 自行決定替代來源："
            "它在檔案不存在時會遞迴 glob 整個套件目錄找同名檔、把 `gs://` 改寫成"
            "公開 HTTPS 下載、或把沒有副檔名的值補成官方權重下載——這三條都會讓"
            "「跑的不是你以為的那顆模型」而輸出檔完全正常。"
        )


def _log_engine_metadata(metadata: dict) -> None:
    """記錄引擎自帶的 metadata，方便日後追溯實際跑的是哪一顆。

    引擎路徑下 `model.ckpt is None`，訓練版本／日期／指標**不在引擎裡**——這些欄位是
    `tools/build_engine.py` 在建置期從來源 `.pt` 的 ckpt 抄進 metadata 的
    （見 `services/engine_metadata.py`）。舊引擎沒有那一段時只會少印幾行。

    以 `.get` 防護讀取，任何例外都只 `warning`，不讓模型載入失敗。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。
    """
    try:
        names = metadata.get("names")
        if names:
            logger.info("模型類別", classes=names)
        vfa = metadata.get("vfa") or {}
        source = vfa.get("source_weights") or {}
        logger.info(
            "引擎身分",
            source_weights=source.get("name"),
            source_sha256=source.get("sha256"),
            built_on=vfa.get("gpu_name"),
            compute_capability=vfa.get("compute_capability"),
            tensorrt=vfa.get("tensorrt"),
            max_batch=metadata.get("batch"),
        )
        train = vfa.get("train") or {}
        logger.info(
            "訓練參數",
            base_model=train.get("base_model"),
            dataset=train.get("dataset"),
        )
        logger.info("訓練 ultralytics 版本", version=train.get("ultralytics"))
        logger.info("訓練日期", date=train.get("date"))
        logger.info("驗證指標", metrics=train.get("metrics"))
    except Exception as exc:
        # 多進程共寫 stdout/stderr：不用 .exception()（完整 traceback 長行可能超過
        # PIPE_BUF 4096 被切斷交錯），改附短 error 字串並保留 WARNING 語意（非致命）。
        logger.warning("記錄引擎 metadata 時發生例外，略過。", error=str(exc))


def _validate_classes(metadata: dict) -> None:
    """驗證 `settings.model.classes` 指定的類別 id 皆存在於引擎的類別定義。

    取值來源從 `model.names` 換成引擎 metadata 的 `names`：語義不變，但**前者會讓
    ultralytics 另建一次 predictor、把引擎多載一次**（多幾秒與一份執行環境的顯存），
    而這個值本來就寫在引擎檔頭裡。

    僅檢查 id 是否存在，不驗證 id 對應的語義名稱是否符合預期——若整顆權重被換成
    另一個「id 剛好也存在但語義不同」的模型（如 COCO 的 `2=car` 對到 CrowdHuman
    的 `2=fbody`），此檢查仍會通過。擋那種情況的是
    `[model].source_weights_sha256`（見 `engine_metadata.validate_engine_metadata`），
    不是這裡。此函式實際防的是 `classes` 設定本身打錯（如超出實際類別範圍的 id）。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。

    Raises:
        ValueError: `settings.model.classes` 內有不存在於 metadata `names` 的類別 id。
    """
    names = metadata.get("names")
    if not names:
        return
    missing = [c for c in settings.model.classes if c not in names]
    if missing:
        raise ValueError(
            f"settings.model.classes 指定的類別 id {missing} 不存在於引擎的"
            f"類別定義 {names}；請確認 classes 設定是否對應到實際載入的引擎。"
        )


def _validate_precision(metadata: dict) -> None:
    """驗證引擎是 FP16 建的。

    **驗 `metadata["args"]["half"]`，不是 backend 的 `fp16` 屬性**：FP16 引擎的 I/O
    binding 仍是 FP32，`AutoBackend` 對 FP16 引擎永遠回報 `fp16 = False`，拿那個值當
    判準等於永遠判定「不是 FP16」。`args` 是 exporter 把匯出參數原樣寫進 metadata 的
    那一份，只有它反映建置時的實際設定。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。

    Raises:
        ValueError: 引擎不是以 `half=True` 匯出的。
    """
    half = (metadata.get("args") or {}).get("half")
    if half is not True:
        raise ValueError(
            f"引擎的匯出參數 half={half}，不是 FP16。正式推論路徑的精度由引擎自帶"
            "（不是設定項），FP32 引擎會讓吞吐掉回改動前的水準而完全沒有訊號。"
            "請用 `tools/build_engine.py` 重建。"
        )


def _check_infer_shape_of(shape: tuple[int, ...]) -> None:
    """把一個實際進入推論的張量形狀記進 log，並核對高寬是否為推論尺寸。

    **這一項取代了改用引擎之前的 `_validate_imgsz`**（驗權重帶的 `imgsz` 是否等於
    `INFER_WIDTH`）。那個檢查在引擎路徑下**語義失效**，而失效的機制正好就是我們選了
    dynamic 引擎的後果：`predictor.py` 只在 backend **不是** dynamic 時才把 metadata 的
    `imgsz` 抄進 `args.imgsz`（`setup_model` 那行的條件是
    `not getattr(self.model, "dynamic", False)`）。dynamic 引擎因此不套用它，`args.imgsz`
    停在預設的 640，實際形狀改由 `pre_transform` 的 `auto` letterbox 決定——照抄那個檢查
    會得到一個看起來有在驗、其實驗不到的檢查，比沒有更危險。形狀改為在建置期固定
    （`tools/build_engine.py` 以 `imgsz=(INFER_HEIGHT, INFER_WIDTH)` 匯出 optimization
    profile 的 opt shape），執行期驗真正進到 backend 的那個值。

    擋的是 640×384 被填成 640×640。`auto` 的條件是
    `same_shapes and args.rect and (format == "pt" or dynamic)`——dynamic 引擎滿足最後
    一項，`rect` 在 predict 模式下是 True，`same_shapes` 由讀取端保證（每一格都縮成同
    一個尺寸）。三者**同時**成立才會保住 640×384；任何一項日後翻掉（ultralytics 改
    `rect` 預設、批次混進不同尺寸的來源），LetterBox 就會填到 `args.imgsz` 的 640×640，
    像素量 1.67 倍——那正是 issue #108 消掉的成本，而症狀只有「變慢」，座標、欄位、
    列數全部正常。

    Args:
        shape: backend binding 的 `(N, C, H, W)`。

    Raises:
        ValueError: 高寬不是 `INFER_HEIGHT` × `INFER_WIDTH`。
    """
    logger.info("推論張量形狀", shape=list(shape))
    if tuple(shape[2:]) != (INFER_HEIGHT, INFER_WIDTH):
        raise ValueError(
            f"實際進入推論的張量形狀是 {tuple(shape)}，預期高寬為 "
            f"({INFER_HEIGHT}, {INFER_WIDTH})。影格在讀取端就縮好了，這裡多出來的"
            "填充是 ultralytics 前處理加的：像素量變成 1.67 倍而輸出完全正常。"
            "請確認引擎是以 dynamic 建的（`tools/build_engine.py` 的預設）。"
        )


def _validate_dynamic(metadata: dict) -> None:
    """驗證引擎是 dynamic 建的。

    **靜態引擎會通過其餘所有載入檢查**，然後在第一個沒湊滿的批次上被 ultralytics 的
    `assert im.shape == s` 擋下——那條訊息只講「input size 不等於 max model size」，看不
    出原因是引擎綁死了批次。而湊不滿批是常態而非例外：T4（n1-standard-8）上實測一次
    跑完出現了 16 種不同的批次大小，1 到 16 全都有，解碼餵不滿批。

    接手這件事的 `_check_infer_shape` 幫不上忙——它排在 `predict` **之後**，靜態引擎
    在 predict 裡就先崩了，永遠輪不到它。所以這一項要在載入時驗。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。

    Raises:
        ValueError: 引擎不是以 `dynamic=True` 匯出的。
    """
    dynamic = (metadata.get("args") or {}).get("dynamic")
    if dynamic is not True:
        raise ValueError(
            f"引擎的匯出參數 dynamic={dynamic}，不是 dynamic 引擎。批次大小會隨供料"
            "變動（湊不滿批是常態），靜態引擎綁死批次，第一個不滿批就會在 ultralytics "
            "的 forward assert 失敗，而那條訊息看不出原因。請用 "
            "`tools/build_engine.py` 重建。"
        )


class YOLODetector:
    """TensorRT 引擎偵測器包裝層，隔離強耦合。

    Attributes:
        model: 已載入引擎的 `ultralytics.YOLO` 實例。
        max_batch: 引擎的最大批次（dynamic 引擎 optimization profile 的 batch 上限）。
            由 `services/inference.py` 拿去跟單次批次比對，兩者同尺度、設定值沒有
            隱含倍數；這裡只負責把引擎自己聲明的上限提供出去。
    """

    def __init__(self):
        """載入 `settings.model.model_path` 指定的 TensorRT 引擎。

        驗證全部排在 `YOLO(...)` **之前**：這幾項都只看引擎檔頭（ultralytics 把
        `json.dumps(metadata)` 寫在引擎前面，見 `services/engine_metadata.py`），
        不合格的引擎不必先吃掉 deserialize 的數秒與數 GB 顯存。

        **不呼叫 `.to(device)`**：引擎不是 PyTorch module，呼叫就崩；它的裝置由
        predict 決定。**也沒有 CPU fallback**：引擎綁 GPU，CPU 上不是「比較慢」而是
        一定失敗，留著 fallback 只是把錯誤延後到第一次 predict。

        Raises:
            ValueError: `model_path` 不是 `.engine`、引擎 metadata 與當下環境不符
                （見 `engine_metadata.validate_engine_metadata`）、引擎不是 FP16
                （`_validate_precision`）、引擎不是 dynamic（`_validate_dynamic`），
                或 `classes` 指定了引擎沒有的類別 id（`_validate_classes`）。
            FileNotFoundError: `model_path` 指定的引擎檔不存在。
            RuntimeError: 沒有可用的 CUDA 裝置。
        """
        model_path = Path(settings.model.model_path)
        _require_engine_file(model_path)
        metadata = read_engine_metadata(model_path)
        validate_engine_metadata(
            metadata,
            current_gpu_environment(),
            settings.model.source_weights_sha256,
        )
        _validate_precision(metadata)
        _validate_dynamic(metadata)
        _log_engine_metadata(metadata)
        _validate_classes(metadata)
        self.max_batch = metadata.get("batch")
        # 已見過的推論張量形狀，用來讓「同一個形狀」只記一次 log（見 _check_infer_shape）
        self._seen_infer_shapes: set[tuple[int, ...]] = set()
        self.model = YOLO(str(model_path))

    def _check_infer_shape(self) -> None:
        """逐批核對實際進入推論的張量形狀；同一個形狀只記一次 log。

        形狀取自 TensorRT backend 自己的 binding：`forward` 每次形狀變動都會把
        `bindings["images"]` 換成當次的 `im.shape`，也就是真正交給
        `context.execute_v2` 的那個形狀。不是從輸入影格重新推導一次——那樣只是把
        `pre_transform` 的邏輯抄第二遍，抄錯了兩邊會一起錯而檢查照樣通過。

        Raises:
            ValueError: 實際進入推論的高寬不是 `INFER_HEIGHT` × `INFER_WIDTH`。
        """
        shape = tuple(self.model.predictor.model.bindings["images"].shape)
        if shape in self._seen_infer_shapes:
            return
        self._seen_infer_shapes.add(shape)
        _check_infer_shape_of(shape)

    def predict(self, batch_frames: list[np.ndarray]) -> list[Results]:
        """對一批影格執行物件偵測，僅保留 `settings.model.classes` 指定的類別
        （預設為 head 與 fbody）。

        回傳的 `Results` 內兩種類別混在一起，由 `services/track_worker.py` 依 `cls`
        拆開——只有 fbody 進 tracker，head 供落腳點推算。

        Args:
            batch_frames: 要偵測的影格清單（BGR，已由讀取端縮成推論尺寸）；
                空清單直接回傳空結果，不呼叫模型。

        Returns:
            與 `batch_frames` 一一對應的 ultralytics `Results` 列表。

        Raises:
            ValueError: 實際進入推論的張量形狀不是預期尺寸（`_check_infer_shape`）。
        """
        if not batch_frames:
            return []
        results = self.model.predict(
            batch_frames,
            verbose=False,
            classes=settings.model.classes,
        )
        self._check_infer_shape()
        return results
