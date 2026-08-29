from pathlib import Path

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
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

# ultralytics 在 predict 路徑上隱含填進去的兩個過濾參數，照抄成模組常數以維持與改動前
# 逐值相同：`conf` 由 `BasePredictor.__init__` 在 `args.conf is None` 時填 0.25，
# `max_det` 是 `cfg/default.yaml` 的預設。**刻意不進設定面**——它們是「與過去的輸出可
# 比」的錨，不是可調參數；做成 `config.toml` 的欄位等於開一個調了就讓新舊檔案不可比、
# 而兩邊看起來都正常的旋鈕。
CONF_THRESHOLD = 0.25
MAX_DET = 300

# end2end 引擎每格輸出的欄數：x1、y1、x2、y2、conf、cls
END2END_COLUMNS = 6


def _require_engine_file(model_path: Path) -> None:
    """`model_path` 必須是**存在的** `.engine` 檔，兩條各自 fail loud。

    副檔名這條擋的是「還指著 `.pt`」：`AutoBackend` 照副檔名選 backend，`.pt` 會走
    PyTorchBackend——載得起來、跑得出結果，只是慢一個量級而完全沒有訊號（正式套件裡
    已經沒有 Torch 推論路徑，見 ADR-011）。

    檔案存在性這條擋的是路徑打錯：`model_path` 是 cwd 相對路徑（四包一律在 repo 根
    執行），在別的 cwd 跑起來就是「檔案不存在」。自己先擋是為了訊息——TensorRT backend
    直接 `open(weight, "rb")`，拋出來的 `FileNotFoundError` 只有一個路徑字串。

    **改用 `AutoBackend` 之前（ADR-013）這道檢查擋的對象大得多**：`YOLO` 走
    `Model._load` → `check_file`，而那條路徑在檔案不存在時會遞迴 glob 整個 ultralytics
    套件目錄找同名檔、把 `gs://` 改寫成公開 HTTPS 下載、把沒有副檔名的值補成官方權重
    下載——三種都會讓「跑的不是你以為的那顆模型」而輸出檔完全正常。`AutoBackend` 的
    `_model_type` 只看副檔名、不解析任何替代來源，那三條因此已經不會發生；**這也是
    日後要改回經 `YOLO` 載入時必須先讀的一段**。

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
            f"找不到引擎檔 {model_path}。`model_path` 是 cwd 相對路徑，四包一律在 "
            "repo 根執行（`uv run` 不改變 cwd）；在別的 cwd 跑就會是這個症狀。"
            "自己先擋是為了訊息——TensorRT backend 直接 open()，拋出來的例外只有"
            "一個路徑字串。"
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


def _validate_dynamic(metadata: dict) -> None:
    """驗證引擎是 dynamic 建的。

    **靜態引擎會通過其餘所有載入檢查**，然後在第一個沒湊滿的批次上被 ultralytics 的
    `assert im.shape == s` 擋下——那條訊息只講「input size 不等於 max model size」，看不
    出原因是引擎綁死了批次。而湊不滿批是常態而非例外：T4（n1-standard-8）上實測一次
    跑完出現了 16 種不同的批次大小，1 到 16 全都有，解碼餵不滿批。

    接手這件事的 `_check_infer_shape` 幫不上忙——它排在 forward **之前**沒錯，但驗的是
    我們自己組出來的張量形狀，那個形狀本來就是對的；崩的是引擎容不下它。所以這一項要
    在載入時、從 metadata 驗。

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


def preprocess_batch(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """把一批 BGR 影格整批送上 `device`，在該裝置上轉成引擎吃的 NCHW float32 [0, 1]。

    語義照抄 ultralytics `BasePredictor.preprocess`（8.4.75）在本專案條件下的行為，但
    只留 CPU 上非做不可的那一趟複製：

    - **不做 `pre_transform` 的 LetterBox**。影格在讀取端就縮成推論尺寸了（issue #108），
      那個 letterbox 在本專案是恆等（不 resize、四邊 padding 全 0），但
      `data/augment.py` 的 `apply_image` 對 3 通道**無條件**走 `cv2.copyMakeBorder`，
      仍然逐格複製一份。
    - **BGR→RGB 與 NHWC→NCHW 改在 GPU 上做**。ultralytics 以 `[..., ::-1]` ＋
      `transpose` ＋ `ascontiguousarray` 在 CPU 上走一趟 strided 複製，是三趟裡局部性
      最差的一趟，而它與九個解碼進程搶同一批 CPU 核。
    - **`np.stack` 保留**：H2D 要一塊連續的來源。

    通道與軸的重排刻意排在還是 uint8 的時候（GPU 上搬一批 11.8 MB，轉成 float32 之後
    是 47 MB），且 `permute` 要排在通道索引**之前**：`[..., [2, 1, 0]]` 再 `permute`
    逐值相同，但結果**不是 contiguous**（本機實測 0.074 ms vs 0.079 ms，那點差距換不
    到補一趟 `.contiguous()`），而 TensorRT 取的是 `data_ptr()`、不看 stride，非連續
    的張量會被當成連續的讀進去。

    **是 `.float()` 不是 `.half()`**：FP16 引擎的 I/O binding 仍是 FP32
    （見 `_validate_precision`），送 FP16 進去會變快又變得不一樣。

    Args:
        frames: 一批 BGR 影格，形狀皆為 `(INFER_HEIGHT, INFER_WIDTH, 3)` 的 uint8。
            空清單由呼叫端擋掉（見 `services/inference.py` 的空批分支）；真的傳進來
            `np.stack` 會拋錯，不會靜默送出空批。
        device: 引擎所在的裝置。

    Returns:
        `(B, 3, INFER_HEIGHT, INFER_WIDTH)` 的 float32 contiguous CUDA 張量。
    """
    batch = np.stack(frames)
    return (
        torch.from_numpy(batch)
        .to(device)
        .permute(0, 3, 1, 2)[:, [2, 1, 0]]
        .float()
        .div_(255)
    )


def postprocess_batch(
    raw: np.ndarray, conf: float, classes: list[int], max_det: int
) -> list[np.ndarray]:
    """把 end2end 引擎的整批原始輸出過濾成逐格的 N×6（xyxy、conf、cls）。

    照抄 ultralytics `utils/nms.py` end2end 分支的三行，加上
    `DetectionPredictor.construct_result` 的 `scale_boxes`。三件事不可調動：

    1. **順序是 conf 過濾 → `[:max_det]` 截斷 → classes 過濾**。把 classes 提到截斷
       之前，同一批輸入會多留下原本被擠出前 300 名的目標類別框——列數、欄位、格式全部
       正常，只是內容與改動前不同。
    2. **conf 是嚴格大於**：恰好等於門檻的那一列要丟掉。
    3. **`scale_boxes` 那一步保留**。在本專案它退化成 clip：`orig_img` 就是推論尺度的
       影格，`gain = min(384/384, 640/640) = 1`、
       `pad_x = round((640 - 640) / 2 - 0.1) = 0`（`pad_y` 同理），減 0、除以 1 之後
       只剩把座標夾回 `[0, W]`／`[0, H]`。成本近零，拿掉會讓超出邊界的框流到下游。

    `iou` 與 `agnostic_nms` 在 end2end 分支**不被使用**（NMS 已經在引擎裡跑完），所以
    這裡沒有它們的對應物。

    Args:
        raw: forward 的原始輸出轉成的 `(B, num_det, 6)` CPU 陣列。
        conf: 信心門檻，**嚴格大於**才留。
        classes: 要保留的類別 id。
        max_det: 每格最多保留幾個框（截斷排在 classes 過濾之前）。

    Returns:
        逐格一個 `(N, 6)` 陣列，與 `raw` 的批次順序一一對應。每一格都是 boolean mask
        產生的新陣列，與 `raw` 那塊 buffer 沒有別名關係。
    """
    class_ids = np.asarray(classes, dtype=raw.dtype)
    outputs = []
    for pred in raw:
        kept = pred[pred[:, 4] > conf][:max_det]
        kept = kept[(kept[:, 5:6] == class_ids).any(1)]
        kept[:, [0, 2]] = kept[:, [0, 2]].clip(0, INFER_WIDTH)
        kept[:, [1, 3]] = kept[:, [1, 3]].clip(0, INFER_HEIGHT)
        outputs.append(kept)
    return outputs


def _check_infer_tensor(im: torch.Tensor) -> None:
    """核對即將交給 forward 的張量的型別、佈局與位置。

    偵測器的介面是 `preprocess` ＋ `infer` 兩步而不是一支合成的 `predict`（環形緩衝的
    slot 歸還卡在兩者中間，見 ADR-013），所以 `infer` 有可能收到不是 `preprocess`
    產出的東西。這道檢查擋的就是那種誤用，四項都是「跑得完但結果不對」而不是會自己
    拋錯的情況：

    - **dtype 不是 float32**：uint8 或 float16 餵進 FP32 binding，`execute_v2` 照樣
      執行，只是把同一塊位元組重新解讀成別的數，輸出仍是形狀正常的框。
    - **不是 contiguous**：TensorRT 拿的是 `im.data_ptr()`，它不看 stride——非連續張量
      會被當成連續的讀進去，等於餵了一張錯位的影像。
    - **不是 4D**：與 binding 的 `(N, C, H, W)` 對不上。
    - **不在 CUDA 上**：把 CPU 位址交給 `execute_v2` 是未定義行為。

    先驗內容性質、最後驗位置，是為了讓前三項在沒有 GPU 的機器上也測得到。

    Args:
        im: `preprocess` 產出的推論張量。

    Raises:
        ValueError: dtype、佈局或維度不符。
        RuntimeError: 張量不在 CUDA 上。
    """
    if im.dtype != torch.float32:
        raise ValueError(
            f"推論張量的 dtype 是 {im.dtype}，預期 float32。FP16 引擎的 I/O binding "
            "仍是 FP32，送 float16 進去只會讓同一塊位元組被重新解讀，輸出仍是形狀"
            "正常的框（見 `_validate_precision`）。"
        )
    if not im.is_contiguous():
        raise ValueError(
            "推論張量不是 contiguous。TensorRT 取的是 data_ptr() 而不看 stride，"
            "非連續張量會被當成連續的讀進去，等於餵了一張錯位的影像。"
        )
    if im.dim() != 4:
        raise ValueError(
            f"推論張量是 {im.dim()} 維，預期 4 維 (N, C, H, W)。"
        )
    if not im.is_cuda:
        raise RuntimeError(
            "推論張量不在 CUDA 上。引擎的 binding 位址是裝置記憶體，把 CPU 位址交給 "
            "execute_v2 是未定義行為。"
        )


def _check_infer_shape_of(shape: tuple[int, ...]) -> None:
    """把一個實際進入推論的張量形狀記進 log，並核對高寬是否為推論尺寸。

    自建前處理之後，形狀不再由 ultralytics 的 `pre_transform` 決定，而是讀取端縮出來
    的尺寸原樣堆成一批（`preprocess_batch` 不做任何 resize 或填充）。這道檢查因此從
    「擋 LetterBox 把 640×384 填成 640×640」變成**「擋讀取端的縮放沒有生效」**，而它
    仍然不是多餘的：ultralytics 匯出 dynamic 引擎時把高與寬一併列為 dynamic axes
    （`exporter.py` 的 `{"images": {0: "batch", 2: "height", 3: "width"}}`），只要落在
    optimization profile 的範圍內，錯誤的高寬**不會**被 TensorRT 的 assert 擋下——
    引擎照跑，吐出來的框停在錯誤的尺度上，而列數、欄位、格式全部正常。

    這一項也仍然取代著改用引擎之前的 `_validate_imgsz`（驗權重帶的 `imgsz`）：dynamic
    引擎不套用 metadata 的 `imgsz`（`predictor.py` 只在 backend **不是** dynamic 時才
    把它抄進 `args.imgsz`），照抄那個檢查會得到一個看起來有在驗、其實驗不到的檢查。
    形狀改為在建置期固定（`tools/build_engine.py` 以
    `imgsz=(INFER_HEIGHT, INFER_WIDTH)` 匯出 optimization profile 的 opt shape），
    執行期驗真正要送進 backend 的那個值。

    Args:
        shape: 推論張量的 `(N, C, H, W)`。

    Raises:
        ValueError: 高寬不是 `INFER_HEIGHT` × `INFER_WIDTH`。
    """
    logger.info("推論張量形狀", shape=list(shape))
    if tuple(shape[2:]) != (INFER_HEIGHT, INFER_WIDTH):
        raise ValueError(
            f"實際進入推論的張量形狀是 {tuple(shape)}，預期高寬為 "
            f"({INFER_HEIGHT}, {INFER_WIDTH})。影格在讀取端就縮好了，高寬對不上代表"
            "縮放沒有生效；dynamic 引擎的高寬維也是動態的，這個形狀跑得完，只是框會"
            "停在錯誤的尺度上而輸出檔完全正常。"
        )


class YOLODetector:
    """TensorRT 引擎偵測器包裝層，隔離強耦合。

    ultralytics 這一側只用到 `AutoBackend` 的 forward：前處理與後處理都是本模組自己的
    程式碼（ADR-013），`YOLO`／`predictor`／`Results` 都不在正式推論路徑上。

    Attributes:
        model: 已載入引擎的 `AutoBackend` 實例。
        device: 引擎所在的裝置，前處理把整批影格送到這裡。
        max_batch: 引擎的最大批次（dynamic 引擎 optimization profile 的 batch 上限）。
            由 `services/inference.py` 拿去跟單次批次比對，兩者同尺度、設定值沒有
            隱含倍數；這裡只負責把引擎自己聲明的上限提供出去。
    """

    def __init__(self):
        """載入 `settings.model.model_path` 指定的 TensorRT 引擎。

        驗證全部排在載入引擎**之前**：這幾項都只看引擎檔頭（ultralytics 把
        `json.dumps(metadata)` 寫在引擎前面，見 `services/engine_metadata.py`），
        不合格的引擎不必先吃掉 deserialize 的數秒與數 GB 顯存。

        **直接建 `AutoBackend`，不經 `YOLO`／`predictor`**：後者會連帶帶進一組
        `conf`／`iou`／`classes`／`max_det`，那些參數在自建後處理之後看起來還在生效、
        實際上是我們自己算的；`predictor.dataset.im0` 與 `predictor.batch[1]` 這兩個
        指向共享記憶體的內部別名（ADR-010 Decision 4 明寫擋不住的那條）也一併消失。
        代價是 deserialize 從第一批推論提前到這裡——兩者都在推論子進程內，
        `pipeline.py` 的錯誤路徑不變，而提前等於失敗得更早。

        **`fp16=False`**：FP16 引擎的 I/O binding 仍是 FP32，`AutoBackend` 對它永遠
        回報 `fp16 = False`，這裡填的值只是把既有事實寫明——填 True 會讓
        `AutoBackend.forward` 在送進 backend 前把張量 `.half()` 掉。

        **不呼叫 `.to(device)`**：引擎不是 PyTorch module，呼叫就崩；裝置在建構時指定。
        **也沒有 CPU fallback**：引擎綁 GPU，CPU 上不是「比較慢」而是一定失敗。

        Raises:
            ValueError: `model_path` 不是 `.engine`、引擎 metadata 與當下環境不符
                （見 `engine_metadata.validate_engine_metadata`）、引擎不是 FP16
                （`_validate_precision`）、引擎不是 dynamic（`_validate_dynamic`）、
                `classes` 指定了引擎沒有的類別 id（`_validate_classes`），或引擎不是
                end2end（`_warmup_and_require_end2end`）。
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
        self.device = torch.device("cuda:0")
        self.model = AutoBackend(str(model_path), device=self.device, fp16=False)
        self._warmup_and_require_end2end()

    def _warmup_and_require_end2end(self) -> None:
        """跑一次 zeros forward：既是 warmup，也是「這顆引擎自帶 NMS」的 fail loud。

        判準用**實際跑出來的輸出形狀**，不讀 metadata 的 `end2end` 欄位：後者是匯出時
        寫進檔頭的一個值，改它不會改變引擎，而我們正是要驗引擎本身。

        不是 end2end 的引擎會吐 `(B, 4 + nc, num_anchors)` 的原始張量，
        `postprocess_batch` 照樣跑得完——那三行只讀第 4、5 欄，於是把某個類別分數當成
        conf、另一個當成 cls，得到一堆座標是 xywh 的框，而列數、欄位、格式全部正常。

        批次取 `min(settings.model.batch, max_batch)`：批次上限與引擎對不上這件事由
        `InferencePipeline` 在 `start_loop` 開頭擋，那裡的訊息指得出要改哪一邊，不該
        被這裡的 warmup 搶先變成 ultralytics 的 assert。

        Raises:
            ValueError: forward 的輸出不是單一張量，或最後一維不是
                `END2END_COLUMNS`。
        """
        batch = settings.model.batch
        if self.max_batch is not None:
            batch = min(batch, self.max_batch)
        dummy = torch.zeros(
            (batch, 3, INFER_HEIGHT, INFER_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        raw = self.model.forward(dummy)
        if not isinstance(raw, torch.Tensor) or raw.shape[-1] != END2END_COLUMNS:
            shape = None if not isinstance(raw, torch.Tensor) else tuple(raw.shape)
            raise ValueError(
                f"引擎的 forward 輸出是 {type(raw).__name__}、形狀 {shape}，不是 "
                f"end2end 的 (B, num_det, {END2END_COLUMNS})。這顆引擎需要完整的 NMS，"
                "而本模組只套用 end2end 的三行過濾（conf → 截斷 → classes）——那會把"
                "類別分數當成 conf 與 cls，得到一堆完全不同的框，而輸出檔的欄位、"
                "列數、格式全部正常。請用 `tools/build_engine.py` 重建（它以 "
                "`nms=True` 匯出）。"
            )

    def _check_infer_shape(self, im: torch.Tensor) -> None:
        """逐批核對推論張量的形狀；同一個形狀只記一次 log。

        驗的是我們自己組出來、即將交給 `execute_v2` 的那個張量。改用引擎之前這裡讀的
        是 TensorRT backend 的 `bindings["images"]`（forward 內部才更新），自建前處理
        之後那個值只是本張量形狀的複本，直接驗來源更短。

        Args:
            im: `preprocess` 產出的推論張量。

        Raises:
            ValueError: 實際進入推論的高寬不是 `INFER_HEIGHT` × `INFER_WIDTH`。
        """
        shape = tuple(im.shape)
        if shape in self._seen_infer_shapes:
            return
        self._seen_infer_shapes.add(shape)
        _check_infer_shape_of(shape)

    def preprocess(self, batch_frames: list[np.ndarray]) -> torch.Tensor:
        """把一批影格轉成推論張量（整批一次 H2D，其餘在 GPU 上做）。

        **回來的當下影格的像素已經複製走了**（`np.stack` 一次，接著同步的 H2D），
        呼叫端因此可以在這裡就歸還環形緩衝的 slot，不必等 forward 完成——這正是
        `predict` 拆成兩步的原因，也是它不保留為合成方法的原因（見 ADR-013）。

        Args:
            batch_frames: 要偵測的影格清單（BGR，已由讀取端縮成推論尺寸）。

        Returns:
            `(B, 3, INFER_HEIGHT, INFER_WIDTH)` 的 float32 CUDA 張量。
        """
        return preprocess_batch(batch_frames, self.device)

    def infer(self, im: torch.Tensor) -> list[np.ndarray]:
        """forward ＋ 整批一次 D2H ＋ 逐格過濾，回傳逐格的偵測框。

        **`.cpu()` 那一步同時做兩件事**：把整批輸出複製走，以及整批只同步一次。
        TensorRT backend 回傳的是可重用的 binding buffer
        （`nn/backends/tensorrt.py` 的 `self.bindings[x].data`），下一次 forward 就地
        覆寫它——後處理若延後到下一批之後才做，會靜默拿到別批的輸出。

        改動前的路徑是逐格在 GPU 上做 boolean index 再逐格 `.cpu()`，一批 16 格等於
        16 次隱含同步，而實際搬的資料只有 115 KB。

        Args:
            im: `preprocess` 產出的推論張量。

        Returns:
            與該批影格一一對應的 `(N, 6)` 陣列（xyxy、conf、cls），座標位於推論尺度。

        Raises:
            ValueError: 張量的 dtype／佈局／維度不符（`_check_infer_tensor`），或高寬
                不是預期尺寸（`_check_infer_shape`）。
            RuntimeError: 張量不在 CUDA 上。
        """
        _check_infer_tensor(im)
        self._check_infer_shape(im)
        raw = self.model.forward(im)
        return postprocess_batch(
            raw.cpu().numpy(), CONF_THRESHOLD, settings.model.classes, MAX_DET
        )
