from collections import deque
from pathlib import Path

import numpy as np
import torch
from vfa_observability import StructuredLogger

from video_analyze.models.config import settings
from video_analyze.services.engine_metadata import (
    current_gpu_environment,
    read_engine_metadata,
    validate_engine_metadata,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.trt_runner import END2END_COLUMNS, TrtRunner

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

# 同時在途的批次數上限，即流水的深度。2 就夠讓 host 與 GPU 完全重疊（GPU 在算第 k 批
# 時 host 在準備第 k+1 批），再深只會讓每一塊 ping-pong 緩衝多一份、並讓「結果晚幾批
# 才回收」拖長 slot 的持有時間。緩衝、event 都是每個深度各一組，這個常數改了要一起看
# `submit` 裡每一處 `% PIPELINE_DEPTH`。
PIPELINE_DEPTH = 2


def _require_engine_file(model_path: Path) -> None:
    """`model_path` 必須是**存在的** `.engine` 檔，兩條各自 fail loud。

    兩條都只為了訊息。改用本套件自己的 runner 之後（ADR-014）沒有任何「載到別的東西
    還跑得起來」的路徑了——`.pt` 與不存在的檔都會在下一步 `read_engine_metadata` 崩，
    但那兩個訊息一個是「檔頭沒有 ultralytics metadata」、一個只有路徑字串，都指不出
    真正的原因：

    - 副檔名這條指的是「`[model].model_path` 還指著 `.pt`」（正式套件裡已經沒有 Torch
      推論路徑，見 ADR-011）。
    - 檔案存在性這條指的是路徑打錯或跑錯目錄：`model_path` 是 cwd 相對路徑，四包一律
      在 repo 根執行。

    **經 `YOLO` 載入時這道檢查擋的對象大得多**：`Model._load` → `check_file` 在檔案不
    存在時會遞迴 glob 整個 ultralytics 套件目錄找同名檔、把 `gs://` 改寫成公開 HTTPS
    下載、把沒有副檔名的值補成官方權重下載——三種都會讓「跑的不是你以為的那顆模型」而
    輸出檔完全正常。ADR-013 改直接建 backend、ADR-014 再改成自己開檔之後那三條都不會
    發生；**這也是日後要改回經 `YOLO` 載入時必須先讀的一段**。

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
            "自己先擋是為了訊息——下一步讀檔頭時 open() 拋出來的例外只有一個"
            "路徑字串。"
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

    **驗 `metadata["args"]["half"]`，不是 I/O binding 的 dtype**：FP16 引擎的 I/O
    binding 仍是 FP32（`TrtRunner` 那道 dtype 檢查驗的就是這件事），拿 binding 的
    dtype 當判準等於永遠判定「不是 FP16」。`args` 是 exporter 把匯出參數原樣寫進
    metadata 的那一份，只有它反映建置時的實際設定。

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

    **靜態引擎會通過其餘所有載入檢查**（warmup 那一批剛好是滿批），然後在第一個沒湊
    滿的批次上被 `TrtRunner` 的 `set_input_shape` 擋下。而湊不滿批是常態而非例外：
    T4（n1-standard-8）上實測一次跑完出現了 16 種不同的批次大小，1 到 16 全都有，
    解碼餵不滿批——等於整天的分析跑到一半才失敗。

    `TrtRunner` 的形狀檢查幫不上忙——它排在 enqueue **之前**沒錯，但驗的是我們自己
    組出來的張量形狀，那個形狀本來就是對的；容不下它的是引擎。

    **這一項是唯一還從 metadata 驗的**（其餘四道已移進 `TrtRunner`、改讀引擎自己宣告的
    值，見 ADR-014 Decision 1）。引擎端也答得出這件事——profile 的 `min[0] == max[0]`
    就是「批次綁死」——但那要先 deserialize；留在這裡是為了讓不合格的引擎不必先吃掉
    數秒與數 GB 顯存。代價是有人手改檔頭的 `args.dynamic` 就繞得過，而那個情境下引擎
    照樣會在第一個不滿批被 `set_input_shape` 擋下（訊息差一點，不是靜默錯誤）。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。

    Raises:
        ValueError: 引擎不是以 `dynamic=True` 匯出的。
    """
    dynamic = (metadata.get("args") or {}).get("dynamic")
    if dynamic is not True:
        raise ValueError(
            f"引擎的匯出參數 dynamic={dynamic}，不是 dynamic 引擎。批次大小會隨供料"
            "變動（湊不滿批是常態），靜態引擎綁死批次，第一個不滿批就會讓 "
            "`TrtRunner.enqueue` 的 set_input_shape 失敗，而那條訊息看不出原因"
            "（整天的分析已經跑到一半）。請用 `tools/build_engine.py` 重建。"
        )


def stack_frames(frames: list[np.ndarray], out: np.ndarray) -> None:
    """把一批 BGR 影格堆進 `out` 的前綴，這是唯一還在 CPU 上的那一趟複製。

    **這一步就是共享記憶體的歸還點**：回來的當下影格的像素已經複製走了，reader 可以
    拿回 slot（見 `YOLODetector.submit` 與 ADR-010 Decision 2）。

    寫進呼叫端給的 pinned buffer 而不是回傳一塊新的陣列，是為了讓接下來的 H2D 可以
    非同步——pageable 記憶體的 `copy_(non_blocking=True)` 會被 CUDA 悄悄降級成同步
    複製，流水就退化回「host 做事時 GPU 閒著」。buffer 的重用由 event 守住
    （`submit` 在寫之前等上一次用到這塊的 H2D 完成）。

    Args:
        frames: 一批 BGR 影格，形狀皆為 `(INFER_HEIGHT, INFER_WIDTH, 3)` 的 uint8。
            空清單由呼叫端擋掉（見 `services/inference.py` 的空批分支）；真的傳進來
            `np.stack` 會拋錯，不會靜默送出空批。
        out: `(max_batch, INFER_HEIGHT, INFER_WIDTH, 3)` 的 uint8 pinned buffer，
            只有前 `len(frames)` 格會被寫到。
    """
    np.stack(frames, out=out[: len(frames)])


def to_infer_tensor(u8_nhwc: torch.Tensor) -> torch.Tensor:
    """把 GPU 上的一批 uint8 NHWC BGR 影格轉成引擎吃的 NCHW float32 [0, 1]。

    語義照抄 ultralytics `BasePredictor.preprocess`（8.4.75）在本專案條件下的行為，但
    只留 CPU 上非做不可的那一趟複製（`stack_frames`）：

    - **不做 `pre_transform` 的 LetterBox**。影格在讀取端就縮成推論尺寸了（issue #108），
      那個 letterbox 在本專案是恆等（不 resize、四邊 padding 全 0），但
      `data/augment.py` 的 `apply_image` 對 3 通道**無條件**走 `cv2.copyMakeBorder`，
      仍然逐格複製一份。
    - **BGR→RGB 與 NHWC→NCHW 改在 GPU 上做**。ultralytics 以 `[..., ::-1]` ＋
      `transpose` ＋ `ascontiguousarray` 在 CPU 上走一趟 strided 複製，是三趟裡局部性
      最差的一趟，而它與九個解碼進程搶同一批 CPU 核。
    - **`np.stack` 保留**（在 `stack_frames`）：H2D 要一塊連續的來源。

    通道與軸的重排刻意排在還是 uint8 的時候（GPU 上搬一批 11.8 MB，轉成 float32 之後
    是 47 MB），且 `permute` 要排在通道索引**之前**：`[..., [2, 1, 0]]` 再 `permute`
    逐值相同，但結果**不是 contiguous**（本機實測 0.074 ms vs 0.079 ms，那點差距換不
    到補一趟 `.contiguous()`），而 TensorRT 取的是 `data_ptr()`、不看 stride，非連續
    的張量會被當成連續的讀進去。

    **是 `.float()` 不是 `.half()`**：FP16 引擎的 I/O binding 仍是 FP32
    （見 `_validate_precision`），送 FP16 進去會變快又變得不一樣。

    Args:
        u8_nhwc: `(B, INFER_HEIGHT, INFER_WIDTH, 3)` 的 uint8 BGR 張量，即
            `stack_frames` 堆好之後搬上 GPU 的那一塊（呼叫端傳的是常駐 device buffer
            的前綴 view）。

    Returns:
        `(B, 3, INFER_HEIGHT, INFER_WIDTH)` 的 float32 contiguous 張量，與 `u8_nhwc`
        同一個裝置。
    """
    return u8_nhwc.permute(0, 3, 1, 2)[:, [2, 1, 0]].float().div_(255)


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


class YOLODetector:
    """TensorRT 引擎偵測器包裝層，同時是推論進程那條兩批深度流水的定序者。

    ultralytics 已經不在正式推論路徑上：前處理與後處理是本模組自己的程式碼
    （ADR-013），引擎的載入與 enqueue 是 `services/trt_runner.py`（ADR-014）。這一層
    只剩「緩衝、stream、event 與批次序號」——也就是「哪一批的資料在哪一塊記憶體、什麼
    時候可以覆寫它」這件事。

    介面是 `submit` ＋ `collect` 兩步：`submit` 把一批排進 GPU 就回來（**不等它算完**），
    `collect` 才取回最舊那一批的結果。兩者之間主迴圈去湊下一批、歸還 slot、送上一批的
    payload，那段 host 工作與 GPU 的 forward 重疊，這正是本層存在的理由。

    每一種緩衝都有 `PIPELINE_DEPTH` 份，第 k 批用第 `k % PIPELINE_DEPTH` 份：

    | 緩衝 | 位置 | 誰寫 | 誰讀 | 重用前要等 |
    |---|---|---|---|---|
    | `_pinned_in` | host pinned | `stack_frames`（CPU） | H2D（copy stream） | `_h2d_done` |
    | `_dev_u8` | device | H2D | 前處理（compute stream） | `_fwd_done` |
    | `_dev_out` | device | forward | D2H（copy stream） | `_d2h_done` |
    | `_pinned_out` | host pinned | D2H | `collect`（CPU） | 深度上限本身 |

    最後一列不必等 event：深度上限逼得第 k 批的 `submit` 之前必須先 `collect` 第
    k − `PIPELINE_DEPTH` 批，而 `collect` 讀完就把結果複製成新陣列了。

    Attributes:
        runner: 引擎與 execution context（`TrtRunner`）。
        device: 引擎所在的裝置。
        max_batch: 引擎 optimization profile 的 batch 上限。由 `services/inference.py`
            拿去跟單次批次比對，兩者同尺度、設定值沒有隱含倍數；這裡只負責把引擎自己
            聲明的上限提供出去。
    """

    def __init__(self):
        """載入 `settings.model.model_path` 指定的 TensorRT 引擎並配置流水用的緩衝。

        **看 metadata 的那幾道排在載入引擎之前**（`_require_engine_file`、
        `validate_engine_metadata`、`_validate_precision`、`_validate_dynamic`、
        `_validate_classes`）：它們都只看引擎檔頭（ultralytics 把 `json.dumps(metadata)`
        寫在引擎前面，見 `services/engine_metadata.py`），不合格的引擎不必先吃掉
        deserialize 的數秒與數 GB 顯存。**看引擎自己宣告的那幾道在 `TrtRunner` 裡**
        （一入一出、輸入 dtype、profile 的 opt 高寬、輸出最後一維是不是 6）：那些值
        deserialize 之後才拿得到，而且要的正是引擎本身而不是檔頭裡的字串。

        **兩條 stream 都是新建的非預設 stream**：TensorRT 對 `stream_handle=0` 會插入
        `cudaDeviceSynchronize`，整條流水會退化成「送出一批就等它算完」，而輸出檔一
        模一樣、只是吞吐掉回改動前。`torch.cuda.Stream()` 不會回傳 default stream，
        這裡仍然明著擋一次——這是全流水唯一沒有其他訊號的失效方式。

        緩衝一律照引擎的 `max_batch` 配置（不是 `[model].batch`）：`submit` 因此接得下
        引擎容得下的任何批次，而不必跟著設定值走。批次 16 下每份影格緩衝
        11.8 MB（pinned 兩份、device 兩份，合計約 47 MB），輸出側每份不到 0.12 MB。

        **不呼叫 `.to(device)`**：引擎不是 PyTorch module，呼叫就崩；裝置在建構時指定。
        **也沒有 CPU fallback**：引擎綁 GPU，CPU 上不是「比較慢」而是一定失敗。

        Raises:
            ValueError: `model_path` 不是 `.engine`、引擎 metadata 與當下環境不符
                （見 `engine_metadata.validate_engine_metadata`）、引擎不是 FP16
                （`_validate_precision`）、引擎不是 dynamic（`_validate_dynamic`）、
                `classes` 指定了引擎沒有的類別 id（`_validate_classes`），或引擎的
                I/O 宣告不合格（見 `TrtRunner.__init__`）。
            FileNotFoundError: `model_path` 指定的引擎檔不存在。
            RuntimeError: 沒有可用的 CUDA 裝置，或建出來的 stream 是 default stream。
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
        self.device = torch.device("cuda:0")
        self.runner = TrtRunner(model_path)
        self.max_batch = self.runner.max_batch
        self._init_pipeline()
        self._warmup()

    def _init_pipeline(self) -> None:
        """依 `self.runner` 與 `self.device` 配置兩條 stream、三組 event 與四種緩衝。

        與 `__init__` 分開，是為了讓 GPU 上的定序測試能換一顆假 runner 進來——那支測試
        要驗的是本函式配出來的東西怎麼被 `submit` 用，與引擎無關。
        """
        self._copy_stream = torch.cuda.Stream()
        self._compute_stream = torch.cuda.Stream()
        for name, stream in (
            ("copy", self._copy_stream),
            ("compute", self._compute_stream),
        ):
            if stream.cuda_stream == 0:
                raise RuntimeError(
                    f"{name} stream 建出來是 default stream。TensorRT 對 "
                    "`stream_handle=0` 會插入 `cudaDeviceSynchronize`，流水會退化成"
                    "序列而完全沒有訊號。"
                )
        frame_shape = (self.max_batch, INFER_HEIGHT, INFER_WIDTH, 3)
        out_shape = (self.max_batch, self.runner.output_num_det, END2END_COLUMNS)
        self._pinned_in = [
            torch.empty(frame_shape, dtype=torch.uint8, pin_memory=True)
            for _ in range(PIPELINE_DEPTH)
        ]
        self._pinned_in_np = [buffer.numpy() for buffer in self._pinned_in]
        self._dev_u8 = [
            torch.empty(frame_shape, dtype=torch.uint8, device=self.device)
            for _ in range(PIPELINE_DEPTH)
        ]
        self._pinned_out = [
            torch.empty(out_shape, dtype=torch.float32, pin_memory=True)
            for _ in range(PIPELINE_DEPTH)
        ]
        self._pinned_out_np = [buffer.numpy() for buffer in self._pinned_out]
        self._dev_out = [
            torch.empty(out_shape, dtype=torch.float32, device=self.device)
            for _ in range(PIPELINE_DEPTH)
        ]
        # 未 record 過的 event 在 `wait_event`／`synchronize` 都是 no-op，所以最初的
        # PIPELINE_DEPTH 批不必特判「沒有上一批可等」
        self._h2d_done = [torch.cuda.Event() for _ in range(PIPELINE_DEPTH)]
        self._fwd_done = [torch.cuda.Event() for _ in range(PIPELINE_DEPTH)]
        self._d2h_done = [torch.cuda.Event() for _ in range(PIPELINE_DEPTH)]
        # 在途批次，每筆是 (序號, 緩衝索引, 批次格數, 前處理張量)
        self._in_flight: deque[tuple[int, int, int, torch.Tensor]] = deque()
        self._next_seq = 0

    @property
    def in_flight(self) -> int:
        """已 `submit` 但還沒 `collect` 的批次數。"""
        return len(self._in_flight)

    def submit(self, batch_frames: list[np.ndarray]) -> int:
        """把一批影格排進 GPU（H2D → 前處理 → forward → D2H），**不等它算完**。

        **回來的當下影格的像素已經複製走了**（`stack_frames` 寫進 pinned buffer），
        呼叫端因此可以在這裡就歸還環形緩衝的 slot。歸還點與 ADR-013 相同，只是複製的
        目的地從一塊臨時陣列換成常駐的 pinned buffer，而那塊 buffer 的重用由本函式
        開頭的 `_h2d_done` 守住。

        五個同步點分成兩類。**本輪的生產／消費依賴，拿掉就一定錯**（兩條都是跨 stream
        的，而 `collect` 只同步得到上一輪）：

        1. 前處理讀 `_dev_u8[b]` 前等 `_h2d_done[b]`——本輪的 H2D 寫完了嗎。
        2. D2H 讀 `_dev_out[b]` 前等 `_fwd_done[b]`——本輪的 forward 算完了嗎。

        **上一輪的緩衝重用，現行迴圈下三條都由傳遞性成立**（深度上限逼得第 k 批的
        `submit` 之前必先 `collect` 第 k−2 批，而那裡 host 同步過 `_d2h_done[b]`，
        copy stream 又是循序的），仍然明著寫出來——深度、stream 數或收批時機一改，
        傳遞性就沒了，而漏掉不會報錯，只會讓某一批讀到已被覆寫的記憶體、輸出檔完全正常：

        3. 寫 `_pinned_in[b]` 前等 `_h2d_done[b]`（上一輪這塊的 H2D 讀完了嗎）。
        4. H2D 寫 `_dev_u8[b]` 前等 `_fwd_done[b]`（上一輪這塊被前處理讀完了嗎；
           forward 排在前處理之後，等它是更保守的同一件事）。
        5. forward 寫 `_dev_out[b]` 前等 `_d2h_done[b]`（上一輪這塊搬回 host 了嗎）。

        每個 `wait_event` 都排在對應的 `record` 之後：`wait_event` 擷取的是**呼叫當下**
        event 的狀態，而未 record 過的 event 是 no-op——順序寫反不會有任何錯誤訊息。

        Args:
            batch_frames: 要偵測的影格清單（BGR，已由讀取端縮成推論尺寸）。

        Returns:
            這一批的序號，由 0 起遞增。呼叫端要拿它跟 `collect` 回傳的序號核對——
            結果配到別批的影格是這條流水最難察覺的錯法（輸出檔完全正常，只有時間戳
            對到隔壁批）。

        Raises:
            RuntimeError: 在途批次已達 `PIPELINE_DEPTH`（呼叫端漏了 `collect`）。
            ValueError: 批次超過緩衝容量，或推論張量的形狀不符（見 `TrtRunner`）。
        """
        if len(self._in_flight) >= PIPELINE_DEPTH:
            raise RuntimeError(
                f"在途批次已達 {PIPELINE_DEPTH}，要先 collect 才能再 submit。深度是"
                "緩衝份數決定的：再送一批會覆寫還沒被取回的那一批的輸入緩衝。"
            )
        num_frames = len(batch_frames)
        seq = self._next_seq
        buffer_index = seq % PIPELINE_DEPTH
        self._h2d_done[buffer_index].synchronize()
        stack_frames(batch_frames, self._pinned_in_np[buffer_index])
        # 影格的像素已經在 pinned buffer 裡，呼叫端從這裡起可以歸還 slot
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._fwd_done[buffer_index])
            self._dev_u8[buffer_index][:num_frames].copy_(
                self._pinned_in[buffer_index][:num_frames], non_blocking=True
            )
            self._h2d_done[buffer_index].record(self._copy_stream)
        with torch.cuda.stream(self._compute_stream):
            self._compute_stream.wait_event(self._h2d_done[buffer_index])
            self._compute_stream.wait_event(self._d2h_done[buffer_index])
            im = to_infer_tensor(self._dev_u8[buffer_index][:num_frames])
            self.runner.enqueue(
                im, self._dev_out[buffer_index][:num_frames], self._compute_stream
            )
            self._fwd_done[buffer_index].record(self._compute_stream)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._fwd_done[buffer_index])
            self._pinned_out[buffer_index][:num_frames].copy_(
                self._dev_out[buffer_index][:num_frames], non_blocking=True
            )
            self._d2h_done[buffer_index].record(self._copy_stream)
        self._next_seq += 1
        # `im` 留在在途紀錄裡：它的配置與使用都只在 compute stream 上，caching
        # allocator 的重用本來就是 stream-ordered（所以不需要 `record_stream`），
        # 持有到 `collect` 只是不讓「這塊記憶體還在被 GPU 讀」這件事依賴 allocator 的
        # 內部行為
        self._in_flight.append((seq, buffer_index, num_frames, im))
        return seq

    def collect(self) -> tuple[int, list[np.ndarray]]:
        """等最舊那一批的 D2H 完成，把它過濾成逐格的偵測框。

        等的是 `_d2h_done`（單一 event），不是整個裝置——`torch.cuda.synchronize()`
        會連帶等掉還在途的下一批，流水就白做了。

        後處理讀的是 pinned buffer 的前綴，而 boolean mask 產生的是新陣列，所以回傳
        的每一格與那塊 buffer 沒有別名關係，下一輪覆寫它不影響已回傳的結果。

        Returns:
            `(序號, 逐格的 (N, 6) 陣列)`；序號與 `submit` 回傳的一一對應，座標位於
            推論尺度。

        Raises:
            IndexError: 沒有在途批次（呼叫端多 collect 了一次）。
        """
        seq, buffer_index, num_frames, _im = self._in_flight.popleft()
        self._d2h_done[buffer_index].synchronize()
        return seq, postprocess_batch(
            self._pinned_out_np[buffer_index][:num_frames],
            CONF_THRESHOLD,
            settings.model.classes,
            MAX_DET,
        )

    def _warmup(self) -> None:
        """跑一次 zeros 批把 kernel 選擇與各緩衝的第一次觸碰算在載入期，不算在第一批。

        走的是正式的 `submit`／`collect`，所以引擎的 I/O 宣告、stream 定序、緩衝配置
        任何一項對不上都在這裡就炸，而不是等到第一批真影格。

        「這顆引擎自帶 NMS」的判準不在這裡——它已經前移到 `TrtRunner` 的載入期，讀的
        是引擎自己宣告的輸出 binding 形狀（不是 JSON metadata 的 `end2end` 欄位，改它
        不會改變引擎）。

        批次取 `min(settings.model.batch, max_batch)`：批次上限與引擎對不上這件事由
        `InferencePipeline` 在 `start_loop` 開頭擋，那裡的訊息指得出要改哪一邊，不該
        被這裡的 warmup 搶先。
        """
        batch = min(settings.model.batch, self.max_batch)
        frames = [
            np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)
            for _ in range(batch)
        ]
        self.submit(frames)
        self.collect()
