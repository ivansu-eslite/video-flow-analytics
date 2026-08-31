"""本套件自己的 TensorRT execution context 包裝，取代 ultralytics 的 `AutoBackend`。

會自己接手這一層，是因為 `AutoBackend.forward` 的介面把「送出」與「拿回」綁成一次
呼叫：它走 `execute_v2`（阻塞到 GPU 算完），輸出寫在 backend 自己持有、下一次 forward
就地覆寫的 binding buffer 上。兩者都讓推論進程只能一件接一件做——host 湊批時 GPU 閒著、
GPU 算時 host 閒著。要讓兩邊重疊就得換成 `execute_async_v3`，並且由呼叫端指定輸出要寫
到哪一塊記憶體（ping-pong 緩衝的其中一塊），這兩件事 `AutoBackend` 都不提供。

本模組只做「把一批推論排進指定的 stream」這一件事：緩衝、stream、event 與流水的定序
都在 `services/detector.py`。這裡不持有任何跨批次的狀態（除了引擎與 context 本身），
所以「哪一批的結果在哪一塊 buffer」不可能在這裡出錯。

**引擎檔的檔頭格式沿用 ultralytics 的**（`json.dumps(metadata)` 的長度以 4 bytes
little-endian 寫在最前面，接著 JSON，然後才是序列化的引擎），解析用的是
`services/engine_metadata.py` 的同一組常數與同一支長度檢查——`tools/build_engine.py`
仍以 ultralytics 的 exporter 產出引擎，格式不是我們定的，各寫一份只會漂移。
"""

from __future__ import annotations

from pathlib import Path

import torch
from vfa_observability import StructuredLogger

from video_analyze.services.engine_metadata import read_metadata_length
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH

logger = StructuredLogger(component="trt_runner")

# end2end 引擎每格輸出的欄數：x1、y1、x2、y2、conf、cls
END2END_COLUMNS = 6


def _check_infer_tensor(im: torch.Tensor) -> None:
    """核對即將交給 `execute_async_v3` 的輸入張量的型別、佈局與位置。

    偵測器的介面是 `submit` ＋ `collect` 兩步而不是一支合成的 `predict`（環形緩衝的
    slot 歸還卡在兩者中間，見 ADR-013、ADR-014），所以 runner 有可能收到不是由前處理
    產出的東西。這道檢查擋的就是那種誤用，四項都是「跑得完但結果不對」而不是會自己
    拋錯的情況：

    - **dtype 不是 float32**：uint8 或 float16 餵進 FP32 binding，引擎照樣執行，只是
      把同一塊位元組重新解讀成別的數，輸出仍是形狀正常的框。
    - **不是 contiguous**：TensorRT 拿的是 `im.data_ptr()`，它不看 stride——非連續張量
      會被當成連續的讀進去，等於餵了一張錯位的影像。
    - **不是 4D**：與 binding 的 `(N, C, H, W)` 對不上。
    - **不在 CUDA 上**：把 CPU 位址交給引擎是未定義行為。

    先驗內容性質、最後驗位置，是為了讓前三項在沒有 GPU 的機器上也測得到。

    Args:
        im: 前處理產出的推論張量。

    Raises:
        ValueError: dtype、佈局或維度不符。
        RuntimeError: 張量不在 CUDA 上。
    """
    if im.dtype != torch.float32:
        raise ValueError(
            f"推論張量的 dtype 是 {im.dtype}，預期 float32。FP16 引擎的 I/O binding "
            "仍是 FP32，送 float16 進去只會讓同一塊位元組被重新解讀，輸出仍是形狀"
            "正常的框（見 `engine_metadata.validate_engine_precision`）。"
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
            "execute_async_v3 是未定義行為。"
        )


def _check_infer_shape_of(shape: tuple[int, ...]) -> None:
    """把一個實際進入推論的張量形狀記進 log，並核對高寬是否為推論尺寸。

    自建前處理之後，形狀不再由 ultralytics 的 `pre_transform` 決定，而是讀取端縮出來
    的尺寸原樣堆成一批（前處理不做任何 resize 或填充）。這道檢查因此從「擋 LetterBox
    把 640×384 填成 640×640」變成**「擋讀取端的縮放沒有生效」**，而它仍然不是多餘的：
    ultralytics 匯出 dynamic 引擎時把高與寬一併列為 dynamic axes
    （`exporter.py` 的 `{"images": {0: "batch", 2: "height", 3: "width"}}`），只要落在
    optimization profile 的範圍內，錯誤的高寬**不會**被 TensorRT 擋下——引擎照跑，吐出
    來的框停在錯誤的尺度上，而列數、欄位、格式全部正常。

    這一項也仍然取代著改用引擎之前的 `_validate_imgsz`（驗權重帶的 `imgsz`）：dynamic
    引擎不套用 metadata 的 `imgsz`，照抄那個檢查會得到一個看起來有在驗、其實驗不到的
    檢查。形狀改為在建置期固定（`tools/build_engine.py` 以
    `imgsz=(INFER_HEIGHT, INFER_WIDTH)` 匯出 optimization profile 的 opt shape），
    執行期驗真正要送進引擎的那個值。

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


class TrtRunner:
    """一顆 TensorRT 引擎與它的 execution context，只提供「非同步排一批進 stream」。

    **單一 context、由呼叫端序列化呼叫**：shape 與 binding 位址是在
    `execute_async_v3` 當下擷取的，同一執行緒依序 enqueue 多批（中間不同步）不在
    TensorRT 的未定義行為條款內，改 shape 也不影響已經送出的那批。反過來說「兩個
    context 各綁一組 buffer」才是錯的——同時在途的多個 context 必須各用一個
    optimization profile，而本專案的引擎只有一個。

    Attributes:
        max_batch: 引擎 optimization profile 的 batch 上限。**讀 profile 而不是 JSON
            metadata 的 `batch`**：後者是檔頭裡的一個字串欄位，改它不會改變引擎，而
            批次超限的症狀（`set_input_shape` 回 False 之後拿上一批的 shape 繼續跑）
            正是要靠這個值先擋下的。
        input_height: profile opt shape 的高。
        input_width: profile opt shape 的寬。
        output_num_det: 引擎宣告的每格最大偵測數（輸出的第二維），供呼叫端配置輸出
            緩衝。
    """

    def __init__(self, engine_path: Path):
        """讀檔頭、deserialize 引擎、建 execution context，並做四道載入期檢查。

        `logger`／`runtime`／`engine`／`context` 四個引用都掛在自己身上：TensorRT 的
        物件不持有建立者的所有權，`runtime` 先被回收的話 `engine` 會在之後的任一次
        呼叫上崩在原生層，而症狀與引擎本身無關。

        Args:
            engine_path: `.engine` 檔路徑（副檔名與存在性由呼叫端先擋，見
                `detector._require_engine_file`）。

        Raises:
            ValueError: 檔頭不是 ultralytics 的 metadata 格式、引擎的 I/O 張量不是
                恰好一入一出、輸入 dtype 不是 float32、輸入 opt 高寬不是推論尺寸，
                或輸出的最後一維不是 `END2END_COLUMNS`。
            RuntimeError: 引擎 deserialize 失敗，或建不出 execution context
                （兩者都只回 `None` 而不拋錯）。
        """
        import tensorrt as trt

        with open(engine_path, "rb") as handle:
            meta_len = read_metadata_length(handle, engine_path)
            handle.seek(meta_len, 1)
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(
                f"{engine_path} deserialize 失敗。引擎綁 GPU 架構與 TensorRT 版本，"
                "兩者的比對已在載入前做過（見 `engine_metadata.validate_engine_metadata`），"
                "走到這裡代表引擎檔本身損壞。詳細原因由 TensorRT 的 logger 印在上方。"
            )
        # 與上面同型：失敗時回 `None` 而不拋錯（配不出 context 的裝置記憶體是最常見的
        # 原因）。不擋的話症狀是 warmup 期間的 `'NoneType' object has no attribute
        # 'set_input_shape'`，指不出真正的原因
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                f"{engine_path} 建不出 execution context。引擎本身已 deserialize 成功，"
                "走到這裡多半是裝置記憶體不足（context 自帶一份 activation 用的暫存）。"
            )

        names = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
        ]
        inputs = [
            n
            for n in names
            if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        outputs = [n for n in names if n not in inputs]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"引擎的 I/O 張量是 {len(inputs)} 入 {len(outputs)} 出（{names}），"
                "本模組只接一入一出。多輸入的引擎在這裡只會綁到其中一個，其餘 binding "
                "停在未設定的位址上；多輸出則代表這不是本 repo 建置工具產出的偵測引擎。"
            )
        self._input_name, self._output_name = inputs[0], outputs[0]

        input_dtype = self._engine.get_tensor_dtype(self._input_name)
        if input_dtype != trt.DataType.FLOAT:
            raise ValueError(
                f"引擎輸入 binding 的 dtype 是 {input_dtype}，不是 FP32。FP16 引擎的 "
                "I/O binding 仍是 FP32（見 `engine_metadata.validate_engine_precision`）；真的收到"
                "別的 dtype 代表前處理送進去的位元組會被重新解讀，而輸出仍是形狀正常"
                "的框。"
            )

        _min, opt, maximum = self._engine.get_tensor_profile_shape(self._input_name, 0)
        self.max_batch = int(maximum[0])
        self.input_height, self.input_width = int(opt[2]), int(opt[3])
        if (self.input_height, self.input_width) != (INFER_HEIGHT, INFER_WIDTH):
            raise ValueError(
                f"引擎 optimization profile 的 opt 高寬是 "
                f"({self.input_height}, {self.input_width})，本套件送進去的一律是 "
                f"({INFER_HEIGHT}, {INFER_WIDTH})。高寬是 dynamic axes，這顆引擎照樣"
                "跑得完、框也對，只是所有 kernel 都是為別的尺寸挑的——症狀只有變慢。"
                "請用 `tools/build_engine.py` 重建。"
            )

        output_shape = tuple(self._engine.get_tensor_shape(self._output_name))
        if len(output_shape) != 3 or output_shape[-1] != END2END_COLUMNS:
            raise ValueError(
                f"引擎輸出 binding 的形狀是 {output_shape}，不是 end2end 的 "
                f"(B, num_det, {END2END_COLUMNS})。這顆引擎需要完整的 NMS，而本套件"
                "只套用 end2end 的三行過濾（conf → 截斷 → classes）——那會把類別分數"
                "當成 conf 與 cls，得到一堆座標是 xywh 的框，而輸出檔的欄位、列數、"
                "格式全部正常。請用 `tools/build_engine.py` 重建（它以 `nms=True` "
                "匯出）。"
            )
        self.output_num_det = int(output_shape[1])
        # 已見過的推論張量形狀，用來讓「同一個形狀」只記一次 log
        self._seen_infer_shapes: set[tuple[int, ...]] = set()

    def enqueue(
        self, im: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        """把一批推論排進 `stream`，輸出寫到 `out`；**不等它算完**。

        三個 TensorRT 呼叫的回傳值都要檢查：`set_input_shape` 超出 profile 只回
        `False`，接著 `execute_async_v3` 仍回 `True` 並用**上一個 shape** 跑——輸出形狀
        正常、內容是別批的。這三個 `if` 是那條路徑唯一的訊號。

        Args:
            im: 前處理產出的 `(B, 3, INFER_HEIGHT, INFER_WIDTH)` float32 contiguous
                CUDA 張量。
            out: 這一批的輸出緩衝，`(B, output_num_det, END2END_COLUMNS)` 的 float32
                contiguous CUDA 張量。呼叫端負責保證它在 forward 完成前不被覆寫。
            stream: 要排進去的 CUDA stream，**不可是 default stream**
                （`stream_handle=0` 會讓 TensorRT 插入 `cudaDeviceSynchronize`，流水
                直接退化成序列）。

        Raises:
            ValueError: `im` 的 dtype／佈局／維度／高寬不符，或 `out` 的形狀不是這一批
                應有的 `(B, output_num_det, END2END_COLUMNS)`。
            RuntimeError: `im` 不在 CUDA 上，`stream` 是 default stream，或三個
                TensorRT 呼叫任一個回 `False`。
        """
        # stream 這條排最前面：它與張量無關，而且是唯一「什麼都對、只是不會重疊」的
        # 失效方式，擺前面才能在沒有 GPU 的機器上也測得到
        if stream.cuda_stream == 0:
            raise RuntimeError(
                "enqueue 收到 default stream。TensorRT 對 `stream_handle=0` 會插入 "
                "`cudaDeviceSynchronize`，流水會退化成「送出一批就等它算完」而完全"
                "沒有訊號——吞吐掉回改動前的水準，輸出檔一模一樣。"
            )
        _check_infer_tensor(im)
        self._check_infer_shape(im)
        expected_out = (im.shape[0], self.output_num_det, END2END_COLUMNS)
        if tuple(out.shape) != expected_out:
            raise ValueError(
                f"輸出緩衝的形狀是 {tuple(out.shape)}，這一批應該是 {expected_out}。"
                "對不上時引擎照樣寫得完（輸出 binding 只是一個位址），只是這一批的"
                "結果會溢出到緩衝的下一段或只填了一部分，而後處理讀到的列數完全正常。"
            )
        if not self._context.set_input_shape(self._input_name, tuple(im.shape)):
            raise RuntimeError(
                f"set_input_shape 失敗（形狀 {tuple(im.shape)}，profile batch 上限 "
                f"{self.max_batch}）。它只回 False 不拋錯，而接下來的 "
                "execute_async_v3 仍會回 True 並沿用**上一批**的 shape 跑。"
            )
        if not self._context.set_tensor_address(self._input_name, im.data_ptr()):
            raise RuntimeError(f"set_tensor_address 失敗（輸入 {self._input_name}）。")
        if not self._context.set_tensor_address(self._output_name, out.data_ptr()):
            raise RuntimeError(f"set_tensor_address 失敗（輸出 {self._output_name}）。")
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(
                "execute_async_v3 失敗。它回 False 代表這一批根本沒有排進 stream，"
                "而輸出緩衝仍留著上一批的內容——不擋下來的話後處理會拿到重複的框。"
            )

    def _check_infer_shape(self, im: torch.Tensor) -> None:
        """逐批核對推論張量的形狀；同一個形狀只記一次 log。

        Args:
            im: 前處理產出的推論張量。

        Raises:
            ValueError: 實際進入推論的高寬不是 `INFER_HEIGHT` × `INFER_WIDTH`。
        """
        shape = tuple(im.shape)
        if shape in self._seen_infer_shapes:
            return
        self._seen_infer_shapes.add(shape)
        _check_infer_shape_of(shape)
