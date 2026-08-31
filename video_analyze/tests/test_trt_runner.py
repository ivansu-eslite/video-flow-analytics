"""自建 TensorRT runner：載入期對引擎 I/O 宣告的三道檢查，與 enqueue 前的張量核對。

接手 `AutoBackend` 之後（ADR-014），「這顆引擎能不能用」有一半的判準從 ultralytics 搬
到了本模組。搬過來的每一項都是**「引擎跑得完、輸出檔也完全正常，只有數值或成本不對」**：
多輸入的引擎只會綁到其中一個 binding、輸出最後一維不是 6 代表引擎沒有內建 NMS（後處理
會把類別分數當成 conf）、profile 的空間維不對則是一整天都跑在為別的尺寸挑的 kernel 上
（收窄 profile 之後，「沒收窄過的舊引擎」也是同一類——見 ADR-015）。

載入期那幾支用假的 `trt.Runtime` 跑（引擎檔頭是真的、序列化的引擎是假的），因此在 CI 上
也測得到；真引擎與 GPU 定序的兩支標了 `skipif`，只在地端跑。
"""

import json
import struct
from pathlib import Path

import pytest
import tensorrt as trt
import torch

from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.trt_runner import (
    END2END_COLUMNS,
    TrtRunner,
    _check_infer_shape_of,
    _check_infer_tensor,
    check_profile_shapes,
)

_NO_GPU = not torch.cuda.is_available()
REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeContext:
    """記下被設過的 shape 與位址，`execute_async_v3` 一律成功。"""

    def __init__(self):
        self.shapes = {}
        self.addresses = {}
        self.streams = []

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)
        return True

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def execute_async_v3(self, stream_handle):
        self.streams.append(stream_handle)
        return True


class _FakeEngine:
    """只回答 `TrtRunner.__init__` 會問的那幾個問題。

    `tensors` 每筆是 `(名稱, mode, dtype, shape)`；輸入的 profile 由 `profile` 指定
    （`(min, opt, max)`）。
    """

    def __init__(self, tensors, profile):
        self._tensors = tensors
        self._profile = profile
        self.context = _FakeContext()

    @property
    def num_io_tensors(self):
        return len(self._tensors)

    def get_tensor_name(self, index):
        return self._tensors[index][0]

    def _entry(self, name):
        return next(t for t in self._tensors if t[0] == name)

    def get_tensor_mode(self, name):
        return self._entry(name)[1]

    def get_tensor_dtype(self, name):
        return self._entry(name)[2]

    def get_tensor_shape(self, name):
        return self._entry(name)[3]

    def get_tensor_profile_shape(self, name, index):
        return self._profile

    def create_execution_context(self):
        return self.context


def _tensors(
    *,
    num_inputs: int = 1,
    input_dtype=trt.DataType.FLOAT,
    output_shape=(-1, 300, END2END_COLUMNS),
):
    entries = [
        (f"images{i}" if i else "images", trt.TensorIOMode.INPUT, input_dtype, (-1, 3, -1, -1))
        for i in range(num_inputs)
    ]
    entries.append(
        ("output0", trt.TensorIOMode.OUTPUT, trt.DataType.FLOAT, output_shape)
    )
    return entries


def _engine_file(tmp_path):
    """帶著真的 ultralytics 檔頭、但引擎本體是幾個位元組的假檔。"""
    meta = json.dumps({"batch": 16, "args": {"half": True, "dynamic": True}}).encode()
    path = tmp_path / "fake_sm75.engine"
    path.write_bytes(struct.pack("<i", len(meta)) + meta + b"\x00engine")
    return path


def _runner(monkeypatch, tmp_path, engine) -> TrtRunner:
    """用假的 `trt.Runtime` 建一顆 runner；`engine` 為 `None` 代表 deserialize 失敗。"""

    class _FakeRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, payload):
            return engine

    monkeypatch.setattr(trt, "Runtime", _FakeRuntime)
    return TrtRunner(_engine_file(tmp_path))


# 收窄後的 profile（ADR-015）：空間維三個界全部釘在推論尺寸，batch 維 1–16。
_SPATIAL = (3, INFER_HEIGHT, INFER_WIDTH)
_PROFILE = ((1, *_SPATIAL), (16, *_SPATIAL), (16, *_SPATIAL))

# 收窄之前 ultralytics 給的那組界：下界是一個永遠跑不起來的形狀、上界是 `workspace`
# 當倍數的副產物，opt 則已經是推論尺寸。
_WIDE_PROFILE = ((1, 3, 32, 32), (16, *_SPATIAL), (16, 3, 768, 1280))


def test_runner_reads_the_batch_ceiling_and_input_size_from_the_profile(
    monkeypatch, tmp_path
):
    """批次上限與高寬取自 optimization profile，不是 JSON metadata 的 `batch`。

    後者是檔頭裡的一個字串欄位，改它不會改變引擎——而「批次超限」的失效方式是
    `set_input_shape` 回 False 之後照樣拿上一批的 shape 跑，輸出形狀正常、內容是別批的。
    """
    runner = _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), _PROFILE))

    assert runner.max_batch == 16
    assert (runner.input_height, runner.input_width) == (INFER_HEIGHT, INFER_WIDTH)
    assert runner.output_num_det == 300


def test_runner_rejects_an_engine_with_more_than_one_input(monkeypatch, tmp_path):
    """多輸入的引擎要在載入期擋下：只會綁到其中一個，其餘 binding 停在未設定的位址。"""
    with pytest.raises(ValueError, match="I/O 張量"):
        _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(num_inputs=2), _PROFILE))


def test_runner_rejects_an_engine_without_builtin_nms(monkeypatch, tmp_path):
    """輸出最後一維不是 6 要擋下——那顆引擎沒有內建 NMS。

    它吐的是 `(B, 4 + nc, num_anchors)` 的原始張量，而 `postprocess_batch` 那三行只讀
    第 4、5 欄，照樣跑得完：把某個類別分數當成 conf、另一個當成 cls，得到一堆座標是
    xywh 的框，而列數、欄位、格式全部正常。

    判準用引擎自己宣告的 binding 形狀，不讀 metadata 的 `end2end` 欄位：後者改了不會
    改變引擎，而這裡要驗的正是引擎本身。
    """
    engine = _FakeEngine(_tensors(output_shape=(-1, 7, 5040)), _PROFILE)

    with pytest.raises(ValueError, match="end2end"):
        _runner(monkeypatch, tmp_path, engine)


def test_runner_rejects_an_engine_whose_input_binding_is_not_fp32(monkeypatch, tmp_path):
    """FP16 引擎的 I/O binding 仍是 FP32；真的收到別的 dtype 代表位元組會被重新解讀。"""
    engine = _FakeEngine(_tensors(input_dtype=trt.DataType.HALF), _PROFILE)

    with pytest.raises(ValueError, match="FP32"):
        _runner(monkeypatch, tmp_path, engine)


def test_runner_rejects_an_engine_built_for_another_input_size(monkeypatch, tmp_path):
    """profile 的 opt 高寬不是推論尺寸要擋下——症狀只有「變慢」。

    高寬是 dynamic axes，這顆引擎照樣吃得下 640×384、框也對，只是所有 kernel 都是為
    別的尺寸挑的。沒有這道檢查的話，整天的分析會安靜地跑在錯的 kernel 上。
    """
    profile = ((1, 3, 32, 32), (16, 3, 768, 1280), (16, 3, 768, 1280))

    with pytest.raises(ValueError, match="空間維"):
        _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), profile))


def test_runner_rejects_an_engine_whose_profile_was_never_narrowed(
    monkeypatch, tmp_path
):
    """opt 對、min／max 沒收窄的舊引擎也要擋下。

    這是收窄 profile（ADR-015）之前每一顆引擎的形狀：opt 已經是 384×640，所以它會通過
    其餘全部載入檢查，症狀只有「forward 慢 2.3%～6.1%、每個 execution context 多吃約
    1 GB 裝置記憶體」——與這個 `__init__` 裡別的檢查擋的是同一類失效。沒有這道檢查的
    話，改動就變成「新引擎比較快，而舊引擎照樣載得起來」，兩者無從分辨。
    """
    with pytest.raises(ValueError, match="空間維"):
        _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), _WIDE_PROFILE))


def test_profile_check_rejects_a_batch_lower_bound_above_one():
    """batch 下界大於 1 的引擎，第一個湊不滿的批次就送不進去。

    湊不滿批是常態而非例外（T4 上實測一次跑完出現 1 到 16 全部 16 種批次）。這條在
    `enqueue` 裡本來就 fail loud，但那是部署之後才看得到；載入期驗同一件事，建置工具
    才擋得下。
    """
    with pytest.raises(ValueError, match="batch 下界"):
        check_profile_shapes((2, *_SPATIAL), (16, *_SPATIAL), (16, *_SPATIAL))


def test_profile_check_rejects_an_opt_batch_below_the_ceiling():
    """opt batch 不等於上界代表 kernel 是為一個不是主要工作點的批次挑的。"""
    with pytest.raises(ValueError, match="opt batch"):
        check_profile_shapes((1, *_SPATIAL), (8, *_SPATIAL), (16, *_SPATIAL))


def test_runner_reports_a_failed_deserialize_instead_of_crashing_later(
    monkeypatch, tmp_path
):
    """`deserialize_cuda_engine` 回 None 不會拋錯——不擋的話下一行才會崩在原生層。"""
    with pytest.raises(RuntimeError, match="deserialize"):
        _runner(monkeypatch, tmp_path, None)


def test_runner_reports_a_context_that_could_not_be_created(monkeypatch, tmp_path):
    """`create_execution_context()` 同樣是失敗回 None 而不拋錯（多半是顯存不夠）。

    不擋的話症狀是 warmup 期間的 `'NoneType' object has no attribute
    'set_input_shape'`，指不出真正的原因。
    """
    engine = _FakeEngine(_tensors(), _PROFILE)
    engine.context = None

    with pytest.raises(RuntimeError, match="execution context"):
        _runner(monkeypatch, tmp_path, engine)


def _infer_tensor(**overrides) -> torch.Tensor:
    kwargs = {"dtype": torch.float32}
    kwargs.update(overrides)
    return torch.zeros((2, 3, INFER_HEIGHT, INFER_WIDTH), **kwargs)


def test_infer_tensor_check_rejects_a_half_precision_tensor():
    """FP16 引擎的 I/O binding 仍是 FP32：送 float16 進去只是把位元組重新解讀。"""
    with pytest.raises(ValueError, match="float32"):
        _check_infer_tensor(_infer_tensor(dtype=torch.float16))


def test_infer_tensor_check_rejects_a_non_contiguous_tensor():
    """TensorRT 取的是 `data_ptr()`、不看 stride——非連續張量會被當成連續的讀進去。

    `[..., [2, 1, 0]]` 再 `permute` 正好產出這種張量：逐值與正確版相同，佈局不同。
    """
    non_contiguous = _infer_tensor().permute(0, 1, 3, 2)

    with pytest.raises(ValueError, match="contiguous"):
        _check_infer_tensor(non_contiguous)


def test_infer_tensor_check_rejects_a_tensor_that_is_not_four_dimensional():
    with pytest.raises(ValueError, match="4 維"):
        _check_infer_tensor(torch.zeros((3, INFER_HEIGHT, INFER_WIDTH)))


def test_infer_tensor_check_rejects_a_cpu_tensor():
    """把 CPU 位址交給引擎是未定義行為。

    這一項排在最後，前三項才能在沒有 GPU 的機器上也測得到。
    """
    with pytest.raises(RuntimeError, match="CUDA"):
        _check_infer_tensor(_infer_tensor())


def test_infer_shape_check_accepts_the_reader_side_shape(capsys):
    """640×384 進、640×384 出：影格在讀取端就縮好了，前處理不該再加填充。"""
    _check_infer_shape_of((16, 3, INFER_HEIGHT, INFER_WIDTH))

    assert str(INFER_HEIGHT) in capsys.readouterr().out


def test_infer_shape_check_rejects_a_padded_square_shape():
    """被填成 640×640 要當場擋下。

    dynamic 引擎的高寬維也是動態的（`exporter.py` 把 height／width 一併列為 dynamic
    axes），落在 profile 範圍內的錯誤高寬**不會**被 TensorRT 擋下——引擎照跑，吐出來的
    框停在錯誤的尺度上，而列數、欄位、格式全部正常。
    """
    with pytest.raises(ValueError, match="640"):
        _check_infer_shape_of((16, 3, 640, 640))


def test_enqueue_refuses_the_default_stream(monkeypatch, tmp_path):
    """default stream 會讓 TensorRT 插入 `cudaDeviceSynchronize`，流水退化成序列。

    症狀只有「吞吐掉回改動前」，輸出檔一模一樣，所以只能在這裡擋。這一項排在張量檢查
    **之前**，沒有 GPU 的機器上才測得到（張量那幾道有一項是「必須在 CUDA 上」）。
    """
    runner = _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), _PROFILE))

    class _DefaultStream:
        cuda_stream = 0

    with pytest.raises(RuntimeError, match="default stream"):
        runner.enqueue(
            _infer_tensor(),
            torch.zeros((2, 300, END2END_COLUMNS)),
            _DefaultStream(),
        )


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_enqueue_rejects_an_output_buffer_of_a_different_batch_size(
    monkeypatch, tmp_path
):
    """輸入與輸出的批次對不上要擋：引擎照樣寫得完，只是這一批的結果填不滿或溢出。"""
    runner = _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), _PROFILE))
    im = torch.zeros(
        (2, 3, INFER_HEIGHT, INFER_WIDTH), dtype=torch.float32, device="cuda:0"
    )
    out = torch.zeros((3, 300, END2END_COLUMNS), device="cuda:0")

    with pytest.raises(ValueError, match="輸出緩衝的形狀"):
        runner.enqueue(im, out, torch.cuda.Stream())


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_the_runner_matches_the_ultralytics_backend_bit_for_bit():
    """同一顆引擎、同一批輸入，自建 runner 的原始輸出與 `AutoBackend.forward` 逐位元
    相同。

    這支釘的是「換掉 `AutoBackend` 沒有改變引擎跑出來的數」——binding 綁錯、shape 設錯、
    stream 對不上都會讓輸出變成別的數而形狀完全正常。`execute_async_v3` 與 `execute_v2`
    走的是同一顆 context，差別只在誰等它算完。

    引擎綁 GPU 架構，CI 上沒有卡也沒有引擎檔，因此整支 skip——**這代表本項只在地端
    驗過**（PR 內文會寫明）。
    """
    from ultralytics.nn.autobackend import AutoBackend

    from video_analyze.models.config import settings

    # `model_path` 是 repo 根的相對路徑（正式執行一律在那裡跑），而 pytest 走
    # `--directory video_analyze`，cwd 是套件目錄，所以這裡自己接回去
    engine_path = REPO_ROOT / settings.model.model_path
    if not engine_path.is_file():
        pytest.skip(f"找不到引擎檔 {engine_path}")

    device = torch.device("cuda:0")
    runner = TrtRunner(engine_path)
    torch.manual_seed(20260829)
    im = torch.rand(
        (4, 3, INFER_HEIGHT, INFER_WIDTH), dtype=torch.float32, device=device
    ).contiguous()
    out = torch.empty(
        (4, runner.output_num_det, END2END_COLUMNS),
        dtype=torch.float32,
        device=device,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner.enqueue(im, out, stream)
    stream.synchronize()

    reference = AutoBackend(str(engine_path), device=device, fp16=False).forward(im)

    assert torch.equal(out, reference)


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_a_batch_larger_than_the_profile_is_rejected_instead_of_reusing_the_last_shape(
    monkeypatch, tmp_path
):
    """`set_input_shape` 回 False 要當場拋錯。

    這是本模組最危險的一條路徑：TensorRT 只回 `False`、不拋錯，而接下來的
    `execute_async_v3` 仍回 `True` 並沿用**上一批**的 shape 跑——輸出形狀正常、內容是
    別批的框。
    """

    class _RefusingContext(_FakeContext):
        def set_input_shape(self, name, shape):
            return False

    engine = _FakeEngine(_tensors(), _PROFILE)
    engine.context = _RefusingContext()
    runner = _runner(monkeypatch, tmp_path, engine)
    im = torch.zeros(
        (2, 3, INFER_HEIGHT, INFER_WIDTH), dtype=torch.float32, device="cuda:0"
    )
    out = torch.zeros((2, 300, END2END_COLUMNS), device="cuda:0")

    with pytest.raises(RuntimeError, match="set_input_shape"):
        runner.enqueue(im, out, torch.cuda.Stream())


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_enqueue_passes_the_given_stream_and_the_real_buffer_addresses(
    monkeypatch, tmp_path
):
    """交給 TensorRT 的是呼叫端指定的 stream 與那兩塊 buffer 的實際位址。

    位址取錯（例如取到某個中間張量）會讓引擎讀寫別塊記憶體，而輸出緩衝停在上一批的
    內容——後處理拿到的是形狀正常的重複框。
    """
    engine = _FakeEngine(_tensors(), _PROFILE)
    runner = _runner(monkeypatch, tmp_path, engine)
    im = torch.zeros(
        (2, 3, INFER_HEIGHT, INFER_WIDTH), dtype=torch.float32, device="cuda:0"
    )
    out = torch.zeros((2, 300, END2END_COLUMNS), device="cuda:0")
    stream = torch.cuda.Stream()

    runner.enqueue(im, out, stream)

    assert engine.context.shapes == {"images": (2, 3, INFER_HEIGHT, INFER_WIDTH)}
    assert engine.context.addresses == {
        "images": im.data_ptr(),
        "output0": out.data_ptr(),
    }
    assert engine.context.streams == [stream.cuda_stream]


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_the_same_shape_is_only_logged_once(monkeypatch, tmp_path, capsys):
    """同一個形狀只記一次 log：一天數十萬批，每批一行會把 log 淹掉。"""
    runner = _runner(monkeypatch, tmp_path, _FakeEngine(_tensors(), _PROFILE))
    out = torch.zeros((2, 300, END2END_COLUMNS), device="cuda:0")
    im = torch.zeros(
        (2, 3, INFER_HEIGHT, INFER_WIDTH), dtype=torch.float32, device="cuda:0"
    )
    stream = torch.cuda.Stream()

    runner.enqueue(im, out, stream)
    runner.enqueue(im, out, stream)

    assert capsys.readouterr().out.count("推論張量形狀") == 1
