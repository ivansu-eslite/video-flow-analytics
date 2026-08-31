"""引擎載入前的五道檢查、執行期的張量核對，以及自建的前處理與後處理。

正式推論路徑收斂成 TensorRT FP16 引擎之後（ADR-011），載入那一層擋的全是**「跑起來
完全正常、輸出檔也完全正常，只有數值或成本不對」**的組合：載到 `.pt`（慢一個量級）、
載到別的模型（類別語義不同）、載到 FP32 引擎（吞吐掉回改動前）、高寬不是 640×384、
引擎沒有內建 NMS（過濾語義整個對不上）。這些都不會拋錯，所以只能靠這幾道檢查在載入
當下擋下。

前處理與後處理改成本套件自己算之後（ADR-013），多了一組**與 ultralytics 8.4.75 逐值
比對**的測試：那兩段是照抄第三方的語義，抄錯的後果同樣是「變快又變得不一樣，而輸出
檔的欄位、列數、格式全部正常」。兩者都是模組層純函式，測試在 CPU 上跑，不需要 GPU
也不需要引擎。

引擎自帶 metadata 與環境的比對在 test_engine_metadata.py。
"""

import json
import struct

import numpy as np
import pytest
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from video_analyze.config.constants import FBODY_CLASS_ID, HEAD_CLASS_ID
from video_analyze.models.config import settings
from video_analyze.services.batching import TARGET_BATCH
from video_analyze.services.detector import (
    CONF_THRESHOLD,
    MAX_DET,
    PIPELINE_DEPTH,
    YOLODetector,
    _log_engine_metadata,
    _require_engine_file,
    _validate_classes,
    _validate_dynamic,
    postprocess_batch,
    stack_frames,
    to_infer_tensor,
)
from video_analyze.services.engine_metadata import (
    VFA_METADATA_KEY,
    VFA_METADATA_SCHEMA,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.trt_runner import END2END_COLUMNS

_NO_GPU = not torch.cuda.is_available()


def _metadata(**overrides) -> dict:
    metadata = {
        "names": {0: "head", 1: "vbody", 2: "fbody"},
        "batch": 16,
        "args": {"half": True, "dynamic": True, "batch": 16},
        VFA_METADATA_KEY: {
            "schema": VFA_METADATA_SCHEMA,
            "source_weights": {"name": "baseline.pt", "sha256": "a" * 64},
            "compute_capability": "7.5",
            "gpu_name": "Tesla T4",
            "tensorrt": "10.13.3.9",
            "tensorrt_package": "tensorrt-cu12",
            "torch_cuda_major": "12",
            "driver": "550.54.15",
            "train": {
                "base_model": "best.pt",
                "dataset": "data.yaml",
                "ultralytics": "8.4.90",
                "date": "2026-07-14",
                "metrics": {"mAP50": 0.806},
            },
        },
    }
    metadata.update(overrides)
    return metadata


def _fake_engine(tmp_path, metadata: dict | None = None):
    meta = json.dumps(metadata if metadata is not None else _metadata()).encode()
    path = tmp_path / "fake_sm75.engine"
    path.write_bytes(struct.pack("<i", len(meta)) + meta + b"\x00engine")
    return path


def test_require_engine_file_rejects_a_torch_weight(tmp_path):
    """還指著 `.pt` 要當場擋下。

    那條路**載得起來也跑得出結果**，只是慢一個量級——正式套件裡已經沒有 Torch 推論
    路徑（ADR-011），沒有這道檢查的話 `model_path` 忘了改就只表現為「怎麼沒變快」。
    """
    weights = tmp_path / "baseline.pt"
    weights.write_bytes(b"not really a weight")

    with pytest.raises(ValueError, match=r"\.engine"):
        _require_engine_file(weights)


def test_require_engine_file_rejects_a_missing_engine_naming_the_real_reason(tmp_path):
    """檔案不存在要在載入之前擋下，且訊息要指得出真正的原因。

    不經 `YOLO` 載入之後（ADR-013、ADR-014）拿掉這道檢查不會再讓 ultralytics 去找替代
    來源，但下一步 `read_engine_metadata` 直接 `open()`，拋出來的 `FileNotFoundError`
    只有一個路徑字串，看不出「`model_path` 是 cwd 相對路徑、跑錯目錄了」這個實際上最
    常見的原因。

    斷言訊息內容而不只是例外型別：`open()` 自己也拋 `FileNotFoundError`，只驗型別的話
    這道檢查被拿掉仍然全綠。
    """
    with pytest.raises(FileNotFoundError, match="cwd") as excinfo:
        _require_engine_file(tmp_path / "missing_sm75.engine")

    assert "repo 根" in str(excinfo.value)


def test_yolo_detector_rejects_a_torch_weight_before_touching_the_gpu(
    monkeypatch, tmp_path
):
    """副檔名這條排在最前面，沒有 GPU 的機器上也擋得到。"""
    weights = tmp_path / "baseline.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(settings.model, "model_path", str(weights))

    with pytest.raises(ValueError, match=r"\.engine"):
        YOLODetector()


def test_yolo_detector_raises_when_the_engine_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings.model, "model_path", str(tmp_path / "missing_sm75.engine")
    )

    with pytest.raises(FileNotFoundError, match="cwd"):
        YOLODetector()


def test_yolo_detector_aborts_without_cuda_instead_of_falling_back_to_cpu(
    monkeypatch, tmp_path
):
    """CUDA 不可用要中止，不是 fallback CPU。

    改吃引擎之後，CPU 從「比較慢」變成「一定失敗」——引擎綁 GPU，deserialize 之後
    第一次 predict 就會崩。留著 fallback 只是把同一個錯誤延後到跑了一段之後。
    """
    import torch

    monkeypatch.setattr(settings.model, "model_path", str(_fake_engine(tmp_path)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        YOLODetector()


def test_yolo_detector_calls_the_shared_precision_check(monkeypatch, tmp_path):
    """載入序列真的會呼叫 `engine_metadata.validate_engine_precision`，不是掛在
    `engine_metadata.py` 卻沒被接進 `__init__`。精度細節本身的 case（FP16 通過／
    FP32 擋下／INT8 擋下／缺欄擋下）在 test_engine_metadata.py，這裡只釘「載入序列有
    呼叫到它」——拿掉這行呼叫，這支測試要紅，而不會被其餘各驗各欄位的載入檢查頂替。
    """
    monkeypatch.setattr(settings.model, "model_path", str(_fake_engine(tmp_path)))
    monkeypatch.setattr(
        "video_analyze.services.detector.current_gpu_environment", lambda: None
    )
    monkeypatch.setattr(
        "video_analyze.services.detector.validate_engine_metadata", lambda *a, **k: None
    )

    def _raise_marker(metadata):
        raise ValueError("precision-check-was-called")

    monkeypatch.setattr(
        "video_analyze.services.detector.validate_engine_precision", _raise_marker
    )

    with pytest.raises(ValueError, match="precision-check-was-called"):
        YOLODetector()


def test_validate_classes_passes_when_all_ids_present(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [0, 2])

    _validate_classes(_metadata())  # 不應拋例外


def test_validate_classes_raises_when_id_missing_from_engine_names(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2, 5])

    with pytest.raises(ValueError, match=r"\[5\]"):
        _validate_classes(_metadata())


def test_validate_classes_skips_when_engine_names_unavailable(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2])

    _validate_classes(_metadata(names=None))  # 缺 names 時無法驗證，略過而非拋例外


def test_log_engine_metadata_prints_the_injected_training_provenance(capsys):
    """訓練版本／日期／指標在引擎路徑下**不在引擎裡**，是建置期從 `.pt` 的 ckpt
    抄進 metadata 的。這幾行是「這批結果是哪個訓練版本跑的」唯一的線索。"""
    _log_engine_metadata(_metadata())

    out = capsys.readouterr().out
    assert "fbody" in out
    assert "baseline.pt" in out
    assert "Tesla T4" in out
    assert "8.4.90" in out
    assert "2026-07-14" in out
    assert "0.806" in out


def test_log_engine_metadata_survives_an_engine_without_the_injected_block(capsys):
    """記 metadata 是追溯用的，缺欄位只少印幾行，不能讓模型載入失敗。"""
    metadata = _metadata()
    del metadata[VFA_METADATA_KEY]

    _log_engine_metadata(metadata)  # 不拋例外

    assert "fbody" in capsys.readouterr().out


def test_validate_dynamic_accepts_a_dynamic_engine():
    _validate_dynamic(_metadata())


def test_validate_dynamic_rejects_a_static_engine():
    """靜態引擎要在載入時擋下，不能留給推論當下的 assert。

    它會通過其餘全部載入檢查（warmup 那一批剛好是滿批），然後在第一個沒湊滿的批次上
    被 `TrtRunner.enqueue` 的 `set_input_shape` 擋下——訊息看得出形狀與上限，看不出
    原因是引擎綁死了批次。而湊不滿批是常態而非例外：T4（n1-standard-8）上實測一次跑完
    出現 16 種不同的批次大小，1 到 16 全都有，等於整天的分析跑到一半才失敗。

    `TrtRunner` 的形狀檢查幫不上忙——它驗的是我們自己組出來的張量形狀，那個形狀本來
    就是對的；容不下它的是引擎。
    """
    with pytest.raises(ValueError, match="dynamic"):
        _validate_dynamic(_metadata(args={"half": True, "dynamic": False}))


def _ultralytics_preprocess_reference(frames: list[np.ndarray]) -> torch.Tensor:
    """ultralytics `BasePredictor.preprocess`（8.4.75）在本專案條件下的等價運算。

    `pre_transform` 的 LetterBox 在這裡是恆等（影格已是推論尺寸、`auto` 讓四邊
    padding 全 0），所以參照只留它之後的五步：BGR→RGB、BHWC→BCHW、contiguous、
    `.float()`、`/255`。
    """
    im = np.stack(frames)
    im = im[..., ::-1].transpose(0, 3, 1, 2)
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).float() / 255


def _synthetic_frames(count: int) -> list[np.ndarray]:
    """三個通道值各不相同的合成影格——通道翻錯或軸換錯才抓得出來。"""
    rng = np.random.default_rng(20260829)
    return [
        rng.integers(0, 256, (INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def _preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    """在 CPU 上把兩段自建前處理接起來，等同正式路徑的「`stack_frames` ＋ H2D ＋
    `to_infer_tensor`」——中間那段搬動不改變任何一個位元組。"""
    buffer = np.empty(
        (len(frames) + 2, INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8
    )
    stack_frames(frames, buffer)
    return to_infer_tensor(torch.from_numpy(buffer[: len(frames)]))


def test_preprocess_matches_the_ultralytics_reference_element_for_element():
    """自建前處理與 ultralytics 的路徑**逐值相同**，不是「差異不大」。

    每個元素都是 `uint8 → float32 → /255` 的獨立運算，兩邊都 round-to-nearest，所以
    這裡沒有容差可談——差一格就代表通道、軸或型別有一項寫錯，而那種錯誤跑起來完全
    正常，只是偵測結果不一樣。
    """
    frames = _synthetic_frames(3)

    got = _preprocess(frames)

    assert torch.equal(got, _ultralytics_preprocess_reference(frames))


def test_preprocess_hands_over_a_contiguous_float32_nchw_tensor():
    """dtype、佈局與軸順序三件事各自都能無聲地毀掉推論結果。

    float16 會被 FP32 binding 重新解讀；非 contiguous 會被 TensorRT 當成連續的讀
    （它只拿 `data_ptr()`）——`[..., [2, 1, 0]]` 再 `permute` 就是這種張量，逐值相同
    但佈局不同，所以這裡連 `is_contiguous()` 一起釘。
    """
    frames = _synthetic_frames(2)

    got = _preprocess(frames)

    assert got.dtype == torch.float32
    assert got.is_contiguous()
    assert tuple(got.shape) == (2, 3, INFER_HEIGHT, INFER_WIDTH)
    # 通道真的翻過：RGB 的第 0 個通道要等於 BGR 影格的第 2 個通道
    np.testing.assert_allclose(got[0, 0].numpy(), frames[0][:, :, 2] / 255)


def test_stack_frames_only_writes_the_prefix_of_the_buffer():
    """`stack_frames` 只寫前 B 格，其餘位置不動。

    緩衝一律照引擎的 `max_batch` 配置，而湊不滿批是常態。寫超過或整塊覆寫都會讓上一
    批殘留的影格被當成本批的資料送進引擎——多出來的那幾格框看起來完全正常。
    """
    buffer = np.full((4, INFER_HEIGHT, INFER_WIDTH, 3), 7, dtype=np.uint8)
    frames = [
        np.full((INFER_HEIGHT, INFER_WIDTH, 3), value, dtype=np.uint8)
        for value in (1, 2)
    ]

    stack_frames(frames, buffer)

    assert buffer[0, 0, 0, 0] == 1
    assert buffer[1, 0, 0, 0] == 2
    assert (buffer[2:] == 7).all()


def _raw_batch(rows: list[list[tuple[float, ...]]]) -> np.ndarray:
    """把逐格的 `(x1, y1, x2, y2, conf, cls)` 列表堆成引擎輸出的 `(B, num_det, 6)`。"""
    return np.array(rows, dtype=np.float32)


def _ultralytics_postprocess_reference(
    raw: np.ndarray, conf: float, classes: list[int], max_det: int
) -> list[np.ndarray]:
    """ultralytics end2end 分支 ＋ `construct_result` 的 `scale_boxes` 的等價運算。"""
    preds = non_max_suppression(
        torch.from_numpy(raw.copy()),
        conf_thres=conf,
        iou_thres=0.7,
        classes=classes,
        agnostic=False,
        max_det=max_det,
        nc=0,
        end2end=True,
    )
    outputs = []
    for pred in preds:
        pred[:, :4] = scale_boxes(
            (INFER_HEIGHT, INFER_WIDTH), pred[:, :4], (INFER_HEIGHT, INFER_WIDTH, 3)
        )
        outputs.append(pred[:, :6].numpy())
    return outputs


def test_postprocess_matches_the_ultralytics_end2end_reference():
    """含四種情況的一批：留下的、低於門檻的、類別不對的、座標超出畫面的。

    座標那一列釘的是 `scale_boxes` 那一步沒有被省掉——在本專案它退化成 clip，省掉
    的話超界的框會原樣流到下游，而列數與欄位完全正常。
    """
    raw = _raw_batch(
        [
            [
                (10.0, 20.0, 60.0, 120.0, 0.90, float(FBODY_CLASS_ID)),
                (-8.0, -3.0, 700.0, 400.0, 0.80, float(HEAD_CLASS_ID)),
                (30.0, 40.0, 50.0, 60.0, 0.10, float(FBODY_CLASS_ID)),
                (30.0, 40.0, 50.0, 60.0, 0.95, 1.0),
            ],
            [
                (0.0, 0.0, 0.0, 0.0, 0.01, 1.0),
                (5.0, 5.0, 15.0, 25.0, 0.55, float(FBODY_CLASS_ID)),
                (1.0, 2.0, 3.0, 4.0, 0.05, float(HEAD_CLASS_ID)),
                (2.0, 2.0, 4.0, 4.0, 0.02, float(FBODY_CLASS_ID)),
            ],
        ]
    )
    classes = [HEAD_CLASS_ID, FBODY_CLASS_ID]

    got = postprocess_batch(raw, CONF_THRESHOLD, classes, MAX_DET)
    expected = _ultralytics_postprocess_reference(raw, CONF_THRESHOLD, classes, MAX_DET)

    assert [len(g) for g in got] == [2, 1]
    for one, other in zip(got, expected, strict=True):
        np.testing.assert_array_equal(one, other)


def test_postprocess_drops_the_row_that_sits_exactly_on_the_threshold():
    """conf 是**嚴格大於**：恰好等於門檻的那一列要丟掉。

    寫成 `>=` 只會讓極少數邊界框多留下來——總列數幾乎不變，下游的計數卻與過去的檔案
    對不上，而兩邊看起來都正常。
    """
    raw = _raw_batch(
        [
            [
                (10.0, 10.0, 20.0, 20.0, CONF_THRESHOLD, float(FBODY_CLASS_ID)),
                (10.0, 10.0, 20.0, 20.0, CONF_THRESHOLD + 1e-6, float(FBODY_CLASS_ID)),
            ]
        ]
    )

    got = postprocess_batch(raw, CONF_THRESHOLD, [FBODY_CLASS_ID], MAX_DET)

    assert len(got[0]) == 1
    assert got[0][0][4] > CONF_THRESHOLD


def test_class_filter_runs_after_the_max_det_truncation():
    """順序是 conf → 截斷 → classes，不是 conf → classes → 截斷。

    這批輸入讓兩種順序給出不同答案：前兩列是 conf 較高的別種類別，第三列才是目標
    類別。現行順序截斷後只剩那兩列別種類別，過濾完是空的；把 classes 提到截斷之前
    會留下第三列。差別只在「多／少幾個低分框」，輸出的欄位、格式全部正常。
    """
    raw = _raw_batch(
        [
            [
                (0.0, 0.0, 10.0, 10.0, 0.90, 1.0),
                (0.0, 0.0, 10.0, 10.0, 0.80, 1.0),
                (0.0, 0.0, 10.0, 10.0, 0.70, float(FBODY_CLASS_ID)),
            ]
        ]
    )

    got = postprocess_batch(raw, CONF_THRESHOLD, [FBODY_CLASS_ID], max_det=2)

    assert len(got[0]) == 0
    # 參照組照同一個順序，兩邊一起釘住
    expected = _ultralytics_postprocess_reference(
        raw, CONF_THRESHOLD, [FBODY_CLASS_ID], max_det=2
    )
    np.testing.assert_array_equal(got[0], expected[0])

class _CountingRunner:
    """假 runner：在指定 stream 上把每一格的第一個像素寫進輸出的 conf 欄。

    - **conf 由輸入算出來**，所以「第 k 批的結果配到第 k+1 批的影格」在斷言上看得見。
    - **刻意用 `torch.cuda._sleep` 拖慢**：緩衝重用的錯誤（上一批還沒讀完就被覆寫）
      只有在 GPU 落後 host 時才顯現，跑得太快的假 runner 會讓那種錯誤剛好不發生。
    - 其餘 299 列留 0，`postprocess_batch` 的 conf 過濾會把它們濾掉，所以每一格恰好
      回一列——列數本身就是一道檢查。
    """

    max_batch = TARGET_BATCH
    output_num_det = 8

    def __init__(self, sleep_cycles: int = 40_000_000):
        self._sleep_cycles = sleep_cycles

    def enqueue(self, im, out, stream) -> None:
        assert stream.cuda_stream != 0, "假 runner 也不該收到 default stream"
        torch.cuda._sleep(self._sleep_cycles)
        out.zero_()
        # 第 0 通道是 RGB 的 R，即 BGR 影格的第 2 個通道；合成影格三通道同值
        out[:, 0, 4] = 0.26 + im[:, 0, 0, 0] * 0.1
        out[:, 0, 5] = float(FBODY_CLASS_ID)


def _detector_with_runner(runner) -> YOLODetector:
    """繞過 `__init__` 的引擎載入，只把流水那一半組起來。

    走完整的 `__init__` 需要一顆真的引擎；這裡要驗的是 `_init_pipeline` 配出來的緩衝、
    stream 與 event 怎麼被 `submit`／`collect` 用，與引擎無關。
    """
    detector = YOLODetector.__new__(YOLODetector)
    detector.device = torch.device("cuda:0")
    detector.runner = runner
    detector.max_batch = runner.max_batch
    detector._init_pipeline()
    return detector


def _expected_conf(runner, frames: list[np.ndarray]) -> list[float]:
    """同步參照：同一顆假 runner、一批一次、每批之間完全同步。"""
    device = torch.device("cuda:0")
    im = (
        torch.from_numpy(np.stack(frames))
        .to(device)
        .permute(0, 3, 1, 2)[:, [2, 1, 0]]
        .float()
        .div_(255)
    )
    out = torch.empty(
        (len(frames), runner.output_num_det, END2END_COLUMNS), device=device
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner.enqueue(im, out, stream)
    stream.synchronize()
    boxes = postprocess_batch(
        out.cpu().numpy(), CONF_THRESHOLD, [FBODY_CLASS_ID], MAX_DET
    )
    return [float(one[0][4]) for one in boxes]


def _frames(values: list[int]) -> list[np.ndarray]:
    return [
        np.full((INFER_HEIGHT, INFER_WIDTH, 3), value, dtype=np.uint8)
        for value in values
    ]


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_pipelined_batches_keep_their_own_data_across_buffer_reuse(monkeypatch):
    """連續 `submit`／`collect` 混不同批次大小，每一批拿回的仍是自己的資料。

    這支釘的是 event 定序與 ping-pong 緩衝的重用。深度 2 代表第 k 批與第 k−2 批共用
    同一組緩衝，任何一條依賴漏掉（例如 `_pinned_in` 在上一輪 H2D 還沒讀完就被
    `stack_frames` 覆寫、或 `_dev_out` 在 D2H 還沒搬完就被下一次 forward 覆寫）都會
    讓某一批拿到別批的內容，而列數、欄位、格式全部正常。

    參照組是同一顆假 runner 一批一次、每批之間完全同步跑出來的，所以兩邊的差只可能
    來自流水本身。
    """
    monkeypatch.setattr(settings.model, "classes", [FBODY_CLASS_ID])
    runner = _CountingRunner()
    detector = _detector_with_runner(runner)
    # 各批的值域**不可重疊**：每批都從 1 開始的話，「第 k 批的緩衝被第 k±2 批覆寫」
    # 在數值上是恆等的（前綴逐值相同，只有尾巴幾格會少），斷言就退化成「列數對不對」
    # 而列數是 Python 這側記帳決定的，與 event 無關——實測那種測資下把 `wait_event`
    # 全部拿掉仍然全綠
    batches = [
        _frames(list(range(start, start + size)))
        for start, size in ((1, 5), (40, 2), (80, 7), (120, 3), (160, 6))
    ]

    results: dict[int, list[np.ndarray]] = {}
    pending: list[int] = []
    for frames in batches:
        if len(pending) >= PIPELINE_DEPTH:
            seq, boxes = detector.collect()
            assert seq == pending.pop(0)
            results[seq] = boxes
        pending.append(detector.submit(frames))
    while pending:
        seq, boxes = detector.collect()
        assert seq == pending.pop(0)
        results[seq] = boxes

    for index, frames in enumerate(batches):
        got = [float(one[0][4]) for one in results[index]]
        assert got == _expected_conf(runner, frames), f"第 {index} 批拿到別批的資料"


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_submitting_beyond_the_pipeline_depth_is_rejected(monkeypatch):
    """深度滿了還 submit 要當場擋下——再送一批會覆寫還沒被取回的那批的輸入緩衝。"""
    monkeypatch.setattr(settings.model, "classes", [FBODY_CLASS_ID])
    detector = _detector_with_runner(_CountingRunner(sleep_cycles=0))
    for _ in range(PIPELINE_DEPTH):
        detector.submit(_frames([1]))

    with pytest.raises(RuntimeError, match="在途批次"):
        detector.submit(_frames([1]))


@pytest.mark.skipif(_NO_GPU, reason="需要 CUDA 裝置")
def test_warmup_runs_one_real_batch_through_the_pipeline(monkeypatch):
    """warmup 走的是正式的 `submit`／`collect`，跑完不留在途批。

    留下在途批的話，第一批真影格會直接撞上深度上限；而 warmup 若改成繞過流水自己跑
    一次 forward，緩衝配置與定序的問題就要等到第一批真影格才浮現。
    """
    monkeypatch.setattr(settings.model, "classes", [FBODY_CLASS_ID])
    detector = _detector_with_runner(_CountingRunner(sleep_cycles=0))

    detector._warmup()

    assert detector.in_flight == 0
