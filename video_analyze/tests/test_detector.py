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
from video_analyze.services.detector import (
    CONF_THRESHOLD,
    END2END_COLUMNS,
    MAX_DET,
    YOLODetector,
    _check_infer_shape_of,
    _check_infer_tensor,
    _log_engine_metadata,
    _require_engine_file,
    _validate_classes,
    _validate_dynamic,
    _validate_precision,
    postprocess_batch,
    preprocess_batch,
)
from video_analyze.services.engine_metadata import (
    VFA_METADATA_KEY,
    VFA_METADATA_SCHEMA,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH


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

    改用 `AutoBackend` 之後（ADR-013）拿掉這道檢查不會再讓 ultralytics 去找替代來源
    ——`_model_type` 只看副檔名——但 TensorRT backend 直接 `open()`，拋出來的
    `FileNotFoundError` 只有一個路徑字串，看不出「`model_path` 是 cwd 相對路徑、跑錯
    目錄了」這個實際上最常見的原因。

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


def test_validate_precision_accepts_an_fp16_engine():
    _validate_precision(_metadata())


def test_validate_precision_rejects_an_fp32_engine():
    """精度驗的是 `metadata["args"]["half"]`，不是 backend 的 `fp16` 屬性。

    FP16 引擎的 I/O binding 仍是 FP32，`AutoBackend` 對 FP16 引擎永遠回報
    `fp16 = False`——拿那個值當判準會**永遠**判定「不是 FP16」，於是這道檢查只能被
    拿掉或永遠失敗，兩種都等於沒有在驗精度。
    """
    with pytest.raises(ValueError, match="half"):
        _validate_precision(_metadata(args={"half": False, "dynamic": True}))


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


def test_infer_shape_check_accepts_the_reader_side_shape(capsys):
    """640×384 進、640×384 出：影格在讀取端就縮好了，前處理不該再加填充。"""
    _check_infer_shape_of((16, 3, INFER_HEIGHT, INFER_WIDTH))

    assert str(INFER_HEIGHT) in capsys.readouterr().out


def test_infer_shape_check_rejects_a_padded_square_shape():
    """被填成 640×640 要當場擋下。

    dynamic 引擎不套用 metadata 的 `imgsz`（`predictor.py` 只在 backend **不是**
    dynamic 時才把它抄進 `args.imgsz`），所以 `args.imgsz` 停在預設的 640，實際形狀
    由 `pre_transform` 的 `auto` letterbox 決定。`auto` 要
    `same_shapes and args.rect and (format == "pt" or dynamic)` 三者同時成立才保住
    640×384；任何一項日後翻掉，LetterBox 就會填到 640×640，像素量 1.67 倍。
    **症狀只有「變慢」**，座標、欄位、列數全部正常——正是 issue #108 消掉的成本靜悄悄
    回來的方式。

    這一項取代了 `_validate_imgsz`：那個檢查在 dynamic 引擎下驗的值不決定實際形狀。
    """
    with pytest.raises(ValueError, match="640"):
        _check_infer_shape_of((16, 3, 640, 640))


def test_validate_dynamic_accepts_a_dynamic_engine():
    _validate_dynamic(_metadata())


def test_validate_dynamic_rejects_a_static_engine():
    """靜態引擎要在載入時擋下，不能留給推論當下的 assert。

    它會通過其餘全部載入檢查，然後在第一個沒湊滿的批次上被 ultralytics 的
    `assert im.shape == s` 擋下——訊息只講「input size 不等於 max model size」，看不出
    原因是引擎綁死了批次。而湊不滿批是常態而非例外：T4（n1-standard-8）上實測一次跑完
    出現 16 種不同的批次大小，1 到 16 全都有。

    接手這件事的 `_check_infer_shape` 幫不上忙——它排在 `predict` **之後**，靜態引擎
    在 predict 裡就先崩了。
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


def test_preprocess_matches_the_ultralytics_reference_element_for_element():
    """自建前處理與 ultralytics 的路徑**逐值相同**，不是「差異不大」。

    每個元素都是 `uint8 → float32 → /255` 的獨立運算，兩邊都 round-to-nearest，所以
    這裡沒有容差可談——差一格就代表通道、軸或型別有一項寫錯，而那種錯誤跑起來完全
    正常，只是偵測結果不一樣。
    """
    frames = _synthetic_frames(3)

    got = preprocess_batch(frames, torch.device("cpu"))

    assert torch.equal(got, _ultralytics_preprocess_reference(frames))


def test_preprocess_hands_over_a_contiguous_float32_nchw_tensor():
    """dtype、佈局與軸順序三件事各自都能無聲地毀掉推論結果。

    float16 會被 FP32 binding 重新解讀；非 contiguous 會被 TensorRT 當成連續的讀
    （它只拿 `data_ptr()`）——`[..., [2, 1, 0]]` 再 `permute` 就是這種張量，逐值相同
    但佈局不同，所以這裡連 `is_contiguous()` 一起釘。
    """
    frames = _synthetic_frames(2)

    got = preprocess_batch(frames, torch.device("cpu"))

    assert got.dtype == torch.float32
    assert got.is_contiguous()
    assert tuple(got.shape) == (2, 3, INFER_HEIGHT, INFER_WIDTH)
    # 通道真的翻過：RGB 的第 0 個通道要等於 BGR 影格的第 2 個通道
    np.testing.assert_allclose(got[0, 0].numpy(), frames[0][:, :, 2] / 255)


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
    """把 CPU 位址交給 `execute_v2` 是未定義行為。

    這一項排在最後，前三項才能在沒有 GPU 的機器上也測得到。
    """
    with pytest.raises(RuntimeError, match="CUDA"):
        _check_infer_tensor(_infer_tensor())


class _FakeBackend:
    """只回傳預先給好的 forward 輸出，用來驗載入期的 end2end 檢查。"""

    def __init__(self, output):
        self.output = output

    def forward(self, im: torch.Tensor):
        return self.output


def _detector_with_backend(output) -> YOLODetector:
    """繞過 `__init__` 組一個只有 warmup 需要的欄位的偵測器。

    走完整的 `__init__` 需要一顆真的引擎與一張 GPU，而這裡要驗的是**收到不是
    end2end 的輸出時會不會擋下**——那與引擎怎麼載入無關。
    """
    detector = YOLODetector.__new__(YOLODetector)
    detector.max_batch = 2
    detector.device = torch.device("cpu")
    detector.model = _FakeBackend(output)
    return detector


def test_warmup_rejects_an_engine_without_builtin_nms():
    """引擎不是 end2end 要在載入期擋下，不能留給第一批推論。

    沒有內建 NMS 的引擎吐的是 `(B, 4 + nc, num_anchors)` 的原始張量，而
    `postprocess_batch` 那三行只讀第 4、5 欄——照樣跑得完，只是把某個類別分數當成
    conf、另一個當成 cls，得到一堆座標是 xywh 的框，而列數、欄位、格式全部正常。

    判準用實際跑出來的輸出形狀而不是 metadata 的 `end2end` 欄位：後者改了不會改變
    引擎，而這裡要驗的正是引擎本身。
    """
    detector = _detector_with_backend(torch.zeros((2, 7, 5040)))

    with pytest.raises(ValueError, match="end2end"):
        detector._warmup_and_require_end2end()


def test_warmup_accepts_an_end2end_engine():
    detector = _detector_with_backend(torch.zeros((2, MAX_DET, END2END_COLUMNS)))

    detector._warmup_and_require_end2end()  # 不應拋例外
