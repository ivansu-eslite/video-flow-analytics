"""引擎載入前的四道檢查，以及執行期的張量形狀核對。

正式推論路徑收斂成 TensorRT FP16 引擎之後（ADR-011），這一層擋的全是**「跑起來完全
正常、輸出檔也完全正常，只有數值或成本不對」**的組合：載到 `.pt`（慢一個量級）、
載到別的模型（類別語義不同）、載到 FP32 引擎（吞吐掉回改動前）、退回 640×640
（像素量 1.67 倍）。這些都不會拋錯，所以只能靠這幾道檢查在載入當下擋下。

引擎自帶 metadata 與環境的比對在 test_engine_metadata.py。
"""

import json
import struct

import pytest

from video_analyze.models.config import settings
from video_analyze.services.detector import (
    YOLODetector,
    _check_infer_shape_of,
    _log_engine_metadata,
    _require_engine_file,
    _validate_classes,
    _validate_precision,
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
    """**這道 `is_file()` 前置檢查不可移除**，它擋的不只是「靜默下載官方權重」。

    ultralytics 的 `check_file` 在檔案不存在時有三種替代行為，三種都會讓「跑的不是
    你以為的那顆模型」而輸出檔完全正常：遞迴 glob 整個套件目錄找同名檔、把 `gs://`
    改寫成公開 HTTPS 下載、把沒有副檔名的值補成官方權重下載。載入引擎一樣會走
    `check_file`（`Model._load` 對非 `.pt` 一律呼叫它），所以改吃引擎之後這道檢查
    沒有變得多餘。

    斷言訊息內容而不只是例外型別：`check_file` 自己找不到檔時也拋 `FileNotFoundError`，
    只驗型別的話這道檢查被拿掉仍然全綠。
    """
    with pytest.raises(FileNotFoundError, match="glob") as excinfo:
        _require_engine_file(tmp_path / "missing_sm75.engine")

    assert "gs://" in str(excinfo.value)


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

    with pytest.raises(FileNotFoundError, match="glob"):
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
    """退回 640×640 要當場擋下。

    ultralytics 只在 `format == "pt"` 或 backend 的 `dynamic` 為真時才走 `auto`
    letterbox；換成靜態引擎就會把短邊填到 640，像素量 1.67 倍。**症狀只有「變慢」**，
    座標、欄位、列數全部正常——這正是 issue #108 消掉的成本靜悄悄回來的方式。

    這一項取代了 `_validate_imgsz`：dynamic 引擎不套用 metadata 的 `imgsz`，照抄那個
    檢查會得到一個看起來有在驗、其實驗不到的檢查。
    """
    with pytest.raises(ValueError, match="640"):
        _check_infer_shape_of((16, 3, 640, 640))
