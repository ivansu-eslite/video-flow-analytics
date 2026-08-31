"""`tools/build_engine.py::verify_engine` 對精度旗標的把關。

C18（PR #154）把精度檢查改成宣告式、建置期與載入期共用同一份宣告
（`engine_metadata.EXPECTED_PRECISION_ARGS`／`validate_engine_precision`），但只有
載入端那支呼叫被測到（test_engine_metadata.py／test_detector.py）——`verify_engine`
裡的 `validate_engine_precision(metadata)` 這行被刪掉不會有任何測試紅，因為 repo 裡
沒有這支檔案。這裡補上 `verify_engine` 本身的把關：INT8 引擎（`half=True` 且
`int8=True`）與缺 `int8` 欄的 metadata 都要在碰 GPU（`YOLO(...)`）之前被擋下。

精度判準本身的完整 case（FP16 通過／FP32 擋下／INT8 擋下／缺欄擋下）在
test_engine_metadata.py，這裡只釘「`verify_engine` 有呼叫到它、且排在 GPU 探測之前」。
"""

import json
import struct

import build_engine
import pytest

from video_analyze.services.engine_metadata import (
    VFA_METADATA_KEY,
    VFA_METADATA_SCHEMA,
    GpuEnvironment,
)

_ENV = GpuEnvironment(
    compute_capability="7.5",
    gpu_name="Tesla T4",
    tensorrt="10.13.3.9",
    tensorrt_package="tensorrt-cu12",
    torch_cuda_major="12",
    driver="550.54.15",
)
_SHA = "a" * 64


def _metadata(args: dict) -> dict:
    return {
        "names": {0: "head", 1: "vbody", 2: "fbody"},
        "batch": 16,
        "args": args,
        VFA_METADATA_KEY: {
            "schema": VFA_METADATA_SCHEMA,
            "source_weights": {"name": "baseline.pt", "sha256": _SHA},
            "compute_capability": _ENV.compute_capability,
            "gpu_name": _ENV.gpu_name,
            "tensorrt": _ENV.tensorrt,
            "tensorrt_package": _ENV.tensorrt_package,
            "torch_cuda_major": _ENV.torch_cuda_major,
            "driver": _ENV.driver,
            "train": {},
        },
    }


def _write_engine(path, metadata: dict):
    """照 ultralytics 的格式寫一顆假引擎：4 bytes 長度 ＋ JSON ＋ 引擎本體。

    本體是無法被真正解出的假 bytes——測試預期在讀到這裡之前就被精度檢查擋下；
    如果 `YOLO` 沒被монkeypatch 掉還跑到這裡，會是明確的載入失敗而不是靜默通過。
    """
    meta = json.dumps(metadata).encode("utf-8")
    path.write_bytes(struct.pack("<i", len(meta)) + meta + b"\x00not-a-real-engine")
    return path


def _reject_gpu_probe(*args, **kwargs):
    pytest.fail("precision 檢查應該在碰 YOLO(engine_path) 之前就中止")


def test_verify_engine_rejects_an_int8_engine(monkeypatch, tmp_path):
    """INT8 引擎可以同時把 FP16 flag 設成真——2026-08-30 實測擋不住的正是這種
    metadata 形狀。`verify_engine` 要在碰 GPU 之前擋下，不能靠載入端事後補一刀。"""
    monkeypatch.setattr(build_engine, "current_gpu_environment", lambda: _ENV)
    monkeypatch.setattr(build_engine, "YOLO", _reject_gpu_probe)
    engine = _write_engine(
        tmp_path / "fake_sm75.engine",
        _metadata({"half": True, "int8": True, "dynamic": True, "batch": 16}),
    )

    with pytest.raises(ValueError, match="int8"):
        build_engine.verify_engine(engine, batch=16, expected_sha256=_SHA)


def test_verify_engine_rejects_metadata_missing_the_int8_flag(monkeypatch, tmp_path):
    """缺 `int8` 欄不是「沒開那個精度」，是匯出路徑與預期不同，一律擋下而不是放行。"""
    monkeypatch.setattr(build_engine, "current_gpu_environment", lambda: _ENV)
    monkeypatch.setattr(build_engine, "YOLO", _reject_gpu_probe)
    engine = _write_engine(
        tmp_path / "fake_sm75.engine",
        _metadata({"half": True, "dynamic": True, "batch": 16}),
    )

    with pytest.raises(ValueError, match="int8"):
        build_engine.verify_engine(engine, batch=16, expected_sha256=_SHA)
