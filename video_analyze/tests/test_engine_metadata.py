"""引擎自帶 metadata 的讀取與驗證。

這幾支釘的是**「跑的不是你以為的那顆引擎」**這一類：引擎是二進位產物，拿錯一顆
（別張卡建的、別份權重建的、別版 TensorRT 建的）不會有任何症狀——載得起來、跑得出
結果、parquet 的欄位與列數全部正常，只有數值不對。所以每一項不符都要**各自**中止，
而且訊息要指出是哪一項：SM 不符要在目標卡上重建，TensorRT 版本不符要對齊映像檔，
權重 hash 不符是拿錯了引擎，三種處置完全不同。
"""

import json
import struct

import pytest

from video_analyze.services.engine_metadata import (
    VFA_METADATA_KEY,
    VFA_METADATA_SCHEMA,
    GpuEnvironment,
    read_engine_metadata,
    validate_engine_metadata,
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


def _vfa(**overrides) -> dict:
    block = {
        "schema": VFA_METADATA_SCHEMA,
        "source_weights": {"name": "baseline.pt", "sha256": _SHA},
        "compute_capability": _ENV.compute_capability,
        "gpu_name": _ENV.gpu_name,
        "tensorrt": _ENV.tensorrt,
        "tensorrt_package": _ENV.tensorrt_package,
        "torch_cuda_major": _ENV.torch_cuda_major,
        "driver": _ENV.driver,
        "train": {"ultralytics": "8.4.90", "date": "2026-07-14"},
    }
    block.update(overrides)
    return block


def _metadata(**vfa_overrides) -> dict:
    return {
        "names": {0: "head", 1: "vbody", 2: "fbody"},
        "batch": 16,
        "args": {"half": True, "dynamic": True, "batch": 16},
        VFA_METADATA_KEY: _vfa(**vfa_overrides),
    }


def _write_engine(path, metadata: dict | None, body: bytes = b"\x00engine-bytes"):
    """照 ultralytics 的格式寫一顆假引擎：4 bytes 長度 ＋ JSON ＋ 引擎本體。"""
    blob = b""
    if metadata is not None:
        meta = json.dumps(metadata).encode("utf-8")
        blob = struct.pack("<i", len(meta)) + meta
    path.write_bytes(blob + body)
    return path


def test_read_engine_metadata_round_trips_and_restores_int_class_ids(tmp_path):
    """`names` 的 key 要還原成 int。

    JSON 沒有整數 key，`json.dumps({0: "head"})` 寫出的是 `{"0": "head"}`。維持字串的話
    `_validate_classes` 拿 int 去比對會**全數落空**，而且是往「查無此 id」的方向落空——
    剛好會被誤判成 classes 設定寫錯，而不是被看出是 key 型別的問題。
    """
    engine = _write_engine(tmp_path / "e.engine", _metadata())

    metadata = read_engine_metadata(engine)

    assert metadata["names"] == {0: "head", 1: "vbody", 2: "fbody"}
    assert metadata[VFA_METADATA_KEY]["tensorrt"] == "10.13.3.9"


def test_read_engine_metadata_rejects_a_bare_tensorrt_engine(tmp_path):
    """裸引擎（沒有 ultralytics 檔頭）要當場擋下，而不是報 JSON 解析錯。"""
    engine = tmp_path / "bare.engine"
    engine.write_bytes(b"\xff\xff\xff\x7f" + b"\x00" * 64)

    with pytest.raises(ValueError, match="metadata"):
        read_engine_metadata(engine)


def test_validate_passes_when_everything_matches():
    validate_engine_metadata(_metadata(), _ENV, _SHA)


def test_validate_rejects_a_different_compute_capability():
    """別張卡建的引擎。TensorRT 沒有指定 target SM 的 API，只能在目標卡上重建。"""
    metadata = _metadata(compute_capability="12.0", gpu_name="NVIDIA GeForce RTX 5090")

    with pytest.raises(ValueError, match="compute capability"):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_rejects_a_different_tensorrt_version():
    """別版 TensorRT 建的引擎：deserialize 不保證成功，成功了也不保證數值一致。"""
    metadata = _metadata(tensorrt="10.16.0")

    with pytest.raises(ValueError, match="TensorRT"):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_rejects_a_different_source_weight():
    """別份權重建的引擎：類別 id 可能剛好都存在而語義不同，classes 過濾不會有訊號。"""
    metadata = _metadata(
        source_weights={"name": "other.pt", "sha256": "b" * 64},
    )

    with pytest.raises(ValueError, match="source_weights_sha256"):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_rejects_a_different_tensorrt_wheel_variant():
    """`tensorrt-cu12` 與 `tensorrt-cu13` 是不同套件，runtime 不可互換。

    比的是**實際安裝的套件名**，不是 `torch.version.cuda`。兩者可以不一樣——本機現況
    就是 torch `2.12.1+cu130` 搭 `tensorrt-cu12`——拿 torch 的 CUDA 建置版本當變體的
    代理，既擋不到真正的 cu12／cu13 對調，又會在 torch 換版時誤擋既有引擎。
    """
    metadata = _metadata(tensorrt_package="tensorrt-cu13")

    with pytest.raises(ValueError, match="tensorrt-cu13"):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_only_warns_when_the_torch_cuda_build_differs(capsys):
    """torch 的 CUDA 建置版本不在 TensorRT 對引擎的約束裡，只記錄不擋。"""
    validate_engine_metadata(_metadata(torch_cuda_major="13"), _ENV, _SHA)

    out = capsys.readouterr().out
    assert '"severity": "WARNING"' in out
    assert "torch" in out


def test_validate_rejects_an_engine_without_the_vfa_block():
    """沒有注入欄位的引擎（例如手動 `yolo export` 出來的）無法確認身分，一律擋下。"""
    metadata = _metadata()
    del metadata[VFA_METADATA_KEY]

    with pytest.raises(ValueError, match=VFA_METADATA_KEY):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_rejects_an_older_metadata_schema():
    """欄位語義改版後，舊引擎要被擋下而不是以缺欄位的方式靜默通過。"""
    metadata = _metadata(schema=VFA_METADATA_SCHEMA - 1)

    with pytest.raises(ValueError, match="schema"):
        validate_engine_metadata(metadata, _ENV, _SHA)


def test_validate_only_warns_when_the_driver_differs(capsys):
    """驅動版本**只記錄不擋**。

    引擎的硬約束來自 TensorRT 本身（SM 與 TRT 版本），驅動不在其中；而建置機
    （GCE T4 VM）與正式節點（Vertex CustomJob 的容器）本來就不保證同一份映像檔。
    做成致命條件會堵死唯一可行的建置路徑。
    """
    metadata = _metadata(driver="535.183.01")

    validate_engine_metadata(metadata, _ENV, _SHA)

    out = capsys.readouterr().out
    assert '"severity": "WARNING"' in out
    assert "535.183.01" in out


def test_validate_warns_when_the_source_weight_is_not_pinned(capsys):
    """沒釘 hash 就只能記錄。**這件事要在 log 上看得見**，否則「沒在驗」與
    「驗過了」在輸出上完全一樣。"""
    validate_engine_metadata(_metadata(), _ENV, expected_weights_sha256="")

    out = capsys.readouterr().out
    assert '"severity": "WARNING"' in out
    assert "source_weights_sha256" in out
