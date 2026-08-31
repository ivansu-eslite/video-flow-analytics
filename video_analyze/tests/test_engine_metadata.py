"""引擎自帶 metadata 的讀取與驗證。

這幾支釘的是**「跑的不是你以為的那顆引擎」**這一類：引擎是二進位產物，拿錯一顆
（別張卡建的、別份權重建的、別版 TensorRT 建的）不會有任何症狀——載得起來、跑得出
結果、parquet 的欄位與列數全部正常，只有數值不對。所以每一項不符都要**各自**中止，
而且訊息要指出是哪一項：SM 不符要在目標卡上重建，TensorRT 版本不符要對齊映像檔，
權重 hash 不符是拿錯了引擎，三種處置完全不同。
"""

import json
import struct
from dataclasses import replace

import pytest

from video_analyze.services.engine_metadata import (
    VFA_METADATA_KEY,
    VFA_METADATA_SCHEMA,
    GpuEnvironment,
    _tensorrt_package,
    read_engine_metadata,
    validate_engine_metadata,
    validate_engine_precision,
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


def _metadata(args: dict | None = None, **vfa_overrides) -> dict:
    return {
        "names": {0: "head", 1: "vbody", 2: "fbody"},
        "batch": 16,
        "args": args
        if args is not None
        else {"half": True, "int8": False, "dynamic": True, "batch": 16},
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


def test_validate_only_warns_when_the_tensorrt_package_is_undetectable(capsys):
    """讀不到套件名時只記錄不擋。

    「測不出來」不等於「不相容」：TensorRT 若不是用 wheel 裝的（tarball／deb，或 slim
    映像檔把 `*.dist-info` 拿掉），runtime 完全正常卻讀不到名字。做成致命條件會讓整天
    的分析起不來，方向與驅動版本那條一樣。
    """
    environment = replace(_ENV, tensorrt_package=None)

    validate_engine_metadata(_metadata(), environment, _SHA)

    assert '"severity": "WARNING"' in capsys.readouterr().out


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


class _FakeDist:
    def __init__(self, name):
        self.metadata = {"Name": name}


def test_tensorrt_package_ignores_the_bindings_and_libs_subpackages(monkeypatch):
    """認的是主套件，不是同樣以 `tensorrt-cu` 開頭的兩個附屬套件。

    `tensorrt-cu12-bindings` 與 `tensorrt-cu12-libs` 跟主套件一起裝，而
    `importlib.metadata.distributions()` 的順序不保證——用前綴比對會隨機拿到附屬套件
    的名字（實測就拿到過 `tensorrt-cu12-libs`）。那個值本身不算錯，但它會讓「引擎的
    建置環境」這欄的內容隨機漂移，兩端各自漂到不同的值就變成假的不符。
    """
    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [
            _FakeDist("tensorrt_cu12_libs"),
            _FakeDist("tensorrt_cu12_bindings"),
            _FakeDist("tensorrt_cu12"),
        ],
    )

    assert _tensorrt_package() == "tensorrt-cu12"


def test_tensorrt_package_returns_none_when_not_installed(monkeypatch, capsys):
    """取不到只記 warning、回 None——這欄的缺漏會在比對時以「不符」被擋下，
    不需要在這裡多拋一次。"""
    monkeypatch.setattr("importlib.metadata.distributions", lambda: [_FakeDist("torch")])

    assert _tensorrt_package() is None
    assert '"severity": "WARNING"' in capsys.readouterr().out


def test_validate_precision_accepts_an_fp16_engine():
    validate_engine_precision(_metadata())


def test_validate_precision_rejects_an_fp32_engine():
    """精度驗的是 `metadata["args"]["half"]`，不是 I/O binding 的 dtype。

    FP16 引擎的 I/O binding 仍是 FP32（`TrtRunner` 那道 dtype 檢查驗的就是這件事）——
    拿 binding 的 dtype 當判準會**永遠**判定「不是 FP16」，於是這道檢查只能被拿掉或
    永遠失敗，兩種都等於沒有在驗精度。
    """
    args = {"half": False, "int8": False, "dynamic": True}

    with pytest.raises(ValueError, match="half"):
        validate_engine_precision(_metadata(args=args))


def test_validate_precision_rejects_an_int8_engine():
    """INT8 引擎可以同時把 FP16 flag 設成真——INT8 只是多開一個 flag。只驗 `half` 的
    舊版檢查對這種引擎完全照過：2026-08-30 的實測，一顆 INT8 引擎在不改本 repo 任何
    一行的情況下被正式路徑載入、跑完九路、產出格式完全正常的 parquet，只有數字變了。
    """
    args = {"half": True, "int8": True, "dynamic": True}

    with pytest.raises(ValueError, match="int8"):
        validate_engine_precision(_metadata(args=args))


def test_validate_precision_rejects_when_the_int8_flag_is_missing():
    """缺 `int8` 欄不是「沒開那個精度」，是這顆引擎的匯出路徑與預期不同，此時無法
    判定實際精度，一律擋下而不是靜默放行。"""
    args = {"half": True, "dynamic": True}

    with pytest.raises(ValueError, match="int8"):
        validate_engine_precision(_metadata(args=args))


def test_validate_precision_rejects_when_args_is_entirely_missing():
    """整段 `args` 都不存在時，`half`／`int8` 兩項一起以缺欄回報，不是無聲通過。"""
    metadata = {"names": {}, VFA_METADATA_KEY: _vfa()}

    with pytest.raises(ValueError, match="half"):
        validate_engine_precision(metadata)
