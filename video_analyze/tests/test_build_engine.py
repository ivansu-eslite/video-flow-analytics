"""`tools/build_engine.py` 的純資料部分：optimization profile 的三個界，以及
`verify_engine` 對精度旗標的把關。

C18（PR #154）把精度檢查改成宣告式、建置期與載入期共用同一份宣告
（`engine_metadata.EXPECTED_PRECISION_ARGS`／`validate_engine_precision`），但只有
載入端那支呼叫被測到（test_engine_metadata.py／test_detector.py）——`verify_engine`
裡的 `validate_engine_precision(metadata)` 這行被刪掉不會有任何測試紅，因為 repo 裡
沒有這支檔案。這裡補上 `verify_engine` 本身的把關：INT8 引擎（`half=True` 且
`int8=True`）與缺 `int8` 欄的 metadata 都要在碰 GPU（`YOLO(...)`）之前被擋下。

精度判準本身的完整 case（FP16 通過／FP32 擋下／INT8 擋下／缺欄擋下）在
test_engine_metadata.py，這裡只釘「`verify_engine` 有呼叫到它、且排在 GPU 探測之前」。

`profile_shapes` 的部分釘的是 C21／ADR-015 的那次收窄：空間維三個界都是推論尺寸、
batch 維是 `1`–`--batch`。判準本身（含舊的寬 profile 引擎要被擋下）在
test_trt_runner.py 的 `check_profile_shapes`，這裡只釘建置端產生的值餵得過它。

`check_train_imgsz` 釘的是訓練 `imgsz` 與 `INFER_WIDTH`／`INFER_HEIGHT` 的比對：一致放行、
不一致拋錯、權重沒有 `train_args.imgsz`（各種缺法）只印警告放行。用假 `ckpt` 屬性即可，
不必碰真的 `.pt` 權重或 GPU。
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
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.trt_runner import check_profile_shapes

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
    """一份完整的引擎檔頭。**必填鍵一個都不能少**：`verify_engine` 的第一道
    `validate_engine_header` 會擋下缺鍵的檔頭，而這幾支測試要驗的是後面那道精度檢查。
    """
    return {
        "stride": 32,
        "task": "detect",
        "imgsz": [INFER_HEIGHT, INFER_WIDTH],
        "channels": 3,
        "end2end": True,
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
    如果 `YOLO` 沒被 monkeypatch 掉還跑到這裡，會是明確的載入失敗而不是靜默通過。
    """
    meta = json.dumps(metadata).encode("utf-8")
    path.write_bytes(struct.pack("<i", len(meta)) + meta + b"\x00not-a-real-engine")
    return path


def _reject_gpu_probe(*args, **kwargs):
    pytest.fail("precision 檢查應該在碰 YOLO(engine_path) 之前就中止")


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            {"half": True, "int8": True, "dynamic": True, "batch": 16}, id="int8_true"
        ),
        pytest.param({"half": True, "dynamic": True, "batch": 16}, id="int8_missing"),
    ],
)
def test_verify_engine_rejects_a_non_fp16_engine(args, monkeypatch, tmp_path):
    """INT8 引擎可以同時把 FP16 flag 設成真——2026-08-30 實測擋不住的正是這種 metadata
    形狀。缺 `int8` 欄同理擋下：不是「沒開那個精度」，是匯出路徑與預期不同，無法判定
    實際精度。兩種情況都要在碰 GPU（`YOLO(engine_path)`）之前被 `verify_engine` 擋下，
    不能靠載入端事後補一刀。"""
    monkeypatch.setattr(build_engine, "current_gpu_environment", lambda: _ENV)
    monkeypatch.setattr(build_engine, "YOLO", _reject_gpu_probe)
    engine = _write_engine(tmp_path / "fake_sm75.engine", _metadata(args))

    with pytest.raises(ValueError, match="int8"):
        build_engine.verify_engine(engine, batch=16, expected_sha256=_SHA)


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_profile_shapes_pin_the_spatial_dims_and_leave_batch_dynamic(batch):
    """空間維三個界都是推論尺寸（收窄的全部內容），batch 維仍是 1–`--batch`。

    收窄之前 ultralytics 給的是 `min=(1,3,32,32)／max=(16,3,768,1280)`：下界是一個
    永遠跑不起來的形狀（32×32 不到 300 個 anchor，end2end head 的 `TopK(K=300)` 在
    建置期就明著警告），上界是 `workspace` 當倍數的副產物。兩個界都在限制 tactic 的
    選擇範圍，而執行期的高寬只有一種。
    """
    minimum, opt, maximum = build_engine.profile_shapes(batch)

    assert minimum == (1, 3, INFER_HEIGHT, INFER_WIDTH)
    assert opt == (batch, 3, INFER_HEIGHT, INFER_WIDTH)
    assert maximum == (batch, 3, INFER_HEIGHT, INFER_WIDTH)


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_profile_shapes_pass_the_load_time_check(batch):
    """建置端產生的界，載入端要收得下。

    兩邊各寫一份判準的話，收窄過的引擎會在自己的載入檢查上被擋下，而症狀是「剛建好的
    引擎跑不起來」——所以這裡直接把 `profile_shapes` 的輸出餵給
    `trt_runner.check_profile_shapes`。
    """
    check_profile_shapes(*build_engine.profile_shapes(batch))


def test_profile_shapes_rejects_a_batch_below_one():
    """`--batch` 小於 1 在組出 profile 之前就擋下——那個 profile 任何一批都送不進去。"""
    with pytest.raises(ValueError, match="必須 >= 1"):
        build_engine.profile_shapes(0)


class _FakeModel:
    """`check_train_imgsz` 只讀 `model.ckpt`，不必是真的 `ultralytics.YOLO` 實例。"""

    def __init__(self, ckpt):
        self.ckpt = ckpt


def test_check_train_imgsz_passes_when_matching():
    model = _FakeModel({"train_args": {"imgsz": max(INFER_WIDTH, INFER_HEIGHT)}})
    build_engine.check_train_imgsz(model)  # 不拋錯


def test_check_train_imgsz_rejects_a_mismatch():
    model = _FakeModel({"train_args": {"imgsz": 320}})
    with pytest.raises(SystemExit, match="320"):
        build_engine.check_train_imgsz(model)


@pytest.mark.parametrize(
    "ckpt",
    [
        pytest.param(None, id="no_ckpt"),
        pytest.param({}, id="no_train_args"),
        pytest.param({"train_args": {}}, id="no_imgsz"),
        pytest.param({"train_args": None}, id="train_args_not_a_dict"),
    ],
)
def test_check_train_imgsz_warns_and_passes_when_train_args_missing(ckpt, capsys):
    model = _FakeModel(ckpt)
    build_engine.check_train_imgsz(model)  # 不拋錯，只印警告
    assert "警告" in capsys.readouterr().out


def test_verify_engine_rejects_a_header_missing_the_end2end_flag(monkeypatch, tmp_path):
    """檔頭缺 `end2end` 要在碰 GPU 之前擋下，理由與精度那兩支同型。

    缺鍵**不會**讓載入崩：`apply_metadata` 走 `metadata.get(...)`、這幾個鍵全部有預設
    值。`end2end` 缺漏時算出 `False`，predictor 會對 `(B, 300, 6)` 的 end2end 輸出再跑
    一次完整 NMS，把類別分數當成 conf 與 cls——建置期驗收與 `compare_backend.py` 整批
    比錯而沒有任何例外。所以 `verify_engine` 的第一道就是這個檔頭檢查。
    """
    monkeypatch.setattr(build_engine, "current_gpu_environment", lambda: _ENV)
    monkeypatch.setattr(build_engine, "YOLO", _reject_gpu_probe)
    metadata = _metadata({"half": True, "int8": False, "dynamic": True, "batch": 16})
    del metadata["end2end"]
    engine = _write_engine(tmp_path / "fake_sm75.engine", metadata)

    with pytest.raises(ValueError, match="end2end"):
        build_engine.verify_engine(engine, batch=16, expected_sha256=_SHA)
