#!/usr/bin/env python3
"""從 `.pt` 權重建出正式推論用的 TensorRT FP16 引擎，比對沒過就不產出。

**同一支工具、兩台機器、兩顆引擎**：TensorRT 引擎不可跨架構重用——`IBuilderConfig`
沒有指定 target SM 的 API，`HardwareCompatibilityLevel` 只有 `NONE`／`AMPERE_PLUS`／
`SAME_COMPUTE_CAPABILITY`，而 `AMPERE_PLUS` 不含 Turing。所以正式引擎（T4／sm75）只能
在 T4 上建，開發機（RTX 5090／sm120）建的那顆只能給地端用、不可上線。引擎檔名把 SM
編進去（`..._sm75.engine`）就是為了讓兩顆不可能被混用——下游 argus 的 promotion 用
`Path(source_model_uri).stem` 當 `model_version`，兩顆同名會直接撞在一起。

**引擎由本工具自己建，不走 ultralytics 的引擎匯出**（ADR-015）：ONNX 中繼檔仍由
ultralytics 匯出，之後的 builder／network／optimization profile／config 都在
`build_serialized_engine` 裡，檔頭由 `services/engine_metadata.py` 寫。接管的理由只有
一個——optimization profile 的空間維。ultralytics 把它訂成
`min=(1,3,32,32)／opt=(16,3,384,640)／max=(16,3,768,1280)`，兩個界都不是誰選的：上界是
`workspace`（未指定即 2）當倍數的副產物，下界是一個永遠跑不起來的形狀（32×32 不到 300
個 anchor，end2end head 的 `TopK(K=300)` 在建置期就明著警告）。而執行期的高寬只有一種
（`trt_runner._check_infer_shape_of` 對任何非 384×640 fail loud），所以把三個界全部釘在
384×640 不減少任何實際可用的形狀，換到的是 T4 上 forward 快 2.3%～6.1%、每個 execution
context 的裝置記憶體從 1,329 MB 降到 332 MB。

三個刻意的建置參數：

- **`dynamic=True`**。不是為了彈性：靜態引擎的 `TensorRTBackend.forward` 會 assert
  輸入形狀與綁定完全相同，而**湊不滿批是常態而非例外**——T4（n1-standard-8，與正式
  節點同機型）上實測一次跑完出現了 16 種不同的批次大小，1 到 16 全都有，8 核的解碼
  餵不滿批。靜態引擎會在第一個不滿批就崩。
  代價是形狀不再被 metadata 釘住：`predictor.py` 的 `setup_model` 只在 backend
  **不是** dynamic 時才把 metadata 的 `imgsz` 抄進 `args.imgsz`，所以 dynamic 引擎的
  實際形狀改由 `pre_transform` 的 `auto` letterbox 決定（見下一條與 `detector.py` 的
  `_check_infer_shape`）。
- **`imgsz=(INFER_HEIGHT, INFER_WIDTH)`**。形狀在建置期固定成 optimization profile 的
  **三個界**（min／opt／max，見 `profile_shapes`）。載入端**不驗** metadata 的
  `imgsz`——dynamic 引擎不套用它（上一條），照抄 `_validate_imgsz` 會得到一個看起來有
  在驗、其實驗不到的檢查；驗的是引擎自己宣告的 profile（`trt_runner.check_profile_shapes`）
  與逐批核對**實際進入推論的 (H, W)**（見 `detector.py` 的 `_check_infer_shape`）。
- **`batch=<最大批次>`**。dynamic 引擎的 batch 維度上限就是這個值（optimization
  profile 的 max shape 取 `shape[0]`），所以它與 `config.toml` 的 `[model] batch`
  要一起定：兩者同尺度（設定值就是實際的單次推理批次），這裡要容得下設定值，
  超過引擎上限會在 `forward` 的 assert 當場失敗。

用法（**在 repo 根目錄執行**，用 `--package` 不用 `--directory`：`--directory` 會把
cwd 切進套件資料夾，而 `[model].model_path` 與 `bucket_dir` 都是 cwd 相對路徑）：

    uv run --package video_analyze python video_analyze/tools/build_engine.py \
        --weights 20260714-153811_yolo26m_baseline.pt \
        --output-dir . \
        --batch 16 \
        --bucket bucket_20260801_small
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# `compare_backend` 與本檔同目錄。tools/ 刻意不是 package（它不隨 wheel 出貨，見
# ADR-011），直接執行腳本時 sys.path[0] 就是 tools/，所以這樣 import 得到。
from compare_backend import (
    DEFAULT_MAX_FOOT_DEV_P99_PX,
    check_report,
    compare_backends,
    print_summary,
)
from ultralytics import YOLO

from video_analyze.services.engine_metadata import (
    build_engine_metadata,
    build_vfa_metadata,
    current_gpu_environment,
    read_engine_metadata,
    sha256_of,
    validate_engine_header,
    validate_engine_metadata,
    validate_engine_precision,
    write_engine_with_metadata,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.trt_runner import TrtRunner

# 比對還沒跑完之前，產物先掛這個尾綴。**不用 `.tmp`**：那個尾綴在本 repo 已經是
# `TrackingResultCollector` 的暫存檔語義（issue #113 正在處理它的殘檔），借用會讓兩件
# 不同的事共用一個名字。留在磁碟上的 `.unverified` 一眼就知道是「建出來但沒驗過」。
_UNVERIFIED_SUFFIX = ".unverified"


def read_train_info(model: YOLO) -> dict:
    """從 `.pt` 的 ckpt 抄出訓練追溯資訊，準備注入引擎 metadata。

    引擎路徑下 `model.ckpt is None`，訓練版本／日期／指標整組取不到——這些欄位原本由
    `detector._log_model_metadata` 從 ckpt 讀出來記進 log，是「這批結果是哪個訓練版本
    跑的」唯一的線索。在建置期抄一份進 metadata，載入端才補得回來。

    訓練機器上的絕對路徑只留檔名，避免把訓練環境路徑寫進會被散布的產物。

    Args:
        model: 已載入 `.pt` 的 `ultralytics.YOLO` 實例。

    Returns:
        可 `json.dumps` 的 dict；ckpt 缺欄位時對應的值為 `None`。
    """
    ckpt = getattr(model, "ckpt", None)
    if not isinstance(ckpt, dict):
        print("[警告] 權重沒有 ckpt metadata，訓練追溯資訊會是空的。")
        return {}
    train_args = ckpt.get("train_args")
    train_args = train_args if isinstance(train_args, dict) else {}

    def _basename(value):
        if not isinstance(value, str) or not value:
            return value
        return Path(value).name

    return {
        "base_model": _basename(train_args.get("model")),
        "dataset": _basename(train_args.get("data")),
        "ultralytics": ckpt.get("version"),
        "date": ckpt.get("date"),
        "metrics": ckpt.get("train_metrics"),
    }


def check_train_imgsz(model: YOLO) -> None:
    """訓練用的 `imgsz` 與 `INFER_WIDTH`／`INFER_HEIGHT` 之間原本沒有任何自動檢查：
    兩者不一致代表推論解析度偏離訓練解析度，模型要在沒見過的尺度上推論。

    權重裡沒有 `train_args.imgsz`（外部來源、舊權重）時印警告放行，不擋建置。

    Args:
        model: 已載入 `.pt` 的 `ultralytics.YOLO` 實例。

    Raises:
        SystemExit: `train_args.imgsz` 與 `max(INFER_WIDTH, INFER_HEIGHT)` 不一致。
    """
    ckpt = getattr(model, "ckpt", None)
    train_args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
    train_imgsz = train_args.get("imgsz") if isinstance(train_args, dict) else None
    if train_imgsz is None:
        print("[警告] 權重沒有 train_args.imgsz，略過訓練／推論解析度比對。")
        return
    infer_imgsz = max(INFER_WIDTH, INFER_HEIGHT)
    if train_imgsz != infer_imgsz:
        raise SystemExit(
            f"訓練 imgsz（{train_imgsz}）與 max(INFER_WIDTH, INFER_HEIGHT)"
            f"（{infer_imgsz}）不一致，推論解析度會偏離訓練解析度。"
        )


def engine_filename(weights: Path, compute_capability: str) -> str:
    """引擎檔名：來源權重的 stem ＋ `_sm<SM>`。

    Args:
        weights: 來源 `.pt`。
        compute_capability: 建置機的 compute capability（如 `"7.5"`）。

    Returns:
        如 `20260714-153811_yolo26m_baseline_sm75.engine`。
    """
    return f"{weights.stem}_sm{compute_capability.replace('.', '')}.engine"


def profile_shapes(batch: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """optimization profile 的 min／opt／max（`(N, C, H, W)`）。

    空間維三個界一律 `(3, INFER_HEIGHT, INFER_WIDTH)`；batch 維是 `1`–`batch`，opt 取
    上界。判準與載入端共用 `trt_runner.check_profile_shapes`，那裡也寫著為什麼下界
    必須是 1（湊不滿批是常態）。

    Args:
        batch: 引擎綁的最大批次。

    Returns:
        `(min, opt, max)` 三個 shape。

    Raises:
        ValueError: `batch` 小於 1。
    """
    if batch < 1:
        raise ValueError(f"--batch 是 {batch}，必須 >= 1。")
    spatial = (3, INFER_HEIGHT, INFER_WIDTH)
    return ((1, *spatial), (batch, *spatial), (batch, *spatial))


def export_onnx(weights: Path, batch: int, workdir: Path) -> tuple[Path, dict]:
    """用 ultralytics 匯出 ONNX 中繼檔，並攔下它為這顆模型組好的 metadata。

    **`half=False` 是關鍵。** `format="engine"` 下 `half=True` 只作用於 builder flag
    （`exporter.py` 只對 `fmt in {onnx, torchscript}` 呼叫 `model.half()`），所以
    ultralytics 的中繼 ONNX 本來就是 FP32；改走 `format="onnx"` 之後同一個參數會真的
    把模型 `.half()`，得到 FP16 輸入 binding 的另一種引擎（實測比 FP32 binding 慢）。
    引擎的 FP16 由 `build_serialized_engine` 的 builder flag 給。

    **metadata 攔下來而不自己重推。** 走 `on_export_start` callback：公開的 `export()`
    沒有任意欄位入口，而 `Exporter.__call__` 是先組完 `self.metadata` 才
    `run_callbacks("on_export_start")`。`stride`／`task`／`names`／`channels`／
    `end2end` 這幾個模型衍生欄位算錯的症狀全是靜默的（見
    `engine_metadata.validate_engine_header`），照抄 ultralytics 的結果比自己重推安全。

    **匯出在暫存目錄裡做，權重原地不動。** ultralytics 把產物寫在來源權重旁邊
    （`self.file.with_suffix(...)`），權重放在共用的 artifacts 目錄是常態，那樣做會
    **無聲蓋掉／刪掉同名的既有檔案**——實測在建置機上就是這樣輾掉了兩份先前效能量測
    留下的產物。先把權重複製進暫存目錄再匯出，ultralytics 就只動得到那份副本。

    Args:
        weights: 來源 `.pt`。
        batch: 引擎的最大批次（ONNX 的 dynamic axes 用它當 dummy 輸入的 batch）。
        workdir: 暫存目錄；權重副本與 ONNX 都落在這裡。

    Returns:
        `(ONNX 檔路徑, exporter.metadata 的內容)`。

    Raises:
        RuntimeError: callback 沒被呼叫到（ultralytics 改了 `on_export_start` 的時點），
            此時檔頭會少掉全部模型衍生欄位。
    """
    staged_weights = workdir / weights.name
    shutil.copy2(weights, staged_weights)
    model = YOLO(str(staged_weights))

    captured: dict = {}

    def _capture(exporter) -> None:
        captured.update(exporter.metadata)

    model.add_callback("on_export_start", _capture)
    exported = Path(
        model.export(
            format="onnx",
            half=False,
            dynamic=True,
            batch=batch,
            imgsz=(INFER_HEIGHT, INFER_WIDTH),
            simplify=True,
            device=0,
            verbose=False,
        )
    )
    if not captured:
        raise RuntimeError(
            "ONNX 匯出沒有觸發 on_export_start，拿不到 ultralytics 組好的 metadata。"
            "檔頭會少掉 stride／task／names／channels／end2end，而那些欄位缺漏不會讓"
            "載入崩、只會被預設值靜默補上。"
        )
    return exported, captured


def build_serialized_engine(onnx_path: Path, batch: int) -> bytes:
    """從 ONNX 建出 FP16 引擎的序列化位元組（不含檔頭）。

    刻意複製 ultralytics `utils/export/engine.py::onnx2engine` 在本工具的呼叫下實際
    生效的設定，只改 optimization profile 與 `profiling_verbosity` 兩項（另外 builder
    logger 的 severity 從 ultralytics 的 `INFO` 降到 `WARNING`，那一項不影響引擎）。
    三處值得寫下來：

    - **不設 `set_memory_pool_limit`。** ultralytics 的 `workspace` 預設 `None` →
      `workspace_bytes = 0` → 整段略過，TensorRT 的預設上限就是整張卡可用的記憶體。
      這不是疏漏而是收益的來源：workspace 上限同時是 **tactic 的篩選條件**，設一個
      保守常數會排除吃 workspace 的 tactic，而「解鎖 tactic」正是收窄 profile 量到
      增益的機制。相位 1 在 T4 上量的那批也是不設上限建的，設了就不是被量到的東西。
    - **`profiling_verbosity` 取 `DETAILED`**（ultralytics 在 FP16 路徑上不動它，預設
      是 `LAYER_NAMES_ONLY`）。與相位 1 的量測組態一致，並讓日後的逐層 profiling 拿得
      到層名以外的資訊；代價是引擎檔略大，不影響 tactic 選擇。

    Args:
        onnx_path: `export_onnx` 產出的中繼檔。
        batch: 引擎綁的最大批次。

    Returns:
        序列化的引擎位元組。

    Raises:
        RuntimeError: ONNX parse 失敗、網路不是單一輸入，或 builder 建不出引擎。
            **前兩者 TensorRT 都只回 `False`／`None` 而不拋錯**，不自己擋的話 T4 上
            一次失敗只會得到一句指不出原因的訊息。
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n  - ".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse 失敗（{onnx_path}）：\n  - {errors}")
    if network.num_inputs != 1:
        raise RuntimeError(
            f"ONNX 的輸入是 {network.num_inputs} 個，本工具只設一個 optimization "
            "profile 形狀，多輸入的網路會有 binding 停在未設定的形狀上。"
        )

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    minimum, opt, maximum = profile_shapes(batch)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, min=minimum, opt=opt, max=maximum)
    config.add_optimization_profile(profile)

    print(f"[建置] optimization profile min={minimum} opt={opt} max={maximum}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "build_serialized_network 回 None（它失敗時不拋錯）。原因由 TensorRT 的 "
            "logger 印在上方。"
        )
    return bytes(serialized)


def export_engine(weights: Path, batch: int, vfa_metadata: dict, dest: Path) -> None:
    """匯出 ONNX、自建 FP16 dynamic 引擎、寫上帶 `vfa` 欄位的檔頭，產物落在 `dest`。

    ONNX 中繼檔跟著暫存目錄一起消失，不必也不該自己去刪權重旁邊的同名檔。

    Args:
        weights: 來源 `.pt`。
        batch: 引擎的最大批次。
        vfa_metadata: 要注入的欄位（`build_vfa_metadata` 的輸出）。
        dest: 產物的落點。
    """
    with tempfile.TemporaryDirectory(prefix="vfa-engine-") as tmp:
        onnx_path, onnx_metadata = export_onnx(weights, batch, Path(tmp))
        # 檔頭先組好也先驗過，才進約 7 分鐘的建置：組錯而等到建完才發現要整個重跑
        metadata = build_engine_metadata(onnx_metadata, vfa_metadata)
        engine = build_serialized_engine(onnx_path, batch)
        write_engine_with_metadata(metadata, engine, dest)


def verify_engine(engine_path: Path, batch: int, expected_sha256: str) -> dict:
    """在比對之前先驗引擎本身：檔頭、metadata、精度、profile、批次上限、跑得起來。

    **精度檢查與載入端共用同一份宣告、同一支函式**（`engine_metadata.EXPECTED_PRECISION_ARGS`
    ／`validate_engine_precision`）：逐項比對 `metadata["args"]` 而不是只驗 `half`——
    INT8 引擎可以同時把 FP16 flag 設成真，只驗 `half` 對這種引擎完全照過。

    **optimization profile 走正式載入路徑驗**：`TrtRunner(engine_path)` 就是推論進程
    載引擎的那一支，建一顆等於把載入期四道檢查（含
    `trt_runner.check_profile_shapes`）全跑一遍，判準取的是**引擎自己宣告的 profile**
    而不是本工具傳給 builder 的設定值。驗完就丟掉，讓它的 execution context 在
    ultralytics 那顆 probe 起來之前先還回去。

    Args:
        engine_path: 待驗的 `.engine`。
        batch: 預期的最大批次。
        expected_sha256: 來源權重的 SHA-256。

    Returns:
        引擎的 metadata dict。

    Raises:
        ValueError: 檔頭必填鍵缺漏或型別不符（`validate_engine_header`），metadata 與
            當下環境不符（`validate_engine_metadata`），精度旗標與
            `EXPECTED_PRECISION_ARGS` 不符（`validate_engine_precision`），
            optimization profile 不符（`trt_runner.check_profile_shapes`），或
            dynamic／批次上限不是預期值。
    """
    metadata = read_engine_metadata(engine_path)
    validate_engine_header(metadata)
    validate_engine_metadata(metadata, current_gpu_environment(), expected_sha256)
    validate_engine_precision(metadata)

    export_args = metadata.get("args") or {}
    if export_args.get("dynamic") is not True:
        raise ValueError(
            "引擎不是 dynamic 建的。靜態引擎會讓 640×384 的影格被填充成 640×640"
            "（像素量 1.67 倍），而且批次大小一有變動就在 forward 的 assert 失敗。"
        )
    if metadata.get("batch") != batch:
        raise ValueError(
            f"引擎的最大批次是 {metadata.get('batch')}，要求的是 {batch}。"
        )

    runner = TrtRunner(engine_path)
    profile_batch = runner.max_batch
    print(
        f"[驗證] 引擎宣告的 profile：batch 上限 {profile_batch}、"
        f"高寬 {runner.input_height}×{runner.input_width}"
    )
    del runner
    if profile_batch != batch:
        raise ValueError(
            f"引擎 optimization profile 的 batch 上限是 {profile_batch}，要求的是 "
            f"{batch}。檔頭的 `batch` 只是一個字串欄位，改它不會改變引擎——這一項取的"
            "是引擎自己宣告的值。"
        )

    # 真的用 ultralytics 載一次並用滿批跑一格：建置期驗收與 `compare_backend.py` 都走
    # 這條路徑，它與正式推論路徑（`TrtRunner`）不同，要各驗一次
    import numpy as np

    probe = YOLO(str(engine_path))
    frames = [np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), np.uint8) for _ in range(batch)]
    # **`imgsz` 一定要帶。** predictor 的 warmup 不走 letterbox，直接用 `args.imgsz`
    # （`engine/predictor.py` 的 `self.model.warmup(imgsz=(bs, ch, *self.imgsz))`），而
    # dynamic 引擎不把檔頭的 `imgsz` 抄進 `args.imgsz`，所以不指定就是預設的 640×640。
    # profile 收窄之前那個形狀還落在 max 界內，收窄之後就出界了——而
    # `nn/backends/tensorrt.py` 呼叫 `set_input_shape` **不檢查回傳值**，接著以 context
    # 上一個 shape 執行，症狀是讀到別的緩衝（batch 1 時直接越界讀取）。帶上 384×640 之後
    # warmup 與 predict 同形狀；對 16:9 來源 letterbox 的結果與預設值逐值相同
    probe.predict(frames, verbose=False, device=0, imgsz=(INFER_HEIGHT, INFER_WIDTH))
    shape = tuple(probe.predictor.model.bindings["images"].shape)
    print(f"[驗證] 滿批 {batch} 實際進入推論的張量形狀：{shape}")
    if shape[2:] != (INFER_HEIGHT, INFER_WIDTH):
        raise ValueError(
            f"滿批推論的實際形狀是 {shape}，預期高寬為 "
            f"({INFER_HEIGHT}, {INFER_WIDTH})。dynamic 引擎的形狀由 pre_transform "
            "決定，形狀對不上代表填充行為與預期不同。"
        )
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path, help="來源 .pt 權重")
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument(
        "--batch",
        type=int,
        required=True,
        help="引擎的最大批次。要容得下 config.toml 的 [model].batch（兩者同尺度，"
        "設定值就是實際的單次推理批次）",
    )
    ap.add_argument(
        "--bucket", type=Path, default=None, help="比對取樣用的 bucket；省略即 --skip-compare"
    )
    ap.add_argument("--cameras", default="test_cam001,test_cam002,test_cam004")
    ap.add_argument("--frames-per-camera", type=int, default=60)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--classes", default="0,2")
    ap.add_argument(
        "--max-foot-dev-p99-px",
        type=float,
        default=DEFAULT_MAX_FOOT_DEV_P99_PX,
        help=f"落腳點偏差的 **p99** 上限（px），預設 {DEFAULT_MAX_FOOT_DEV_P99_PX}。"
        "另外三道門檻（尾巴比例、配對率、框數差）見 compare_backend.check_report",
    )
    ap.add_argument(
        "--skip-compare",
        action="store_true",
        help="只 build ＋ deserialize，不跑對 Torch FP32 的比對。**產物保留 "
        f"`{_UNVERIFIED_SUFFIX}` 尾綴、不會改成正式檔名**，只用於「確認這張卡建得出"
        "引擎」的 smoke（T4 驗收清單的第一步）。",
    )
    ap.add_argument("--report-out", type=Path, default=None, help="比對報告 JSON 輸出路徑")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"找不到權重檔 {args.weights}")
    if args.bucket is None and not args.skip_compare:
        raise SystemExit(
            "沒給 --bucket 就沒有畫面可比對。要嘛給 --bucket，要嘛明講 --skip-compare"
            "（那樣的產物不可上線）。"
        )

    environment = current_gpu_environment()
    print(
        f"建置機：{environment.gpu_name}（sm{environment.compute_capability}）、"
        f"TensorRT {environment.tensorrt}（{environment.tensorrt_package}）、"
        f"driver {environment.driver}、torch {torch.__version__}"
    )

    weights_sha256 = sha256_of(args.weights)
    source_model = YOLO(str(args.weights))
    check_train_imgsz(source_model)
    vfa_metadata = build_vfa_metadata(
        args.weights, environment, read_train_info(source_model)
    )
    del source_model  # 匯出會自己再載一次，這裡只是為了讀 ckpt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / engine_filename(
        args.weights, environment.compute_capability
    )
    staged = final_path.with_suffix(final_path.suffix + _UNVERIFIED_SUFFIX)
    export_engine(args.weights, args.batch, vfa_metadata, staged)

    try:
        verify_engine(staged, args.batch, weights_sha256)
        if not args.skip_compare:
            report = compare_backends(
                bucket=args.bucket,
                model_a=args.weights,
                model_b=staged,
                cameras=[c.strip() for c in args.cameras.split(",") if c.strip()],
                classes=[int(c) for c in args.classes.split(",") if c.strip()],
                frames_per_camera=args.frames_per_camera,
                stride=args.stride,
                allow_tf32=False,  # 跨機器比對的前提，不開放由 CLI 關掉
            )
            print_summary(report)
            # 寫報告排在判準之前（沒過也要看得到數字），但**它的失敗不可以害死引擎**：
            # 這一段在「沒過就刪產物」的 except 內，報告路徑唯讀或沒權限的話，四道門檻
            # 其實全過的引擎會被刪掉，還印出「未通過驗收」——T4 上約 7 分鐘的建置要重跑，
            # 而錯誤訊息指向錯的原因
            if args.report_out:
                try:
                    args.report_out.parent.mkdir(parents=True, exist_ok=True)
                    args.report_out.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"已寫出 {args.report_out}")
                except OSError as exc:
                    print(f"[警告] 報告寫不出來（{exc}），比對結果只在上面的輸出裡。")
            failures = check_report(report, max_p99_px=args.max_foot_dev_p99_px)
            if failures:
                raise ValueError(
                    "對 Torch FP32 的比對沒過，不產出引擎：\n  - "
                    + "\n  - ".join(failures)
                )
    except BaseException:
        # 沒過就沒有產物：`.unverified` 一併刪掉，避免有人拿去改名上線
        staged.unlink(missing_ok=True)
        print(f"[失敗] 已刪除未通過驗收的產物 {staged}", file=sys.stderr)
        raise

    if args.skip_compare:
        # **不改名**。只印一行警告的話，磁碟上會留下一顆與通過比對的產物長得一模一樣的
        # 引擎，事後無從分辨——而 `--skip-compare` 正是為了 smoke 而存在，smoke 的產物
        # 混進正式檔名是這支工具最容易造成的事故。留著尾綴也讓它跑不起來：
        # `YOLODetector` 要求副檔名是 `.engine`，`.engine.unverified` 會被當場擋下。
        print(
            f"\n[未驗收] --skip-compare：沒跑對 Torch FP32 的比對，產物保留在 {staged}"
            "，**不會改成正式檔名，也載不進正式推論路徑**。要上線請拿掉 "
            "--skip-compare 重跑。"
        )
        return
    staged.rename(final_path)
    print(f"\n引擎已產出：{final_path}")


if __name__ == "__main__":
    main()
