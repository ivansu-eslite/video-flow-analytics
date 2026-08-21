#!/usr/bin/env python3
"""從 `.pt` 權重建出正式推論用的 TensorRT FP16 引擎，比對沒過就不產出。

**同一支工具、兩台機器、兩顆引擎**：TensorRT 引擎不可跨架構重用——`IBuilderConfig`
沒有指定 target SM 的 API，`HardwareCompatibilityLevel` 只有 `NONE`／`AMPERE_PLUS`／
`SAME_COMPUTE_CAPABILITY`，而 `AMPERE_PLUS` 不含 Turing。所以正式引擎（T4／sm75）只能
在 T4 上建，開發機（RTX 5090／sm120）建的那顆只能給地端用、不可上線。引擎檔名把 SM
編進去（`..._sm75.engine`）就是為了讓兩顆不可能被混用——下游 argus 的 promotion 用
`Path(source_model_uri).stem` 當 `model_version`，兩顆同名會直接撞在一起。

三個刻意的建置參數：

- **`dynamic=True`**。不是為了彈性，是兩件事各自都逼出這個選擇：(1) 靜態引擎的
  `TensorRTBackend.forward` 會 assert 輸入形狀與綁定完全相同，而推理迴圈的批次大小
  本來就會因湊批而變動（某一路讀完時剩幾格就送幾格）；(2) ultralytics 的
  `pre_transform` 只在 `format == "pt"` 或 `dynamic` 為真時才走 `auto` 模式，靜態引擎
  會讓 640×384 的影格被填充成 640×640，像素量 1.67 倍——正是 issue #108 消掉的那個成本。
- **`imgsz=(INFER_HEIGHT, INFER_WIDTH)`**。形狀在建置期固定（optimization profile 的
  opt shape），不是在執行期驗。載入端不驗 metadata 的 `imgsz`：dynamic 引擎不套用它，
  實際形狀由 `pre_transform` 決定，照抄 `_validate_imgsz` 會得到一個看起來有在驗、
  其實驗不到的檢查。執行期改為逐批記錄**實際進入推論的 (H, W)**（見 `detector.py`）。
- **`batch=<最大批次>`**。dynamic 引擎的 batch 維度上限就是這個值（optimization
  profile 的 max shape 取 `shape[0]`），所以它與 `config.toml` 的 `[model] batch`
  要一起定：實際單次推理批次是設定值的 2 倍（ultralytics 對 in-memory list source
  一次 forward 整個 list），超過引擎上限會在 `forward` 的 assert 當場失敗。

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
    VFA_METADATA_KEY,
    build_vfa_metadata,
    current_gpu_environment,
    read_engine_metadata,
    sha256_of,
    validate_engine_metadata,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH

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


def engine_filename(weights: Path, compute_capability: str) -> str:
    """引擎檔名：來源權重的 stem ＋ `_sm<SM>`。

    Args:
        weights: 來源 `.pt`。
        compute_capability: 建置機的 compute capability（如 `"7.5"`）。

    Returns:
        如 `20260714-153811_yolo26m_baseline_sm75.engine`。
    """
    return f"{weights.stem}_sm{compute_capability.replace('.', '')}.engine"


def export_engine(weights: Path, batch: int, vfa_metadata: dict) -> Path:
    """匯出 FP16 dynamic 引擎，並把 `vfa` 欄位注入 metadata。

    注入走 `on_export_start` callback：公開的 `export()` 沒有任意欄位入口，而
    `Exporter.__call__` 是先組完 `self.metadata` 才 `run_callbacks("on_export_start")`，
    所以那個時點改得動它。

    Args:
        weights: 來源 `.pt`。
        batch: 引擎的最大批次。
        vfa_metadata: 要注入的欄位（`build_vfa_metadata` 的輸出）。

    Returns:
        ultralytics 產出的 `.engine` 路徑（在 `weights` 旁邊）。
    """
    model = YOLO(str(weights))

    def _inject(exporter) -> None:
        exporter.metadata[VFA_METADATA_KEY] = vfa_metadata

    model.add_callback("on_export_start", _inject)
    exported = Path(
        model.export(
            format="engine",
            half=True,
            dynamic=True,
            batch=batch,
            imgsz=(INFER_HEIGHT, INFER_WIDTH),
            device=0,
            verbose=False,
        )
    )
    # ultralytics 匯出引擎要先過 ONNX，中繼檔會留在權重旁邊（80 MB 起跳）。留著沒有用途
    # ——本工具下次跑會重新匯出——但很容易被誤以為是產物之一
    exported.with_suffix(".onnx").unlink(missing_ok=True)
    return exported


def verify_engine(engine_path: Path, batch: int, expected_sha256: str) -> dict:
    """在比對之前先驗引擎本身：metadata、精度、批次上限、實際跑得起來。

    **精度驗 `metadata["args"]["half"]`，不是 backend 的 `fp16` 屬性**：FP16 引擎的
    I/O binding 仍是 FP32，`AutoBackend` 對它永遠回報 `False`，拿那個值當判準等於
    永遠判定「不是 FP16」。

    Args:
        engine_path: 待驗的 `.engine`。
        batch: 預期的最大批次。
        expected_sha256: 來源權重的 SHA-256。

    Returns:
        引擎的 metadata dict。

    Raises:
        ValueError: metadata 與當下環境不符（`validate_engine_metadata`），或精度／
            dynamic／批次上限不是預期值。
    """
    metadata = read_engine_metadata(engine_path)
    validate_engine_metadata(metadata, current_gpu_environment(), expected_sha256)

    export_args = metadata.get("args") or {}
    if export_args.get("half") is not True:
        raise ValueError(
            f"引擎不是 FP16 建的（metadata.args.half = {export_args.get('half')}）。"
            "正式推論路徑只收 FP16 引擎（ADR-011）。"
        )
    if export_args.get("dynamic") is not True:
        raise ValueError(
            "引擎不是 dynamic 建的。靜態引擎會讓 640×384 的影格被填充成 640×640"
            "（像素量 1.67 倍），而且批次大小一有變動就在 forward 的 assert 失敗。"
        )
    if metadata.get("batch") != batch:
        raise ValueError(
            f"引擎的最大批次是 {metadata.get('batch')}，要求的是 {batch}。"
        )

    # 真的載一次並用滿批跑一格：deserialize 成功、且 optimization profile 涵蓋
    # 「設定值換算出來的實際批次 × 推論尺寸」，這兩件事只有跑過才知道
    import numpy as np

    probe = YOLO(str(engine_path))
    frames = [np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), np.uint8) for _ in range(batch)]
    probe.predict(frames, verbose=False, device=0)
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
        help="引擎的最大批次。是 config.toml [model].batch 的 2 倍（ultralytics 對 "
        "in-memory list source 一次 forward 整個 list）",
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
    vfa_metadata = build_vfa_metadata(
        args.weights, environment, read_train_info(source_model)
    )
    del source_model  # 匯出會自己再載一次，這裡只是為了讀 ckpt

    exported = export_engine(args.weights, args.batch, vfa_metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / engine_filename(
        args.weights, environment.compute_capability
    )
    staged = final_path.with_suffix(final_path.suffix + _UNVERIFIED_SUFFIX)
    shutil.move(str(exported), staged)

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
            if args.report_out:
                args.report_out.parent.mkdir(parents=True, exist_ok=True)
                args.report_out.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"已寫出 {args.report_out}")
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
