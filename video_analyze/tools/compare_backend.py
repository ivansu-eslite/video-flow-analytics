#!/usr/bin/env python3
"""比對兩種推論後端（PyTorch `.pt` 對 TensorRT `.engine`）的偵測結果差多少。

**這支是從 `outputs/vfa_perf/code/compare_backend.py` 移植進版控的**（那份不進版控，
隨效能量測的產物一起放在 `outputs/`）。移植而非新寫：那份本來就是「Torch 權重對
TensorRT 引擎」的比對器——已處理 `.engine` 不能呼叫 `.to()`、刻意不接受用 `--half`
控制引擎那側的精度（引擎的精度在 build 時就固定了）、比的正是**框底邊中點的座標
偏差**（下游的落腳點、跨線進出、區域佔用都是從這個點算出來的，框稍微大一點小一點
不重要，底邊中點跑掉才會改變判定）。隔壁的 `compare_precision.py` 只比兩種 Torch
精度、`build_model` 會對引擎呼叫 `.to()` 而當場崩，兩支是刻意分職的。

移植時的三處改動：
- `--classes` 預設改成正式設定的 `[0, 2]`（head ＋ fbody），並**逐類別各報一組偏差**。
  原版只跑 fbody（既有的 1.20 px 基準就是這麼量的），但 head 框現在也進落腳點推算
  （ADR-009），只量 fbody 等於有一半的輸入沒被驗到。看 `class_2` 那組即可與舊基準對齊。
- 抽出 `compare_backends()` 供 `build_engine.py` 直接呼叫（「比對沒過就不產出引擎」）。
- 逐格 `zip` 加 `strict=True`：兩個後端逐格對應，長度不同代表有一側漏了整格。

**這支載入 `.pt`，但它不在 `YOLODetector` 的推論路徑上、也不隨 wheel 出貨**
（`tools/` 是 `video_analyze/` 底下、`src/` 之外的目錄，uv_build 只打包 `src/`）。
ADR-011 的分界就是這一條：正式產品只有一套 inference implementation，Torch FP32
是套件外的驗證工具。

⚠ **跨機器比對前必須關掉 TF32**：PyTorch 預設讓卷積在 Ampere＋用 TF32（尾數只有
10 bit），開發機的 RTX 5090 吃得到，但正式環境的 T4 是 Turing、沒有 TF32 硬體，跑的
是真 FP32。若在開發機用預設值量，等於拿「TF32 對 FP16」當成「FP32 對 FP16」，會**低估**
偏差——而低估的方向剛好是「看起來沒問題」。故預設強制關閉；`--allow-tf32` 可以刻意
打開，用來單獨看 TF32 本身偏多少。

用法：

    uv run --package video_analyze python video_analyze/tools/compare_backend.py \
        --bucket bucket_20260801_small \
        --model 20260714-153811_yolo26m_baseline.pt \
        --model-b 20260714-153811_yolo26m_baseline_sm120.engine

（在 repo 根目錄執行；`--package` 不改變 cwd，`--directory` 會。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH

# 每一次 `predict` 都要帶的推論尺寸。**不是可有可無的參數**：predictor 的 warmup 不走
# letterbox，直接用 `args.imgsz`（`engine/predictor.py` 的
# `self.model.warmup(imgsz=(bs, ch, *self.imgsz))`），而 dynamic 引擎不把檔頭的 `imgsz`
# 抄進 `args.imgsz`，所以不指定就是預設的 640×640。引擎的 optimization profile 收窄成
# min=opt=max=384×640 之後（ADR-015）那個形狀出界，而 `nn/backends/tensorrt.py` 呼叫
# `set_input_shape` **不檢查回傳值**，接著以 context 上一個 shape 執行——這裡逐張送
# （batch 1），warmup 緩衝只有 1×3×640×640×4 而引擎照 16×3×384×640 讀，是越界讀取，
# 最好的情況是拿到垃圾。帶上 384×640 之後 warmup 與 predict 同形狀；對 16:9 來源
# letterbox 的結果與預設值逐值相同（兩者在 `auto=True` 下都收斂到 384×640）。
_INFER_IMGSZ = (INFER_HEIGHT, INFER_WIDTH)

# 驗收判準。**看 p99 而不是 max**，這一條要寫清楚理由：既有基準的兩個數字
# （FP16 的 1.20 px、TensorRT 已驗的 1.16 px）分別是 **277 個框的 max** 與 **277 個框的
# p99**，而本工具預設取樣 2000 個框以上。max 是樣本數的函數（樣本越多、尾巴越長），拿
# 277 個框的 max 當 2000 個框的門檻是在比兩個不同的統計量。p99 在兩個樣本數下可比。
#
# max 不設門檻不代表尾巴沒人管：另外一條門檻直接管尾巴的**比例**——偏差超過
# `DEFAULT_OUTLIER_PX` 的配對不得超過 `DEFAULT_MAX_OUTLIER_RATIO_PCT`。單一個離群不擋
# （實測那一個是 conf 0.255 對 0.455 的邊界偵測，模型自己就不確定人腳在哪，不是 FP16
# 的捨入誤差），系統性的尾巴擋得住。
DEFAULT_MAX_FOOT_DEV_P99_PX = 1.20
DEFAULT_OUTLIER_PX = 5.0
DEFAULT_MAX_OUTLIER_RATIO_PCT = 0.5
# 兩條「整批壞掉」的門檻：精度或形狀出錯時偏差不見得變大，但框數與配對率會塌。
DEFAULT_MIN_MATCH_RATE_PCT = 98.0
DEFAULT_MAX_COUNT_DIFF_PCT = 2.0


def load_frames(video: Path, count: int, stride: int) -> list[np.ndarray]:
    """每隔 stride 張取一張，讓取樣涵蓋整段影片而不是集中在開頭。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟 {video}")
    frames: list[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < count:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{video} 讀不到影格")
    return frames


def build_model(weights: Path, classes: list[int], half: bool) -> YOLO:
    """載入模型並讓精度真正生效。

    ultralytics 只在**建立 predictor 的那一次**決定精度：predictor 已存在後再傳
    `half=True` 只會改到 `args.half`，模型權重仍是 float32，等於量到 FP32 對 FP32
    而不自知。因此每種精度各用一個實例，並在第一次呼叫就帶上該精度。
    """
    # 匯出格式（.engine／.onnx）不是 PyTorch module，不能呼叫 .to()；
    # 它們的裝置由 predict 的 device 參數決定。
    model = YOLO(str(weights))
    if weights.suffix == ".pt":
        model = model.to("cuda")
    warmup = np.zeros((1080, 1920, 3), dtype=np.uint8)
    kwargs: dict = {
        "verbose": False,
        "classes": classes,
        "device": 0,
        "imgsz": _INFER_IMGSZ,
    }
    if half:
        kwargs["half"] = True
    model.predict([warmup], **kwargs)

    # TensorRT 引擎的精度由引擎本身決定，`predictor.model.fp16` 不反映它（FP16 引擎的
    # I/O binding 仍是 FP32，AutoBackend 對它永遠回報 False），這道防呆只對 .pt 有意義；
    # 引擎那側的精度改由 build_engine.py 驗 metadata 的 args.half。
    if weights.suffix == ".pt":
        actual = bool(getattr(model.predictor.model, "fp16", False))
        if actual != half:
            raise SystemExit(
                f"精度設定沒有生效：要求 half={half}，實際 fp16={actual}。"
                "不能繼續比對，否則會拿兩份相同精度的結果當成有差異。"
            )
    return model


def detect(
    model: YOLO, frames: list[np.ndarray], classes: list[int], half: bool
) -> list[np.ndarray]:
    """回傳每張畫面的偵測框陣列 [x1, y1, x2, y2, conf, cls]。"""
    kwargs: dict = {"verbose": False, "classes": classes, "imgsz": _INFER_IMGSZ}
    if half:
        kwargs["half"] = True
    out = []
    # 逐張送，避免批次組成不同造成的差異混進來
    for index, f in enumerate(frames):
        r = model.predict([f], **kwargs)[0]
        if index == 0:
            _check_engine_infer_shape(model, f)
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 6), dtype=np.float64))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = r.boxes.conf.cpu().numpy().astype(np.float64).reshape(-1, 1)
        cls = r.boxes.cls.cpu().numpy().astype(np.float64).reshape(-1, 1)
        out.append(np.hstack([xyxy, conf, cls]))
    return out


def _check_engine_infer_shape(model: YOLO, frame: np.ndarray) -> None:
    """引擎那側核對第一張畫面實際進入推論的高寬；`.pt` 那側沒有 binding，直接略過。

    `_INFER_IMGSZ` 只在 `auto=True` 的 letterbox 收斂到 384×640 時才等於推論尺寸，而
    **那不是所有長寬比都成立**：本 bucket 的四台（1920×1080 與 3840×2160）都收斂到
    384×640，但例如 1440×1080 會得到 384×512。收窄 profile 之前那種形狀落在 max 界
    （768×1280）內、跑得對；收窄之後它出界，而 `nn/backends/tensorrt.py` 呼叫
    `set_input_shape` 不檢查回傳值，接著以 context 上一個 shape 執行——四道門檻多半會
    因為偏差爆掉而擋下，但訊息會指向「比對沒過」而不是原因。

    Args:
        model: 已經至少 predict 過一次的 `YOLO`。
        frame: 剛送進去的那張畫面，只用於錯誤訊息。

    Raises:
        SystemExit: 引擎實際吃到的高寬不是 `_INFER_IMGSZ`。
    """
    bindings = getattr(getattr(model.predictor, "model", None), "bindings", None)
    if not bindings or "images" not in bindings:
        return
    shape = tuple(bindings["images"].shape)
    if shape[2:] != _INFER_IMGSZ:
        raise SystemExit(
            f"引擎實際吃到的高寬是 {shape[2:]}，不是 {_INFER_IMGSZ}——來源畫面 "
            f"{frame.shape[1]}×{frame.shape[0]} 經 letterbox 收斂不到推論尺寸。"
            "收窄 profile 之後這個形狀落在 optimization profile 外，而 TensorRT 的 "
            "`set_input_shape` 只回 False、不拋錯。請改用 16:9 的取樣來源。"
        )


def match_boxes(
    a: np.ndarray, b: np.ndarray, iou_thresh: float = 0.5
) -> list[tuple[int, int]]:
    """用 IoU 貪婪配對兩組框，回傳 (a 的索引, b 的索引)。"""
    if len(a) == 0 or len(b) == 0:
        return []
    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]
    ax2, ay2 = a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = b[:, 2][None, :], b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou = inter / (area_a + area_b - inter + 1e-9)

    pairs: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if iou[i, j] < iou_thresh:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        pairs.append((int(i), int(j)))
    return pairs


def _dev_stats(values: list[float]) -> dict:
    """把一組偏差整理成 max／mean／p99／尾巴比例。

    `over_outlier_pct` 是偏差超過 `DEFAULT_OUTLIER_PX` 的配對比例——判準看的是它，
    不是 `max`（理由見本檔開頭的門檻常數）。
    """
    # 空樣本的欄位要與有樣本時**完全一致**：零配對正是這支工具最該講話的時候
    # （引擎壞掉、精度或形狀出錯都長這樣），少一個欄位會讓 print_summary 以 KeyError
    # 收場——報告沒寫出、check_report 沒跑到，本該印的「配對率 0% < 98%」也看不到。
    if not values:
        return {
            "max": 0.0,
            "mean": 0.0,
            "p99": 0.0,
            "n": 0,
            "over_outlier_px": DEFAULT_OUTLIER_PX,
            "over_outlier_n": 0,
            "over_outlier_pct": 0.0,
        }
    over = sum(1 for v in values if v > DEFAULT_OUTLIER_PX)
    return {
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 4),
        "p99": round(float(np.percentile(values, 99)), 3),
        "n": len(values),
        "over_outlier_px": DEFAULT_OUTLIER_PX,
        "over_outlier_n": over,
        "over_outlier_pct": round(over / len(values) * 100, 4),
    }


def compare_backends(
    bucket: Path,
    model_a: Path,
    model_b: Path,
    cameras: list[str],
    classes: list[int],
    frames_per_camera: int = 60,
    stride: int = 7,
    allow_tf32: bool = False,
) -> dict:
    """跑完整份比對並回傳報告 dict。

    Args:
        bucket: 影片來源 bucket 目錄（底下每個攝影機一個子目錄）。
        model_a: 基準模型（`.pt`，FP32）。
        model_b: 對照模型（`.pt` 或 `.engine`）。
        cameras: 要取樣的攝影機子目錄名稱。
        classes: 要保留的類別 id。
        frames_per_camera: 每台攝影機取幾張畫面。
        stride: 每隔幾張取一張。
        allow_tf32: 保留 PyTorch 預設的 TF32 卷積；預設 `False`（強制真 FP32，
            與 T4 的行為一致）。

    Returns:
        報告 dict。判準由 `check_report` 套用，看的是
        `overall.foot_point_dev_px.p99` 與 `.over_outlier_pct`，**不是 `.max`**
        （理由見本檔開頭的門檻常數）。

    Raises:
        RuntimeError: 所有指定的攝影機都找不到影片。
    """
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    model_fp32 = build_model(model_a, classes, half=False)
    # 引擎的精度是 build 時烤進去的，不能也不該用 --half 去控；只有 .pt 對照組才傳
    b_half = model_b.suffix == ".pt"
    model_fp16 = build_model(model_b, classes, half=b_half)

    report: dict = {
        "model": model_a.name,
        "model_b": model_b.name,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "allow_tf32": allow_tf32,
        "classes": classes,
        "cameras": {},
    }
    all_foot_dev: list[float] = []
    all_corner_dev: list[float] = []
    per_class_foot_dev: dict[int, list[float]] = {c: [] for c in classes}
    total_a = total_b = total_matched = 0

    for cam in cameras:
        files = sorted((bucket / cam).rglob("*.mkv"))
        if not files:
            print(f"[略過] {cam} 找不到影片")
            continue
        frames = load_frames(files[0], frames_per_camera, stride)

        det_a = detect(model_fp32, frames, classes, half=False)
        det_b = detect(model_fp16, frames, classes, half=b_half)

        foot_dev: list[float] = []
        corner_dev: list[float] = []
        n_a = n_b = matched = 0
        for a, b in zip(det_a, det_b, strict=True):
            n_a += len(a)
            n_b += len(b)
            for i, j in match_boxes(a, b):
                matched += 1
                # 落腳點在這裡取框底邊中點：兩側都用同一個定義，比的是後端造成的
                # 座標偏差本身，不是落腳點推算法（head 反射推算在 track_worker，
                # 不在偵測輸出裡）
                fa = np.array([(a[i, 0] + a[i, 2]) / 2, a[i, 3]])
                fb = np.array([(b[j, 0] + b[j, 2]) / 2, b[j, 3]])
                dev = float(np.linalg.norm(fa - fb))
                foot_dev.append(dev)
                corner_dev.append(float(np.abs(a[i, :4] - b[j, :4]).max()))
                per_class_foot_dev.setdefault(int(a[i, 5]), []).append(dev)

        report["cameras"][cam] = {
            "frames": len(frames),
            "resolution": f"{frames[0].shape[1]}×{frames[0].shape[0]}",
            "detections_a": n_a,
            "detections_b": n_b,
            "detection_count_diff": n_b - n_a,
            "matched": matched,
            "unmatched_a": n_a - matched,
            "unmatched_b": n_b - matched,
            "foot_point_dev_px": _dev_stats(foot_dev),
            "corner_dev_px_max": round(max(corner_dev), 3) if corner_dev else 0.0,
        }
        all_foot_dev += foot_dev
        all_corner_dev += corner_dev
        total_a += n_a
        total_b += n_b
        total_matched += matched

        c = report["cameras"][cam]
        print(
            f"{cam} ({c['resolution']}) {c['frames']} 張畫面：\n"
            f"  偵測數 A {n_a} → B {n_b}（差 {n_b - n_a:+d}）\n"
            f"  配對成功 {matched}，落腳點偏差 最大 "
            f"{c['foot_point_dev_px']['max']} px、"
            f"平均 {c['foot_point_dev_px']['mean']} px",
            flush=True,
        )

    if not report["cameras"]:
        raise RuntimeError(
            f"{bucket} 底下的 {cameras} 全部找不到影片，比對沒有跑到任何畫面。"
        )

    report["overall"] = {
        "detections_a": total_a,
        "detections_b": total_b,
        "detection_count_diff_pct": (
            round((total_b - total_a) / total_a * 100, 3) if total_a else 0.0
        ),
        "matched": total_matched,
        "match_rate_pct": (
            round(total_matched / total_a * 100, 2) if total_a else 0.0
        ),
        "foot_point_dev_px": _dev_stats(all_foot_dev),
        "corner_dev_px_max": round(max(all_corner_dev), 3) if all_corner_dev else 0.0,
    }
    # 逐類別各一組：class 2（fbody）那組才與既有的 1.20 px 基準同口徑
    report["per_class_foot_point_dev_px"] = {
        f"class_{c}": _dev_stats(v) for c, v in sorted(per_class_foot_dev.items())
    }
    return report


def check_report(
    report: dict,
    max_p99_px: float = DEFAULT_MAX_FOOT_DEV_P99_PX,
    max_outlier_ratio_pct: float = DEFAULT_MAX_OUTLIER_RATIO_PCT,
    min_match_rate_pct: float = DEFAULT_MIN_MATCH_RATE_PCT,
    max_count_diff_pct: float = DEFAULT_MAX_COUNT_DIFF_PCT,
) -> list[str]:
    """對報告套四道驗收門檻，回傳沒過的項目描述（全過則為空清單）。

    判準與計算放在同一支：門檻寫在別處的話，改了取樣方式（框數、攝影機、classes）
    而沒改門檻不會有訊號，而門檻本身就是樣本數的函數（見本檔開頭的常數說明）。

    Args:
        report: `compare_backends` 的輸出。
        max_p99_px: 落腳點偏差 p99 上限（px）。
        max_outlier_ratio_pct: 偏差超過 `DEFAULT_OUTLIER_PX` 的配對比例上限（%）。
        min_match_rate_pct: 配對率下限（%）。
        max_count_diff_pct: 偵測框數差異上限（%，取絕對值）。

    Returns:
        沒過的門檻描述清單。
    """
    overall = report["overall"]
    dev = overall["foot_point_dev_px"]
    failures: list[str] = []
    if dev["p99"] > max_p99_px:
        failures.append(
            f"落腳點偏差 p99 {dev['p99']} px > 上限 {max_p99_px} px"
            f"（最大 {dev['max']} px、平均 {dev['mean']} px）"
        )
    if dev["over_outlier_pct"] > max_outlier_ratio_pct:
        failures.append(
            f"偏差 > {dev['over_outlier_px']} px 的配對佔 "
            f"{dev['over_outlier_pct']}%（{dev['over_outlier_n']}/{dev['n']}）"
            f" > 上限 {max_outlier_ratio_pct}%"
        )
    if overall["match_rate_pct"] < min_match_rate_pct:
        failures.append(
            f"配對率 {overall['match_rate_pct']}% < 下限 {min_match_rate_pct}%"
        )
    if abs(overall["detection_count_diff_pct"]) > max_count_diff_pct:
        failures.append(
            f"偵測框數差 {overall['detection_count_diff_pct']:+.3f}% "
            f"超過 ±{max_count_diff_pct}%"
        )
    return failures


def print_summary(report: dict) -> None:
    """把總計那幾行印出來。"""
    o = report["overall"]
    d = o["foot_point_dev_px"]
    print(
        f"\n總計：偵測數 {o['detections_a']} → {o['detections_b']}"
        f"（{o['detection_count_diff_pct']:+.3f}%），配對率 {o['match_rate_pct']}%\n"
        f"落腳點偏差：**99 百分位 {d['p99']} px（判準看這個）**、"
        f"平均 {d['mean']} px、最大 {d['max']} px\n"
        f"  > {d['over_outlier_px']} px 的配對：{d['over_outlier_n']}/{d['n']}"
        f"（{d['over_outlier_pct']}%）"
    )
    for name, stats in report["per_class_foot_point_dev_px"].items():
        print(
            f"  {name}：配對 {stats['n']}，最大 {stats['max']} px、"
            f"平均 {stats['mean']} px"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path, help="基準（.pt，FP32）")
    ap.add_argument(
        "--model-b", required=True, type=Path, help="對照（.pt 或 .engine）"
    )
    ap.add_argument("--cameras", default="test_cam001,test_cam002,test_cam004")
    ap.add_argument("--frames-per-camera", type=int, default=60)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--classes", default="0,2", help="預設同正式設定的 head + fbody")
    ap.add_argument(
        "--allow-tf32",
        action="store_true",
        help="保留 PyTorch 預設的 TF32 卷積；不加則強制真 FP32（與 T4 的行為一致）",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--check",
        action="store_true",
        help="套驗收門檻，沒過就以非零 exit code 結束（build_engine.py 走的是同一組）",
    )
    args = ap.parse_args()

    report = compare_backends(
        bucket=args.bucket,
        model_a=args.model,
        model_b=args.model_b,
        cameras=[c.strip() for c in args.cameras.split(",") if c.strip()],
        classes=[int(c) for c in args.classes.split(",") if c.strip()],
        frames_per_camera=args.frames_per_camera,
        stride=args.stride,
        allow_tf32=args.allow_tf32,
    )
    print_summary(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已寫出 {args.out}")

    if args.check:
        failures = check_report(report)
        if failures:
            raise SystemExit("驗收未通過：\n  - " + "\n  - ".join(failures))
        print("驗收通過。")


if __name__ == "__main__":
    main()
