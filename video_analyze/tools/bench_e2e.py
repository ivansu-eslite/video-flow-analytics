#!/usr/bin/env python3
"""`video_analyze` 的端到端效能量測：跑矩陣、收中繼資料、把產物解析成可比較的表。

**這支是從 `outputs/vfa_perf/code/` 的三支腳本移植進版控的**（`run_main_local.sh` 跑
單輪、`run_all_main.sh` 跑矩陣、`summarize_main_runs.py` 解析），與 `compare_backend.py`
是同一件事的第二次。那三支寫死了絕對路徑、日期與產物目錄名，又放在 gitignore 的
`outputs/` 底下：下次要重跑（換卡、改批次、驗證某次重構有沒有拖慢）得從頭再寫一次。

⚠ **FPS 口徑：報表的「每秒張數」一律取推論進程**——`component == "inference"` 的
`FPS 整體` 那行。每輪 log 裡還有**追蹤進程**印的 `overall_fps`（`追蹤進程結束`，
`[tracker].shards` 有幾片就有幾行），但它的分母是該進程自己的 wall clock，與推論進程
不是同一個東西，混用會系統性高估 1–2%。這件事寫在註解裡活不久，所以本檔改用四道互鎖：

1. 解析器回傳的 `FpsReading` 是 frozen dataclass，欄位各自帶口徑（`inference_fps`／
   逐片的 `track_workers`），**沒有任何欄位叫 `fps` 或 `overall_fps`**，呼叫端拿不到
   一個泛用的 FPS 可以誤用；要單一追蹤數字得走 `min_track_worker_fps`，名字裡就帶著
   「取最小值」——瓶頸是最慢的那片，平均會把它蓋掉。
2. 分組統計的輸入只由 `_inference_fps_values()` 產生，追蹤值不傳進去；明細表兩欄分開
   標示，而分組平均只有推論那一個。
3. 缺推論行時**不以追蹤值遞補**，該列標為未完成並排除在統計外——遞補正好製造系統性高估。
4. `component` 與 `message` 兩個欄位同時比對；重複的推論行直接拋錯（代表兩次執行寫進
   同一份 log，靜默取一個是同類錯誤），追蹤行則以 `shard_id` 重複來認同一件事。
   **不提供切換口徑的旗標**：給了開關等於承認兩者可互換，而它們分母不同。

其餘四個刻意的取捨：

- **子進程繼承的 settings 環境變數進分組 key**。工具只設四個變數，但整份環境都會被
  繼承：環境裡先有一個 `INPUT__CAMERA_IDS`，那輪量的就是別的工作量，而 FPS 表上看不
  出來。所以 `extra_settings_env()` 把「會生效、工具沒設」的那些收進 meta，並讓它成為
  `group_runs` 分組 key 的一部分——只記進 meta 是不夠的，`report` 照樣會把兩種工作量
  平均成一個看起來合理的數字。
- **資源統計用 `os.wait4`，不是 `getrusage(RUSAGE_CHILDREN)`**。後者是本進程「所有已
  回收子進程」的累計值：矩陣跑十輪，第二筆之後每筆的 CPU 時間是總和、`ru_maxrss` 是
  歷來最大值，靜默偏大且無法相減還原。配套是**不可先呼叫 `Popen.wait()`**（那會先回收
  子進程，`wait4` 隨即拋 `ChildProcessError`）。`ru_maxrss` 是進程樹中**最大的單一
  進程**、不是總和——vfa 是多進程，這個數字不等於整體記憶體足跡。
- **缺 `nvidia-smi` 即 fail loud**，且在矩陣開跑前就檢查。這是 GPU 端到端量測工具，
  取樣不只是輔助：「顯卡／CPU 拆分」用的正是該次執行自己的 SM 取樣，少了它那項分析
  做不了，而 FPS 表上完全看不出來。跑滿 45 分鐘才發現太貴。要豁免請明講 `--no-gpu-log`。
- **產物目錄預設 `outputs/bench_e2e`，絕不可放在 `outputs/<bucket>/` 底下**：每輪開頭會
  清空該目錄，產物會自己刪自己。兩道防護——被清路徑做逃逸檢查，產物檔已存在且未給
  `--overwrite` 即中止（舊腳本靜默覆蓋，那正是「這輪的 meta 配上一輪的 log」的來源）。

本檔**只用 stdlib**：不 import torch／cv2，測試才會是瞬間的，也避免在量測進程之外先載
一次 torch 擾動待測環境（相依版本改用外呼 venv python 取得）。與 `tools/` 另外兩支一樣
不定位 repo 根、不讀 `settings`，全靠 CLI 參數與「在 repo 根執行」這句約束。

用法（**在 repo 根目錄執行**，用 `--package` 不用 `--directory`：`--directory` 會把 cwd
切進套件資料夾，而 `bucket_dir` 與 `OUTPUT_ROOT` 都是 cwd 相對路徑）：

    # 跑矩陣：{40 秒版, 2 分鐘版} × {batch 16, 8} × 2 次
    uv run --package video_analyze python video_analyze/tools/bench_e2e.py run \
        --buckets bucket_20260801_perf40,bucket_20260801_perf \
        --date 2026-08-01 \
        --batches 16,8 \
        --repeat 2

    # 落腳點對照組：跑第二次、寫進同一個產物目錄（report 本就按組態分組）
    uv run --package video_analyze python video_analyze/tools/bench_e2e.py run \
        --buckets bucket_20260801_perf40 --date 2026-08-01 --batches 16 \
        --foot-point bbox_bottom --repeat 2 --label attrib

    # 解析成明細表與分組統計
    uv run --package video_analyze python video_analyze/tools/bench_e2e.py report
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DEFAULT_RUNS_DIR = Path("outputs/bench_e2e")
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_CONFIG = Path("video_analyze/config.toml")
DEFAULT_FOOT_POINT = "head"
DEFAULT_REPEAT = 2
DEFAULT_LABEL = "main"
DEFAULT_EXCLUDE_PREFIXES = "smoke"
# `-s um` 是舊產物用的（只有 sm／mem／fb）；`pucm` 多帶功耗、溫度與時脈，之後要看
# 「掉頻了沒」才有資料。欄位隨 `-s` 而變，故實際用的指標集記進 meta，舊產物不要拿去
# 跟新的比對格式。
DEFAULT_GPU_METRICS = "pucm"
DEFAULT_GPU_INTERVAL_SECONDS = 2

# 被量測的指令。刻意寫死不開參數：換掉它就不是「量 video_analyze 的端到端」了。
RUN_COMMAND: tuple[str, ...] = ("uv", "run", "--package", "video_analyze", "video_analyze")

# 相依版本外呼 venv python 取得（見模組 docstring：本檔不 import torch）。
_VERSION_MODULES = ("torch", "tensorrt", "ultralytics")
_VERSION_PROBE = """
import importlib
for name in {modules!r}:
    try:
        print(f"{{name}}={{importlib.import_module(name).__version__}}")
    except Exception as exc:
        print(f"{{name}}=<import 失敗: {{type(exc).__name__}}>")
""".format(modules=_VERSION_MODULES)

# `AppConfig` 的四個頂層區塊名，即 pydantic-settings 的四段環境變數命名空間
# （`env_nested_delimiter="__"`）。本工具只設其中四個變數，但**整份環境都會被子進程
# 繼承**——環境裡先有一個 `INPUT__CAMERA_IDS`，量到的就是別的工作量。
SETTINGS_ENV_PREFIXES = ("INPUT__", "MODEL__", "FOOT_POINT__", "TRACKER__")
# 非 settings、但會改變量測結果的環境變數。**這份清單是列舉式的、不可能完備**——
# 影響效能的環境變數是開放集合（`OMP_NUM_THREADS`、`LD_LIBRARY_PATH`、
# `CUDA_LAUNCH_BLOCKING`…）。收在這裡的是已知會讓「同一個組態」量出不同數字、
# 而報表上完全看不出來的那幾個；不在清單上的仍會靜默影響結果。
RUNTIME_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS")
# 本工具自己會設的那四個。單一定義：`build_run_env` 設它們，`extra_settings_env` 扣掉
# 它們，兩邊漂移的話「繼承到什麼」就會漏報。
MANAGED_SETTINGS_ENV = ("INPUT__BUCKET_DIR", "INPUT__DATE", "MODEL__BATCH", "FOOT_POINT__METHOD")

_INFERENCE_COMPONENT = "inference"
_INFERENCE_MESSAGE = "FPS 整體"
_TRACK_WORKER_COMPONENT = "track_worker"
_TRACK_WORKER_MESSAGE = "追蹤進程結束"


# --------------------------------------------------------------------------------------
# FPS 口徑
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackWorkerReading:
    """追蹤進程**一片**結束時印的那行（`shards` 有幾片就有幾行）。

    `shard_id` 是那片的編號，分片（issue #141）之前的舊產物沒有這個欄位，為 `None`；
    `cameras` 是 issue #143 才加印的，更早的產物即使有 `shard_id` 也拿不到。
    `tracking_fps`／`frames` 同理，缺欄位就是 `None` 或空 tuple，不猜也不補。
    """

    shard_id: int | None
    overall_fps: float
    tracking_fps: float | None
    frames: int | None
    cameras: tuple[str, ...]

    @property
    def headroom(self) -> float | None:
        """這片的餘裕＝純處理速度 ÷ 實際吞吐（含等上游）。缺 `tracking_fps` 時為 None。"""
        if self.tracking_fps is None or self.overall_fps <= 0:
            return None
        return self.tracking_fps / self.overall_fps


@dataclass(frozen=True)
class FpsReading:
    """一輪 log 裡的推論口徑，加上逐片的追蹤口徑。

    **沒有泛用的 `fps` 欄位是刻意的**：兩個口徑的分母不同（推論進程的 wall clock 對
    追蹤進程的 wall clock），差 1–2%，取名 `fps` 就等於邀請呼叫端隨手取一個。
    `inference_frames`／`inference_elapsed_seconds` 只跟著推論那行走。

    追蹤側**存成逐片而不是先聚合成一個數**：整條 pipeline 的瓶頸是最慢的那片，先平均
    會把它蓋掉，而平均值看起來完全合理。要單一數字的呼叫端走
    `min_track_worker_fps`／`min_track_worker_headroom`，名字裡就帶著取最小值這件事。
    """

    inference_fps: float | None
    inference_frames: int | None
    inference_elapsed_seconds: float | None
    track_workers: tuple[TrackWorkerReading, ...]

    @property
    def completed(self) -> bool:
        """有推論那行才算跑完。追蹤值**不能**拿來遞補（會系統性高估）。"""
        return self.inference_fps is not None

    @property
    def shard_count(self) -> int:
        """印出結束行的片數。

        崩在半路的片走不到那行，這個數字因此可能少於 `[tracker].shards`。不在這裡擋：
        少一片時該輪的 `exit_status` 必然非 0，報表上已經有訊號。
        """
        return len(self.track_workers)

    @property
    def min_track_worker_fps(self) -> float | None:
        """最慢那片的 `overall_fps`；一片都沒有時為 None。"""
        if not self.track_workers:
            return None
        return min(reading.overall_fps for reading in self.track_workers)

    @property
    def min_track_worker_headroom(self) -> float | None:
        """最慢那片的餘裕；**只要有一片缺 `tracking_fps` 就回 None**。

        跳過缺值的片去取剩下幾片的最小值，偏誤方向剛好是「漏掉可能最慢的那片」，
        算出來的餘裕偏大——而這個數字是容量決策的依據，偏大正是會誤事的方向。
        """
        if not self.track_workers:
            return None
        headrooms = [reading.headroom for reading in self.track_workers]
        if any(headroom is None for headroom in headrooms):
            return None
        return min(headroom for headroom in headrooms if headroom is not None)


def parse_fps_log(text: str) -> FpsReading:
    """從一輪的 stdout log 取推論口徑，以及逐片的追蹤口徑。

    非 JSON 的行（uv 的下載進度、ultralytics 的橫幅）直接略過。

    **追蹤結束多行是正常的**（`[tracker].shards` 有幾片就有幾行，見 ADR-012），所以
    收集而不拋錯；分辨「多片」與「兩次執行寫進同一份 log」靠的是 `shard_id`——同一次
    執行的各片編號互異，重複即代表兩次執行，仍拋錯。分片之前的舊產物沒有 `shard_id`，
    那時每輪本來就只有一行，兩行 `None` 同樣算重複。推論那行沒有分片，重複一律拋錯，
    這個前提沒變。
    """
    inference: dict | None = None
    track_workers: list[TrackWorkerReading] = []
    seen_shard_ids: set[int | None] = set()
    for line in text.splitlines():
        if "overall_fps" not in line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        component = entry.get("component")
        message = entry.get("message")
        if component == _INFERENCE_COMPONENT and message == _INFERENCE_MESSAGE:
            if inference is not None:
                raise ValueError(
                    f"同一份 log 出現兩行推論進程的「{_INFERENCE_MESSAGE}」，"
                    "代表兩次執行寫進同一份 log；取其中一個會產生看似合理的數字。"
                )
            inference = entry
        elif component == _TRACK_WORKER_COMPONENT and message == _TRACK_WORKER_MESSAGE:
            raw_shard_id = entry.get("shard_id")
            shard_id = None if raw_shard_id is None else int(raw_shard_id)
            if shard_id in seen_shard_ids:
                raise ValueError(
                    f"同一份 log 出現兩行 shard_id={shard_id} 的「{_TRACK_WORKER_MESSAGE}」，"
                    "代表兩次執行寫進同一份 log（同一次執行的各片編號互異）。"
                )
            seen_shard_ids.add(shard_id)
            raw_frames = entry.get("frames")
            raw_tracking_fps = entry.get("tracking_fps")
            track_workers.append(
                TrackWorkerReading(
                    shard_id=shard_id,
                    overall_fps=float(entry["overall_fps"]),
                    tracking_fps=(
                        None if raw_tracking_fps is None else float(raw_tracking_fps)
                    ),
                    frames=None if raw_frames is None else int(raw_frames),
                    cameras=tuple(entry.get("owned_cameras") or ()),
                )
            )
    return FpsReading(
        inference_fps=None if inference is None else float(inference["overall_fps"]),
        inference_frames=None if inference is None else int(inference["total_frames"]),
        inference_elapsed_seconds=(
            None if inference is None else float(inference["elapsed_seconds"])
        ),
        track_workers=tuple(
            sorted(
                track_workers,
                key=lambda reading: (reading.shard_id is None, reading.shard_id),
            )
        ),
    )


def _format_track_summary(reading: FpsReading) -> str:
    """把逐片追蹤讀數縮成摘要列的一段：最慢那片的吞吐、片數與最小餘裕。

    取最小不取平均：整條 pipeline 的瓶頸是最慢的那片，平均會把它蓋掉，而平均值看起來
    完全合理。片數一起印，是因為「只有一片」與「多片中最慢的一片」是不同的數字；最慢那片
    負責的攝影機也印出來——分片分組是執行當下依路數與 `shards` 算出來的，不印就得回頭翻
    log 才知道是哪幾路擠在一起。

    分片之前的舊 log 沒有 `shard_id`（連帶沒有 `owned_cameras`）、更舊的沒有
    `tracking_fps`，缺哪個就少印哪一段，不以 `None` 硬套格式。
    """
    if not reading.track_workers:
        return "追蹤（最小）－"
    slowest = min(reading.track_workers, key=lambda item: item.overall_fps)
    parts = [f"{reading.shard_count} 片"]
    if slowest.shard_id is not None:
        parts.append(f"最慢 shard {slowest.shard_id}")
    headroom = reading.min_track_worker_headroom
    parts.append(f"最小餘裕 {headroom:.2f}×" if headroom is not None else "餘裕缺值")
    # 攝影機清單擺在最後、與前面的段落用分號隔開：它本身也用頓號分隔，混在同一串裡
    # 會讀不出哪裡是一段的邊界
    cameras = (
        f"；shard {slowest.shard_id}＝{'、'.join(slowest.cameras)}"
        if slowest.shard_id is not None and slowest.cameras
        else ""
    )
    return f"追蹤（最小）{slowest.overall_fps:.2f} 張/秒（{'、'.join(parts)}{cameras}）"

# --------------------------------------------------------------------------------------
# 產物解析
# --------------------------------------------------------------------------------------


def parse_meta(text: str) -> dict[str, str]:
    """`key=value` 逐行的 meta。值裡可以有 `=`（只切第一個），沒有 `=` 的行略過。"""
    return dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line
    )


def format_meta(pairs: Mapping[str, object]) -> str:
    """`parse_meta` 的反向。值一律 `str()`，讓 meta 純粹是文字檔。"""
    return "".join(f"{key}={value}\n" for key, value in pairs.items())


@dataclass(frozen=True)
class GpuUsage:
    """`nvidia-smi dmon` 取樣的摘要。一列資料都沒有時三個統計值皆為 None。

    `samples` 為 0 要讓它在產物裡看得見，不要變成安靜的 0%——取樣沒跑起來與 GPU 真的
    閒置，在表上長得一模一樣。
    """

    samples: int
    mean_sm_percent: float | None
    max_sm_percent: float | None
    max_fb_mb: float | None


def parse_gpu_log(text: str) -> GpuUsage:
    """依**表頭找欄位**而非固定索引：`dmon -s` 給不同指標集，欄位數與順序都會變。

    第一行 `#` 開頭的是欄名（`#Time gpu pwr ... sm mem ... fb ...`），第二行是單位，
    略過。取不到的值（`mtemp` 在部分卡是 `-`）逐格跳過，不讓整列作廢。
    """
    header: list[str] | None = None
    sm_values: list[float] = []
    fb_values: list[float] = []
    samples = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if header is None:
                header = stripped.lstrip("#").split()
            continue
        if header is None:
            continue
        fields = stripped.split()
        if len(fields) != len(header):
            continue
        samples += 1
        row = dict(zip(header, fields, strict=True))
        for column, sink in (("sm", sm_values), ("fb", fb_values)):
            raw = row.get(column)
            if raw is None:
                continue
            try:
                sink.append(float(raw))
            except ValueError:
                continue
    return GpuUsage(
        samples=samples,
        mean_sm_percent=statistics.mean(sm_values) if sm_values else None,
        max_sm_percent=max(sm_values) if sm_values else None,
        max_fb_mb=max(fb_values) if fb_values else None,
    )


@dataclass(frozen=True)
class RunRecord:
    """一輪量測的產物：meta ＋ 兩個口徑的 FPS ＋ GPU 取樣摘要。"""

    name: str
    meta: Mapping[str, str]
    fps_reading: FpsReading
    gpu: GpuUsage | None

    @property
    def succeeded(self) -> bool:
        return self.meta.get("exit_status") == "0"


def load_run_records(runs_dir: Path) -> tuple[list[RunRecord], list[str]]:
    """讀 `runs_dir` 底下每個 `<name>.meta` 及其同名 `.log`／`.gpu.log`。

    回傳 (讀得起來的 records, 每份壞產物一行的說明)。**一份壞掉的產物不該讓整份報表
    印不出來**——`parse_fps_log` 對重複的推論行刻意拋 `ValueError`，而那個訊息裡沒有
    檔名；子進程被 SIGKILL 截斷多位元組字元則會 `UnicodeDecodeError`。兩者原本都會
    讓 `report` 整個掛掉，而使用者無從得知是哪一輪的產物有問題（CLAUDE.md 記過同一型
    的坑：沒有檔名線索的例外）。所以逐檔隔離，壞的跳過但**大聲說出來**，不靜默丟棄。
    """
    records: list[RunRecord] = []
    broken: list[str] = []
    for meta_path in sorted(runs_dir.glob("*.meta")):
        log_path = meta_path.with_suffix(".log")
        if not log_path.exists():
            continue
        gpu_path = meta_path.with_suffix(".gpu.log")
        try:
            records.append(
                RunRecord(
                    name=meta_path.stem,
                    meta=parse_meta(meta_path.read_text(encoding="utf-8")),
                    fps_reading=parse_fps_log(log_path.read_text(encoding="utf-8")),
                    gpu=(
                        parse_gpu_log(gpu_path.read_text(encoding="utf-8"))
                        if gpu_path.exists()
                        else None
                    ),
                )
            )
        except (ValueError, UnicodeDecodeError) as error:
            broken.append(f"{meta_path.stem}：{type(error).__name__}: {error}")
    return records, broken


@dataclass(frozen=True)
class GroupStat:
    """同組態多次量測的平均與離散度。**只有推論口徑**，見模組 docstring 第 2 道互鎖。"""

    bucket: str
    batch: str
    foot_point_method: str
    extra_settings_env: str
    commit: str
    date: str
    label: str
    machine: str
    runs: int
    mean_inference_fps: float
    spread_percent: float
    values: tuple[float, ...]


def _inference_fps_values(records: Iterable[RunRecord]) -> list[float]:
    """分組統計唯一的取值入口——追蹤進程的值進不到這裡。

    過濾條件與 `group_runs` 收成員用的 `FpsReading.completed` 必須是同一個
    （`is not None`）：用真值判斷的話 `inference_fps == 0.0` 會被這裡濾掉、卻被
    `completed` 收進來，該組只有那一筆時 `statistics.mean([])` 直接拋 `StatisticsError`。
    """
    return [
        r.fps_reading.inference_fps
        for r in records
        if r.fps_reading.inference_fps is not None
    ]


def group_runs(
    records: Iterable[RunRecord], exclude_prefixes: Sequence[str] = ()
) -> list[GroupStat]:
    """按「組態」分組。未完成、非零 exit、排除前綴的 run 不進統計。

    **「組態」不只是 bucket／batch／落腳點**：`commit`／`date`／`label`／`machine` 一併
    進 key。少了它們，這支工具最主要的用途就會靜默失效——用 `--label before` 跑一個
    commit、`--label after` 跑另一個、寫進同一個 `--runs-dir`，四輪會被平均成一組
    `n=4`，**要找的回歸剛好被抹平**。同理跨測試片日期與跨機器的數字也不該混在一起。

    `git_dirty` 刻意**不**進 key：它不區分「未追蹤檔」與「已修改的程式碼」，把工作目錄
    的雜訊變成分組維度只會讓同一份程式碼的多輪散成好幾組，卻不帶任何資訊。
    舊產物沒有 `date`／`label` 鍵，`.get` 的預設值讓它們照舊分在一起。
    """
    groups: dict[tuple[str, ...], list[RunRecord]] = {}
    for record in records:
        if not record.fps_reading.completed or not record.succeeded:
            continue
        if any(record.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        key = (
            record.meta.get("bucket", "?"),
            record.meta.get("model_batch", "?"),
            record.meta.get("foot_point_method", "?"),
            # 繼承到的 settings 環境變數也是組態的一部分（見 `extra_settings_env`）。
            # 舊產物沒這個鍵，預設 "{}" 讓它們照舊分在一起。
            record.meta.get("extra_settings_env", "{}"),
            record.meta.get("commit", "?"),
            record.meta.get("date", "?"),
            record.meta.get("label", "?"),
            record.meta.get("machine", "?"),
        )
        groups.setdefault(key, []).append(record)

    stats: list[GroupStat] = []
    for (bucket, batch, foot, extra_env, commit, day, label, machine), members in sorted(
        groups.items()
    ):
        values = _inference_fps_values(members)
        mean = statistics.mean(values)
        # `mean` 為 0 不是異常輸入：`FpsMeter._safe_div` 在 total_frames == 0 時就回 0.0，
        # 同組兩輪都是 0.0 時這裡除下去會讓整份 report 拋 ZeroDivisionError
        spread = (
            (max(values) - min(values)) / mean * 100 if len(values) > 1 and mean else 0.0
        )
        stats.append(
            GroupStat(
                bucket=bucket,
                batch=batch,
                foot_point_method=foot,
                extra_settings_env=extra_env,
                commit=commit,
                date=day,
                label=label,
                machine=machine,
                runs=len(values),
                mean_inference_fps=mean,
                spread_percent=spread,
                values=tuple(values),
            )
        )
    return stats


# --------------------------------------------------------------------------------------
# 矩陣展開與執行環境
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchCase:
    """矩陣裡的一格。`name` 同時是該輪四個產物檔的 stem。"""

    name: str
    bucket: str
    batch: int
    foot_point_method: str
    repeat_index: int


def expand_matrix(
    buckets: Sequence[str],
    batches: Sequence[int],
    foot_point_method: str,
    repeat: int,
    date: str,
    label: str = DEFAULT_LABEL,
) -> list[BenchCase]:
    """展開矩陣，**重複次數在最外層**。

    這不是寫法偏好而是量測性質：同組態連跑會把熱漂移集中在單一格，離散度那欄就失真了
    ——第一次跑的都是冷的、第二次跑的都是熱的，反而讓機器狀態被平均掉。

    **`name` 含分析日期**（`bucket` 名稱裡的日期是資料集命名，兩者不是同一件事）：多日
    bucket 換 `--date` 重跑時，少了它兩輪的名稱完全相同——不是誤報「產物已存在」擋下你，
    就是配上 `--overwrite` 直接毀掉前一天的產物。
    """
    cases: list[BenchCase] = []
    for repeat_index in range(1, repeat + 1):
        for bucket in buckets:
            for batch in batches:
                tag = bucket.removeprefix("bucket_")
                cases.append(
                    BenchCase(
                        name=(
                            f"{label}_{tag}_d{date.replace('-', '')}"
                            f"_b{batch}_{foot_point_method}_r{repeat_index}"
                        ),
                        bucket=bucket,
                        batch=batch,
                        foot_point_method=foot_point_method,
                        repeat_index=repeat_index,
                    )
                )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        # 撞名的兩格會在矩陣**執行中途**互相覆蓋產物，而開跑前的既存檔檢查抓不到
        # （第一格是它自己寫出來的）。`--buckets a,a` 與 `--buckets bucket_x,x`
        # （去前綴後同 tag）都會走到這裡。
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"矩陣有撞名的格子：{duplicates}；請檢查 --buckets 是否重複或去前綴後同名")
    return cases


def build_run_env(base: Mapping[str, str], case: BenchCase, date: str) -> dict[str, str]:
    """疊上 pydantic-settings 的四個命名空間。值一律字串（`os.environ` 只收字串）。"""
    managed = {
        "INPUT__BUCKET_DIR": case.bucket,
        "INPUT__DATE": date,
        "MODEL__BATCH": str(case.batch),
        "FOOT_POINT__METHOD": case.foot_point_method,
    }
    return {**base, **managed}


def extra_settings_env(env: Mapping[str, str], managed: Iterable[str]) -> str:
    """把「本工具沒設、卻會改變這輪量到什麼」的環境變數收成一行 JSON。

    這條與 FPS 口徑同級：環境裡有一個 `INPUT__CAMERA_IDS`，那輪量的就是別的工作量，
    而 FPS 表上完全看不出來。所以不只記進 meta，還**進 `group_runs` 的分組 key**——
    在不同環境下跑的兩輪，結構上就不可能被平均在一起。

    收兩類：`SETTINGS_ENV_PREFIXES` 開頭的 pydantic-settings 變數，以及
    `RUNTIME_ENV_KEYS` 列的執行環境變數（`CUDA_VISIBLE_DEVICES` 決定跑哪張卡——
    連 GPU 取樣取的是不是同一張都取決於它）。

    **比對不分大小寫**：pydantic-settings 預設 case-insensitive，`input__camera_ids`
    小寫一樣會生效；只比對大寫等於留一個繞過整套防護的後門。

    ⚠ **這不是完備的**：影響量測的環境變數是開放集合，`RUNTIME_ENV_KEYS` 只收已知
    重要的那幾個。不在其中的變數仍會靜默改變數字。

    回傳 JSON 而非 `K=V;K=V`：meta 是逐行 `key=value`，值裡有換行會把整份格式撐破。
    """
    managed_keys = {key.upper() for key in managed}
    extras = {
        key: value
        for key, value in env.items()
        if key.upper() not in managed_keys
        and (key.upper().startswith(SETTINGS_ENV_PREFIXES) or key.upper() in RUNTIME_ENV_KEYS)
    }
    return json.dumps(extras, ensure_ascii=False, sort_keys=True)


def check_name_component(value: str, flag: str) -> str:
    """驗證會進檔名或 meta 的 CLI 字串。

    `--label` 直接是產物檔 stem 的一部分：`--label '../<bucket>/x'` 會把產物寫進每輪
    開頭 `rmtree` 的目錄底下，`bucket_output_dir` 的逃逸檢查與 `check_runs_dir_disjoint`
    兩道都攔不到（它們看的是 bucket 與 runs-dir，不是 label），矩陣跑完只剩最後一輪。
    `bucket` 有整套檢查而 label 沒有，是不一致——這裡補齊。

    換行另外擋：meta 是逐行 `key=value`，值裡有換行會把整份格式撐破。
    """
    if not value or value != value.strip():
        raise SystemExit(f"{flag} 不可為空或前後帶空白：{value!r}")
    if "/" in value or "\\" in value or ".." in value:
        raise SystemExit(f"{flag} 不可含路徑分隔或 `..`（它會進產物檔名）：{value!r}")
    if any(char in value for char in "\r\n"):
        raise SystemExit(f"{flag} 不可含換行（meta 是逐行 key=value）：{value!r}")
    return value


def bucket_output_dir(bucket: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """算出每輪開頭要清空的目錄，並擋下逃逸。

    這個路徑會被 `shutil.rmtree`，而 bucket 來自 CLI 字串：含 `/` 或 `..`、或解析後
    不在 `output_root` 底下（例如 `outputs/<bucket>` 本身是指向別處的 symlink），一律中止。

    **`output_root` 刻意沒有對應的 CLI 旗標**（只留參數給測試注入）：被量測的進程寫到
    哪裡是由 `config/constants.py` 的 `OUTPUT_ROOT = Path("outputs")` 決定，那是常數、
    沒有環境變數可以覆寫。開一個旗標只會讓工具清 `<root>/<bucket>`、子進程仍寫
    `outputs/<bucket>`，每輪清空與 parquet 複製同時靜默失效。要讓它真的可調得先改
    `src/` 的常數，那是另一件事。
    """
    if not bucket or bucket != bucket.strip():
        raise SystemExit("bucket 名稱不可為空或帶前後空白")
    if "/" in bucket or "\\" in bucket or ".." in bucket:
        raise SystemExit(f"bucket 名稱 {bucket!r} 含路徑分隔或 `..`；只接受單一目錄名")
    candidate = (output_root / bucket).resolve()
    root = output_root.resolve()
    if candidate == root or root not in candidate.parents:
        raise SystemExit(f"{candidate} 不在 {root} 底下，拒絕清空")
    return candidate


def check_runs_dir_disjoint(runs_dir: Path, cleared_dirs: Iterable[Path]) -> None:
    """產物目錄不可落在每輪會被清空的目錄底下，否則產物會自己刪自己。"""
    runs = runs_dir.resolve()
    for cleared in cleared_dirs:
        if runs == cleared or cleared in runs.parents:
            raise SystemExit(
                f"產物目錄 {runs_dir} 位於每輪開頭會被清空的 {cleared} 底下；"
                "請改用 --runs-dir 指到別處（預設 outputs/bench_e2e 就是分開的）。"
            )


def shell_exit_code(wait_status: int) -> int:
    """把 `os.wait4` 的原始 status word 轉成 shell 慣例，讓舊產物的 `exit_status` 對得上。

    訊號終止記 `128+N`（`SIGKILL` → 137），與 bash 的 `$?` 一致；舊腳本用的是
    `/usr/bin/time` ＋ `$?`，兩邊的數字要能直接比較。
    """
    if os.WIFSIGNALED(wait_status):
        return 128 + os.WTERMSIG(wait_status)
    if os.WIFEXITED(wait_status):
        return os.WEXITSTATUS(wait_status)
    return -1


# --------------------------------------------------------------------------------------
# 執行
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceUsage:
    """一輪的資源用量。欄位名帶單位，因為 `ru_*` 的單位是平台相關的陷阱。

    `max_rss_kb` 是 `ru_maxrss`：進程樹中**最大的單一進程**，不是總和。vfa 是多進程
    （推論、追蹤、各路解碼），這個數字不等於整體記憶體足跡。
    """

    exit_status: int
    wall_seconds: float
    max_rss_kb: int
    user_seconds: float
    system_seconds: float
    cpu_percent: int
    major_page_faults: int
    minor_page_faults: int
    voluntary_context_switches: int
    involuntary_context_switches: int
    fs_inputs: int
    fs_outputs: int


def run_measured(command: Sequence[str], env: Mapping[str, str], log_path: Path) -> ResourceUsage:
    """跑一輪並收資源用量。

    用 `os.wait4` 而非 `Popen.wait()` ＋ `getrusage(RUSAGE_CHILDREN)`（見模組 docstring）。
    **不可在此之前呼叫 `wait()`／`communicate()`**：那會先回收子進程，`wait4` 隨即拋
    `ChildProcessError`。`returncode` 因此要手動指派，否則 `Popen.__del__` 會抱怨。
    """
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command), stdout=log_file, stderr=subprocess.STDOUT, env=dict(env)
        )
        _, wait_status, usage = os.wait4(process.pid, 0)
    wall_seconds = time.monotonic() - started
    process.returncode = shell_exit_code(wait_status)
    cpu_seconds = usage.ru_utime + usage.ru_stime
    return ResourceUsage(
        exit_status=process.returncode,
        wall_seconds=wall_seconds,
        max_rss_kb=int(usage.ru_maxrss),
        user_seconds=usage.ru_utime,
        system_seconds=usage.ru_stime,
        cpu_percent=round(cpu_seconds / wall_seconds * 100) if wall_seconds > 0 else 0,
        major_page_faults=int(usage.ru_majflt),
        minor_page_faults=int(usage.ru_minflt),
        voluntary_context_switches=int(usage.ru_nvcsw),
        involuntary_context_switches=int(usage.ru_nivcsw),
        fs_inputs=int(usage.ru_inblock),
        fs_outputs=int(usage.ru_oublock),
    )


@contextlib.contextmanager
def gpu_sampler(
    log_path: Path, metrics: str, interval_seconds: int, enabled: bool
) -> Iterator[None]:
    """背景跑 `nvidia-smi dmon`，`finally` 收尾。

    `finally` 而非結尾直接 kill：例外、`SystemExit`、`KeyboardInterrupt` 都要收得掉，
    等價於原本 shell 的 `trap ... EXIT`。留著不收會在矩陣跑到一半中斷時堆出一串孤兒。
    """
    if not enabled:
        yield
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            # `-i 0`：多 GPU 主機上 dmon 每個取樣間隔會每張卡各吐一列，`parse_gpu_log`
            # 不看 gpu 欄，於是 samples 膨脹 ×N、閒置卡被平均進 SM 使用率（90% + 0% → 45）。
            # ⚠ 本機只有單卡，這條是依 dmon 語義做的防禦，沒有在多卡機器上實測過。
            ["nvidia-smi", "dmon", "-i", "0", "-s", metrics,
             "-d", str(interval_seconds), "-o", "T"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            yield
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _probe_versions(python: Path | str) -> dict[str, str]:
    completed = subprocess.run(
        [str(python), "-c", _VERSION_PROBE], capture_output=True, text=True, check=False
    )
    return parse_meta(completed.stdout)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def _host_cpu(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str:
    """這輪跑在什麼 CPU 上（`/proc/cpuinfo` 的第一個 `model name`）。

    2026-08-26 在 T4 上量到同組態跨天差 7.9%，最可能的解釋是 VM 重開機換了實體主機——
    但**事後補不回來**：meta 沒有任何一欄記得住那輪跑在哪台機器上，VM 一關就查不到
    （`gcloud describe` 對 TERMINATED 的機器回 `Unknown CPU Platform`）。`--machine` 是
    人給的代號，同一個代號跨天可能落在不同世代的主機上。

    雲端客體看到的粒度很粗（「Intel(R) Xeon(R) CPU @ 2.30GHz」這種，SKU 被抹掉了），
    但足以分辨世代，也就足以判斷「這兩輪能不能相減」。讀不到就記 `<未知>`——這是給人
    判讀用的線索，不是量測的前提條件。
    """
    try:
        text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<未知>"
    for line in text.splitlines():
        if line.startswith("model name"):
            _, _, value = line.partition(":")
            if value.strip():
                return " ".join(value.split())
    return "<未知>"


def _model_classes(config_path: Path, env: Mapping[str, str]) -> str:
    """回傳這輪實際生效的 `[model].classes`（JSON 字串）。

    與 `_engine_path` 同一個理由要記進 meta：偵測幾個類別（`[0, 2]` 對 `[2]`）是功能
    差異，而 FPS 表上完全看不出來——同一支工具、同一顆引擎，少偵測一個類別就是比較
    快，事後沒有這個欄位就分不出「改動變快了」與「量的東西變少了」。環境變數
    `MODEL__CLASSES` 會覆寫設定檔（pydantic-settings 對複合型別吃 JSON），所以先看它。
    """
    override = env.get("MODEL__CLASSES")
    if override is not None:
        # 設了卻是空白＝誤用。回退到設定檔的話 meta 會記下一組沒被量到的類別，正是這欄
        # 要防的事；空字串更糟——產物看起來完整，只有這欄默默沒有內容
        if not override.strip():
            raise SystemExit(
                "環境變數 MODEL__CLASSES 是空的，量測記不下偵測類別"
                "（要用設定檔的值就不要設這個變數）"
            )
        return override.strip()
    if not config_path.is_file():
        raise SystemExit(f"找不到設定檔 {config_path}（本工具要在 repo 根目錄執行）")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    classes = config.get("model", {}).get("classes")
    if not classes:
        raise SystemExit(f"{config_path} 的 [model].classes 是空的，量測記不下偵測類別")
    return json.dumps(classes)


def _engine_path(config_path: Path, env: Mapping[str, str]) -> tuple[str, str]:
    """回傳 (引擎路徑, 這個值是從哪來的)。

    **以子進程實際會載入的那顆為準，不是設定檔寫什麼就記什麼**：環境裡的
    `MODEL__MODEL_PATH` 會覆寫 `config.toml`，只讀設定檔的話 meta 會記下一顆從未被
    量測過的引擎的 sha256——而引擎身分正是這份 meta 最需要可信的欄位之一。
    """
    override = env.get("MODEL__MODEL_PATH")
    if override is not None:
        # 設了卻是空白＝誤用。回退到設定檔的話 meta 會記下一顆從未被量測過的引擎，
        # 正是這欄要防的事；空字串更糟——`Path("").is_file()` 一定是 False，
        # sha256 會被記成「找不到引擎檔」，看起來像是真的量到卻找不到引擎檔案。
        if not override.strip():
            raise SystemExit(
                "環境變數 MODEL__MODEL_PATH 是空的，量測記不下引擎身分"
                "（要用設定檔的值就不要設這個變數）"
            )
        return override.strip(), "env:MODEL__MODEL_PATH"
    if not config_path.is_file():
        raise SystemExit(f"找不到設定檔 {config_path}（本工具要在 repo 根目錄執行）")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    model_path = config.get("model", {}).get("model_path")
    if not model_path:
        raise SystemExit(f"{config_path} 的 [model].model_path 是空的，量測記不下引擎身分")
    return str(model_path), f"config:{config_path}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_segments(bucket: str, day: date) -> int:
    """數 `<bucket>/<攝影機目錄>/<YYYY>/<MM>/<DD>/` 底下的檔案數。

    兩件事刻意不做：不限定副檔名（registry 的 `storage.file_ext` 可以是 mkv 以外的
    值，寫死 `.mkv` 會讓別的 bucket 靜默記成 0），也不遞迴整個 bucket（多日 bucket
    會高估數倍——被量測的只有 `--date` 那一天）。

    這是給人看的追溯欄位，不是契約：它數的是 bucket 裡**所有**攝影機目錄，而該輪實際
    讀了哪幾路由 registry 與 `INPUT__CAMERA_IDS` 決定（後者記在 `extra_settings_env`）。
    """
    day_glob = f"*/{day:%Y}/{day:%m}/{day:%d}/*"
    return sum(1 for path in Path(bucket).glob(day_glob) if path.is_file())


def artifact_paths(runs_dir: Path, name: str) -> dict[str, Path]:
    """一輪產出的四個檔。單一定義：碰撞檢查、清除、寫入三處都從這裡拿。"""
    return {
        "meta": runs_dir / f"{name}.meta",
        "log": runs_dir / f"{name}.log",
        "gpu": runs_dir / f"{name}.gpu.log",
        "parquet": runs_dir / f"{name}.parquet",
    }


def clear_artifacts(runs_dir: Path, name: str) -> None:
    """覆蓋前把**整組**產物刪掉。

    只靠寫入時截斷是不夠的：`.log`／`.meta` 每輪都會重寫，但 `.gpu.log` 在
    `--no-gpu-log` 時根本不開檔、`.parquet` 在該輪沒寫出 parquet 時也不會被碰。
    留著就是上一輪的產物配這一輪的 meta——正是這支工具宣稱要防的錯配。
    """
    for path in artifact_paths(runs_dir, name).values():
        path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    batches = [int(item) for item in args.batches.split(",") if item.strip()]
    if not buckets or not batches:
        raise SystemExit("--buckets 與 --batches 都不可為空")
    if args.repeat < 1:
        raise SystemExit("--repeat 至少為 1")
    try:
        # 這個字串同時是 `INPUT__DATE` 與輸出目錄名（`outputs/<bucket>/<date>/`），
        # 格式鬆掉的話 parquet 會複製不到而工具不會有任何訊號。
        day = date.fromisoformat(args.date)
    except ValueError as exc:
        raise SystemExit(f"--date 要是 YYYY-MM-DD：{exc}") from exc
    if f"{day:%Y-%m-%d}" != args.date:
        raise SystemExit(f"--date 要正規化成 YYYY-MM-DD（收到 {args.date!r}）")
    if args.gpu_log and shutil.which("nvidia-smi") is None:
        raise SystemExit(
            "找不到 nvidia-smi。這是 GPU 端到端量測工具，少了 SM 取樣就做不了"
            "「顯卡／CPU 拆分」那項分析，而 FPS 表上完全看不出來；"
            "確定要在沒有取樣的情況下量，請明講 --no-gpu-log。"
        )

    check_name_component(args.label, "--label")
    check_name_component(args.machine, "--machine")

    cleared_dirs = [bucket_output_dir(bucket) for bucket in buckets]
    check_runs_dir_disjoint(args.runs_dir, cleared_dirs)
    cleared_by_bucket = dict(zip(buckets, cleared_dirs, strict=True))

    cases = expand_matrix(
        buckets, batches, args.foot_point, args.repeat, args.date, args.label
    )
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [
            path
            for case in cases
            for path in artifact_paths(args.runs_dir, case.name).values()
            if path.exists()
        ]
        if existing:
            raise SystemExit(
                f"產物已存在（{len(existing)} 個檔，例如 {existing[0]}）。"
                "覆蓋請明講 --overwrite——靜默覆蓋正是「這輪的 meta 配上一輪的 log」的來源。"
            )

    engine, engine_source = _engine_path(args.config, os.environ)
    engine_sha256 = _sha256(Path(engine)) if Path(engine).is_file() else "<找不到引擎檔>"
    model_classes = _model_classes(args.config, os.environ)
    host_cpu = _host_cpu()
    versions = _probe_versions(sys.executable)
    commit = _git_output("rev-parse", "HEAD")
    # 逐層打 tag 的量測（每層只差一項改動）光看 commit sha 判讀不出是哪一層；沒打 tag
    # 的一般執行留一個看得出來的值，不要讓這欄變成空字串
    layer = _git_output("describe", "--tags", "--exact-match", "HEAD") or "<未打 tag>"
    git_dirty = len(_git_output("status", "--porcelain").splitlines())
    segments = {bucket: _count_segments(bucket, day) for bucket in buckets}
    inherited_env = extra_settings_env(os.environ, MANAGED_SETTINGS_ENV)
    if inherited_env != "{}":
        # 不擋下來——量子集有時就是本意；但它進 meta 也進分組 key，不會被靜默平均掉
        print(f"⚠ 環境裡另有會生效的 settings 變數，已記進 meta 與分組 key：{inherited_env}")

    print(
        f"矩陣共 {len(cases)} 輪："
        f"{len(buckets)} bucket × {len(batches)} batch × {args.repeat} 次"
    )
    worst_status = 0
    for index, case in enumerate(cases, start=1):
        paths = artifact_paths(args.runs_dir, case.name)
        print(f"=== [{index}/{len(cases)}] {case.name} ===", flush=True)

        clear_artifacts(args.runs_dir, case.name)

        # 每輪都從乾淨的輸出開始：舊 parquet 留著會讓「這輪到底寫了幾列」失去意義
        cleared = cleared_by_bucket[case.bucket]
        if cleared.exists():
            shutil.rmtree(cleared)

        meta: dict[str, object] = {
            "name": case.name,
            "label": args.label,
            "machine": args.machine,
            "host_cpu": host_cpu,
            "codebase": "vfa-main",
            "layer": layer,
            "commit": commit,
            "git_dirty": git_dirty,
            "bucket": case.bucket,
            "date": args.date,
            "model_batch": case.batch,
            "model_classes": model_classes,
            "foot_point_method": case.foot_point_method,
            "repeat_index": case.repeat_index,
            "extra_settings_env": inherited_env,
            "engine": engine,
            "engine_sha256": engine_sha256,
            # 引擎身分是從設定檔還是環境變數來的：兩者不一致時，記下哪一個真的生效了
            "engine_source": engine_source,
            **versions,
            "segments": segments[case.bucket],
            "gpu_metrics": args.gpu_metrics if args.gpu_log else "<未取樣>",
            "gpu_sample_seconds": args.gpu_interval if args.gpu_log else "<未取樣>",
            "command": " ".join(RUN_COMMAND),
            "started": _now(),
        }
        paths["meta"].write_text(format_meta(meta), encoding="utf-8")

        env = build_run_env(os.environ, case, args.date)
        with gpu_sampler(paths["gpu"], args.gpu_metrics, args.gpu_interval, args.gpu_log):
            usage = run_measured(RUN_COMMAND, env, paths["log"])

        gpu = (
            parse_gpu_log(paths["gpu"].read_text(encoding="utf-8")) if args.gpu_log else None
        )
        tail: dict[str, object] = {
            "finished": _now(),
            "exit_status": usage.exit_status,
            "wall_seconds": round(usage.wall_seconds, 2),
            "max_rss_kb": usage.max_rss_kb,
            "user_seconds": round(usage.user_seconds, 2),
            "system_seconds": round(usage.system_seconds, 2),
            "cpu_percent": usage.cpu_percent,
            "major_page_faults": usage.major_page_faults,
            "minor_page_faults": usage.minor_page_faults,
            "voluntary_context_switches": usage.voluntary_context_switches,
            "involuntary_context_switches": usage.involuntary_context_switches,
            "fs_inputs": usage.fs_inputs,
            "fs_outputs": usage.fs_outputs,
        }
        if gpu is not None:
            # 取樣一列都沒有時 samples=0 要留在產物裡，別讓它變成安靜的 0%
            tail["gpu_samples"] = gpu.samples
        with paths["meta"].open("a", encoding="utf-8") as handle:
            handle.write(format_meta(tail))

        parquet = cleared / args.date / "tracking_results.parquet"
        if parquet.is_file():
            shutil.copy2(parquet, paths["parquet"])
            with paths["meta"].open("a", encoding="utf-8") as handle:
                handle.write(format_meta({"parquet_bytes": parquet.stat().st_size}))

        reading = parse_fps_log(paths["log"].read_text(encoding="utf-8"))
        if reading.completed:
            print(
                f"推論 {reading.inference_fps:.2f} 張/秒"
                f"（{reading.inference_frames} 格 / {reading.inference_elapsed_seconds:.1f} 秒）"
                f"、{_format_track_summary(reading)}"
                f"、wall {usage.wall_seconds:.1f} 秒、max RSS "
                f"{usage.max_rss_kb / 1024:.0f} MB、exit {usage.exit_status}"
            )
        else:
            print(f"未完成：log 裡沒有推論進程的「{_INFERENCE_MESSAGE}」，exit {usage.exit_status}")
        worst_status = max(worst_status, usage.exit_status)

    print(f"矩陣完成，產物在 {args.runs_dir}")
    return worst_status


def command_report(args: argparse.Namespace) -> int:
    if not args.runs_dir.is_dir():
        raise SystemExit(f"找不到產物目錄 {args.runs_dir}")
    records, broken = load_run_records(args.runs_dir)
    if broken:
        # 印在表格之前而不是之後：讀者要先知道這份報表少了哪幾輪
        print(f"⚠ 有 {len(broken)} 份產物讀不起來，未列入下表：")
        for line in broken:
            print(f"  {line}")
    if not records:
        raise SystemExit(f"{args.runs_dir} 底下沒有成對的 .meta／.log")
    exclude = [item.strip() for item in args.exclude_prefixes.split(",") if item.strip()]

    # 欄寬依實際名稱長度，不寫死：run 名稱含 label／日期／bucket tag，寫死的話
    # 稍長一點的組合就會把整張表撐歪
    width = max([len(record.name) for record in records] + [len("run")])
    # 追蹤側三欄都是「最慢那片」的口徑，欄名短到看不出來，所以在表頭上方寫一次
    print(
        "（infer_fps＝推論進程口徑，即報表的每秒張數；trk_min＝各追蹤片 overall_fps 的"
        "最小值、shd＝片數、hdrm＝各片 tracking_fps÷overall_fps 的最小值）"
    )
    print(
        f"{'run':<{width}} {'bucket':<18} {'b':<3} {'foot':<12} "
        f"{'infer_fps':>9} {'trk_min':>9} {'shd':>3} {'hdrm':>5} {'sec':>6} {'frames':>8} "
        f"{'rss_MB':>7} {'sm%':>5} {'exit':>4}"
    )
    for record in records:
        meta = record.meta
        reading = record.fps_reading
        if not reading.completed:
            print(f"{record.name:<{width}} 未完成或失敗（exit {meta.get('exit_status', '?')}）")
            continue
        rss_kb = meta.get("max_rss_kb")
        rss = f"{int(rss_kb) / 1024:.0f}" if rss_kb else "-"
        sm = (
            f"{record.gpu.mean_sm_percent:.0f}"
            if record.gpu is not None and record.gpu.mean_sm_percent is not None
            else "-"
        )
        track_min = reading.min_track_worker_fps
        track = f"{track_min:.2f}" if track_min is not None else "-"
        shards = str(reading.shard_count) if reading.track_workers else "-"
        headroom_min = reading.min_track_worker_headroom
        headroom = f"{headroom_min:.2f}" if headroom_min is not None else "-"
        print(
            f"{record.name:<{width}} {meta.get('bucket', '?').removeprefix('bucket_'):<18} "
            f"{meta.get('model_batch', '?'):<3} {meta.get('foot_point_method', '?'):<12} "
            f"{reading.inference_fps:>9.2f} {track:>9} {shards:>3} {headroom:>5} "
            f"{reading.inference_elapsed_seconds:>6.1f} {reading.inference_frames:>8} "
            f"{rss:>7} {sm:>5} {meta.get('exit_status', '?'):>4}"
        )

    print(f"\n=== 分組平均（同組態多次；每秒張數＝推論進程口徑；排除前綴 {exclude}） ===")
    stats = group_runs(records, exclude)
    # `commit`／`date`／`label`／`machine` 都是分組維度，但多數時候整份產物只有一個值，
    # 印出來只是噪音。只在某個維度真的出現多值時才印它——那正是「兩組不該被比較的數字
    # 並排在一起」的時候，讀者必須看得到是哪個維度把它們分開的。
    varying = [
        field
        for field in ("label", "commit", "date", "machine")
        if len({getattr(stat, field) for stat in stats}) > 1
    ]
    for stat in stats:
        # 繼承到的 settings 環境變數會把同 bucket/batch/foot 的 run 拆成不同組
        # （見 `extra_settings_env`）；不印出來的話兩列會長得一模一樣。
        extra = "" if stat.extra_settings_env == "{}" else f" env={stat.extra_settings_env}"
        for field in varying:
            value = getattr(stat, field)
            extra += f" {field}={value[:12] if field == 'commit' else value}"
        print(
            f"{stat.bucket.removeprefix('bucket_'):<18} batch={stat.batch:<3} "
            f"{stat.foot_point_method:<12} n={stat.runs} "
            f"平均={stat.mean_inference_fps:>7.2f} 離散={stat.spread_percent:>4.1f}% "
            f"{[round(value, 2) for value in stat.values]}{extra}"
        )
    return 0


def main() -> None:
    # 本檔所有輸出都是中文，而 `LC_ALL=POSIX` 之類的環境會讓 stdout 退回 ascii：
    # `report` 印不出表、`run` 則是印進度時才炸。檔案 I/O 已逐處指定 utf-8
    # （與同目錄另兩支工具一致），這裡把 stdout／stderr 一併釘住。
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑矩陣量測（單輪＝各軸長度為 1 的退化情形）")
    run.add_argument("--buckets", required=True, help="測試片目錄，逗號分隔（cwd 相對）")
    run.add_argument("--date", required=True, help="INPUT__DATE，例如 2026-08-01")
    run.add_argument("--batches", required=True, help="MODEL__BATCH，逗號分隔")
    run.add_argument(
        "--foot-point",
        default=DEFAULT_FOOT_POINT,
        help=f"FOOT_POINT__METHOD，預設 {DEFAULT_FOOT_POINT}。**不是矩陣的第四條軸**："
        "對照組只掛在單一格上，當成軸會讓每輪從 5 個 run 變 8 個；"
        "要跑對照組請再跑一次本工具、寫進同一個產物目錄（report 本就按組態分組）",
    )
    run.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help=f"預設 {DEFAULT_REPEAT}")
    run.add_argument("--label", default=DEFAULT_LABEL, help=f"run 名稱前綴，預設 {DEFAULT_LABEL}")
    run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="引擎身分的來源")
    run.add_argument("--machine", default="local", help="記進 meta 的機器代號")
    run.add_argument(
        "--gpu-metrics", default=DEFAULT_GPU_METRICS, help=f"dmon -s，預設 {DEFAULT_GPU_METRICS}"
    )
    run.add_argument(
        "--gpu-interval", type=int, default=DEFAULT_GPU_INTERVAL_SECONDS, help="dmon -d（秒）"
    )
    run.add_argument(
        "--no-gpu-log",
        dest="gpu_log",
        action="store_false",
        help="明確豁免 GPU 取樣（少了它「顯卡／CPU 拆分」那項分析就做不了）",
    )
    run.add_argument("--overwrite", action="store_true", help="允許覆蓋同名產物")
    run.set_defaults(func=command_run)

    report = sub.add_parser("report", help="把產物解析成明細表與分組統計")
    report.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    report.add_argument(
        "--exclude-prefixes",
        default=DEFAULT_EXCLUDE_PREFIXES,
        help=f"不進分組統計的 run 名稱前綴，逗號分隔，預設 {DEFAULT_EXCLUDE_PREFIXES}",
    )
    report.set_defaults(func=command_report)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
