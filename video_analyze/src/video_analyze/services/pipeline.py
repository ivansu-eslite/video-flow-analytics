import datetime
import multiprocessing as mp
import os
import signal
from dataclasses import dataclass
from pathlib import Path

from vfa_observability import StructuredLogger
from vfa_registry import load_registry

from video_analyze.config.constants import OUTPUT_ROOT, TRACKING_RESULTS_FILENAME
from video_analyze.models.config import settings
from video_analyze.services.batching import (
    RING_SLOTS,
    TARGET_BATCH,
    TRACK_QUEUE_SLOTS,
)
from video_analyze.services.detector import YOLODetector
from video_analyze.services.frame_ring import FrameRing, create_ring_buffers
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.output_parts import (
    claim_parts_dir,
    merge_parts,
    parts_dir_for,
    plan_routes,
    shard_part_path,
)
from video_analyze.services.track_worker import (
    TRACK_FAILED,
    TRACK_SIGNAL_PUT_TIMEOUT,
    fanout_track_signal,
    run_track_worker,
)
from video_analyze.services.video_reader import (
    FrameShape,
    SegmentInfo,
    discover_segments,
    probe_frame_shape,
    probe_stream_fps,
    run_video_reader,
)

logger = StructuredLogger(component="pipeline")


@dataclass
class AnalysisResult:
    """`analyze_daily` 的回傳結果。

    Attributes:
        date: 分析的日期。
        camera_ids: 已分析的攝影機清單，`stream_dirname` 格式，與 parquet 的
            `camera_id` 欄位保持一致；下游若要用這裡回傳的值去 join
            `tracking_results.parquet`，格式不同會導致靜默地全數落空。
        tracking_results_path: 追蹤結果 parquet 的路徑（字串，非 `Path`
            物件；需要 `Path` 操作時呼叫端須自行包一層 `Path(...)`）。
    """

    date: datetime.date
    camera_ids: list[str]
    tracking_results_path: str


def run_inference_pipeline(
    data_queues: list[mp.Queue],
    free_queues: list[mp.Queue],
    ring_buffers: list,
    stream_names: list[str],
    track_queues: list[mp.Queue],
    route: list[int],
) -> None:
    """推理子進程的進入點：建構偵測器與環形緩衝後啟動推理主迴圈。

    以 `mp.Process(target=run_inference_pipeline, ...)` 於子進程執行，故
    參數需為可 pickle 的型別（環形緩衝以 `mp.RawArray` 傳遞）。

    追蹤器與結果收集不在這裡：它們搬到獨立的追蹤進程了（`run_track_worker`），
    本進程只出偵測框。`frame_shapes` 也跟著搬過去——它的兩個消費端（parquet 的尺寸
    欄位、座標反算參數）都在那一側。

    **啟動階段的例外也要送得出 `TRACK_FAILED`**：追蹤進程在本進程之前就 start 了，
    此刻正阻塞在 `track_queue.get()`。issue #113 揭露了為什麼上游死掉不會讓它自己
    醒過來——`track_queue` 的 pipe 寫入端 fd 被父進程與九個讀取進程一起繼承，寫入端
    永遠不會全部關閉，`get()` 收不到 EOF。改吃 TensorRT 引擎之後，「載入失敗」從罕見
    變成常見的一類（引擎檔不在、SM 對不上、TensorRT 版本與映像檔不一致），而它們全部
    發生在 `start_loop` 之前——那裡的 `except` 涵蓋不到。沒有這一段的話，追蹤進程會一直
    等到父進程 `_terminate_all` 的 SIGTERM。**要送到每一片**，與 `start_loop` 內的
    `except` 共用同一支 fan-out helper。

    Args:
        data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
        free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
        ring_buffers: 各路 `create_ring_buffers` 建立的共享記憶體。
        stream_names: 各路攝影機的 `stream_dirname`。
        track_queues: 各片追蹤進程的佇列，索引即片編號。
        route: 各路歸屬的片編號，索引為 stream_id。
    """
    try:
        detector = YOLODetector()
        # 緩衝存的是讀取端已縮好的影格，故用推論尺寸而非 frame_shapes；沿用原始解析度會讓
        # `np.frombuffer(...).reshape` 在本子進程當場拋 ValueError
        rings = [
            FrameRing(buffer, RING_SLOTS, INFER_HEIGHT, INFER_WIDTH)
            for buffer in ring_buffers
        ]
        pipeline = InferencePipeline(
            stream_names=stream_names,
            detector=detector,
            track_queues=track_queues,
            route=route,
        )
    except BaseException:
        # 只包組裝這一段：`start_loop` 自己的 except 已經會送一次，包進來會在同一次
        # 失敗送出兩個訊號
        fanout_track_signal(
            track_queues, TRACK_FAILED, timeout=TRACK_SIGNAL_PUT_TIMEOUT
        )
        raise
    pipeline.start_loop(data_queues, free_queues, rings)


def _describe_exit(process: mp.Process) -> str:
    """把一個異常結束的子進程描述成「角色 + pid + 結束方式」一段文字。

    角色取 `mp.Process(name=...)`，不另外維護一份 pid→角色的對照表：`name` 同時被
    兩個消費端讀到——父進程這裡的 `p.name`，以及子進程預設 traceback 的表頭
    （`BaseProcess._bootstrap` 的 `except:` 分支寫 `Process %s:`），一個 kwarg 讓那份
    交錯的 stderr 也認得出是哪一路，不必再靠時間順序猜。

    **負 exitcode 才轉訊號名**，而「被訊號終止」不一定表現為負值：
    `run_track_worker` 攔了 SIGTERM 轉 `raise SystemExit(128 + signum)`，`_bootstrap`
    對帶 int code 的 `SystemExit` 走專門分支，`exitcode` 是正的 143。措辭因此不暗示
    所有訊號死亡都顯示得出訊號名。
    """
    detail = f"{process.name} pid={process.pid} exitcode={process.exitcode}"
    if process.exitcode is not None and process.exitcode < 0:
        try:
            detail += f" {signal.Signals(-process.exitcode).name}"
        except ValueError:
            # 這是失敗路徑上最後一道訊息，自己爆掉會把根因蓋掉：對不上的訊號值就
            # 只留原本的負 exitcode
            pass
    return detail


def _raise_if_abnormal(processes: list[mp.Process]) -> None:
    """任一子進程已結束且 exitcode 非 0 就中止整批分析。

    判定條件是 `not p.is_alive() and p.exitcode`——0 為 falsy（正常結束），負值成立
    （被訊號終止）。訊息帶得出每個進程的角色與所屬對象，因為 `pid` 對不回「哪一路
    攝影機」、也對不回「這是讀取、推理還是追蹤」，而真正的根因在子進程自己的
    traceback 上、與另外十幾個進程的輸出交錯在同一份 stderr。

    Raises:
        RuntimeError: 有子進程異常結束。
    """
    abnormal = [p for p in processes if not p.is_alive() and p.exitcode]
    if abnormal:
        detail = "；".join(_describe_exit(p) for p in abnormal)
        raise RuntimeError(f"子進程異常結束（{detail}），分析已中止。")


def _terminate_all(processes: list[mp.Process]) -> None:
    """終止所有仍在存活的子進程，避免中斷後留下孤兒進程。"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            logger.warning("進程未在時限內結束，強制 kill", name=p.name, pid=p.pid)
            p.kill()
            p.join()


def analyze_daily(
    date: datetime.date,
    bucket_dir: str,
    camera_ids: list[str] | None = None,
) -> AnalysisResult:
    """以「一天」為單位執行多路 YOLO 偵測 + ByteTrack 追蹤分析。

    以參數傳入 `bucket_dir`（而非讀全域 `settings`），讓本函式可重複以不同
    bucket 呼叫。內部會拆成「一路一個讀取子進程 + 1 個推理子進程 + `[tracker].shards`
    個追蹤子進程」，逐段掃描指定日期的影片、輸出追蹤明細 parquet。

    **本進程（主進程）認領這一天並持鎖到合併完成**（`output_parts.claim_parts_dir`）：
    各追蹤進程只寫自己的 part 檔，正式檔名由這裡合併產生，所以認領、跑、合併三段要被
    同一把鎖蓋住，而只有主進程橫跨全程。子進程靠 `fork` 繼承那個 fd，主進程被 SIGKILL
    之後鎖仍在孤兒子進程手上（見該模組 docstring）。

    Args:
        date: 要分析的日期。
        bucket_dir: 本機模擬 GCS bucket 的根目錄。
        camera_ids: 要分析的攝影機清單；`None` 或空清單代表 registry 內
            全部攝影機。

    Returns:
        本次分析的結果摘要（見 `AnalysisResult`）。

    Raises:
        FileNotFoundError: `bucket_dir` 底下找不到 `camera_registry.yaml`。
        ValueError: `camera_registry.yaml` 沒有任何攝影機、`camera_ids`
            指定了查無對應設備登錄的 ID，或任一攝影機在該日期沒有任何
            影片片段。
        RuntimeError: 任一子進程異常結束。
        KeyboardInterrupt: 收到中斷訊號，會先優雅終止所有子進程再重新拋出。
    """
    bucket_path = Path(bucket_dir)
    registry = load_registry(bucket_path)
    cameras = registry.resolve_cameras(camera_ids)
    if not cameras:
        raise ValueError("camera_registry.yaml 沒有任何攝影機，無法執行分析。")

    # 輸出路徑掛上 bucket 名稱，避免不同 bucket 的輸出互相覆蓋
    output_root = OUTPUT_ROOT / bucket_path.name

    stream_names: list[str] = []
    segments_per_stream: list[list[SegmentInfo]] = []
    frame_shapes: list[FrameShape] = []
    stream_fps: list[float] = []
    for cam in cameras:
        segments = discover_segments(
            bucket_path, cam.stream_dirname, date, registry.storage.file_ext
        )
        if not segments:
            raise ValueError(f"{cam.stream_dirname} 在 {date} 沒有任何影片片段")
        stream_names.append(cam.stream_dirname)
        segments_per_stream.append(segments)
        # 探測首格取該路的原始解析度（假設整天固定）：供 parquet 的尺寸欄位與追蹤
        # 進程的座標反算用，**不是**用來配置環形緩衝（緩衝照推論尺寸配）
        frame_shapes.append(probe_frame_shape(segments[0]))
        # 追蹤進程的路由權重（registry 沒有 fps 欄位）。只讀容器標頭、不解碼影格
        stream_fps.append(probe_stream_fps(segments[0]))

    results_path = output_root / date.isoformat() / TRACKING_RESULTS_FILENAME

    num_streams = len(stream_names)
    # 三個常數都由 `[model].batch` 推導（見 `services/batching.py`），而「批次調小了」
    # 唯一的症狀是跑得慢——不會有例外、輸出也完全正常。啟動時記一筆，讓事後查得到當次
    # 實際用的是多少。
    logger.info(
        "批次相關常數",
        target_batch=TARGET_BATCH,
        ring_slots=RING_SLOTS,
        track_queue_slots=TRACK_QUEUE_SLOTS,
    )
    # 啟動時決定、整天不變。分片數改變不改變任何一格的結果，只影響列順序與吞吐
    route = plan_routes(stream_fps, settings.tracker.shards)
    num_shards = max(route) + 1
    for shard_id in range(num_shards):
        owned = [sid for sid in range(num_streams) if route[sid] == shard_id]
        # 記下來才比得了兩輪量測：分配不同時餘裕本來就不同，而輸出檔看不出差別
        logger.info(
            "追蹤分片的路由",
            shard_id=shard_id,
            cameras=[stream_names[sid] for sid in owned],
            fps_sum=round(sum(stream_fps[sid] for sid in owned), 2),
        )

    # 認領要在配置共享記憶體與起任何子進程**之前**：這一天已經有人在跑的話，此刻中止
    # 什麼都還沒建立。fd 一路持到合併完成——子進程繼承的就是它
    lock_fd = claim_parts_dir(results_path)
    parts_dir = parts_dir_for(results_path)
    part_paths = [shard_part_path(parts_dir, shard_id) for shard_id in range(num_shards)]
    try:
        # 影格走共享記憶體環形緩衝，queue 只傳輕量索引，避免每格 6MB 走 pickle + pipe
        data_queues = [mp.Queue() for _ in range(num_streams)]
        free_queues = [mp.Queue() for _ in range(num_streams)]
        # 讀取端已把影格縮成推論尺寸，緩衝只需存 640×384（4K 每格 23.73 MiB → 0.70 MiB）。
        # frame_shapes 仍然要留著、且必須是原始解析度：它要寫進 parquet，也是追蹤進程
        # 反算座標的依據（見 `services/letterbox.py`）。合計裝不進 /dev/shm 的組態由
        # `create_ring_buffers` 在配置第一塊之前擋下（見 frame_ring.require_shm_capacity）
        ring_buffers = create_ring_buffers(
            len(frame_shapes), RING_SLOTS, INFER_HEIGHT, INFER_WIDTH
        )
        processes: list[mp.Process] = []

        # 一片一條 queue，傳的是偵測框（每格幾十個框、幾 KB）而非影格——影格在推論完成
        # 後就沒有用途（slot 當下就歸還了，見 ADR-010）。
        # **上限不可省**：影格側的背壓只覆蓋到推論為止，沒有這個上限，追蹤一落後 payload
        # 就無上限堆在推論進程，而 TRACK_FAILED 也會晚到父進程 terminate 之後（見
        # `services/batching.py` 的 `TRACK_QUEUE_SLOTS`）。**上限不除以 N**：它擋的兩件
        # 事都是每條 queue 各自的性質，除以 N 反而讓正常抖動更容易變成兩個進程互等
        track_queues: list[mp.Queue] = [
            mp.Queue(maxsize=TRACK_QUEUE_SLOTS) for _ in range(num_shards)
        ]

        try:
            for shard_id in range(num_shards):
                track_proc = mp.Process(
                    target=run_track_worker,
                    args=(
                        track_queues[shard_id],
                        stream_names,
                        frame_shapes,
                        part_paths[shard_id],
                        shard_id,
                        # 收到不屬於自己的 stream_id 就拋錯：路由送錯片是靜默的
                        # （見 `services/track_worker.py`）
                        frozenset(
                            sid for sid in range(num_streams) if route[sid] == shard_id
                        ),
                    ),
                    # 角色與所屬對象一起帶著走：父進程的異常彙總讀 `p.name`，子進程
                    # 預設 traceback 的表頭也印同一個值（見 `_describe_exit`）
                    name=f"track[shard{shard_id}]",
                )
                track_proc.start()
                processes.append(track_proc)

            infer_proc = mp.Process(
                target=run_inference_pipeline,
                args=(
                    data_queues,
                    free_queues,
                    ring_buffers,
                    stream_names,
                    track_queues,
                    route,
                ),
                name="inference",
            )
            infer_proc.start()
            processes.append(infer_proc)

            for i, segments in enumerate(segments_per_stream):
                reader_proc = mp.Process(
                    target=run_video_reader,
                    args=(
                        i,
                        # 攝影機名在讀取進程內沒有第二個來源（那側只有整數
                        # `stream_id`），失敗紀錄要指得出是哪一路就得傳進去
                        stream_names[i],
                        segments,
                        data_queues[i],
                        free_queues[i],
                        ring_buffers[i],
                        RING_SLOTS,
                        # 不是緩衝尺寸（緩衝照推論尺寸配），是讀取端逐格核對「整天解析度
                        # 固定」用的探測值——letterbox 會把任何尺寸抹平，write_slot 的
                        # 形狀檢查不再擋得住中途換解析度
                        frame_shapes[i],
                    ),
                    name=f"reader[{stream_names[i]}]",
                )
                reader_proc.start()
                processes.append(reader_proc)

            while any(p.is_alive() for p in processes):
                for p in processes:
                    p.join(timeout=0.5)
                _raise_if_abnormal(processes)

            # 補一次檢查：避免最後一個進程恰好在上一輪之後才異常結束而被誤判為成功
            _raise_if_abnormal(processes)
        except KeyboardInterrupt:
            logger.warning("收到中斷訊號（Ctrl+C），正在優雅關閉所有子進程...")
            _terminate_all(processes)
            raise
        except Exception:
            _terminate_all(processes)
            raise

        # 各片都正常收尾了（否則上面的檢查會擋下），這裡才把 part 合併成正式檔名。
        # 合併失敗不留下正式檔，parts 目錄則由下一次執行認領時清掉
        merge_parts(results_path, part_paths)
    finally:
        # 鎖的存續期到合併完成為止。正式執行時進程接著就結束了，kernel 也會做同一件事；
        # 明寫是為了讓 in-process 呼叫（測試、批次跑多天）不把 fd 一路累積下去
        os.close(lock_fd)

    return AnalysisResult(
        date=date,
        camera_ids=stream_names,
        tracking_results_path=str(results_path),
    )
