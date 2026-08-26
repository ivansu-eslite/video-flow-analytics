import datetime
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from vfa_observability import StructuredLogger
from vfa_registry import load_registry

from video_analyze.config.constants import OUTPUT_ROOT, TRACKING_RESULTS_FILENAME
from video_analyze.services.batching import (
    RING_SLOTS,
    TARGET_BATCH,
    TRACK_QUEUE_SLOTS,
)
from video_analyze.services.detector import YOLODetector
from video_analyze.services.frame_ring import FrameRing, create_ring_buffers
from video_analyze.services.inference import InferencePipeline
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH
from video_analyze.services.track_worker import TRACK_FAILED, run_track_worker
from video_analyze.services.video_reader import (
    FrameShape,
    SegmentInfo,
    discover_segments,
    probe_frame_shape,
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
    track_queue: mp.Queue,
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
    等到父進程 `_terminate_all` 的 SIGTERM。

    Args:
        data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
        free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
        ring_buffers: 各路 `create_ring_buffers` 建立的共享記憶體。
        stream_names: 各路攝影機的 `stream_dirname`。
        track_queue: 送往追蹤進程的佇列。
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
            track_queue=track_queue,
        )
    except BaseException:
        # 只包組裝這一段：`start_loop` 自己的 except 已經會送一次，包進來會在同一次
        # 失敗送出兩個訊號
        track_queue.put(TRACK_FAILED)
        raise
    pipeline.start_loop(data_queues, free_queues, rings)


def _terminate_all(processes: list[mp.Process]) -> None:
    """終止所有仍在存活的子進程，避免中斷後留下孤兒進程。"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            logger.warning("進程未在時限內結束，強制 kill", pid=p.pid)
            p.kill()
            p.join()


def analyze_daily(
    date: datetime.date,
    bucket_dir: str,
    camera_ids: list[str] | None = None,
) -> AnalysisResult:
    """以「一天」為單位執行多路 YOLO 偵測 + ByteTrack 追蹤分析。

    以參數傳入 `bucket_dir`（而非讀全域 `settings`），讓本函式可重複以不同
    bucket 呼叫。內部會拆成 N 個讀取子進程 + 1 個推理子進程 + 1 個追蹤子進程，
    逐段掃描指定日期的影片、輸出追蹤明細 parquet。

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

    def _raise_if_abnormal(procs: list[mp.Process]) -> None:
        abnormal = [p for p in procs if not p.is_alive() and p.exitcode]
        if abnormal:
            detail = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in abnormal)
            raise RuntimeError(f"子進程異常結束（{detail}），分析已中止。")

    # 追蹤進程與推論進程之間只走這一條 queue，傳的是偵測框（每格幾十個框、幾 KB）
    # 而非影格——影格在推論完成後就沒有用途（slot 當下就歸還了，見 ADR-010）。
    # **上限不可省**：影格側的背壓只覆蓋到推論為止，沒有這個上限，追蹤一落後 payload 就
    # 無上限堆在推論進程，而 TRACK_FAILED 也會晚到父進程 terminate 之後（見
    # `services/batching.py` 的 `TRACK_QUEUE_SLOTS`）
    track_queue: mp.Queue = mp.Queue(maxsize=TRACK_QUEUE_SLOTS)

    try:
        track_proc = mp.Process(
            target=run_track_worker,
            args=(track_queue, stream_names, frame_shapes, results_path),
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
                track_queue,
            ),
        )
        infer_proc.start()
        processes.append(infer_proc)

        for i, segments in enumerate(segments_per_stream):
            reader_proc = mp.Process(
                target=run_video_reader,
                args=(
                    i,
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

    return AnalysisResult(
        date=date,
        camera_ids=stream_names,
        tracking_results_path=str(results_path),
    )
