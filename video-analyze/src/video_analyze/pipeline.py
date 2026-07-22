import dataclasses
import datetime
import json
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vfa_registry import load_registry

from video_analyze.config import settings
from video_analyze.detector import YOLODetector
from video_analyze.inference import InferencePipeline
from video_analyze.io.frame_ring import (
    RING_SLOTS,
    FrameRing,
    create_ring_buffer,
)
from video_analyze.io.video_reader import (
    SegmentInfo,
    discover_segments,
    probe_frame_shape,
    run_video_reader,
)
from video_analyze.io.video_writer import mirrored_output_path
from video_analyze.tracker import MultiStreamByteTracker

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")


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
        output_video_paths: 已輸出的標註影片路徑清單；`save_video=False` 時
            為空清單，且只列出實際成功寫出的檔案（0 幀等未產生輸出的片段
            會被略過，不在清單內）。
    """

    date: datetime.date
    camera_ids: list[str]
    tracking_results_path: str
    output_video_paths: list[str] = field(default_factory=list)


def run_inference_pipeline(
    data_queues: list[mp.Queue],
    free_queues: list[mp.Queue],
    ring_buffers: list,
    frame_shapes: list[tuple[int, int]],
    stream_names: list[str],
    output_root: Path,
    results_path: Path,
) -> None:
    """推理子進程的進入點：建構偵測器/追蹤器/環形緩衝後啟動推理主迴圈。

    以 `mp.Process(target=run_inference_pipeline, ...)` 於子進程執行，故
    參數需為可 pickle 的型別（環形緩衝以 `mp.RawArray` 傳遞）。

    Args:
        data_queues: 各路讀取進程送出的資料佇列，索引為 stream_id。
        free_queues: 各路歸還環形緩衝 slot 用的佇列，索引為 stream_id。
        ring_buffers: 各路 `create_ring_buffer` 建立的共享記憶體。
        frame_shapes: 各路的 `(height, width)`，索引與 `ring_buffers` 對應。
        stream_names: 各路攝影機的 `stream_dirname`。
        output_root: 標註影片輸出根目錄。
        results_path: 追蹤結果 parquet 的目標路徑。
    """
    detector = YOLODetector()
    tracker = MultiStreamByteTracker(num_streams=len(stream_names))
    rings = [
        FrameRing(buffer, RING_SLOTS, height, width)
        for buffer, (height, width) in zip(ring_buffers, frame_shapes)
    ]
    pipeline = InferencePipeline(
        stream_names=stream_names,
        detector=detector,
        tracker=tracker,
        output_root=output_root,
        results_path=results_path,
    )
    pipeline.start_loop(data_queues, free_queues, rings)


def _terminate_all(processes: list[mp.Process]) -> None:
    """終止所有仍在存活的子進程，避免中斷後留下孤兒進程。"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            logger.warning("進程 %s 未在時限內結束，強制 kill。", p.pid)
            p.kill()
            p.join()


def analyze_daily(
    date: datetime.date,
    bucket_dir: str,
    camera_ids: list[str] | None = None,
) -> AnalysisResult:
    """以「一天」為單位執行多路 YOLO 偵測 + ByteTrack 追蹤分析。

    以參數傳入 `bucket_dir`（而非讀全域 `settings`），讓本函式可重複以不同
    bucket 呼叫。內部會拆成 N 個讀取子進程 + 1 個推理子進程，逐段掃描指定
    日期的影片、輸出追蹤明細 parquet 與（依設定）逐片段標註影片。

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
    frame_shapes: list[tuple[int, int]] = []
    for cam in cameras:
        segments = discover_segments(
            bucket_path, cam.stream_dirname, date, registry.storage.file_ext
        )
        if not segments:
            raise ValueError(f"{cam.stream_dirname} 在 {date} 沒有任何影片片段")
        stream_names.append(cam.stream_dirname)
        segments_per_stream.append(segments)
        # 依首格解析度一次配置該路的共享環形緩衝（假設整天解析度固定）
        frame_shapes.append(probe_frame_shape(segments[0]))

    results_path = output_root / date.isoformat() / "tracking_results.parquet"

    num_streams = len(stream_names)
    # 影格走共享記憶體環形緩衝，queue 只傳輕量索引，避免每格 6MB 走 pickle + pipe
    data_queues = [mp.Queue() for _ in range(num_streams)]
    free_queues = [mp.Queue() for _ in range(num_streams)]
    ring_buffers = [
        create_ring_buffer(RING_SLOTS, height, width)
        for height, width in frame_shapes
    ]
    processes: list[mp.Process] = []

    def _raise_if_abnormal(procs: list[mp.Process]) -> None:
        abnormal = [p for p in procs if not p.is_alive() and p.exitcode]
        if abnormal:
            detail = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in abnormal)
            raise RuntimeError(f"子進程異常結束（{detail}），分析已中止。")

    try:
        infer_proc = mp.Process(
            target=run_inference_pipeline,
            args=(
                data_queues,
                free_queues,
                ring_buffers,
                frame_shapes,
                stream_names,
                output_root,
                results_path,
            ),
        )
        infer_proc.start()
        processes.append(infer_proc)

        for i, segments in enumerate(segments_per_stream):
            height, width = frame_shapes[i]
            reader_proc = mp.Process(
                target=run_video_reader,
                args=(
                    i,
                    segments,
                    data_queues[i],
                    free_queues[i],
                    ring_buffers[i],
                    RING_SLOTS,
                    height,
                    width,
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

    output_video_paths = (
        [
            str(out_path)
            for segments in segments_per_stream
            for seg in segments
            if (out_path := mirrored_output_path(output_root, seg.relpath)).exists()
        ]
        if settings.output.save_video
        else []
    )
    return AnalysisResult(
        date=date,
        camera_ids=stream_names,
        tracking_results_path=str(results_path),
        output_video_paths=output_video_paths,
    )


def run_analyze() -> None:
    """`analyze` 子命令的進入點：從 `config.toml` 取參數後呼叫 `analyze_daily`。

    Raises:
        ValueError: `config.toml` 的 `[input].date` 未設定。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")
    try:
        result = analyze_daily(
            date=settings.input.date,
            bucket_dir=settings.input.bucket_dir,
            camera_ids=settings.input.camera_ids,
        )
    except KeyboardInterrupt:
        sys.exit(130)
    logger.info("當日所有影片皆已處理完畢！")
    logger.info(
        "分析結果:\n%s",
        json.dumps(
            dataclasses.asdict(result), indent=2, default=str, ensure_ascii=False
        ),
    )


if __name__ == "__main__":
    run_analyze()
