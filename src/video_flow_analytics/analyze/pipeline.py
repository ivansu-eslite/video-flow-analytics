import dataclasses
import datetime
import json
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path

from video_flow_analytics.analyze.detector import YOLODetector
from video_flow_analytics.analyze.inference import InferencePipeline
from video_flow_analytics.analyze.tracker import MultiStreamByteTracker
from video_flow_analytics.core.config import settings
from video_flow_analytics.core.registry import load_registry
from video_flow_analytics.io.frame_ring import (
    RING_SLOTS,
    FrameRing,
    create_ring_buffer,
)
from video_flow_analytics.io.video_reader import (
    SegmentInfo,
    discover_segments,
    probe_frame_shape,
    run_video_reader,
)
from video_flow_analytics.io.video_writer import mirrored_output_path

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")


@dataclass
class AnalysisResult:
    """analyze_daily 的回傳結果（原 DailyAnalysisResponse，改為就地定義的
    dataclass；不再是 pydantic 模型，也不再為 HTTP 化預留）。"""

    date: datetime.date
    # 實際分析過的攝影機，格式與 tracking_results.parquet 的 camera_id 欄位一致
    # （即 stream_dirname，`<location>_<camera_id>`），可直接用來 join 追蹤結果；
    # 與用來篩選攝影機的裸 camera_id 格式不同
    camera_ids: list[str]
    # 各 camera 的追蹤結果明細（Parquet）
    tracking_results_path: str
    # 開發驗證用的標註影片；save_video=False 時為空清單，
    # 只列出實際成功寫出的檔案（略過 0 幀等未產生輸出的片段）
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
    """以「一天」為單位執行多路追蹤分析。"""
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
    # 每路各一組：資料 queue（傳 slot 索引 + metadata）、空 slot queue、共享環形緩衝。
    # 影格走共享記憶體，queue 只傳輕量索引，避免每格 6MB 走 pickle + pipe。
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

        # 迴圈結束代表所有進程都已死亡；即使最後一個進程剛好在上一輪檢查「之後」
        # 才異常結束（因此走不到下一輪 while 判斷），這裡再補一次檢查，避免把
        # 「最後一步才失敗」誤判為成功。
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
        # 與 parquet 的 camera_id 欄位（stream_dirname）保持一致，避免下游用
        # 這裡回傳的值去 join tracking_results.parquet 時因格式不同而全數落空
        camera_ids=stream_names,
        tracking_results_path=str(results_path),
        output_video_paths=output_video_paths,
    )


def run_analyze() -> None:
    """analyze 子命令：從 config.toml 取參數後呼叫 analyze_daily。"""
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
