import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_flow_analytics.core.config import settings

logger = logging.getLogger(__name__)

# 每路 writer 待編碼影格緩衝上限；正常不會積壓，設上限只防病態情況吃光記憶體
_WRITER_QUEUE_MAXSIZE = 60


def mirrored_output_path(output_root: Path, segment_relpath: str | Path) -> Path:
    """輸出片段的實際路徑：鏡射輸入片段相對 bucket 的路徑，只把根換成 output_root。"""
    return output_root / segment_relpath


@dataclass
class _OpenSegment:
    relpath: str
    writer: cv2.VideoWriter


class MultiStreamVideoWriter:
    """為每一支輸入片段各自輸出一支標註影片，路徑鏡射輸入（見 mirrored_output_path）。

    mp4v 編碼是 CPU 重工，inline 執行會卡住 GPU（實測單日吞吐腰斬），故每路各起一個
    背景 writer 執行緒編碼，與下一批 GPU 推理重疊（cv2.VideoWriter.write 會釋放 GIL）。

    fail-loud：writer 執行緒例外會記錄下來，主緒在下次 write() 或 close_all() 重新拋出。
    """

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.enabled = settings.output.save_video
        # 每路一支 queue 與一條背景編碼執行緒（收到第一格時惰性建立）
        self._queues: dict[int, queue.Queue] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()

        if not self.enabled:
            logger.info("save_video=False，僅執行推理，不儲存影片。")

    def write(
        self, stream_id: int, segment_relpath: str, frame: np.ndarray, fps: float
    ) -> None:
        if not self.enabled:
            return
        # 若某路 writer 執行緒已失敗，立刻在主緒重拋、中止整個推理（fail-loud）
        self._raise_if_failed()
        q = self._queues.get(stream_id)
        if q is None:
            q = queue.Queue(maxsize=_WRITER_QUEUE_MAXSIZE)
            self._queues[stream_id] = q
            thread = threading.Thread(
                target=self._stream_worker, args=(q,), daemon=True
            )
            self._threads[stream_id] = thread
            thread.start()
        q.put((segment_relpath, frame, fps))

    def _stream_worker(self, q: queue.Queue) -> None:
        """單一路的背景編碼迴圈：依 segment_relpath 逐片段開關檔並寫入影格。"""
        current: _OpenSegment | None = None
        failed = False
        while True:
            item = q.get()
            if item is None:  # 收尾訊號
                break
            if failed:
                # 已失敗：持續排空 queue（丟棄）以免生產端阻塞在 put 上，直到收尾
                continue
            segment_relpath, frame, fps = item
            try:
                if current is None or current.relpath != segment_relpath:
                    if current is not None:
                        self._close(current)
                    current = _OpenSegment(
                        relpath=segment_relpath,
                        writer=self._open_writer(segment_relpath, frame, fps),
                    )
                current.writer.write(frame)
            except BaseException as exc:  # noqa: BLE001 - 記錄後由主緒重拋
                with self._error_lock:
                    if self._error is None:
                        self._error = exc
                failed = True
        if current is not None:
            self._close(current)

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _open_writer(
        self, segment_relpath: str, frame: np.ndarray, fps: float
    ) -> cv2.VideoWriter:
        output_path = mirrored_output_path(self.output_root, segment_relpath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(
                f"無法建立輸出影片 {output_path}（可能為編碼器/容器不相容）"
            )
        return writer

    def _close(self, current: _OpenSegment) -> None:
        current.writer.release()
        output_path = mirrored_output_path(self.output_root, current.relpath)
        logger.info("已輸出: %s", output_path)

    def close_stream(self, stream_id: int) -> None:
        """某一路正常讀完：通知該路 writer 執行緒收尾並等它把緩衝的影格寫完。"""
        if not self.enabled:
            return
        self._join_stream(stream_id)
        self._raise_if_failed()

    def close_all(self) -> None:
        """全部串流正常跑完：等所有 writer 執行緒寫完；任一路失敗即重拋。"""
        if not self.enabled:
            return
        for stream_id in list(self._queues):
            self._join_stream(stream_id)
        self._raise_if_failed()

    def abort(self) -> None:
        """例外路徑的清理：停掉並等所有 writer 執行緒，但不重拋（避免遮蓋原始例外）。"""
        if not self.enabled:
            return
        for stream_id in list(self._queues):
            self._join_stream(stream_id)

    def _join_stream(self, stream_id: int) -> None:
        """送出收尾訊號並 join 該路 writer 執行緒；idempotent。"""
        q = self._queues.pop(stream_id, None)
        thread = self._threads.pop(stream_id, None)
        if q is not None:
            q.put(None)
        if thread is not None:
            thread.join()
