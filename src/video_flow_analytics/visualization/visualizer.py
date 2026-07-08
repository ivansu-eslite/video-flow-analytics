import cv2
import numpy as np


class TrackAnnotator:
    """視覺化繪圖模組"""

    @staticmethod
    def draw_bboxes(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """在影格上原地畫出每個追蹤框與其 track_id。

        Args:
            frame: 要畫框的影格（BGR），會被直接修改（in-place）並回傳。
            tracks: `MultiStreamByteTracker.update` 的輸出，每列至少為
                `[x1, y1, x2, y2, track_id, ...]`；空陣列時原樣回傳 `frame`。

        Returns:
            畫好框的影格（與傳入的 `frame` 為同一物件）。
        """
        if len(tracks) == 0:
            return frame

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return frame
