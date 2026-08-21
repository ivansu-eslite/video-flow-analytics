"""把「縮放到推論尺寸」與「座標映射回原始解析度」放在同一個模組。

原本這件事由 ultralytics 在推論進程內做（`predict` 的前處理會 letterbox，輸出的框再由
它自己反算回原圖座標），實測每格 1.84 ms、佔推論進程 8.5%，而推論進程是序列瓶頸。把縮放
搬到 N 個讀取進程之後，這條反算就沒有人做了——ultralytics 只知道它收到的是
`INFER_WIDTH` × `INFER_HEIGHT`，`orig_shape` 也是這個尺寸，輸出的框自然停在那個尺度上。

**下游要的是原始解析度像素**（`x1`/`y1`/`x2`/`y2`、`foot_x`/`foot_y` 進 parquet，再由
zone／line 兩包拿去跟攝影機幾何比對），所以縮放與反算是同一件事的兩半，必須共用同一組
參數。分成兩個檔各寫一次的話，哪天有人只改其中一邊，輸出的座標會整批偏掉而完全不會報錯。

共享記憶體也跟著受益：原本每格存原始解析度（4K 約 23.73 MiB），改存 640×384 後是
0.70 MiB。
"""

import cv2
import numpy as np

# 推論輸入尺寸。ultralytics 對 16:9 來源在 rect 模式下也是縮到這個尺寸（640 為長邊、
# 短邊 pad 到 stride 32 的倍數即 384），此處寫死是為了讓讀取端與推論端有共同約定。
# **假設來源是 16:9**：1080p 與 4K 都是，故填充量相同、只有 `scale` 不同。日後接入非
# 16:9 的攝影機時這個假設要重新檢查（`letterbox_params` 本身算得對，但不同來源會算出
# 不同的填充量，混形狀又會回來）。
INFER_WIDTH = 640
INFER_HEIGHT = 384

_PAD_VALUE = 114  # 與 ultralytics LetterBox 的預設填充值一致


def letterbox_params(orig_height: int, orig_width: int) -> tuple[float, float, float]:
    """算出等比縮放的比例與置中填充量。

    與 ultralytics `LetterBox` 同一套算法：取長短邊比例的較小者，縮完置中填充。
    1080p 與 4K 都是 16:9，因此兩者都縮成 640×360、上下各填 12——**填充量相同、
    只有 `scale` 不同**，這也是混解析度批次在 rect 模式下能共用同一形狀的原因。

    Args:
        orig_height: 原始影格高度（pixel）。
        orig_width: 原始影格寬度（pixel）。

    Returns:
        `(scale, pad_x, pad_y)`；`scale` 是原圖到推論尺寸的縮放比例，兩個 pad 是
        單邊填充量（可能是 .5，故為 float）。
    """
    scale = min(INFER_WIDTH / orig_width, INFER_HEIGHT / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)
    pad_x = (INFER_WIDTH - new_width) / 2
    pad_y = (INFER_HEIGHT - new_height) / 2
    return scale, pad_x, pad_y


def letterbox(frame: np.ndarray) -> np.ndarray:
    """把影格等比縮到 `INFER_WIDTH` × `INFER_HEIGHT` 並置中填充。

    Args:
        frame: 原始 BGR 影格。

    Returns:
        `(INFER_HEIGHT, INFER_WIDTH, 3)` 的新陣列；已是該尺寸時原樣回傳（不複製）。
    """
    height, width = frame.shape[:2]
    if (height, width) == (INFER_HEIGHT, INFER_WIDTH):
        return frame
    scale, pad_x, pad_y = letterbox_params(height, width)
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        # 縮小一律用 INTER_LINEAR，與 ultralytics LetterBox 一致；換插值法會讓偵測
        # 結果與改動前對不起來，而那個差異看起來會像是「把縮放搬到讀取端造成的」
        interpolation=cv2.INTER_LINEAR,
    )
    top = round(pad_y - 0.1)
    bottom = round(pad_y + 0.1)
    left = round(pad_x - 0.1)
    right = round(pad_x + 0.1)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )


def clip_to_content_inplace(boxes, pad_x: float, pad_y: float) -> None:
    """把推論尺度的框就地裁進 letterbox 的內容區（填充帶以內）。

    改動前這件事由 ultralytics 做：`predict` 反算座標時順手 `clip_boxes` 把框裁進原圖
    範圍，所以進 tracker 的偵測、以及寫進 parquet 的框都在畫面內。搬走 letterbox 之後
    它只會裁到 640×384 的邊界，也就是**連上下各 12 px 的填充帶都算合法**，於是原本被
    裁掉的部分會在反算時被放大回去——4K 每個推論像素等於 6 個原始像素，實測 4K 攝影機
    約 15% 的框因此外擴，最大 8.3 px（1080p 為 3 倍、影響較小）。

    裁在**送進 tracker 之前**（而非反算之後）是為了與改動前同一個位置：tracker 當時看到
    的就是裁過的框，Kalman 狀態才對得起來。head 框也一起裁到（呼叫端在 `split_detections`
    之前呼叫），配對判準因此也與改動前一致。

    Args:
        boxes: 形狀 `(N, >=4)`、前四欄為 `x1, y1, x2, y2` 的陣列或 torch tensor
            （兩者的 `.clip(min, max)` 語義相同）。
        pad_x: 單邊水平填充量，來自 `letterbox_params`。
        pad_y: 單邊垂直填充量。
    """
    if len(boxes) == 0:
        return
    boxes[:, 0] = boxes[:, 0].clip(pad_x, INFER_WIDTH - pad_x)
    boxes[:, 2] = boxes[:, 2].clip(pad_x, INFER_WIDTH - pad_x)
    boxes[:, 1] = boxes[:, 1].clip(pad_y, INFER_HEIGHT - pad_y)
    boxes[:, 3] = boxes[:, 3].clip(pad_y, INFER_HEIGHT - pad_y)


def unscale_boxes_inplace(
    boxes: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> None:
    """把推論尺度的框座標就地改回原始解析度。

    `boxes` 的前四欄必須是 `x1, y1, x2, y2`；其餘欄（track_id、分數等）不動。
    就地修改是因為呼叫端拿到的是 tracker 每格輸出的小陣列，複製一份沒有意義。

    Args:
        boxes: 形狀 `(N, >=4)` 的陣列，前四欄為框座標。
        scale: `letterbox_params` 給的縮放比例。
        pad_x: 單邊水平填充量。
        pad_y: 單邊垂直填充量。
    """
    if boxes.size == 0:
        return
    boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_x) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_y) / scale


def unscale_points_inplace(
    points: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> None:
    """把推論尺度的點座標就地改回原始解析度。

    落腳點（`FootPointEstimator.estimate` 的輸出）與框走的是同一組參數，但欄位佈局
    不同（`[N, 2]` 的 `x, y`），故另開一支——拿 `unscale_boxes_inplace` 硬套會把
    `x, y` 當成 `x1, y1` 而漏掉一半的欄，且不會有任何錯誤。

    Args:
        points: 形狀 `(N, 2)` 的陣列，兩欄為 `x, y`。
        scale: `letterbox_params` 給的縮放比例。
        pad_x: 單邊水平填充量。
        pad_y: 單邊垂直填充量。
    """
    if points.size == 0:
        return
    points[:, 0] = (points[:, 0] - pad_x) / scale
    points[:, 1] = (points[:, 1] - pad_y) / scale
