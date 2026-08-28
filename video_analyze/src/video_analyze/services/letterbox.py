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

縮放做在**解碼出來的 nv12 平面上**（`letterbox_nv12`），不是先轉成 BGR 再縮：搬動的
資料量只有一半，`cvtColor` 也只需處理縮小後的畫面。這裡沒有 BGR 版的縮放函式，因為
讀取端拿到的一律是 nv12（NVDEC 硬解的輸出格式），留一份 BGR 版只會讓兩條路各自漂移。
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


def letterbox_params(orig_height: int, orig_width: int) -> tuple[float, int, int]:
    """算出等比縮放的比例與左／上緣的填充量。

    與 ultralytics 同一套算法：取長短邊比例的較小者，縮完置中填充。1080p 與 4K 都是
    16:9，因此兩者都縮成 640×360、上下各填 12——**填充量相同、只有 `scale` 不同**，
    這也是不同解析度的來源在 rect 模式下能共用同一形狀的原因。

    填充量取 `round(pad - 0.1)` 的整數而非 `.5` 的分數，是為了與 ultralytics 的
    `ops.scale_boxes` 逐字一致：置中填充在單邊除不盡時，內容實際是從整數那一格開始的
    （左／上取小、右／下取大）。回傳分數的話，非 16:9 來源的每個 x 都會偏約 0.75 個
    原始像素——16:9 剛好整除（pad 0 與 12），看不出來。

    Args:
        orig_height: 原始影格高度（pixel）。
        orig_width: 原始影格寬度（pixel）。

    Returns:
        `(scale, pad_x, pad_y)`；`scale` 是原圖到推論尺寸的縮放比例，兩個 pad 是
        **左緣與上緣**的填充量（縮放後的內容就從這個座標開始）。
    """
    scale = min(INFER_WIDTH / orig_width, INFER_HEIGHT / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)
    pad_x = round((INFER_WIDTH - new_width) / 2 - 0.1)
    pad_y = round((INFER_HEIGHT - new_height) / 2 - 0.1)
    return scale, pad_x, pad_y


def letterbox_nv12(planes: np.ndarray, height: int, width: int) -> np.ndarray:
    """把 nv12 影格在兩個平面上縮到推論尺度，轉成 BGR 後置中填充到推論尺寸。

    縮放與色彩轉換的順序是反過來的：先在 Y 與交錯的 UV 兩個平面上各自 `cv2.resize`，
    縮完才 `cvtColor` 成 BGR。與「先轉 BGR 再縮」得到的畫面不逐位元相同——色度是在
    1/2 解析度上插值的——但省下的是縮放與色彩轉換兩邊的搬運量（nv12 每像素 1.5 byte
    對 BGR 的 3 byte），實測讀取端 CPU 地端 2.03 → 0.88 核、T4 6.093 → 1.343 核。

    縮放後的尺寸與 `letterbox_params` 給的是同一組（`round(邊長 × scale)`），填充量
    因此也沿用同一組參數，反算才對得回原始解析度。

    Args:
        planes: 解碼出的 nv12 影格（`av` 的 `frame.to_ndarray()`），形狀為
            `(height * 3 // 2, width)`：前 `height` 列是 Y，其後 `height // 2` 列是
            交錯的 UV（色度本來就是 1/2 解析度）。
        height: 影格高度（pixel）。
        width: 影格寬度（pixel）。

    Returns:
        `(INFER_HEIGHT, INFER_WIDTH, 3)` 的 BGR 陣列。

    Raises:
        ValueError: `planes` 的形狀不是 `(height, width)` 對應的 nv12 佈局（呼叫端拿
            到的不是 nv12，或尺寸與宣告的不符）；或等比縮放後的寬高不是偶數。
    """
    if planes.shape != (height + height // 2, width):
        raise ValueError(
            f"nv12 影格的形狀應為 {(height + height // 2, width)}（前 {height} 列 Y、"
            f"其後 {height // 2} 列交錯 UV），實得 {planes.shape}：呼叫端拿到的畫面"
            "不是 nv12，或尺寸與宣告的不符。"
        )
    scale, pad_x, pad_y = letterbox_params(height, width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    if new_width % 2 or new_height % 2:
        # 色度平面是 2×2 取樣，寬高其一為奇數就拼不回 nv12 佈局。此處只能 fail loud，
        # 不能把尺寸調成偶數：那會讓實際縮放比例與 `letterbox_params` 的 `scale` 分岔，
        # 而反算用的是後者——每個座標靜默偏掉，輸出卻完全正常。16:9 來源（1080p 與 4K）
        # 縮完是 640×360，兩邊都是偶數；非 16:9 要重新對過縮放與填充參數。
        raise ValueError(
            f"{width}×{height} 等比縮到 {new_width}×{new_height} 後寬高不是偶數，"
            "無法在 nv12 的色度平面（2×2 取樣）上縮放：本模組假設來源為 16:9，"
            "接入其他長寬比要重新對過縮放與反算參數。"
        )
    luma = cv2.resize(
        planes[:height],
        (new_width, new_height),
        # 縮小一律用 INTER_LINEAR，與 ultralytics LetterBox 一致；換插值法會讓偵測
        # 結果與改動前對不起來，而那個差異看起來會像是「把縮放搬到讀取端造成的」
        interpolation=cv2.INTER_LINEAR,
    )
    chroma = cv2.resize(
        # 交錯的 UV 攤成兩通道才縮得對：當成單通道縮的話 U 與 V 會互相混進對方
        planes[height:].reshape(height // 2, width // 2, 2),
        (new_width // 2, new_height // 2),
        interpolation=cv2.INTER_LINEAR,
    )
    packed = np.empty((new_height + new_height // 2, new_width), dtype=planes.dtype)
    packed[:new_height] = luma
    packed[new_height:] = chroma.reshape(new_height // 2, new_width)
    resized = cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12)
    # 右／下緣吃掉除不盡的那半格（單邊除不盡時比左／上多 1），與 ultralytics 的
    # `round(pad + 0.1)` 等值。內容的起點就是 `letterbox_params` 給的 pad，反算才對得回去
    bottom = INFER_HEIGHT - new_height - pad_y
    right = INFER_WIDTH - new_width - pad_x
    return cv2.copyMakeBorder(
        resized,
        pad_y,
        bottom,
        pad_x,
        right,
        cv2.BORDER_CONSTANT,
        value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )


def content_box(orig_height: int, orig_width: int) -> tuple[int, int, int, int]:
    """縮放後的畫面內容在推論尺度上佔的矩形（填充帶以內的範圍）。

    右／下邊界不能用 `INFER_WIDTH - pad_x` 反推：置中填充除不盡時右／下比左／上多
    1 px，那樣算會讓裁切邊界鬆掉一格（4K 下等於原始解析度的 6 px）。

    Args:
        orig_height: 原始影格高度（pixel）。
        orig_width: 原始影格寬度（pixel）。

    Returns:
        `(x1, y1, x2, y2)`，推論尺度上的內容區。
    """
    scale, pad_x, pad_y = letterbox_params(orig_height, orig_width)
    return (
        pad_x,
        pad_y,
        pad_x + round(orig_width * scale),
        pad_y + round(orig_height * scale),
    )


def clip_to_content_inplace(boxes, content: tuple[int, int, int, int]) -> None:
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
        content: 該路的內容區 `(x1, y1, x2, y2)`，由 `content_box` 算出。
    """
    if len(boxes) == 0:
        return
    x1, y1, x2, y2 = content
    boxes[:, 0] = boxes[:, 0].clip(x1, x2)
    boxes[:, 2] = boxes[:, 2].clip(x1, x2)
    boxes[:, 1] = boxes[:, 1].clip(y1, y2)
    boxes[:, 3] = boxes[:, 3].clip(y1, y2)


def unscale_boxes_inplace(
    boxes: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> None:
    """把推論尺度的框座標就地改回原始解析度。

    `boxes` 的前四欄必須是 `x1, y1, x2, y2`；其餘欄（track_id、分數等）不動。
    就地修改是因為呼叫端拿到的是 tracker 每格輸出的小陣列，複製一份沒有意義。

    Args:
        boxes: 形狀 `(N, >=4)` 的陣列，前四欄為框座標。
        scale: `letterbox_params` 給的縮放比例。
        pad_x: 左緣填充量。
        pad_y: 上緣填充量。
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
        pad_x: 左緣填充量。
        pad_y: 上緣填充量。
    """
    if points.size == 0:
        return
    points[:, 0] = (points[:, 0] - pad_x) / scale
    points[:, 1] = (points[:, 1] - pad_y) / scale
