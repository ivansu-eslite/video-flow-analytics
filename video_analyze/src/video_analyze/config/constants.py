"""非 Pydantic 的靜態常數：輸出根、追蹤結果檔名與 parquet schema。

與各套件慣例一致，把輸出契約（路徑、檔名、欄位 schema）從邏輯模組抽出集中於此；
模組私有的調校常數（湊批等待、flush 門檻等）仍留在各自模組。
"""

from pathlib import Path

import polars as pl

# 輸出根目錄（cwd 相對）：實際輸出會再掛上 bucket 名稱，避免不同 bucket 互相覆蓋。
OUTPUT_ROOT = Path("outputs")

# 追蹤結果 parquet 的檔名。
TRACKING_RESULTS_FILENAME = "tracking_results.parquet"

# 各追蹤進程的 part 檔所在目錄，與正式輸出同一層（`<date>/` 底下）。整天分片跑的期間
# 才存在，合併完就刪掉；名字由正式檔名推導，兩者因此不可能各自漂移。
TRACKING_RESULTS_PARTS_DIRNAME = Path(TRACKING_RESULTS_FILENAME).stem + ".parts"

# parts 目錄裡的鎖檔名，永遠 0 byte。認領這一天的是**主進程**（見
# `services/output_parts.py`），不是任何一個追蹤進程——認領、跑、合併三段要被同一把鎖
# 蓋住，只有主進程橫跨全程。
PARTS_LOCK_FILENAME = ".lock"

# 權重的類別 id（CrowdHuman：0=head, 1=vbody, 2=fbody）。fbody 是追蹤目標，head 只
# 用來推算落腳點、不進 tracker（否則同一個人會多出一條頭部軌跡）。`_validate_classes`
# 只驗證 id 存在於權重，換權重時這兩個常數要跟著對。
HEAD_CLASS_ID = 0
FBODY_CLASS_ID = 2

TRACKING_RESULTS_SCHEMA = {
    "camera_id": pl.Utf8,
    "frame_id": pl.Int64,
    # timestamp 為台北在地時間：檔名為 UTC，已在 services/video_reader.py 解析時轉換成
    # 台北（見該檔 _FILENAME_TZ / _LOCAL_TZ），schema 標記需與來源 tzinfo 一致。
    "timestamp": pl.Datetime("us", "Asia/Taipei"),
    "track_id": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
    # 落腳點（人站在地面的位置）：由 head 框推算；配不到頭時先沿用該軌跡上次的偏移量，
    # 連偏移量都沒有（或已過期）才退回 `((x1+x2)/2, y2)`。
    # 算在這裡而非讓下游各自從 bbox 現算，是因為推算需要 head 框，而 head 不進追蹤
    # 結果；下游（line_counting／zone_mapping／overlay）一律讀這兩欄，見 ADR-009。
    "foot_x": pl.Float64,
    "foot_y": pl.Float64,
    # 該路影像尺寸（整天固定，見 services/video_reader.py 的 probe_frame_shape）。
    # 逐列重複同一個常數，靠 parquet 的 dictionary/RLE 壓縮吸收；下游（line_counting）
    # 是純 CPU 套件、不掛載影片，只能從這裡取得尺寸做解析度相關的參數換算。
    "frame_width": pl.Int64,
    "frame_height": pl.Int64,
}
