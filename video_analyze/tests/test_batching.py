"""批次相關常數的推導關係。

`[model].batch` 就是單次推理批次（issue #116 之前中間隔一個 ×2 的換算），另外兩個常數
由它推導。這裡釘的是推導關係而非絕對值——batch 是可調的，但調它的人不該需要知道另外
兩個常數存在。
"""

from video_analyze.models.config import settings
from video_analyze.services.batching import RING_SLOTS, TARGET_BATCH


def test_target_batch_is_the_configured_value_with_no_conversion():
    """設定值與實際送進 `predict` 的批次之間不再有係數。

    中間隔一個換算時，「引擎綁的最大批次」「`build_engine.py --batch`」「config 的
    `batch`」三者分屬兩種尺度，調參要心算；而算錯的症狀只是慢，不會有任何錯誤。
    """
    assert TARGET_BATCH == settings.model.batch


def test_ring_slots_hold_two_batches():
    """環形緩衝要裝得下一批之外再留同量給 reader 備下一批。

    推理進程整批推論完才歸還 slot（ADR-010），一批會同時扣住同一路最多 `TARGET_BATCH`
    個 slot；總數不足其 2 倍時，reader 在整個推論期間都拿不到空位而完全停擺，**且不會
    有任何錯誤訊息**。issue #116 之前這條由 `start_loop` 開頭的執行期不變量檢查守著，
    公式收斂到單一模組後改由這裡守。
    """
    assert RING_SLOTS == TARGET_BATCH * 2

