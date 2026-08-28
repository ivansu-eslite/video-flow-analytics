# ADR-012：追蹤進程依攝影機分片，輸出改走 part 檔＋主進程合併

- 狀態：已採用
- 日期：2026-08-28
- 影響範圍：`video_analyze`（`services/track_worker.py`、`services/inference.py`、
  `services/pipeline.py`、新增 `services/output_parts.py`）

## 脈絡

追蹤與落盤在 issue #109 移出推論進程之後，追蹤進程成為下一個候選瓶頸。T4 上的實測是
`tracking_fps` 310 對 `overall_fps` 216.7——餘裕只有 1.43 倍。推論進程的前後處理搬上
GPU 之後推論會再快一截，追蹤就會直接接手當瓶頸，屆時那項改動量到的數字會被它蓋住而
低估。

追蹤本身天然可切：各路的 `BYTETracker`（`services/tracker.py`）與 `FootPointEstimator`
（`services/foot_point.py`）狀態按 stream_id 獨立，路與路之間沒有任何共享狀態。切不開
的是**輸出**——整天九路的結果原本由單一 `TrackingResultCollector` 收進單一
`tracking_results.parquet.tmp`，而 `pq.ParquetWriter` 是行程內的檔案 handle，N 個進程
不可能共寫同一份。

## 決策

追蹤進程改為 N 個（`[tracker].shards`，預設 2），各自負責一組固定的攝影機；每片寫自己
的 part 檔，主進程在全部到齊後合併成下游看的那一個檔名。

```
outputs/<bucket>/<date>/
  tracking_results.parquet        ← 契約不變，下游只看這個
  tracking_results.parts/         ← 只在跑到一半時存在
    .lock                         ← flock 認領這一天（0 byte，主進程持有）
    shard0.parquet                ← 該片正常收尾後 rename 出來的
    shard1.parquet.tmp            ← 該片還在寫
```

四件事跟著定下來：

- **路由是啟動時算好的靜態表**，權重取各路的 fps（所有路都跑整天全部片段，工作量與
  fps 成正比），依 `(-fps, stream_id)` 排序後貪婪分配。tie-break 用 stream_id，讓同
  組態的兩輪量測分到一樣的組合。fps 由新增的 `probe_stream_fps` 讀容器標頭取得，不
  解碼任何影格。
- **鎖的對象換成 `parts/.lock`、持有者換成主進程**。認領、跑、合併三段要被同一把鎖
  蓋住，而只有主進程橫跨全程。
- **stream_id 維持全域編號**：每片都收到完整的 `stream_names`／`frame_shapes`，
  payload 與索引都不必轉換，代價只是每片多建幾個用不到的 `BYTETracker`（初始化只是
  幾個空 list）。
- **中斷語義不變**：任何中斷仍讓整天作廢（各片 discard 自己的 part，下次執行清空
  parts 目錄重跑）。

## 理由

### 分片不改變任何一格的結果

三件事各自獨立，合起來讓「分幾片」與輸出內容無關：每路的 tracker 與落腳點推算器狀態
獨立；單一路的 payload 在推論端保序（同一路的框依序 `put` 進同一條 queue）；part 檔
按片分開、合併順序由主進程決定。**改變的只有列順序**——原本是九路按到達順序交錯，
現在是逐片相接、片內才交錯。下游 zone／line 都走 `group_by` 向量化，不依賴列順序。

驗收因此是「N=1 與 N=2 各跑一輪，對齊 `(camera_id, timestamp)` 後逐值相同」，不是逐
byte 比對——`track_id` 的指派本來就不可重現（見根 `CLAUDE.md`）。

### 輸出為什麼必須改成 part 檔

`pq.ParquetWriter` 持有行程內的檔案 handle，N 個進程共寫同一份做不到。可行的只有兩種：
每片各寫一份再合併，或多起一個進程集中落盤。後者被否決——見下節。

### 鎖為什麼跟著主進程走

主進程持鎖後 `fork` 出的所有子進程繼承同一個 open file description，而 flock 屬於
description 不屬於 fd，因此只要任一進程還開著它鎖就在。**issue #113 立起來的保護沒有
退化**：主進程被 SIGKILL 之後孤兒子進程仍守著鎖，另一個執行照樣擋得下。⚠ 這依賴
`fork`；改成 spawn 會靜默失去它。

`claim_parts_dir` 沿用了 `claim_tmp_slot` 的兩個判準，理由一字不變：判斷「有沒有人正
在寫」只能看鎖不能看檔名或 mtime；拿到鎖之後要再確認手上的 inode 還是那個檔名指向的
東西。清殘骸則多一條這一版才有的——**不能把 `.lock` 自己刪掉**。`rmtree` 會把它一起
帶走，鎖就留在一個沒有檔名的 inode 上，另一個執行馬上能在新建的 inode 上取得鎖，兩邊
都以為自己獨佔，而兩邊的輸出檔都完全正常。這與舊版殘檔用 `ftruncate` 而非 `unlink`
清是同一件事。

### 合併的暫存檔放在 parts 目錄裡

不寫 `tracking_results.parquet.tmp`（改版前整天檔的暫存路徑）：那條路徑上可能還躺著
改版前留下的殘檔，而它可能仍被舊版的孤兒追蹤進程持有著 flock、新版已經不看那把鎖，
所以認領時只警告不刪——那就不該再寫過去。放在 parts 目錄裡還有第二個好處：合併中途
崩掉留下的半成品由下一次認領一併清掉，放在目錄外則沒有任何一步會回來收。

### 三處只會靜默出錯的地方

- **路由送錯片**：那片也有全部路的 tracker，會照樣追蹤、照樣寫進自己的 part，只是該
  路的軌跡被切成兩段而 `track_id` 分裂，合併後的檔案完全正常。唯一的訊號是追蹤進程
  入口的 `owned_stream_ids` 檢查——不能依賴 `MultiStreamByteTracker.update` 對未知
  stream_id 回空陣列那條靜默路徑，在「每片都建滿 tracker」的設計下連那條都走不到。
- **`TRACK_FAILED` 卡在已死的那片**：fan-out 是序列的 `put`，而每條 queue 都有上限。
  前一片若已經死了、它的 queue 又是滿的，無 timeout 的 `put` 會永久阻塞，後面的片一個
  都收不到訊號。故 `TRACK_FAILED` 帶 1 秒上限、逾時記 warning 續下一片；`TRACK_DONE`
  反過來不帶——走到那裡各片都在正常消化，而這個訊號掉了就缺一支 part。
- **清 parts 時刪掉 `.lock`**：見上一節。

## 被否決的替代方案

- **第五個進程集中落盤**（各追蹤進程把結果送給它、由它單獨寫整天檔）：多一次跨進程
  序列化，而且把「一個進程寫整天」這個瓶頸原樣留下——依片段落盤的斷點續跑之後多
  writer 本來就是必然，現在多繞一層之後還是要拆掉。
- **`stream_id % N` 當路由**：平衡完全取決於 registry 的排列順序。九路的實際組態
  （4K@20 ×3、1080p@30 ×1、1080p@15 ×5）在最壞排列下會差到近兩倍。
- **每片只收自己那幾路的 `stream_names`／`frame_shapes`**：payload 的 stream_id 要跟著
  轉換成片內索引，`_track_one` 與反算參數的索引全部要跟著改，而換來的只是省下幾個空
  的 `BYTETracker`。
- **`TRACK_QUEUE_SLOTS` 除以 N**：它擋的兩件事（backlog 有界、in-band 訊號趕在父進程
  偵測延遲之前送達）都是每條 queue 各自的性質，除以 N 反而讓正常抖動更容易變成兩個
  進程互等。

## 後果

- `tracking_results.parquet` 的 schema、路徑、下游三包的讀取契約都不變；**列順序改變**。
- 跑到一半時輸出目錄多一個 `tracking_results.parts/`，正常跑完就不存在。`.tmp` 殘檔的
  形態也跟著從「輸出目錄下的一個大檔」變成「parts 目錄裡的數個檔」。
- 合併是一段新的序列尾巴（1–2 GB 的讀入再寫出），耗時與列數記進 log。磁碟峰值在合併
  期間約 2 倍（part 與正式檔並存）。**搬運逐批串流**（`iter_batches`）而不是
  `pq.read_table`：整天 4500 萬列攤成 arrow table 是 4.9 GB，而 `TRACKER__SHARDS=1`
  就是一支 part 裝整天——那條路徑輸出完全正確，只是峰值記憶體跟著天數規模走。
- 緩衝隨 N 線性成長：`_FLUSH_EVERY_ROWS` 是每個 collector 各自的門檻，N=2 時同時開著
  兩個。20 萬列以每列十餘個 Python 物件估約 40–80 MB，加倍後仍在百 MB 級，故本次不動
  這個值、只記錄實測 RSS。
- 依片段落盤的斷點續跑疊上來時，換掉的只是「parts 目錄裡有哪些檔」
  （`shard<k>.parquet` → `<camera>/<segment stem>.parquet`），認領、清理、合併、rmtree
  四條路徑原樣沿用；屆時鎖的語義會反轉（殘留的完整 part 從垃圾變成資產），路由權重會
  從 fps 改成 `fps × 剩餘段數`。
