# vfa 主幹與 argus onprem 交付副本的差異對照

本文件記錄 vfa 主幹與 argus 的 onprem 交付副本（`pipelines/onprem/video_flow_analytics/`）
之間刻意存在的差異，供 vfa 端日後改動這些對應區域時判斷是否需要通知 argus 同步。截至
vfa `2255776` 這個同步基準，兩者的差異只有兩層：

| 層 | 是什麼 | 規模 |
| --- | --- | --- |
| 一、機械轉換 | argus 側的結構決策：共用 lib 命名與 ADR 編號換算 | 全樹 |
| 二、交付適配 | 副本刻意改寫的內容：執行 cwd、視角、交付環境的預設值與共享記憶體 | 19 檔 |

除這兩層之外，argus 那份副本的每一個檔案都必須與 vfa 主幹內容套完第一層後**逐字相同**——
這是那份副本自己的 review 判準，操作細節（合併方式、驗證指令）留在 argus repo，本文件只記
「vfa 端可能需要知道的部分」。

## 第一層：機械轉換（vfa 端不需要處理）

argus 把 vfa 的共用 lib 提升到自己的 `pipelines/shared/`，`vfa_registry`／`vfa_config` 併成
`models`、`vfa_observability` 改名 `observability`；ADR 編號則是 argus 自己的全域流水號，與
vfa 的編號不同步（vfa 的 [ADR-009](../adr/shared/009-head-based-foot-point.md)／
[ADR-010](../adr/video_analyze/010-zero-copy-frame-lifetime.md)／
[ADR-011](../adr/video_analyze/011-single-inference-backend.md) 在 argus 側各自對到另一個
編號）。這層轉換連 `.py` 的 import 與註解都會套用，且是全機械、無需判斷的規則。vfa 端看到
argus 那份副本的 import 路徑或 ADR 連結編號與這裡不同，是這層轉換的結果，不是內容分岐，
不需要 vfa 端維護或執行——操作規則存放在 argus repo 自己的移植文件。

## 第二層：交付適配（19 處）

以下各項是「vfa 的哪些設計假設，在 argus 的交付環境／部署形態下不成立」，argus 因此在自己
的副本裡刻意改寫。**vfa 端未來改動這些對應區域、且改變了現有假設時，這份表能提示「這裡在
argus 有一份對應適配，可能要通知同步」**——理由欄逐字保留自原始移植紀錄。

### 執行環境與敘述類（#1–11）

| # | 檔案 | 適配內容 | 理由 |
| --- | --- | --- | --- |
| 1 | [ADR-007](../adr/shared/007-remove-registry-snapshot.md) | Context／Options／Decision／Consequences 四段改寫 | vfa 原文是從「部署端（argus）已經這樣做了」的視角寫的，抄進 argus 之後變成自我指涉。argus 改寫成部署端視角：指名 `pipelines/serverless/cloud_run_job/jobs/flow_report` 走 `--registry-root` 那條線、`date_{YYYY-MM-DD}/` 分層負責當日凍結 |
| 2–4 | `zone_mapping`／`line_counting`／`flow_report` 的 `README.md` | 執行 cwd 措辭、`bucket_dir` 說明、`camera_registry.yaml` 格式的指路、＋ 整章 golden sample 驗證 | 執行 cwd 是 argus 副本的真實路徑（`pipelines/onprem/video_flow_analytics/`），不是 repo 根；格式指路改指 `../video_analyze/README.md`（vfa 根 README「設定」章在該副本刻意不存在，見 #11）；golden 章是 argus 專屬內容（GCS bucket、日期分層約定、參數綁定的判讀原則），vfa 沒有 |
| 5–7 | 同三包的 `config.toml` | `[input]` 註解的執行 cwd 與 `bucket_dir` 說明 | 同上；`bucket_dir` 不再寫成「本機模擬 GCS bucket」——在 argus 裡 GCS 是真的存在的東西，這個措辭會誤導 |
| 8 | `.gitignore` | `camera_registry.yaml` 一行；overlay 註解引用 argus 自己的 ADR | argus 副本的 bucket 目錄只放 registry（影片不隨副本進來），需要單獨擋 |
| 9 | `video_analyze/config.toml` | ① `[input]` 註解的執行 cwd 與 `bucket_dir` 說明 ② `[model].model_path` 指 `_sm75`（vfa 開發機指 `_sm120`）＋改寫的註解 | ① 同 #5；② vfa 的 `config.toml` 填的是開發機那顆（RTX 5090／sm120），argus 的交付對象是 T4／CustomJob。引擎綁 SM、載入時比對，指錯直接中止——留 sm120 等於交付一份必定啟動失敗的設定 |
| 10 | `video_analyze/README.md` | 七處：① 資料來源敘述 ② `pyyaml` 註解不引用 vfa 的根 `CLAUDE.md` ③ 引擎放置位置 ④ 執行 cwd ⑤ `bucket_dir` 說明 ⑥ 內聯的 `lines[]` 四列規格表＋指向 `../zone_mapping/README.md`／`../line_counting/README.md` ⑦ pytest cwd ＋ 整章 golden sample 驗證 ⑧ 「環境需求」多一列與一節「共享記憶體（`/dev/shm`）」 | ①③④⑤⑦ 是執行位置與資料來源措辭；② vfa 的根 `CLAUDE.md` 沒有隨副本進來，引用會斷；⑧ 交付形態是容器，容器預設的 `/dev/shm` 是 64 MiB、argus 現行組態要 202.5 MiB，不足時**靜默**降速（見 #16–17）；vfa 開發機跑在裸機上，`/dev/shm` 預設是實體記憶體的一半，撞不到；⑥ 這張表是 argus 副本內的唯一正本——三包 README 都指向這裡 |
| 11 | `README.md`（根） | 整份是另一份文件（章節為 目錄定位／四階段怎麼串／在此 workspace 執行／跨套件不變量／架構決策紀錄） | vfa 的根 README 是套件使用手冊（含「設定」章）；argus 副本的根 README 角色是「這個目錄在 argus 裡是什麼、與雲端副本的關係」，**不得長出「設定」章**——三包 README 的交叉連結刻意指向 `../video_analyze/README.md`，長出來會變成兩份規格 |

`.gitignore` 的 `outputs` 尾斜線與 `bucket_*/` 上方三行說明註解曾在某次同步被移除又改回
vfa 版：兩處寫不出適配理由——vfa 是**刻意**不帶尾斜線的，讓 `outputs` 掛成 symlink 到外部碟
時也擋得住，帶了斜線只匹配目錄、保護會消失。這類「寫不出理由的差異」argus 那邊會改回
vfa 版，不會留在交付適配清單裡。

### `tools/` 用法字串的執行 cwd（#12–14）

| # | 檔案 | 適配內容 | 理由 |
| --- | --- | --- | --- |
| 12–14 | `video_analyze/tools/{build_engine,compare_backend,bench_e2e}.py` | 用法字串的執行 cwd（共 5 處，含一則錯誤訊息） | vfa 寫「在 repo 根目錄執行」並附可複製的 `uv run --package video_analyze …`，在 argus repo 根照做會直接失敗：`error: The workspace does not have a member video-analyze`——argus 的 uv workspace 根是 `pipelines/onprem/video_flow_analytics/` |

### 交付環境的預設值與共享記憶體（#15–18）

| # | 檔案 | 適配內容 | 理由 |
| --- | --- | --- | --- |
| 15 | `video_analyze/src/video_analyze/models/config.py` | `ModelConfig` 兩個預設值：`batch` 1 → 16、`source_weights_sha256` 由空字串改成交付引擎的來源權重 hash | 預設值同時是「找不到 `config.toml` 時套用的值」（`load_config` 只記 warning 就以全預設啟動），而交付形態是容器，`config.toml` 不一定跟著進去。vfa 的預設值在那個情境下都會靜默劣化：`batch=1` 退成單格推論，環形緩衝格數與 track queue 容量跟著縮，不報錯、只是吞吐垮掉；`source_weights_sha256` 留空則引擎的來源權重不再比對，唯一擋得下「換成另一顆 id 剛好都存在、語義卻不同的權重」的檢查消失 |
| 16–17 | `video_analyze/src/video_analyze/services/pipeline.py` ＋ `video_analyze/tests/test_pipeline_shm_log.py` | 配置環形緩衝之前多一行 log（`log_shm_requirement`）：`shm_total_mb`／`shm_available_mb`／`required_mb`／`num_streams`／`ring_slots`，**只記不擋**；加一支測試釘住這五個欄位 | `/dev/shm` 不足時 CPython 靜默改用 `/tmp`，程式照跑、`exit 0`、輸出完全正確，只有吞吐垮掉。vfa 的 `create_ring_buffer` 逐塊記 `backing_dirs`，看得出「這一塊掉出去了」，看不出「這個環境配了多大、這次要多少」——而交付形態是容器，Docker 預設的 64 MiB 正好卡在 202.5 MiB 的需求之下 |
| 18 | `video_analyze/src/video_analyze/main.py` ＋ `video_analyze/tests/test_main_settings_log.py` | `main()` 檢查完 `[input].date` 之後、呼叫 `analyze_daily` 之前，多記一行 `生效設定` log（`model`／`tracker`／`foot_point`／`input` 四區塊的 `model_dump(mode="json")`）；加一支測試釘住 | 交付形態是 Vertex AI CustomJob，除了 log 沒有別的東西可看。`load_config()` 找不到 `config.toml` 只記 warning 就以預設值啟動，環境變數又能逐欄覆寫，事後要知道這次到底用了哪組 tracker 門檻、`foot_point.method`、`batch`、`model_path`、`source_weights_sha256`，只有這一行 |

`config.py` 與 `config.toml`（#9）的 `model_path`／`source_weights_sha256`／`batch` 三欄在
argus 副本裡刻意逐字相同：兩邊填不同顆引擎的話，「有沒有把 `config.toml` 複製進容器」會變成
「換一顆引擎」，而兩條路徑都跑得起來。

#16–17 與 vfa 自身在 `310b576`／`7eba3e0` 兩次改動的處置方向相反，不只是強弱不同：vfa 那兩次
改動是主動擋下 `/dev/shm` 不足；argus 副本因為雲端 `/dev/shm` 的真值還沒實測過，選擇先由
第一次執行讀出來、只記 log 不擋，等有真值再談門檻。vfa 端若調整這條路徑的行為，不能機械地
假設 argus 會照搬——這是一個要重新確認的決定，不只是版本追新。

### argus 專屬新增：`video_analyze/vertex_ai_image/`（vfa 沒有這個目錄）

argus 專屬的容器建置目錄（`Dockerfile`／`pyproject.toml`／`.dockerignore`／`README.md`，
來自 argus `feat/54`），對應雲端那顆 image 要長成什麼樣才跑得動同一條 TensorRT 推論鏈；vfa
沒有對應物，onprem 副本本身在地端裸機上以 `uv run --package` 跑、不經容器，這個目錄不影響
vfa 的 workspace 成員清單。

## 不隨同步進來的東西

- **`CLAUDE.md`**：vfa 的 repo 級指示，對 argus 無意義。
- **`libs/`**：已由 argus 自己的 ADR 提升到 `pipelines/shared/`。vfa 對 `libs/` 的改動要
  argus 端判斷是否適用於 `pipelines/shared/`，**不是自動套用**——曾有一次 vfa 移除了
  `[output]` 區塊的敘述（因為 vfa 四包都沒有 `[output]` 了），但 `pipelines/shared/models`
  是 argus 雲端也共用的，而 `pipelines/vertex_ai/` 至今仍有 `save_video` 與 `[output]`，
  那句敘述在 argus 是正確的，照抄反而會改錯。

## 已知缺口（記著，不是待辦）

- **argus 基底自身的共用 lib 改名沒做完**：`pipelines/shared/` 落地時，散文與 docstring 裡
  還留著 `vfa_registry`／`vfa_config` 的舊名（測試 docstring、README、部分 ADR 自身）。這是
  argus 自己的機械轉換規則只動 import 與依賴宣告、不動散文所致，與 vfa 無關。
- **`video_analyze/tools/bench_e2e.py` 少一道參數檢查**：`--label` 與 `--machine` 都過
  `check_name_component`，`--foot-point` 沒有——但它同樣會進產物檔名（`_b{batch}_{foot}_r{n}`）
  與子進程環境變數。打錯字要等到每個 bucket 的 `outputs/` 都被 `rmtree` 過才會發現。這是
  vfa 自身既有的缺陷，發現於移植比對時，不在這份對照文件修——要修請在 vfa 開 issue。
- **`pipelines/shared/models` 與 `observability` 的獨立 pytest 跑不起來**
  （`ModuleNotFoundError: No module named 'models'`）。這兩個 lib 的程式碼由 vfa 四包的測試
  涵蓋，但 argus 自己那套獨立測試在基底上就是壞的，與 vfa 無關。

## argus 那份副本不在其 CI 涵蓋範圍內

這是 argus 那邊既定的設計、不是待補的缺口——交付副本不執行、也不出貨，不代表 vfa 自己沒有
CI。後果是本文件登記的每一條差異，argus 端漏跑驗證不會有任何訊號，同步時要逐條實跑並把
輸出留在該側 PR 說明裡。

## 相關文件

若要判斷 vfa 主幹與 argus onprem 交付副本（`feat/53`）目前的程式碼差異是否需要回流，見
另一條線（P32）產出的差異表與裁定紀錄；那份文件是決策用的差異表，本文件是說明性的差異
成因對照，兩者用途不同。
