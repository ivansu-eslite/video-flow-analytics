# 移植 roadmap：三包 DDD 對齊並交付 argus/pipelines/onprem/

把 `video-flow-analytics`（vfa）三包（flow-report／zone-mapping／video-analyze）重構成 argus DDD
骨架，並同步交付到 `argus/pipelines/onprem/<pkg>/`。**本檔是整個工程的 master 參考——每次要為某個
工作單元寫 plan 前先讀本檔；§2 的共用決策各 plan 直接引用、不重寫，plan 只寫該包特化。**

---

## 0. 策略（2026-07-22 更新：byte-identical → DDD 對齊）

移交鏈：**vfa → `argus/pipelines/onprem/<pkg>` → serverless cloud-run job**。
vfa→onprem 由本人開發；onprem→serverless 的 DDD 重構由同事執行。

**現行策略**：在 onprem 階段就採用 argus 的**平台無關** DDD 慣例，把同事日後 onprem→serverless 的
diff 壓到只剩雲端接線（GCS／Dockerfile／env 注入／workspace membership）。以 flow-report 當**驗證樣板**
（其 serverless 版同事已完成，可逐項對照量測殘留 diff），確認方法可行再套 zone-mapping／video-analyze
（那兩包才是真正省同事 diff 的對象）。

> **舊策略已作廢**：本檔早期主張「維持 uv+config.toml+src-layout 原樣、不改寫成 argus 慣例」，並以
> byte-identical copy 交付（flow-report 首次移植即此做法，argus PR #17）。改弦理由：同事已示範 serverless
> DDD 重構（origin/main 的 `pipelines/serverless/cloud_run_job/jobs/flow-report`），與其讓他每包重做結構，
> 不如 onprem 先對齊。**vfa 與 onprem 仍維持 src byte-identical——只是兩邊都是 DDD 版。**

---

## 1. 工作單元順序與狀態

> **順序已於 2026-07-22 改版（編號沿用舊的、不重新編，避免既有 plan／issue 交叉引用失效）。**
> **新執行順序：`5(vfa 半) → 6 → 【argus 一批：2 + 4 + 5(argus 半) + 7】`**，
> 即 **vfa 側全部做完定形，argus 側整批暫緩**。
>
> **改版理由**：原順序是 `1→2→3→4→5→6→7`，每包「vfa 先、onprem 後」逐包交付。問題在於
> **成本不對稱——vfa 是自己的 repo、迭代免費；argus 每次交付都要別人 review**。按舊順序，單元 4 會先送一版
> 自帶 registry 副本的 zone-mapping 給人 review，等單元 5 抽 lib 時再改寫同一份程式碼＝**同一段碼 review 兩次**。
>
> **前置條件已於 2026-07-22 達成**：抽取原本卡在「要有兩個 DDD 形狀的消費者才知道 lib 介面」，單元 3 完成後
> 實測 `flow-report/models/registry.py` 與 `zone-mapping/models/registry.py` **程式碼完全相同、所有差異都在
> docstring**；`observability/structured_logging.py` 兩份 **byte-identical**。介面已由現實決定，可以抽了。
>
> **抽取仍排在 video-analyze(6) 之前**（此理由不變）：否則 video-analyze 會把精簡 registry 搬成
> `models/registry.py`＝**第三份副本**。完整版只依賴 pydantic+pyyaml（video-analyze 本來就有），
> 且 zone 解析 lazy、不呼叫就不驗證幾何，行為不變。附帶好處：最難的 video-analyze（GPU／多進程／
> golden 不可重現）排最後，面對已定型的結構。
>
> **argus 側暫緩的兩個實測理由**（動工前必須先解決，見單元 5 列）：
> ① `pipelines/onprem/` **沒有 uv workspace**——serverless 的 libs 掛在
> `pipelines/serverless/cloud_run_job/pyproject.toml` 的 workspace（members = 2 jobs + `libs/argus-gcs`），
> 但 onprem 底下 `flow-report`／`rtsp-stream` 是**各自獨立的 uv 專案、無共同根**。要用 `{ workspace = true }`
> 得先建 workspace 根，會動到**已合併的 flow-report** 與**同事的 rtsp-stream**。這才是抽 lib 的主要成本。
> ② serverless 的 registry 還是 **job-local**——onprem 改吃 `libs/` 會讓 flow-report 的 onprem→serverless
> diff **變大**，與「壓縮同事 diff」的第一目標打架。緩解點是 `libs/argus-gcs` 已是既有 pattern，
> 「serverless 也開 `libs/argus-registry`」對同事不陌生，**但這要先跟他講好，不可單方面決定**。

| # | 單元 | repo | 狀態 |
|---|---|---|---|
| 1 | flow-report DDD 重構 | vfa | ✅ **完成**：骨架 PR #43 + 函式級對齊 PR #45（issue #44，已合併）。main 現為 **`28ca959`**（weekday_zh 已除、yaml 防呆／Polars dedup 已入） |
| 2 | flow-report 同步 **+ golden 遷移 GCS** | argus onprem | ⏸ **地端完成、暫緩交付（2026-07-22 決策）**：三筆 commit 已在本機 worktree，**不 push、不開 PR**，等 argus 整批交付時一起決定形狀（可能要改成吃 `libs/`）。plan `indexed-marinating-mitten.md`（`~/.claude/plans/`，未進版控）。argus worktree `../argus-flow-report`、分支 `feat/flow-report-ddd`（從 origin/main 開）——**已決定跳過建 issue**，故分支名無編號。DDD 同步＋golden 遷移＋README 三筆 commit **已在本機完成**（`4c8b588`／`d9a1274`／`98aa739`），**尚未 push、尚未開 PR**。來源＝vfa main **`28ca959`**（已含對齊，直接複製、勿再改）。**單一 PR 同時做兩件事**：① DDD 結構同步 ② golden 遷移到 GCS（刪 repo 內 `golden/`＋`outputs/` 輸入樹、改 `.gitignore`、改寫 README golden 章節）。捆一起的理由：argus 每次更新需他人 review，且 golden 那段**無程式碼變更**、拆開會把 README golden 章節與驗證各做兩遍。commit 逐關注點分離、PR 標題框在「更新到目前交付標準」單一主題。E2E golden 儲格值比對已實測通過（乾淨重拉 GCS→重跑→比對） |
| 3 | zone-mapping DDD 重構 | vfa | ✅ **完成（2026-07-22）**：issue #46、分支 `refactor/46-zone-mapping-ddd`。骨架＝flow-report 同形（`main`／`config/constants`／`models/{config,registry}`／`services/{zone_map,stats}`／`observability`）＋pydantic-settings＋find_project_root＋ruff 100；registry 已補 yaml 防呆，兩份完整版驗證邏輯完全一致。golden（GCS）重跑 `zone_counts.parquet` **byte-identical**，重構前後各驗一次 |
| 4 | zone-mapping 交付 | argus onprem | ⏸ **暫緩（隨 argus 整批）**——交付時**直接吃共用 lib，不送自帶副本的版本**。原因：`feat/20`／argus #20 原走 byte-identical，改為 DDD；已做的 ruff-100／動態 project-root 可留用。**golden 一開始就用 GCS、無遷移問題**。動工時要修 `feat/20` README 的**過期敘述**：寫「GCS 存取尚未就緒」（已就緒）、路徑寫成 `reference/zone_mapping/`（正確為 `reference/golden_samples/zone_mapping/`）|
| 5 | **抽共用 lib（registry + structured_logging）** | vfa 半 ✅ **完成（2026-07-23）**；argus 半隨整批 | **vfa 半已完成（issue #48）**：`libs/vfa-registry`＋`libs/vfa-observability`，三包以 `[tool.uv.sources]` **path 依賴（editable）** 引用、**未建 uv workspace**（workspace 的單一 root `.venv` 會讓 video-analyze 的 torch 外溢，破壞依賴面收斂；path 依賴也讓 `uv sync --project`／執行 cwd 約束完全不變）。命名 `vfa-*` 而非 `argus-*` 是使用者定案，移進 argus 時若要改成 argus 命名慣例需一次改名 diff。video-analyze 已一併吃 lib、刪除精簡副本（順帶補上它欠缺的 yaml 防呆），**未產生第三份副本**。CLAUDE.md「三包共用碼的處理方式」已改寫並寫明推翻舊決策的理由。golden：zone-mapping `zone_counts.parquet` byte 一致、flow-report `report.xlsx` 儲格值一致；video-analyze 不跑 golden（僅 import 置換，不觸及推理路徑）。以下為原規劃備忘（vfa 半已全數執行、三件待定事項皆已定案），保留供 argus 半參考：抽取範圍除 registry 外**加上 `observability/structured_logging.py`**（flow-report／zone-mapping 兩份實測 byte-identical）。**由你自己做——同事只管 serverless、不會碰 onprem 的重複**。<br>**vfa 半當時要定的三件事（皆已定案）**：① vfa 三包目前也**刻意無 workspace**（CLAUDE.md 明寫「刻意重複而非共用 lib，因三包未來可能各奔不同平台」）——抽 lib 與此直接衝突，**必須一併改寫該段並寫明新理由**；② docstring 合併——兩份程式碼相同、差異全在敘述（各自寫了本包視角），要合成中性版本，`load_registry_from_path` 的 `Raises: ValueError` 採 zone-mapping 那份（flow-report 沒列）；③ **video-analyze 是否為消費者**：其精簡 registry 是完整版的**子集**（148 行 vs 253，無 `Zone`／`parsed_zones`／`parse_and_validate_zones`，`zones: list[Any]` 純忽略），原則上可直接吃完整版，但需逐欄確認 `StorageConfig`／`CameraEntry`／`stream_dirname` 無反向差異。<br>**argus 半：暫緩**，卡在上方 ① onprem 無 workspace ② serverless registry 仍 job-local（需與同事對齊）。serverless 那邊由同事在其湊到 2 個 job 時獨立抽，兩邊時間軸不同 |
| 6 | video-analyze DDD 重構 | vfa | 待做（在單元 5 vfa 半之後）——**直接依賴單元 5 的共用 lib，不建第三份 registry**（原精簡 `registry.py` 刪除）。**順帶收斂：三包 config 的巢狀 model（`InputConfig`／`ZoneConfig`／`ReportConfig`／`TrackerConfig` 等）目前都是預設的 `extra="ignore"`，欄位名打錯（如 `entry_debounce_frame` 少個 s）會被靜默忽略而套用預設值、統計口徑悄悄改變。`extra="forbid"` 只在頂層 `AppConfig` 生效（vfa #46 review 實測）。三包一起加，避免 config 語義分岔；此事**不會被單元 5 的抽取消滅**——抽的只有 registry，config 刻意各包分開** |
| 7 | video-analyze 交付 | argus onprem | ⏸ **暫緩（隨 argus 整批）**。**golden 一開始就用 GCS、無遷移問題**；inputs 是 5.54 GiB 影片樹，用 `gcloud storage rsync --recursive`（且**先關掉** parallel composite upload，見 §2）。yolo 權重在 `reference/models/yolo/`（Vertex AI 版本保存尚未完成，見 #22）|

- 每個單元 = **1 plan → 1 issue → 1 PR**（vfa／argus 不同 repo，各自 PR）。
- 選配：argus 開父 epic issue「DDD 對齊 onprem 三包」掛子 issue（#20 等）做 GitHub 端進度追蹤。
- flow-report 既有 plan `mighty-snuggling-hopcroft.md` 綁了 vfa+argus，實作時依單一 scope 拆成單元 1、2。

---

## 2. 共用 DDD 決策（各 unit 的 plan 引用此節，不重寫）

**目標骨架**（`src/<pkg>/`，對齊 origin/main serverless 版）：
- `main.py`：CLI 外殼（argparse／讀 `settings`→組參數→呼叫 service 入口）。
- `config/constants.py`：**非 Pydantic** 靜態常數（分頁名、表頭、閥值、預設路徑）。
- `models/{config,registry}.py`：所有 Pydantic 模型（skill `argus-pydantic-architecture` **強制**放 `models/`）。
- `services/*.py`：核心邏輯（盡量純函式；I/O 邊界包 try-except）。
- `observability/structured_logging.py`：抄 argus `StructuredLogger`。

**config → pydantic-settings**：`AppConfig(BaseSettings)` + `SettingsConfigDict(toml_file=_get_toml_path(),
env_nested_delimiter="__")` + `settings_customise_sources` 回 `(init, env, TomlConfigSettingsSource)`；樣板抄
origin/main `jobs/flow-report/src/flow_report/models/config.py`。巢狀 model 原封不動；維持全域單例
`settings = load_config()`。新增依賴 `pydantic-settings[toml]`。

**find_project_root**：往上找 `pyproject.toml`，取代 `Path(__file__).parents[N]`（config 搬進 `models/` 後
深度會錯，故此替換是機械必要，非額外選擇）。

**structured_logging**：抄 argus `StructuredLogger`（`__init__(*, component)`；`.info/.warning/.error(msg,
**fields)`、`.exception(msg, *, error)`）。⚠ **video-analyze 多進程需另設計**（見 §5）。

**registry：先 reconcile、後抽 lib（分兩階段）**
- **單元 1、3（flow-report／zone-mapping）**：registry 搬成 job-local `models/registry.py`，並**修成兩份
  一致**（flow-report 的 `resolve_cameras` 補 dup-check，與 zone-mapping 的 `bedbd75` 對齊）。目的是讓單元 5
  抽 lib 時是**零思考搬移**（不用先解分歧）。
- **此時不抽 lib 的理由**：只有 1 個完整版共用者（單元 1 階段），且 serverless 那邊 registry 也還是 job-local
  （`models/registry.py`，未進 `libs/`）——提早抽會讓 onprem 跑到 serverless 前面、破壞對齊。
- **單元 5 才抽**（見 §1）：1–4 完成後達到 argus「2+ 共用且 API 穩定才抽」的門檻，由**你自己**抽
  （同事只做 serverless、不會替 onprem 抽）。
- **單元 6（video-analyze）不再有自己的 registry**：抽取已在單元 5 完成，video-analyze 直接依賴共用 lib、
  刪掉原本的精簡 `registry.py`——**全程不產生第三份副本**。它不呼叫 `parse_and_validate_zones`，zone 幾何
  仍不會被驗證，行為與現況一致。
- serverless 那邊由同事在其湊到 2 個 job 時獨立抽；兩邊各照 2-sharer 門檻收斂，只是時間軸不同。
- **待同步（vfa PR #45／issue #44 遺留）**：flow-report 的 `load_registry_from_path` 已補上
  `yaml.safe_load` 後的 `isinstance(data, dict)` 防呆（空檔／純註解檔會讓 `CameraRegistry(**data)`
  以不清楚的 `TypeError` 崩潰），但 `zone-mapping`／`video-analyze` 的對應讀取邏輯尚未同步加上。
  ✅ zone-mapping 已於單元 3（2026-07-22）補上；✅ **video-analyze 已於單元 5（2026-07-23）
  隨改吃共用 lib 一併補上，本條結案**——三份副本已收斂成 `libs/vfa-registry` 一份。

**慣例對齊**：ruff `line-length=100`；observability 檔名用 `structured_logging.py`（對齊 serverless
flow-report／opencv-job，而非 gce 的 `logging.py`）。

**殘留 diff 原則（刻意保留、非漏做，PR 說明標為「預期差異」）**：
- `InputConfig` 保留 `bucket_dir`（**不**改成 serverless 的 `input_root`/`output_root`）——IO 邊界差異。
- `StructuredLogger` 用 `ensure_ascii=False`（中文 log 地端直接可讀）——serverless 用 `True` 對接 Cloud
  Logging。onprem 刻意保留 `False`。
- `main.py` 保持薄 CLI 外殼、registry 保留 `bucket_dir` helper（`registry_path`/`load_registry`）——
  雲端接線／IO 邊界，留給同事。

**規範位階（採用時心裡有數）**：只有 **pydantic/models 是 skill 強制**（`argus-pydantic-architecture`）；
**DDD 目錄與 structured_logging 是 serverless 域的文件指南**（`docs/pipelines/serverless/README.md`「強烈
建議」／「一律」），管不到 onprem；**find_project_root 非規範**，只是 `parents[N]` 的穩健替代。onprem 現無
架構規範文件——採用這些是「提前對齊下游」，非合規義務。

**文件同步（每個單元的 definition of done 一部分）**：重構要一併更新——
- 該包 vfa `README.md`：模組結構／執行／設定章節（DDD 目錄、entry point 由 `…pipeline:run_report` 改
  `…main:…`、config→pydantic-settings）。
- vfa `CLAUDE.md`：受影響段落（「三包共用碼的處理方式」的 registry/config 描述、常用指令、模組結構引用）；
  「時區不變量」「不可重現」等與重構無關的段落不動。
- argus onprem 該包 `README.md`：見 §3 步驟 5／§4 坑 6（執行章節改寫 + DDD 結構反映）。
- 三份 `registry.py` reconcile 後，記得更新 CLAUDE.md 中「三份無自動同步機制」那句的措辭。

**golden sample 一律放 GCS、不進 git**（權威來源：argus issue #22 的路徑表；2026-07-22 已全部上傳完成）

Bucket：`gs://eslite-minority-report-dev-argus-data-storage`，base：`reference/golden_samples/`

| 階段 | 角色 | 檔案 | 路徑（base 之下）|
|---|---|---|---|
| video-analyze | inputs | 48 支 `.mkv` + `camera_registry.yaml`（5.54 GiB）| `video_analyze/inputs/test_cam00{1..4}/2026/05/01/` |
| video-analyze | expected | `tracking_results.parquet` | `video_analyze/expected/2026-05-01/` |
| zone-mapping | inputs | `tracking_results.parquet`、`camera_registry.yaml` | `zone_mapping/inputs/2026-05-01/` |
| zone-mapping | expected | `zone_counts.parquet`、`camera_registry_used.yaml` | `zone_mapping/expected/2026-05-01/` |
| flow-report | inputs | `zone_counts.parquet`、`camera_registry_used.yaml` | `flow_report/inputs/2026-05-01/` |
| flow-report | expected | `report.xlsx` | `flow_report/expected/`（**不加日期層**，它是跨日累加產物）|

- 路徑設計：影片保留 `test_camXXX/{YYYY}/{MM}/{DD}/` 讓 `bucket_dir` 可直接指向 `inputs/`；逐日產物多一層
  日期讓多日 golden 並存；**下一階段的 `inputs/` 與上一階段的 `expected/` 內容相同、刻意各存一份**，讓每個
  階段獨立取用自己的輸入、不必知道上游路徑。
- **抓取**：`gcloud storage cp`（單檔）／`gcloud storage rsync --recursive`（影片樹）到本機後再 `uv run`
  ——**不改程式讀 GCS**（沿用 #22 界線：pipeline 仍讀本機路徑）。
- **上傳大檔前必須**：`gcloud config set storage/parallel_composite_upload_enabled False`，否則暫存碎塊會寫到
  bucket 根的 `gcloud/tmp/`、被 IAM Condition 擋掉（#22 實測 48 支中 12 支大檔因此失敗）。
- **比對**：xlsx 用 openpyxl 儲格值（非 byte）；parquet 可 byte／逐列比對。
- ✅ **已補齊（2026-07-22，單元 3）**：`zone_mapping/inputs/` 原本只有 `tracking_results.parquet`，
  缺 zone 幾何（`map_zones_daily` 需要 `bucket_dir/camera_registry.yaml`）。已用 GCS→GCS 從
  `zone_mapping/expected/2026-05-01/camera_registry_used.yaml` 複製一份到
  `zone_mapping/inputs/2026-05-01/camera_registry.yaml`（1857 B，MD5 `qp6ddO5nZ9Z/+VYbGrJc4A==`，
  與來源一致）。**argus #22 的路徑表需同步更新。**
- ✅ **已補齊（2026-07-22）**：`flow_report/inputs/` 原本缺 `camera_registry_used.yaml`（但
  `_build_report_frames` 需要它）。已用 GCS→GCS 從 `zone_mapping/expected/2026-05-01/` 複製一份到
  `flow_report/inputs/2026-05-01/`（1857 B，MD5 `qp6ddO5nZ9Z/+VYbGrJc4A==`，與來源一致），符合 #22
  「每階段獨立取用、不必知道上游路徑」的設計原則。**單元 2 直接從 `flow_report/inputs/` 拉兩個檔即可。**
  #22 的盤點表與路徑表**已同步更新**（body + 留言，2026-07-22），上表即與 issue 一致。

**驗收**：xlsx 用 openpyxl `iter_rows(values_only=True)` **儲格值**比對（**非 byte**；見 §4 坑 3）；
vfa↔onprem `diff -rq` src 一致；對照 origin/main serverless 版量殘留 diff 應只剩雲端接線。

⚠ **argus 端一律從 origin/main 開分支**（工作樹目前 checkout 在過期的 `feat/20`，看不到 serverless 參考與
新目錄結構）。

---

## 3. 每個單元的執行步驟

1. **讀本檔 §2** + 該包來源結構（進入點、輸入輸出契約、依賴）。
2. **plan（plan mode）**：只寫該包特化，§2 共用決策用引用。
3. **vfa 先重構**：建 DDD 目錄、逐檔搬移（邏輯不變）、config→pydantic-settings、registry reconcile、
   抽 `config/constants.py`、新增 `observability/`、改測試 import 路徑。
4. **驗證 vfa**：`uv sync` / `uv run pytest` / `uv run ruff check .`；golden 儲格值比對（見 §4 坑 3）。
5. **建 issue（argus，從 origin/main）** → 分支 `feat/{issue#}-...` → **同步 onprem**：src 與 vfa
   byte-identical（`git ls-files` 逐檔、`diff -rq` 驗）＋非 src 檔（config.toml 執行註解、README 執行/golden
   章節、`.gitignore`、golden 樹）。
6. **驗證 onprem**：全新 clone「三行跑通」（cd → uv sync → uv run）；`diff -rq` vfa↔onprem；對照 serverless
   量殘留 diff。
7. **commit（英文、逐關注點）** → PR（`Closes #issue`）→ dispatch review subagent。
8. **review 迴圈**：實測每條建議再決定（見 §4 坑 10）；兩地維護的套件，修正同步回 vfa（真相來源）。

---

## 4. 移植踩過的坑與正確做法（屬移植機制、與 DDD 策略無關，仍全部有效）

- **遇到「要複製套件檔」→ 用 `git ls-files <pkg>/` 的清單逐檔複製，不要 `cp -r` 整個目錄。**
  整包複製會帶入 `.venv`／`__pycache__`／`.ruff_cache`／`.pytest_cache`。複製後用
  `diff -r --exclude=__pycache__ <來源>/src <目標>/src` 驗證逐檔一致，不靠肉眼。

- **遇到「要準備 golden 輸出」→ 用搬過去的輸入重新跑一次產生，不要複製磁碟上現成的 `report.xlsx`。**
  現成產出可能是 stale（實測 flow-report 的 report.xlsx mtime 早於其輸入 `zone_counts.parquet`）。
  先放輸入樹，再 `uv run` 產生 golden。

- **遇到「要比對 xlsx 是否一致」→ 比對各分頁的儲格值，不要做 byte 比對。**
  openpyxl 會把當下時間寫進 `docProps/core.xml`，兩次產出的 bytes 必不同。用 openpyxl 讀出
  `iter_rows(values_only=True)` 逐分頁比對（範例見 `argus/pipelines/onprem/flow-report/README.md` 的
  「golden sample 驗證」節）。

- **【2026-07-22 起變更】遇到「golden 要放哪」→ 一律放 GCS、不進 git**（路徑見 §2）。從 GCS 拉到本機後，
  輸入放套件根的 `outputs/{bucket}/{date}/`——因為 `OUTPUT_ROOT` 是 cwd 相對（見 §5 執行機制）；**本機輸入樹
  不要留 `report.xlsx`**，`append` 模式重跑會疊加重複列，README 的重跑指令要內含
  `rm -f outputs/{bucket}/report.xlsx`。

- **遇到「要 gitignore」→ golden 不進 git，`outputs/` 整個擋掉**（它現在只是從 GCS 拉下來的本機暫存）。
  加完用 `git check-ignore -v` 實測方向。**注意**：`argus/pipelines/onprem/flow-report/` 目前仍有
  `golden/report.xlsx` 與 `outputs/bucket_name1/2026-05-01/*` **進在版控裡**（PR #17 的舊做法），
  **單元 2 要一併 `git rm` 並改寫 `.gitignore`**。

- **遇到「README 照搬」→ 不能逐字搬，要改寫執行相關章節。** 來源 README 描述的是舊的「三套件共用
  `outputs/` 樹、從 repo 根以 `--project` 執行」模型；在 argus 獨立佈局下規則相反（cd 進套件目錄）。
  改寫「安裝與快速開始」「執行位置」，刪掉 `--project`／`--directory`／repo 根路徑，並修掉指向未一併
  搬移之物的引用。（DDD 重構後 entry point 由 `flow_report.pipeline:run_report` 變 `flow_report.main:...`，
  README 執行指令要同步。）

- **遇到「註解提到別的套件或計畫」→ 一併清掉或改寫。** `config.toml` 開頭註解、測試 fixture 註解常引用
  argus 沒有的東西（sibling 套件、`docs/plans/...`）。

- **遇到「commit 訊息語言」→ argus 用英文 `type(scope): message`，vfa 用繁中；同一分支內不混用。**
  argus 慣例見 `argus/.agents/skills/argus-commit-message-footer`。

- **遇到「兩地都要維護同一份 src」→ 以 vfa 為真相來源，逐一 commit 對應搬移，`src/` 用 `diff -r` 驗
  byte-identical；合併來源 PR 用 merge commit，不要 squash**（squash 會把多 commit 壓成一個，逐一對應即斷）。

- **遇到「code review 給了 patch」→ 先實測再決定，不照抄。** flow-report 上兩則建議實測會弄壞程式：用
  `camera_id` join parquet 會 0 列（parquet 的 `camera_id` 欄存的是 `stream_dirname`）；`unique(keep="first")`
  漏 `maintain_order=True` 會讓輸出列序漂移、golden 比對隨機失敗。修正時陳述「函式自身的缺陷」，不靠
  移交史或 config 預設值論證。

- **遇到「golden sample 含真實資料」→ 標記敏感度。** flow-report 的 registry 快照含內網 IP 與真實店點
  zone 名稱。argus 為 PRIVATE 可接受，但**轉公開或對外分享前需脫敏**；寫進 PR 的 Risk。

---

## 5. 各包專屬注意（動工前先確認）

**執行機制（三包共通）**：`config.toml` 由 `find_project_root`（重構後）／`__file__`（重構前）定位、不受
cwd 影響；但輸出根 `OUTPUT_ROOT = Path("outputs")` 與 `bucket_dir` 是 **cwd 相對**。結論：**一律 cd 進套件
目錄執行**（`cd pipelines/onprem/<pkg> && uv run …`），不用 `--project`／`--directory`。

**golden 獨立性**：三包的 golden sample 各自獨立，**不可跨套件串接比對**——各包輸入來源與產生方式本就不同
（flow-report 用全量 `zone_counts.parquet`；zone-mapping 用 `tracking_results.parquet` 尾段切片）。同名檔案
最易誤串：zone-mapping 產出的 `zone_counts.parquet` 與 flow-report 交付的輸入同名、全量版實測 byte 一致，但
改用切片輸入後已不同。此約定要寫進**每一包**的 README golden 章節，措辭需能獨立成立。
> **待辦：flow-report 的 README 尚未補上此段**（交付 zone-mapping 時決定先不動 flow-report）。交付
> video-analyze 時一併補上 flow-report 這段，讓三包一致。

**zone-mapping（#20 暫停中）**：`feat/20` 已有 4 commit（byte-identical port phase 1）——ruff-100、
`resolve project root dynamically`（已是 find_project_root 精神）、golden 設定可留用；改 DDD 時最終形狀改成
`main/config/models/services/observability`，別重頭來。從 origin/main 開新分支，勿續建在過期的 `feat/20` 上。

**video-analyze 的 golden 標的不同**：`tracking_results.parquet` **不可重現**（ByteTrack 的 `track_id` 在
重跑間會變，逐值比對無驗收力）。golden 用聚合後穩定的 `zone_counts.parquet` 當比對標的。詳見主 CLAUDE.md
「不可重現」一節。

**video-analyze 依賴重**（torch／ultralytics／opencv、GPU、多進程）：`uv sync` 時間與環境需求遠高於
flow-report；golden 產生可能需要 GPU。

**`registry.py` 三份差異**：zone-mapping／flow-report 是完整版（含 `parse_and_validate_zones`）；video-analyze
是精簡版。**單元 1、3** 把 flow-report／zone-mapping 各搬成 `models/registry.py` 並依 §2 reconcile 修成兩份
一致；**單元 5** 抽成共用 lib；**單元 6** 的 video-analyze 直接用該 lib、刪掉精簡版，**不產生第三份**。
**zone 名稱全域唯一**由各套件自己的 `parse_and_validate_zones` 驗證；zone-mapping 也會擋跨攝影機重複命名。

**video-analyze 多進程下的 `StructuredLogger`（2026-07-22 已實讀 `_emit`，風險範圍縮小）**：argus 的
`StructuredLogger`（`docs/pipelines/serverless/README.md` §3 要求 serverless 一律使用）是為**單進程** cloud
job 設計、往 stdout 印 JSON-lines。video-analyze 會 spawn 多個子進程跑多路串流（`pipeline.py` 的
`mp.Process`／`mp.Queue`）。

- **不是介面問題，不必為多進程改 lib**：`_emit` 實作是 `print(json.dumps(...), flush=True)` 到
  `sys.stdout`／`sys.stderr`，**類別完全無狀態**（只存一個 `component` 字串），沒有 handler、沒有 file
  handle、沒有 logger 註冊表。因此**沒有 fork 繼承 handler 的問題**，也不需要「每個子進程各持自己的
  logger」這種設計——各模組 import 時本來就各自建一個實例。此結論讓抽 lib（單元 5）**不必等 video-analyze**。
- **剩下的真實風險只有單次寫入的原子性**：多子進程寫同一個 fd，短行實務上不易交錯（Linux pipe 的
  `PIPE_BUF` 為 4096），但 **`.exception()` 帶完整 stacktrace 的長行可能超過而被切斷交錯**。這是**用法**
  問題不是 lib 問題，屬單元 6 要驗與處理的範圍（例如截斷 stacktrace、或錯誤彙整回父進程再印）。
- **單元 6 的實際工作量**：video-analyze 目前用 stdlib `logging`（`getLogger(__name__)` + %-style，
  約 20 處呼叫點分佈於 `config`／`detector`／`inference`／`pipeline`／`tracking_results`／`io/video_writer`），
  改成 `StructuredLogger` 的 kwargs 風格是逐點改寫。

flow-report／opencv-job 是單進程，不會遇到原子性問題，其樣板不涵蓋這點。
