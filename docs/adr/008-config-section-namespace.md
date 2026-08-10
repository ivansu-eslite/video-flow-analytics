# ADR-008: 設定的頂層區塊名是全域命名空間，`[input]` 由共用 lib 提供

## Status

Accepted

## Context

四包都用 pydantic-settings，`SettingsConfigDict(env_nested_delimiter="__")` 讓環境變數
`<區塊>__<欄位>` 對應到 `config.toml` 的 `[區塊] 欄位`。這個機制沒有套件前綴：
`INPUT__CAMERA_IDS` 對四包一律解讀成「`[input]` 底下的 `camera_ids`」，**頂層區塊名等於
一段全域的環境變數命名空間**。四包同屬一個 workspace、部署時共用一份環境設定執行是常態，
而各包的 `models/config.py` 在模組層就 `load_config()`——所以覆寫在別包眼中不合法時，
受害的包連 `import` 都會失敗，不是「該次執行讀不到值」而已。

`[input]` 四包都有，但抽出前是四份各自裁剪的複製品，已經漂出兩個方向：

- `camera_ids` 只有 `video_analyze` 有。
- `bucket_minutes` 於 `a4fa396`（2026-07-31）從 `flow_report` 的 `[zone]` 移進 `[input]`，
  成為第二個獨有欄位。

實測（改動前，四包八組合）：

```
$ INPUT__CAMERA_IDS='["cam1"]' uv run --directory zone_mapping python -c "import zone_mapping.models.config"
input.camera_ids
  Extra inputs are not permitted [type=extra_forbidden, input_value='["cam1"]', input_type=str]
```

`INPUT__CAMERA_IDS` 打死 `zone_mapping`／`line_counting`／`flow_report` 三包，
`INPUT__BUCKET_MINUTES` 打死 `video_analyze`／`zone_mapping`／`line_counting` 三包——而
`flow_report` 在 `[zone]` 區塊守衛裡的錯誤訊息，正好叫維運人員去設 `INPUT__BUCKET_MINUTES`。

同時 `bucket_minutes` 這個值在三包各有一條環境變數路徑（`ZONE__BUCKET_MINUTES`／
`LINE__BUCKET_MINUTES`／`INPUT__BUCKET_MINUTES`），三者描述的是同一個口徑：上游兩包寫檔
時的 `time_bucket` 粒度，`flow_report` 用它驗 `report.period_minutes` 是它的倍數。

這與 CLAUDE.md「共用 lib 存在的理由」記載的 `load_registry_from_path` 是同一個劇本：同一
份程式碼複製四份，各自漂移，直到某一份的漂移打壞別份。`find_project_root`／
`_get_toml_path` 也是逐字相同的四份複製（去註解後 md5 皆為
`2ebe2ac9a38cde4e4cb4d98d84754276`）。

## Options Considered

### Option A：維持四份，只補齊欄位

Description：不抽 lib，把 `camera_ids` 與 `bucket_minutes` 手動補進四份 `InputConfig`。

- Advantages：改動最小，不新增 workspace 成員。
- Disadvantages：只修掉當下這兩個欄位，機制沒變——下次任一包加 `[input]` 欄位又會複製
  同一個缺陷，而且症狀（別包 import 崩潰）與加欄位的動作隔了一層，不容易聯想。四份手抄
  的一致性沒有任何機制把關。

### Option B：`[input]` 抽成共用 lib（本次選擇）

Description：新增 `libs/vfa_config`，其中的 `InputConfig` 是四包欄位的**聯集**，四包一律
吃完整版；`find_project_root`／`get_toml_path` 一併抽出。各包私有的 `[tracker]`／
`[model]`／`[output]`／`[zone]`／`[line]`／`[report]` 留在各包。

- Advantages：「一包加欄位、其他包崩潰」在結構上不可能發生——加欄位的地方就是四包共用的
  那一份。`bucket_minutes` 收斂成單一環境變數路徑 `INPUT__BUCKET_MINUTES`，一次覆寫全部
  三包。與現有 `vfa_registry`／`vfa_observability` 同一個模式，不引入新概念。
- Disadvantages：各包會帶著自己用不到的欄位（`video_analyze` 的 `bucket_minutes`、
  `flow_report` 的 `camera_ids`），與 CLAUDE.md 現行「各包只保留自己 `run_*` 實際讀到的
  區塊」相牴；需要明說這是修訂而非違反。多一個 workspace 成員要維護。

### Option C：改用套件前綴的環境變數

Description：各包設 `env_prefix="ZONE_MAPPING_"` 之類，讓命名空間不再共用。

- Advantages：不必動 `[input]` 的內容，撞名從根本消失。
- Disadvantages：四包全部的環境變數路徑都改名，部署端（argus）與所有既有文件同步改；
  而且沒有解決真正的問題——`bucket_minutes` 仍是三個不同名的變數描述同一個口徑，
  反而讓「一次覆寫全部」變成不可能。撞名是症狀，複製漂移才是病灶。

### Option D：`bucket_minutes` 寫進 parquet 的檔級 metadata（**否決，不是延後**）

Description：`zone_mapping`／`line_counting` 寫 `zone_counts.parquet`／`line_counts.parquet`
時，把當時用的 `bucket_minutes` 寫進檔級 metadata；`flow_report` 從檔案讀，不再自己設定。

- Advantages：能額外消除三份 `config.toml` 的手抄，並解決「設定值是當下的、不是產檔當下
  的」——改了粒度之後補跑舊日期，`flow_report` 讀到的仍是新設定，會與舊 parquet 的實際
  粒度錯位。技術上可行（已實測 `polars 1.42` 的 `write_parquet(metadata=…)` 在 0 列時仍
  可讀回）。
- Disadvantages：這兩份 parquet 是對下游的**檔案契約**，改格式會打到 `flow_report` 以外
  的讀者（BI、golden 比對）。收益是消除一個數字的手抄，代價是動一份有外部讀者的格式，
  不成比例。

## Decision

採 Option B，並定下一條規則：

> **設定的頂層區塊名是跨四包的全域命名空間。** 新增頂層區塊前要確認該名稱在其他三包沒
> 被用過；四包都有的區塊（目前只有 `[input]`）必須由 `libs/vfa_config` 提供單一定義，
> 欄位取四包需求的聯集，**不得各包裁剪**。

`InputConfig` 因此刻意包含各包用不到的欄位。這**修訂**（非違反）CLAUDE.md 原本的「各包
只保留自己 `run_*` 實際讀到的區塊」：該規則要防的是 `flow_report` 扛 `[tracker]`／
`[model]` 這種依賴面外溢，那部分完全保留——私有區塊仍留在各包。`[input]` 是例外，因為
它對應的是共用的環境變數前綴，本來就不是可以各包裁剪的東西。

`bucket_minutes` 同時從 `zone_mapping` 的 `[zone]`、`line_counting` 的 `[line]` 移到
`[input]`，三包共用單一路徑。兩包各加一個搬家提示（沿用舊位置時報出新位置與新的環境
變數名），不為兩處用法造共用的 helper。

`flow_report` 的 `ZONE__` 環境變數警告（PR #77）一併移除：`ZONE__BUCKET_MINUTES` 現在對
`zone_mapping` 也已不是合法設定，會由該包的搬家提示擋下，`flow_report` 不必再警告一份。
`config.toml` 裡的 `[zone]` 區塊守衛保留——那是給沿用更舊設定檔的人，與環境變數無關。

Option D 是**否決，不是延後**：不要因為它「更徹底」而日後重提，否決的理由是動了有外部
讀者的檔案契約，那個代價不會隨時間變小。

## Consequences

Positive:

- 四包對任一 `INPUT__*` 都能正常載入（16 組合實測，改動前有 6 組崩在 import）。維運人員
  可以用單一份環境設定跑完四個階段。
- `bucket_minutes` 收斂成 `INPUT__BUCKET_MINUTES` 一條路徑，一次覆寫三包。
- 四包各有一支區塊契約測試（`set(AppConfig.model_fields)` ＋ `input` 的型別來源），
  新增頂層區塊或把 `input` 改回本地定義都會變紅，逼人回到這條規則。
- `find_project_root`／`get_toml_path` 的四份逐字複製消失。

Negative:

- **三份 `config.toml` 仍各填一次 `bucket_minutes`，且這是被接受的最終狀態。** 唯一能
  消除手抄的做法（Option D）已否決。三份填不一致時仍然沒有訊號——`flow_report` 只驗
  `period_minutes` 是自己那份 `bucket_minutes` 的倍數，不會發現上游實際用的是別的值。
  靠的是慣例（三處註解都寫明要一致）而非機制，這是已知且接受的缺口，不是待辦。
- 各包帶著用不到的欄位。讀 `video_analyze` 的設定時會看到 `bucket_minutes`，需要靠
  `InputConfig` 的 docstring 才知道它不讀。
- 沿用舊設定的人會踩到兩個搬家提示（`[zone]`／`[line]` 的 `bucket_minutes`）。這是刻意
  的 fail loud：靜默忽略會讓 `time_bucket` 粒度悄悄退回 60 分鐘預設值。
- 多一個 workspace 成員（`libs/vfa_config`），四包的 `pydantic` 版本 pin 又多一處要同步。
