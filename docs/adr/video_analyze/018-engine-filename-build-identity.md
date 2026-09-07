# ADR-018: 引擎檔名帶建置身分

## Status

Accepted

## Context

[ADR-011 第 9 節](011-single-inference-backend.md)把引擎檔名訂成
`<權重 stem>_sm<SM>.engine`，解決的是「不同架構的兩顆引擎不可混用」——T4（sm75）與
開發機 5090（sm120）各建一顆，同名的話下游 argus 的 promotion 用
`Path(source_model_uri).stem` 當 `model_version`，兩顆會撞在一起。

那個格式只處理了一半。**同一份權重在同一張卡上重建，兩顆引擎的檔名完全一樣。** 重建
不是罕見動作：TensorRT 的 kernel autotuning 帶計時成分，同樣的輸入本來就不保證產出
逐位元相同的引擎；換 TensorRT 版本、換驅動、改 `--batch`、改 optimization profile
（[ADR-015](015-narrow-engine-profile.md) 就是一次），也都會產出不同的二進位，而檔名
一個字都不會變。

**引擎自帶的 metadata 也分不出哪一顆該留。** 本機三顆 sm75 引擎（2026-08-22、
08-26、08-31 各建一次）與兩顆 sm120 引擎（2026-08-31、09-08）逐欄比對下來，同一組卡的
任兩顆之間**只有一個欄位不同**——ultralytics 寫的 `date` 時間戳。其餘全部一樣：

```
args: {'data': None, 'batch': 16, 'fraction': 1.0, 'half': True, 'int8': False,
       'dynamic': True, 'simplify': True, 'nms': False}
vfa : source_weights.sha256 b14302f4…、compute_capability 7.5、gpu_name Tesla T4、
      tensorrt 10.13.3.9、tensorrt_package tensorrt-cu12、torch_cuda_major 13、
      driver 595.71.05
```

而 `date` 解決不了這件事：它不在檔名裡（檔名照樣撞），拿到檔案的人也覆核不了那個值
對不對——理由與下面 Option B 被否決的理由是同一個。

三顆 sm75 引擎唯一真正的差別是內容本身：

| 引擎 | 建置日 | 內容 sha256 |
|---|---|---|
| A | 2026-08-22 | `c29b066266c61cce1d18a38b714c42deaa21503f64e1aeb05ff3e79e3be0574b` |
| B | 2026-08-26 | `a46e75667a7af64f97a577c0b8bf16560e1285f1116fcbddd8c1ed2dfaf47941` |
| C | 2026-08-31 | `d98a7c879254f72b363ad84474297c765c4f2cabf218014542157690525c67d8` |

開發機（5090／sm120）那對同樣如此：2026-08-31 建的是
`a5523a3f31fe6d5bb4603c29d625f476e261b6497ada37503f640a5f718ed5c9`，本次改動時用同一份
權重、同一張卡重建的是
`043b2a0c053f3f8762c59f35a5d2fee87988b09a4e041bb2e682294841efe730`——舊格式下兩顆都叫
`20260714-153811_yolo26m_baseline_sm120.engine`。

後果有兩層：

- 下游 promotion 的 `model_version` 在版本軸上是同一個值；上傳到物件儲存同名即覆蓋，
  被蓋掉的那顆連同它跑出來的結果一起失去對應。
- 事後出現數字差異時，**沒有任何欄位回答得了「這批結果是哪一顆引擎跑的」**。

## Options Considered

### Option A：檔名帶引擎內容的 sha256 前 8 碼（採用）

Description：`<權重 stem>_sm<SM>_<sha8>.engine`，hash 取引擎檔的完整內容（檔頭 ＋
序列化引擎）。

- Advantages：要辨識的東西就是這顆二進位本身，任何差異都反映在名字上；拿到檔案的人
  可以直接 `sha256sum` 覆核檔名，不必信任產生它的流程。8 碼與物件儲存上已在用的手動
  命名（`..._sm75_a46e7566.engine`）一致。
- Disadvantages：正式檔名要等引擎建完才知道，建置流程得拆成兩段（見下）。

### Option B：檔名帶建置時間戳

Description：`<權重 stem>_sm<SM>_<YYYYmmdd-HHMM>.engine`。

- Disadvantages：**覆核不了**——拿到檔案的人看不出那個時間戳對不對，改一個字沒有任何
  訊號。也不回答真正要問的問題：兩顆時間戳不同的引擎，內容可能相同；同一分鐘內建的兩顆
  會撞名。檔頭裡的 `date` 就是這種東西，它存在但沒有解決問題。

### Option C：檔名帶建置參數的摘要

Description：把 `--batch`、profile 形狀、TensorRT 版本雜湊成一段字尾。

- Disadvantages：**同參數重建照樣同名**，正是要消除的情況本身。

### Option D：把 TensorRT 版本／驅動寫進檔名

Description：`..._trt10.13.3.9_drv595.engine`。

- Disadvantages：那些欄位已經在檔頭裡，載入端逐項比對
  （`engine_metadata.validate_engine_metadata`），寫進檔名只是把名字拉長；而且版本不同
  必然內容不同，內容 hash 已經涵蓋。

## Decision

引擎產物名改為 `<權重 stem>_sm<SM>_<引擎內容 sha256 前 8 碼>.engine`（Option A）。
`_sm<SM>` 擋跨架構混用，`_<sha8>` 擋同權重同卡的兩次建置混用。

**建置流程拆成兩段。** 改動前 `final_path` 在建置之前就算得出來，暫存名是
`final_path + ".unverified"`。內容 hash 要有檔案才算得出來，所以：

1. `engine_basename(weights, cc)` → `<stem>_sm<SM>`（不含副檔名），暫存檔名由它組出
   `<stem>_sm<SM>.engine.unverified`，與建置結果無關，建置前就決定。
2. `engine_filename(weights, cc, engine_sha256)` → `<stem>_sm<SM>_<sha8>.engine`，在
   驗收與比對全部通過**之後**才呼叫，sha 用既有的 `services/engine_metadata.sha256_of`
   算（不另寫一份雜湊工具）。

失敗路徑不變：任何一關沒過仍然刪掉暫存檔，磁碟上不會留下正式檔名的產物；
`--skip-compare` 仍然提早 return、產物停在 `.engine.unverified` 不改名
（ADR-011 第 10 節）。改名成功後印出的那行補上完整 sha256，log 裡就能覆核檔名。

**正式檔名已存在時直接覆蓋，不擋。** 檔名帶的是內容 hash，同名即同內容，覆蓋是等價
操作；只印一行說明。

## Consequences

- 下游 promotion 的 `model_version` 逐次建置唯一，不再互相覆蓋。
- 事後追查「這批結果是哪一顆引擎跑的」有了可覆核的答案：檔名的 8 碼對得上
  `sha256sum`。
- **載入端不驗「檔名的 sha8 是否等於檔案內容」**：`services/detector.py` 的載入序列
  一行不動，這次刻意只把它當命名慣例。手動改過名的引擎不會有訊號，這是明知並接受的
  缺口——載入端已經逐項比對檔頭裡的 SM、TensorRT 版本、驅動與來源權重 hash，那些才是
  會讓推論真的出錯的東西；檔名只是給人看的標籤。
- **舊引擎不強制重建**：沒有 sha8 尾綴的名字照樣載得進去，只是無法辨識是哪一次建置的
  產物。物件儲存上既有的 `..._sm75.engine` 不動、不重產、不改名。
- 8 碼（32 bit）的碰撞機率在個位數顆引擎的規模下可忽略，且與已在用的手動命名一致。

## Related Links

- [ADR-011](011-single-inference-backend.md) 第 9 節（本次修訂的對象）與第 10 節
  （`.unverified` 尾綴與「比對沒過就沒有產物」）。
- [ADR-015](015-narrow-engine-profile.md)——收窄 optimization profile，說明為何同一份
  權重的引擎會因建置參數而不同。
