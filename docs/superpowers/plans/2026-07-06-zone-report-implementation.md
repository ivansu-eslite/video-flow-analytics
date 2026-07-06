# Zone 人流 Excel 報表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一個下游階段，把 `zone_counts.parquet` 彙總成一份跨日累加更新的
Excel 報表（`outputs/{bucket_name}/report.xlsx`），含「每小時人流」「每日尖峰」
「活動事件」三個分頁。

**Architecture:** 新增 `src/video_flow_analytics/report/` 子套件，比照
`zone_mapping/` 的 stats（純運算）/ pipeline（I/O + orchestration）分工。
`stats.py` 負責時區轉換、期間彙總、尖峰計算、用餐時段規則；`pipeline.py` 負責
讀 parquet、驗證 zone 名稱全域唯一、讀寫既有 xlsx（依 `on_duplicate_date` 合併）
並 atomic 寫檔。`cli.py` 新增 `report` 子命令。

**Tech Stack:** Python 3.12、polars、openpyxl（新增依賴）、pydantic、pytest（新增
dev 依賴）。

## Global Constraints

- 所有回覆、註解、commit 訊息使用繁體中文；程式碼識別字（函式名、變數名、API）
  維持英文。
- `uv run ruff check .` 必須全過（line-length=88, select=["E","F","I","W"]），
  每個 commit 前執行。
- Zone 名稱在整份 `camera_registry.yaml` 必須全域唯一（跨攝影機不可重複）；此
  驗證只加在 `report/pipeline.py`，不動 `CameraRegistry` / `CameraEntry`。
- `period_minutes` 必須是 `zone.bucket_minutes` 的倍數，否則在讀資料前
  fail-loud。
- 報表輸出路徑固定為 `outputs/{bucket_name}/report.xlsx`（單一跨日累加檔案，
  非逐日資料夾）。
- 三個分頁名稱固定為：`每小時人流`、`每日尖峰`、`活動事件`。
- 時區轉換一律用台北固定 +8 小時偏移（不查 IANA tz 資料庫，因台北無 DST）。
- `report/stats.py` 為純函式、需要 pytest 單元測試；`report/pipeline.py`
  （檔案 I/O／Excel／CLI 串接）不寫自動化測試，改用手動驗證步驟確認。
- 寫檔採 `.tmp` + atomic rename，比照專案既有的
  `tracking_results.parquet` / `zone_counts.parquet` 慣例。
- 不要修改 `bucket_name1/camera_registry.yaml`（該檔案目前違反新的 zone 全域
  唯一規則，由使用者自行修正；手動驗證時改用暫存副本）。

---

## 檔案總覽

- `pyproject.toml` — 新增 `openpyxl` 依賴、`pytest` dev 依賴、
  `[tool.pytest.ini_options]`
- `config.toml` — 新增 `[report]` 區塊
- `src/video_flow_analytics/core/config.py` — 新增 `ReportConfig`、
  `AppConfig.report`
- `src/video_flow_analytics/report/__init__.py` — 新建（空檔）
- `src/video_flow_analytics/report/stats.py` — 新建：`weekday_zh`、
  `to_taipei`、`rollup_by_period`、`meal_time_reminder`、`peak_per_day`
- `src/video_flow_analytics/report/pipeline.py` — 新建：
  `_validate_unique_zone_names`、`_build_report_frames`、Excel 讀寫合併
  helper、`export_report_daily`、`run_report`
- `src/video_flow_analytics/cli.py` — 新增 `report` 子命令
- `tests/report/test_stats.py` — 新建：`stats.py` 的單元測試
- `CLAUDE.md` — 補充 report 子系統與 zone 全域唯一規則的文件

---

### Task 1: 專案設定（依賴、config.toml、ReportConfig）

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.toml`
- Modify: `src/video_flow_analytics/core/config.py`

**Interfaces:**
- Produces: `ReportConfig`（pydantic model，欄位 `period_minutes: int`、
  `metric: Literal["entries", "unique_visitors"]`、
  `on_duplicate_date: Literal["overwrite", "append", "error"]`）與
  `AppConfig.report: ReportConfig`，供後續所有 Task 讀取設定值。

- [ ] **Step 1: 新增 openpyxl 依賴與 pytest dev 依賴**

```bash
uv add openpyxl
uv add --dev pytest
```

- [ ] **Step 2: 新增 pytest 設定到 pyproject.toml**

在 `pyproject.toml` 檔尾（`[tool.ruff.lint]` 區塊之後）新增：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: `config.toml` 新增 `[report]` 區塊**

在 `config.toml` 檔尾（`[zone]` 區塊之後）新增：

```toml
[report]
period_minutes = 60              # 報表人流彙總的時段粒度（分鐘），需為 zone.bucket_minutes 的倍數
metric = "entries"                # "entries" 或 "unique_visitors"，決定人流量/尖峰人流用哪個統計量
on_duplicate_date = "overwrite"   # 同一天資料已存在時："overwrite"（預設）/ "append" / "error"
```

- [ ] **Step 4: `core/config.py` 新增 `ReportConfig`**

修改 `src/video_flow_analytics/core/config.py`：

在檔案開頭的 import 區塊，把

```python
from pydantic import BaseModel, Field
```

改成

```python
from typing import Literal

from pydantic import BaseModel, Field
```

在 `class ZoneConfig` 定義之後、`class InputConfig` 定義之前，新增：

```python
class ReportConfig(BaseModel):
    # 報表人流彙總的時段粒度（分鐘），需為 zone.bucket_minutes 的倍數
    period_minutes: int = Field(default=60, ge=1)
    # 決定「人流量」「尖峰人流」用哪個統計量
    metric: Literal["entries", "unique_visitors"] = "entries"
    # 同一天資料已存在時的處理方式
    on_duplicate_date: Literal["overwrite", "append", "error"] = "overwrite"
```

修改 `class AppConfig`，新增欄位：

```python
class AppConfig(BaseModel):
    tracker: TrackerConfig
    model: ModelConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
```

- [ ] **Step 5: 驗證設定可以正常載入**

```bash
uv run python -c "
from video_flow_analytics.core.config import settings
print(settings.report)
"
```

Expected: 印出 `ReportConfig(period_minutes=60, metric='entries', on_duplicate_date='overwrite')`

- [ ] **Step 6: ruff 檢查並 commit**

```bash
uv run ruff check .
git add pyproject.toml uv.lock config.toml src/video_flow_analytics/core/config.py
git commit -m "$(cat <<'EOF'
feat(report): 新增報表相依套件與 ReportConfig 設定

為 zone 人流 Excel 報表功能鋪路：加入 openpyxl/pytest 依賴、
config.toml 的 [report] 區塊與對應的 ReportConfig pydantic model。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `report/stats.py` — 基礎工具（`weekday_zh` + `to_taipei`）

**Files:**
- Create: `src/video_flow_analytics/report/__init__.py`
- Create: `src/video_flow_analytics/report/stats.py`
- Test: `tests/report/test_stats.py`

**Interfaces:**
- Produces:
  - `weekday_zh(d: datetime.date) -> str`
  - `to_taipei(df: pl.DataFrame, column: str = "time_bucket") -> pl.DataFrame`
    （回傳新增 `local_time` 欄位的 DataFrame，型別為 naive
    `Datetime(time_unit="us")`，值為輸入欄位 +8 小時）

- [ ] **Step 1: 建立空的 `report/__init__.py`**

```bash
mkdir -p src/video_flow_analytics/report tests/report
touch src/video_flow_analytics/report/__init__.py
```

- [ ] **Step 2: 寫失敗測試 `tests/report/test_stats.py`**

```python
import datetime

import polars as pl

from video_flow_analytics.report.stats import to_taipei, weekday_zh


def test_weekday_zh_matches_known_friday():
    # 2026-05-01 是星期五（與 Sample Report.xlsx 的資料一致）
    assert weekday_zh(datetime.date(2026, 5, 1)) == "星期五"


def test_weekday_zh_covers_full_week():
    base = datetime.date(2026, 5, 4)  # 星期一
    expected = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    for offset, name in enumerate(expected):
        assert weekday_zh(base + datetime.timedelta(days=offset)) == name


def test_to_taipei_adds_eight_hours():
    df = pl.DataFrame(
        {
            "time_bucket": [
                datetime.datetime(2026, 5, 1, 11, 0, tzinfo=datetime.timezone.utc)
            ]
        }
    )
    result = to_taipei(df)
    assert result["local_time"][0] == datetime.datetime(2026, 5, 1, 19, 0)
```

- [ ] **Step 3: 執行測試確認失敗**

```bash
uv run pytest tests/report/test_stats.py -v
```

Expected: `ModuleNotFoundError` 或 `ImportError`（`stats.py` 尚未建立）

- [ ] **Step 4: 實作 `report/stats.py`**

```python
"""Zone 人流報表的核心演算法：時區轉換、期間彙總、尖峰計算、用餐時段規則。

所有函式皆為純運算（不做任何檔案 I/O），方便單元測試；I/O 與 orchestration
在 report/pipeline.py。
"""

import datetime

import polars as pl

_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def weekday_zh(d: datetime.date) -> str:
    return _WEEKDAY_ZH[d.weekday()]


def to_taipei(df: pl.DataFrame, column: str = "time_bucket") -> pl.DataFrame:
    """新增 local_time 欄位：column 轉為台北時區（固定 +8 小時，台北無 DST）。"""
    return df.with_columns(
        (pl.col(column).dt.replace_time_zone(None) + pl.duration(hours=8)).alias(
            "local_time"
        )
    )
```

- [ ] **Step 5: 執行測試確認通過**

```bash
uv run pytest tests/report/test_stats.py -v
```

Expected: 3 個測試全部 PASS

- [ ] **Step 6: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/report/__init__.py src/video_flow_analytics/report/stats.py tests/report/test_stats.py
git commit -m "$(cat <<'EOF'
feat(report): 新增 report/stats.py 的時區轉換與星期轉換工具

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `report/stats.py` — 期間彙總 `rollup_by_period`

**Files:**
- Modify: `src/video_flow_analytics/report/stats.py`
- Test: `tests/report/test_stats.py`

**Interfaces:**
- Consumes: `weekday_zh(d: datetime.date) -> str`（Task 2）
- Produces:
  `rollup_by_period(df: pl.DataFrame, period_minutes: int, metric: str) -> pl.DataFrame`
  輸入需已含 `local_time`（naive datetime）與 `zone`、`metric` 指定的欄位
  （`"entries"` 或 `"unique_visitors"`）。輸出欄位固定為
  `["date", "weekday", "period", "zone", "value"]`（皆為字串，除了 `value`
  為 `Int64`），並依 `(date, period, zone)` 排序。`period` 格式為
  `"HH:MM"`（該期間的起始時間）。

- [ ] **Step 1: 寫失敗測試**

在 `tests/report/test_stats.py` 尾端新增：

```python
from video_flow_analytics.report.stats import rollup_by_period


def _make_zone_counts(rows):
    return pl.DataFrame(
        rows,
        schema={
            "local_time": pl.Datetime("us"),
            "zone": pl.Utf8,
            "entries": pl.Int64,
            "unique_visitors": pl.Int64,
        },
        orient="row",
    )


def test_rollup_by_period_sums_entries_within_hour():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40),
            (datetime.datetime(2026, 5, 1, 19, 45), "checkout", 30, 20),
            (datetime.datetime(2026, 5, 1, 20, 0), "checkout", 5, 5),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="entries")
    checkout_19 = result.filter(
        (pl.col("period") == "19:00") & (pl.col("zone") == "checkout")
    )
    assert checkout_19["value"].to_list() == [180]
    assert checkout_19["date"].to_list() == ["2026-05-01"]
    assert checkout_19["weekday"].to_list() == ["星期五"]


def test_rollup_by_period_supports_unique_visitors_metric():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 15), "checkout", 50, 40),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="unique_visitors")
    assert result["value"].to_list() == [120]


def test_rollup_by_period_keeps_zones_separate():
    df = _make_zone_counts(
        [
            (datetime.datetime(2026, 5, 1, 19, 0), "checkout", 100, 80),
            (datetime.datetime(2026, 5, 1, 19, 0), "entrance", 10, 8),
        ]
    )
    result = rollup_by_period(df, period_minutes=60, metric="entries")
    values_by_zone = dict(zip(result["zone"].to_list(), result["value"].to_list()))
    assert values_by_zone == {"checkout": 100, "entrance": 10}
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/report/test_stats.py -v -k rollup_by_period
```

Expected: FAIL（`ImportError: cannot import name 'rollup_by_period'`）

- [ ] **Step 3: 實作 `rollup_by_period`**

在 `report/stats.py` 尾端新增：

```python
def rollup_by_period(df: pl.DataFrame, period_minutes: int, metric: str) -> pl.DataFrame:
    """把已轉為本地時間的 zone 人流資料，依 period_minutes 彙總成期間×區域的統計。

    輸入需含 local_time（naive datetime）、zone、metric 指定的欄位。
    輸出欄位：date（字串 YYYY-MM-DD）、weekday（中文）、period（字串 HH:MM，
    該期間起始時間）、zone、value（Int64）。
    """
    rolled = (
        df.with_columns(
            pl.col("local_time").dt.truncate(f"{period_minutes}m").alias("period_start")
        )
        .group_by(["zone", "period_start"])
        .agg(pl.col(metric).sum().alias("value"))
        .with_columns(
            pl.col("period_start").dt.strftime("%Y-%m-%d").alias("date"),
            pl.col("period_start").dt.strftime("%H:%M").alias("period"),
        )
        .select("date", "period", "zone", "value")
        .sort(["date", "period", "zone"])
    )
    weekdays = [
        weekday_zh(datetime.date.fromisoformat(d)) for d in rolled["date"].to_list()
    ]
    return rolled.with_columns(pl.Series("weekday", weekdays)).select(
        "date", "weekday", "period", "zone", "value"
    )
```

- [ ] **Step 4: 執行測試確認通過**

```bash
uv run pytest tests/report/test_stats.py -v -k rollup_by_period
```

Expected: 3 個測試全部 PASS

- [ ] **Step 5: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/report/stats.py tests/report/test_stats.py
git commit -m "$(cat <<'EOF'
feat(report): 新增 rollup_by_period 期間彙總

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `report/stats.py` — 尖峰計算（`meal_time_reminder` + `peak_per_day`）

**Files:**
- Modify: `src/video_flow_analytics/report/stats.py`
- Test: `tests/report/test_stats.py`

**Interfaces:**
- Consumes: `rollup_by_period` 的輸出格式
  `["date", "weekday", "period", "zone", "value"]`（Task 3）
- Produces:
  - `meal_time_reminder(hour: int) -> str`：`11 <= hour < 14` →
    `"加強午餐動線"`；`17 <= hour < 20` → `"加強晚餐動線"`；其餘 → `"無"`
  - `peak_per_day(rollup_df: pl.DataFrame) -> pl.DataFrame`：輸出欄位
    `["date", "weekday", "zone", "peak_period", "peak_value", "reminder"]`，
    每個 `(date, zone)` 一列，取 `value` 最大的期間；並列時取時間較早的期間。

- [ ] **Step 1: 寫失敗測試**

在 `tests/report/test_stats.py` 尾端新增：

```python
from video_flow_analytics.report.stats import meal_time_reminder, peak_per_day


def test_meal_time_reminder_boundaries():
    cases = {
        10: "無",
        11: "加強午餐動線",
        13: "加強午餐動線",
        14: "無",
        16: "無",
        17: "加強晚餐動線",
        19: "加強晚餐動線",
        20: "無",
    }
    for hour, expected in cases.items():
        assert meal_time_reminder(hour) == expected, hour


def _make_rollup(rows):
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Utf8,
            "weekday": pl.Utf8,
            "period": pl.Utf8,
            "zone": pl.Utf8,
            "value": pl.Int64,
        },
        orient="row",
    )


def test_peak_per_day_picks_max_value_per_zone():
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "18:00", "checkout", 776),
            ("2026-05-01", "星期五", "19:00", "checkout", 1246),
            ("2026-05-01", "星期五", "20:00", "checkout", 300),
            ("2026-05-01", "星期五", "11:00", "entrance", 282),
        ]
    )
    result = peak_per_day(df).sort("zone")
    checkout = result.filter(pl.col("zone") == "checkout").row(0, named=True)
    assert checkout["peak_period"] == "19:00"
    assert checkout["peak_value"] == 1246
    assert checkout["reminder"] == "加強晚餐動線"

    entrance = result.filter(pl.col("zone") == "entrance").row(0, named=True)
    assert entrance["peak_period"] == "11:00"
    assert entrance["reminder"] == "加強午餐動線"


def test_peak_per_day_ties_pick_earlier_period():
    df = _make_rollup(
        [
            ("2026-05-01", "星期五", "09:00", "entrance", 100),
            ("2026-05-01", "星期五", "15:00", "entrance", 100),
        ]
    )
    result = peak_per_day(df)
    assert result.row(0, named=True)["peak_period"] == "09:00"
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/report/test_stats.py -v -k "meal_time_reminder or peak_per_day"
```

Expected: FAIL（`ImportError`）

- [ ] **Step 3: 實作 `meal_time_reminder` 與 `peak_per_day`**

在 `report/stats.py` 尾端新增：

```python
def meal_time_reminder(hour: int) -> str:
    if 11 <= hour < 14:
        return "加強午餐動線"
    if 17 <= hour < 20:
        return "加強晚餐動線"
    return "無"


def peak_per_day(rollup_df: pl.DataFrame) -> pl.DataFrame:
    """每個 (date, zone) 取 value 最大的期間；並列時取時間較早的期間。"""
    sorted_df = rollup_df.sort(
        ["date", "zone", "value", "period"],
        descending=[False, False, True, False],
    )
    peaks = sorted_df.group_by(["date", "zone"], maintain_order=True).first()
    reminders = [
        meal_time_reminder(int(period.split(":")[0]))
        for period in peaks["period"].to_list()
    ]
    return peaks.with_columns(pl.Series("reminder", reminders)).select(
        "date",
        "weekday",
        "zone",
        pl.col("period").alias("peak_period"),
        pl.col("value").alias("peak_value"),
        "reminder",
    )
```

- [ ] **Step 4: 執行測試確認通過**

```bash
uv run pytest tests/report/test_stats.py -v
```

Expected: 全部測試 PASS（累計 9 個）

- [ ] **Step 5: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/report/stats.py tests/report/test_stats.py
git commit -m "$(cat <<'EOF'
feat(report): 新增尖峰計算與用餐時段提醒規則

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `report/pipeline.py` — 讀取、驗證、彙總 orchestration

**Files:**
- Create: `src/video_flow_analytics/report/pipeline.py`

**Interfaces:**
- Consumes:
  - `video_flow_analytics.core.registry.load_registry(bucket_dir: Path) -> CameraRegistry`
  - `CameraEntry.parsed_zones() -> list[Zone]`（`Zone.name: str`）
  - `report.stats.to_taipei`、`report.stats.rollup_by_period`、
    `report.stats.peak_per_day`（Task 2-4）
- Produces:
  - `_validate_unique_zone_names(registry: CameraRegistry) -> None`（fail-loud）
  - `_build_report_frames(date, bucket_dir, period_minutes, metric, bucket_minutes, output_root) -> tuple[pl.DataFrame, pl.DataFrame]`
    回傳 `(hourly_df, peak_df)`，供 Task 6 的 `export_report_daily` 使用。

- [ ] **Step 1: 實作 `report/pipeline.py`（讀取與驗證部分）**

```python
"""Zone 人流 Excel 報表：離線下游步驟（CLI 的 `report` 子命令）。

讀 `outputs/{bucket}/{date}/zone_counts.parquet`，彙總成跨日累加更新的
`outputs/{bucket}/report.xlsx`。實際的期間彙總／尖峰計算在 report/stats.py；
這裡負責讀檔、驗證、orchestration 與 Excel 讀寫。
"""

import datetime
import logging
from pathlib import Path

import polars as pl

from video_flow_analytics.core.registry import CameraRegistry, load_registry
from video_flow_analytics.report.stats import peak_per_day, rollup_by_period, to_taipei

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")


def _validate_unique_zone_names(registry: CameraRegistry) -> None:
    """報表以 zone 名稱（不含 camera_id）分組，因此要求整份 registry 的 zone
    名稱全域唯一；此驗證只在產報表時檢查，不影響 analyze_daily / zone_mapping。
    """
    names = [zone.name for cam in registry.cameras for zone in cam.parsed_zones()]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise ValueError(
            "camera_registry.yaml 中有跨攝影機重複的 zone 名稱，報表需要 zone "
            f"名稱全域唯一（不只同一攝影機內唯一）: {dupes}"
        )


def _build_report_frames(
    date: datetime.date,
    bucket_dir: str,
    period_minutes: int,
    metric: str,
    bucket_minutes: int,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if period_minutes % bucket_minutes != 0:
        raise ValueError(
            f"report.period_minutes（{period_minutes}）必須是 "
            f"zone.bucket_minutes（{bucket_minutes}）的倍數。"
        )

    bucket_path = Path(bucket_dir)
    registry = load_registry(bucket_path)
    _validate_unique_zone_names(registry)

    counts_path = output_root / bucket_path.name / date.isoformat() / "zone_counts.parquet"
    if not counts_path.exists():
        raise FileNotFoundError(
            f"找不到 zone 人流統計 {counts_path}，請先執行 map_zones_daily 產生當日 parquet。"
        )

    df = to_taipei(pl.read_parquet(counts_path))
    hourly_df = rollup_by_period(df, period_minutes, metric)
    peak_df = peak_per_day(hourly_df)
    return hourly_df, peak_df
```

- [ ] **Step 2: 手動驗證（暫存修正過 zone 名稱的 registry 副本）**

真實的 `bucket_name1/camera_registry.yaml` 目前有跨攝影機重複的 zone 名稱
（`entrance`、`checkout` 同時出現在 cam001 與 cam004），會被 Step 1 新增的
`_validate_unique_zone_names` 擋下來 —— 這是預期行為。手動驗證時用暫存副本
繞開，不要修改使用者的原始檔案：

```bash
SCRATCH=$(mktemp -d)
mkdir -p "$SCRATCH/bucket_name1"
cp bucket_name1/camera_registry.yaml "$SCRATCH/bucket_name1/"   # 只需要 registry，bucket_name1 本身有 6GB 影片不要整包複製
uv run python -c "
import yaml
path = '$SCRATCH/bucket_name1/camera_registry.yaml'
data = yaml.safe_load(open(path, encoding='utf-8'))
for cam in data['cameras']:
    if cam['camera_id'] == 'cam004':
        for zone in cam['zones']:
            zone['name'] = zone['name'] + '_2'
yaml.safe_dump(data, open(path, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
"
uv run python -c "
import datetime
from video_flow_analytics.report.pipeline import _build_report_frames

hourly, peak = _build_report_frames(
    date=datetime.date(2026, 5, 1),
    bucket_dir='$SCRATCH/bucket_name1',
    period_minutes=60,
    metric='entries',
    bucket_minutes=15,
)
print(hourly)
print(peak)
"
rm -rf "$SCRATCH"
```

Expected: 印出兩個 DataFrame，`hourly` 欄位為
`date, weekday, period, zone, value`（`2026-05-01` / `星期五` /
`19:00`、`20:00` 兩個期間 / 5 個 zone），`peak` 欄位為
`date, weekday, zone, peak_period, peak_value, reminder`（每個 zone 一列，
`peak_period` 應為 `19:00`，`reminder` 為 `加強晚餐動線`）。

- [ ] **Step 3: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/report/pipeline.py
git commit -m "$(cat <<'EOF'
feat(report): 新增 report/pipeline.py 的讀取與彙總 orchestration

包含 zone 名稱全域唯一驗證，因為報表以 zone 名稱（不含 camera_id）
分組。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `report/pipeline.py` — Excel 讀寫與既有檔案合併

**Files:**
- Modify: `src/video_flow_analytics/report/pipeline.py`

**Interfaces:**
- Consumes: `_build_report_frames`（Task 5）
- Produces:
  - `export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date, bucket_minutes, output_root=OUTPUT_ROOT) -> Path`
  - `run_report() -> None`

- [ ] **Step 1: 實作 Excel 讀寫 helper 與 `export_report_daily`**

在 `report/pipeline.py` 開頭的 import 區塊，補上：

```python
from typing import Literal

import openpyxl
from openpyxl.styles import Font
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from video_flow_analytics.core.config import settings
```

在檔案尾端（`_build_report_frames` 之後）新增：

```python
SHEET_HOURLY = "每小時人流"
SHEET_PEAK = "每日尖峰"
SHEET_EVENTS = "活動事件"

_HOURLY_HEADERS = ["日期", "星期", "小時", "區域", "人流量"]
_PEAK_HEADERS = ["日期", "星期", "區域", "尖峰時段", "尖峰人流", "每日提醒"]
_EVENTS_HEADERS = ["日期", "星期", "開始時間", "結束時間", "區域", "活動名稱", "活動類型"]


def _init_sheet(wb: Workbook, name: str, headers: list[str]) -> Worksheet:
    ws = wb.create_sheet(name)
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col_idx in range(1, len(headers) + 1):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = 14
    return ws


def _existing_dates(ws: Worksheet) -> set[str]:
    return {row[0].value for row in ws.iter_rows(min_row=2) if row[0].value is not None}


def _remove_rows_for_dates(ws: Worksheet, dates: set[str]) -> None:
    rows_to_delete = [
        row[0].row for row in ws.iter_rows(min_row=2) if row[0].value in dates
    ]
    for row_idx in reversed(rows_to_delete):
        ws.delete_rows(row_idx)


def _append_rows(ws: Worksheet, df: pl.DataFrame) -> None:
    for row in df.iter_rows():
        ws.append(row)


def _sort_rows(ws: Worksheet, key_columns: tuple[int, ...]) -> None:
    if ws.max_row < 2:
        return
    rows = [[cell.value for cell in row] for row in ws.iter_rows(min_row=2)]
    rows.sort(key=lambda r: tuple(r[i] for i in key_columns))
    ws.delete_rows(2, ws.max_row - 1)
    for row in rows:
        ws.append(row)


def _write_report(
    path: Path,
    hourly_new: pl.DataFrame,
    peak_new: pl.DataFrame,
    on_duplicate_date: Literal["overwrite", "append", "error"],
) -> None:
    new_dates = set(hourly_new["date"].to_list())

    if path.exists():
        wb = openpyxl.load_workbook(path)
    else:
        wb = Workbook()
        wb.remove(wb.active)
        _init_sheet(wb, SHEET_HOURLY, _HOURLY_HEADERS)
        _init_sheet(wb, SHEET_PEAK, _PEAK_HEADERS)
        _init_sheet(wb, SHEET_EVENTS, _EVENTS_HEADERS)

    hourly_ws = wb[SHEET_HOURLY]
    peak_ws = wb[SHEET_PEAK]

    if on_duplicate_date == "error":
        conflict = new_dates & (_existing_dates(hourly_ws) | _existing_dates(peak_ws))
        if conflict:
            raise ValueError(
                f"報表中已存在這些日期的資料，未寫入任何內容：{sorted(conflict)}"
                "（可改用 on_duplicate_date='overwrite' 或 'append'）"
            )

    if on_duplicate_date == "overwrite":
        _remove_rows_for_dates(hourly_ws, new_dates)
        _remove_rows_for_dates(peak_ws, new_dates)

    _append_rows(hourly_ws, hourly_new)
    _append_rows(peak_ws, peak_new)

    if on_duplicate_date == "overwrite":
        _sort_rows(hourly_ws, key_columns=(0, 2, 3))  # 日期, 小時, 區域
        _sort_rows(peak_ws, key_columns=(0, 2))  # 日期, 區域

    tmp_path = path.with_name(path.name + ".tmp")
    wb.save(tmp_path)
    tmp_path.replace(path)


def export_report_daily(
    date: datetime.date,
    bucket_dir: str,
    period_minutes: int,
    metric: Literal["entries", "unique_visitors"],
    on_duplicate_date: Literal["overwrite", "append", "error"],
    bucket_minutes: int,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """執行單日 zone 人流報表彙總，回傳 report.xlsx 路徑（跨日累加更新）。"""
    hourly_df, peak_df = _build_report_frames(
        date, bucket_dir, period_minutes, metric, bucket_minutes, output_root
    )

    bucket_name = Path(bucket_dir).name
    report_path = output_root / bucket_name / "report.xlsx"
    _write_report(report_path, hourly_df, peak_df, on_duplicate_date)

    logger.info(
        "Zone 人流報表已寫入 %s（本次日期：%s，%d 列每小時人流、%d 列每日尖峰）。",
        report_path,
        sorted(hourly_df["date"].unique().to_list()),
        hourly_df.height,
        peak_df.height,
    )
    return report_path


def run_report() -> None:
    """report 子命令：從 config.toml 取參數後呼叫 export_report_daily。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")

    export_report_daily(
        date=settings.input.date,
        bucket_dir=settings.input.bucket_dir,
        period_minutes=settings.report.period_minutes,
        metric=settings.report.metric,
        on_duplicate_date=settings.report.on_duplicate_date,
        bucket_minutes=settings.zone.bucket_minutes,
    )


if __name__ == "__main__":
    run_report()
```

- [ ] **Step 2: 手動驗證 — 新檔案建立 + overwrite 重跑同一天不重複**

```bash
SCRATCH=$(mktemp -d)
mkdir -p "$SCRATCH/bucket_name1"
cp bucket_name1/camera_registry.yaml "$SCRATCH/bucket_name1/"   # 只需要 registry，bucket_name1 本身有 6GB 影片不要整包複製
uv run python -c "
import yaml
path = '$SCRATCH/bucket_name1/camera_registry.yaml'
data = yaml.safe_load(open(path, encoding='utf-8'))
for cam in data['cameras']:
    if cam['camera_id'] == 'cam004':
        for zone in cam['zones']:
            zone['name'] = zone['name'] + '_2'
yaml.safe_dump(data, open(path, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
"
mkdir -p "$SCRATCH/outputs/bucket_name1/2026-05-01"
cp outputs/bucket_name1/2026-05-01/zone_counts.parquet "$SCRATCH/outputs/bucket_name1/2026-05-01/"

uv run python -c "
import datetime
from pathlib import Path
from video_flow_analytics.report.pipeline import export_report_daily
import openpyxl

kwargs = dict(
    date=datetime.date(2026, 5, 1),
    bucket_dir='$SCRATCH/bucket_name1',
    period_minutes=60,
    metric='entries',
    bucket_minutes=15,
    output_root=Path('$SCRATCH/outputs'),
)

path = export_report_daily(on_duplicate_date='overwrite', **kwargs)
wb = openpyxl.load_workbook(path)
print('sheets:', wb.sheetnames)
print('每小時人流 rows:', wb['每小時人流'].max_row)
print('每日尖峰 rows:', wb['每日尖峰'].max_row)

# 重跑同一天，overwrite 不應該讓列數變多
path = export_report_daily(on_duplicate_date='overwrite', **kwargs)
wb = openpyxl.load_workbook(path)
print('after rerun 每小時人流 rows:', wb['每小時人流'].max_row)
print('after rerun 每日尖峰 rows:', wb['每日尖峰'].max_row)

# error 模式應該擋下同一天
try:
    export_report_daily(on_duplicate_date='error', **kwargs)
    print('BUG: 應該要 raise ValueError')
except ValueError as e:
    print('error 模式正確擋下:', e)
"
rm -rf "$SCRATCH"
```

Expected:
- `sheets: ['每小時人流', '每日尖峰', '活動事件']`
- 第一次與重跑後的「每小時人流 rows」「每日尖峰 rows」數字相同（overwrite
  沒有造成重複）
- `error` 模式印出 `error 模式正確擋下: ...`，不是 `BUG: ...`

- [ ] **Step 3: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/report/pipeline.py
git commit -m "$(cat <<'EOF'
feat(report): 新增 export_report_daily 的 Excel 讀寫與既有檔案合併

支援 on_duplicate_date 三種模式（overwrite/append/error），
活動事件分頁維持不動、只在新建檔案時建立標題列。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: CLI `report` 子命令 + 真實資料端到端驗證

**Files:**
- Modify: `src/video_flow_analytics/cli.py`

**Interfaces:**
- Consumes: `video_flow_analytics.report.pipeline.run_report`（Task 6）

- [ ] **Step 1: `cli.py` 新增 `report` 子命令**

修改 `src/video_flow_analytics/cli.py`：

```python
    subparsers.add_parser(
        "zone-map",
        help="讀 tracking_results.parquet 套 zone 幾何，輸出每時段每區域人流",
    )
    subparsers.add_parser(
        "report",
        help="讀 zone_counts.parquet 彙總成跨日累加的 Excel 人流報表",
    )
    args = parser.parse_args()

    if args.command == "analyze":
        from video_flow_analytics.analyze.pipeline import run_analyze

        run_analyze()
    elif args.command == "zone-map":
        from video_flow_analytics.zone_mapping.pipeline import run_zone_map

        run_zone_map()
    elif args.command == "report":
        from video_flow_analytics.report.pipeline import run_report

        run_report()
```

- [ ] **Step 2: 驗證 `report` 子命令的 argparse 註冊（不觸碰真實 config.toml）**

```bash
uv run video-flow-analytics report -h
```

Expected: 印出 `report` 子命令的 help 訊息（含 Step 1 寫的
`"讀 zone_counts.parquet 彙總成跨日累加的 Excel 人流報表"`），exit code 0，
不執行任何實際的報表邏輯（argparse 的 `-h` 會在呼叫 `run_report()` 之前就結束
程式）。

- [ ] **Step 3: 驗證 CLI 會正確 dispatch 到 `run_report()`**

真實的 `bucket_name1/camera_registry.yaml` 目前有跨攝影機重複的 zone 名稱，
會被 `_validate_unique_zone_names` 擋下來（這是 Task 5 就確認過的預期行為）。
用這個特性驗證 CLI 有正確把 `report` 指令 dispatch 到
`report.pipeline.run_report()`，同時完全不需要修改真實的 `config.toml`：

```bash
uv run python -c "
import sys
sys.argv = ['video-flow-analytics', 'report']
from video_flow_analytics.cli import main
try:
    main()
    print('BUG: 預期應該在 zone 名稱驗證處失敗')
except ValueError as e:
    print('CLI 正確 dispatch 到 run_report()，並在預期的驗證點失敗：', e)
"
```

Expected: 印出 `CLI 正確 dispatch 到 run_report()，並在預期的驗證點失敗：
camera_registry.yaml 中有跨攝影機重複的 zone 名稱...`，不是 `BUG: ...`。
這段驗證完全沒有寫檔、沒有修改任何檔案（`_validate_unique_zone_names` 在讀
`zone_counts.parquet` 之前就先擋下來了）。

> Task 6 Step 2 已經用暫存的 registry 副本完整驗證過
> `export_report_daily`（含 Excel 讀寫、overwrite/error 行為）真的能跑出正確
> 結果；這裡只需要額外驗證 CLI 這一層的 argparse 註冊與 dispatch 是否正確
> 接上，不需要重複整個端到端流程。

- [ ] **Step 4: ruff 檢查並 commit**

```bash
uv run ruff check .
git add src/video_flow_analytics/cli.py
git commit -m "$(cat <<'EOF'
feat(report): 新增 `report` CLI 子命令

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 更新 `CLAUDE.md` 文件

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新「專案概述」段落**

在「進入點是函式呼叫...」段落，`zone_mapping.pipeline.map_zones_daily` 那行
之後新增一行：

```markdown
- `report.pipeline.export_report_daily(date, bucket_dir, period_minutes, metric, on_duplicate_date, bucket_minutes) -> Path`（讀 `zone_counts.parquet` 彙總成跨日累加更新的 Excel 報表 `outputs/{bucket}/report.xlsx`；純 CPU 運算，不需重跑偵測或 zone mapping）
```

- [ ] **Step 2: 更新「常用指令」段落**

```markdown
uv run video-flow-analytics report     # 彙總 zone 人流成 Excel 報表，參數讀 config.toml 的 [report]
```

- [ ] **Step 3: 更新「套件結構」段落**

在 `- **`zone_mapping/`**：...獨立下游功能。只依賴 `core`。` 之後新增：

```markdown
- **`report/`**：`stats.py`（時區轉換、期間彙總、尖峰計算、用餐時段規則）、
  `pipeline.py`（讀 `zone_counts.parquet`、驗證、寫 Excel），獨立下游功能。
  只依賴 `core`。
```

- [ ] **Step 4: 在「Zone Mapping」段落之後新增「Report」小節**

```markdown
### Report（Excel 人流報表）

- **輸出**：`outputs/{bucket}/report.xlsx`，是跨日累加更新的單一檔案（不像
  `zone_counts.parquet` 逐日各一份），含三個分頁：「每小時人流」「每日尖峰」
  「活動事件」。「活動事件」目前只建標題列、由其他來源填入，`export_report_daily`
  不會動這個分頁。
- **zone 名稱全域唯一（新前提）**：報表的「區域」欄位以 zone 名稱分組、不含
  camera_id，因此要求整份 `camera_registry.yaml` 的 zone 名稱**跨攝影機也不可
  重複**（原本 `parsed_zones()` 只驗證同一攝影機內不重複）。此驗證只加在
  `report/pipeline.py._validate_unique_zone_names`，不影響 `analyze_daily` /
  `zone_mapping` 既有路徑。未來若有 UI 維護 `camera_registry.yaml`，會在該處
  即時擋下重複命名。
- **時區**：`zone_counts.parquet` 以 UTC 曆日切分，報表顯示台北時區
  （固定 +8 小時，無 DST）的本地小時／日期。單次執行涵蓋的 UTC 一天資料，轉換
  後會落在本地「當天 08:00–23:59」與「隔天 00:00–07:59」兩個曆日；若店家凌晨
  無營業不影響正確性，若有 24 小時營運資料則凌晨時段只會計入當次執行的輸出，
  不會等隔天執行時自動合併成完整一天。
- **`on_duplicate_date`**（`config.toml` 的 `[report]`）：重跑同一天時的處理
  方式，`overwrite`（預設）刪除既有相同日期的列後插入新列，`append` 直接加到
  尾端不檢查，`error` 發現重複日期就整個中止不寫入。
```

- [ ] **Step 5: commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: 補充 report 子系統與 zone 全域唯一規則說明

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review 檢查結果

1. **Spec 覆蓋**：`period_minutes`/`metric`/`on_duplicate_date` config（Task 1）、
   `to_taipei`/`rollup_by_period`/`peak_per_day`/`meal_time_reminder`/
   `weekday_zh`（Task 2-4）、zone 全域唯一驗證（Task 5）、Excel 讀寫合併三種
   模式與活動事件分頁不動（Task 6）、CLI 子命令與端到端驗證（Task 7）、
   CLAUDE.md 文件（Task 8）—— spec 各段落均有對應任務。
2. **Placeholder 掃描**：無 TBD/TODO，所有程式碼步驟皆為完整可執行內容。
3. **型別一致性**：`_build_report_frames` 回傳
   `tuple[pl.DataFrame, pl.DataFrame]`（Task 5）與 `export_report_daily`
   內部呼叫方式一致（Task 6）；`rollup_by_period` 輸出欄位
   `date/weekday/period/zone/value`（Task 3）與 `peak_per_day` 的輸入假設
   （Task 4）、`_write_report`/`_sort_rows` 的欄位順序（Task 6：hourly 用
   `(0,2,3)` 對應 `日期,小時,區域`；peak 用 `(0,2)` 對應 `日期,區域`）皆對得上
   Excel 標題順序 `_HOURLY_HEADERS`/`_PEAK_HEADERS`。
