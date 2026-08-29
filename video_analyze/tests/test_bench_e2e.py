"""`tools/bench_e2e.py` 的測試，全部針對純函式（不起子進程、不碰 GPU）。

分四組：FPS 口徑、矩陣展開、環境與安全、產物解析。每條的 docstring 寫「這條壞掉會
怎樣」——量測工具的錯誤幾乎都不會拋例外，只會產生看似合理的數字。

能 import `bench_e2e` 是靠 `conftest.py` 把 `tools/` 插進 `sys.path`（見該檔）。
"""

import datetime
import json

import pytest
from bench_e2e import (
    MANAGED_SETTINGS_ENV,
    BenchCase,
    FpsReading,
    GpuUsage,
    RunRecord,
    TrackWorkerReading,
    _count_segments,
    _engine_path,
    _host_cpu,
    _model_classes,
    artifact_paths,
    bucket_output_dir,
    build_run_env,
    check_name_component,
    check_runs_dir_disjoint,
    clear_artifacts,
    expand_matrix,
    extra_settings_env,
    format_meta,
    group_runs,
    load_run_records,
    parse_fps_log,
    parse_gpu_log,
    parse_meta,
    shell_exit_code,
)

_INFERENCE_LINE = json.dumps(
    {
        "component": "inference",
        "message": "FPS 整體",
        "total_frames": 20590,
        "elapsed_seconds": 40.7,
        "overall_fps": 506.46,
        "severity": "INFO",
    },
    ensure_ascii=False,
)


def _track_worker_line(
    shard_id: int | None = 0,
    *,
    overall_fps: float = 514.34,
    tracking_fps: float | None = 968.18,
    cameras: tuple[str, ...] = ("cam01",),
) -> str:
    """一片追蹤進程的結束行。`shard_id=None` 模擬分片之前的舊產物。"""
    entry = {
        "component": "track_worker",
        "message": "追蹤進程結束",
        "frames": 20590,
        "elapsed_seconds": 40.0,
        "overall_fps": overall_fps,
        "tracking_fps": tracking_fps,
        "severity": "INFO",
    }
    if shard_id is not None:
        entry["shard_id"] = shard_id
        entry["owned_cameras"] = list(cameras)
    if tracking_fps is None:
        del entry["tracking_fps"]
    return json.dumps(entry, ensure_ascii=False)


_TRACK_WORKER_LINE = _track_worker_line()


def _record(name: str, fps: float | None, *, exit_status: str = "0", **meta: str) -> RunRecord:
    base = {"bucket": "bucket_20260801_perf40", "model_batch": "16", "foot_point_method": "head"}
    return RunRecord(
        name=name,
        meta={**base, **meta, "exit_status": exit_status},
        fps_reading=FpsReading(
            inference_fps=fps,
            inference_frames=20590 if fps else None,
            inference_elapsed_seconds=40.7 if fps else None,
            track_workers=(
                TrackWorkerReading(
                    shard_id=0,
                    overall_fps=514.34,
                    tracking_fps=968.18,
                    frames=20590,
                    cameras=("cam01",),
                ),
            ),
        ),
        gpu=GpuUsage(samples=20, mean_sm_percent=16.0, max_sm_percent=18.0, max_fb_mb=2287.0),
    )


# --------------------------------------------------------------------------------------
# FPS 口徑
# --------------------------------------------------------------------------------------


def test_parse_fps_log_takes_inference_not_track_worker():
    """每輪 log 有兩行 `overall_fps`，推論那行才是報表的口徑。

    取到追蹤進程那行不會有任何錯誤訊號——它的分母是自己的 wall clock，數字只高 1–2%，
    看起來完全合理，卻讓所有量測系統性高估。
    """
    reading = parse_fps_log(f"{_INFERENCE_LINE}\n{_TRACK_WORKER_LINE}\n")

    assert reading.inference_fps == 506.46
    assert reading.inference_frames == 20590
    assert reading.inference_elapsed_seconds == 40.7


def test_parse_fps_log_keeps_both_readings_under_separate_names():
    """兩個口徑各自留一個欄位，且**沒有**泛用的 `fps`／`overall_fps` 可以誤用。

    若哪天有人加了一個泛用欄位，這條會擋下：呼叫端一旦能拿到「一個 FPS」，
    第一道型別互鎖就失效了。
    """
    reading = parse_fps_log(f"{_TRACK_WORKER_LINE}\n{_INFERENCE_LINE}\n")

    assert reading.inference_fps == 506.46
    assert reading.min_track_worker_fps == 514.34
    assert not hasattr(reading, "fps")
    assert not hasattr(reading, "overall_fps")


def test_parse_fps_log_does_not_backfill_inference_from_track_worker():
    """推論那行缺席時不以追蹤值遞補，該輪標為未完成。

    遞補的方向剛好是系統性高估：崩在半路的那幾輪會混進統計，還把平均往上拉。
    """
    reading = parse_fps_log(f"{_TRACK_WORKER_LINE}\n")

    assert reading.inference_fps is None
    assert reading.min_track_worker_fps == 514.34
    assert reading.completed is False


def test_parse_fps_log_collects_one_reading_per_shard():
    """追蹤進程分片後每片各印一行，全部收下來，餘裕取最慢那片。

    這條壞掉的樣子是 `bench_e2e run` 第一輪跑完就崩（issue #143 之前 `parse_fps_log`
    對第二行追蹤結束拋 `ValueError`，`command_run` 沒接），矩陣後續輪次一輪都不跑。
    平均掉各片同樣不行：瓶頸是最慢的那片，平均會把它蓋掉。
    """
    log = (
        f"{_track_worker_line(0, overall_fps=520.0, tracking_fps=1040.0)}\n"
        f"{_INFERENCE_LINE}\n"
        f"{_track_worker_line(1, overall_fps=500.0, tracking_fps=800.0, cameras=('cam02',))}\n"
    )

    reading = parse_fps_log(log)

    assert reading.shard_count == 2
    assert [item.shard_id for item in reading.track_workers] == [0, 1]
    assert [item.cameras for item in reading.track_workers] == [("cam01",), ("cam02",)]
    assert reading.min_track_worker_fps == 500.0
    assert reading.min_track_worker_headroom == pytest.approx(1.6)
    assert reading.inference_fps == 506.46


def test_parse_fps_log_headroom_is_none_when_any_shard_lacks_tracking_fps():
    """任一片缺 `tracking_fps` 就不給餘裕，而不是拿剩下幾片算最小值。

    跳過缺值的片會漏掉「可能最慢的那片」，算出的餘裕偏大——這個數字是容量決策的依據，
    偏大正是會誤事的方向。
    """
    log = (
        f"{_track_worker_line(0, overall_fps=520.0, tracking_fps=1040.0)}\n"
        f"{_track_worker_line(1, overall_fps=500.0, tracking_fps=None)}\n"
    )

    reading = parse_fps_log(log)

    assert reading.shard_count == 2
    assert reading.min_track_worker_fps == 500.0
    assert reading.min_track_worker_headroom is None


def test_parse_fps_log_rejects_duplicate_inference_line():
    """同一份 log 出現兩行推論 `FPS 整體`，代表兩次執行寫進同一份檔案。

    靜默取其中一個就是「這輪的 meta 配上另一輪的 log」——產物看起來完好，數字對不上
    卻查不出原因。放寬追蹤那側之後這道更要在：推論行沒有分片，重複只有一種解釋。
    """
    with pytest.raises(ValueError, match="兩行推論進程"):
        parse_fps_log(f"{_INFERENCE_LINE}\n{_INFERENCE_LINE}\n")


def test_parse_fps_log_rejects_duplicate_shard_id():
    """兩行 `shard_id` 相同的追蹤結束＝兩次執行寫進同一份 log，仍要拋錯。

    同一次執行的各片編號互異，所以放寬「多行追蹤結束」之後，`shard_id` 是分辨「N 片」
    與「跑了兩次」的依據；不擋的話兩次執行會被當成兩片，最小值取到的是另一輪的數字。
    """
    with pytest.raises(ValueError, match="shard_id=0"):
        parse_fps_log(f"{_TRACK_WORKER_LINE}\n{_TRACK_WORKER_LINE}\n")


def test_parse_fps_log_rejects_duplicate_track_line_without_shard_id():
    """分片之前的舊產物沒有 `shard_id`，那時每輪只有一行，兩行同樣算兩次執行。"""
    old_line = _track_worker_line(None)

    with pytest.raises(ValueError, match="shard_id=None"):
        parse_fps_log(f"{old_line}\n{old_line}\n")


def test_parse_fps_log_skips_non_json_lines():
    """log 前段混著 uv 的下載進度與 ultralytics 橫幅，不是 JSON 的行要略過而非炸掉。"""
    noisy = (
        "Downloading video-analyze (1.2MiB)\n"
        "not json at all overall_fps=999\n"
        f"{_INFERENCE_LINE}\n"
    )

    assert parse_fps_log(noisy).inference_fps == 506.46


def test_group_runs_averages_inference_values_only():
    """分組平均只吃推論值——第二道互鎖，與解析器的欄位命名互為備援。

    這裡兩輪的追蹤值都是 514.34；若統計誤用追蹤口徑，平均會變成 514.34 而非 506.46。
    """
    stats = group_runs([_record("main_a", 505.0), _record("main_b", 507.92)])

    assert len(stats) == 1
    assert stats[0].mean_inference_fps == pytest.approx(506.46)
    assert stats[0].values == (505.0, 507.92)


def test_group_runs_excludes_failed_incomplete_and_prefixed_runs():
    """非零 exit、缺推論行、排除前綴的 run 都不進統計。

    smoke 那輪跑的是縮短的資料量，混進來會把平均整個帶偏；失敗的那輪則多半只跑了
    一部分影片，FPS 反而偏高。
    """
    records = [
        _record("main_ok", 506.46),
        _record("main_failed", 700.0, exit_status="1"),
        _record("main_incomplete", None),
        _record("smoke_warmup", 300.0),
    ]

    stats = group_runs(records, exclude_prefixes=["smoke"])

    assert len(stats) == 1
    assert stats[0].runs == 1
    assert stats[0].values == (506.46,)


def test_group_runs_does_not_average_across_labels():
    """`--label` 是分組維度：before／after 不可以被平均成一組。

    這條壞掉的後果正好是這支工具最主要用途的反面——拿它比較兩個 commit 有沒有拖慢，
    四輪合成一組 `n=4`，要找的回歸剛好被平均掉，而報表上完全看不出來。
    `commit`／`date`／`machine` 同理，共用同一個 key。
    """
    records = [
        _record("before_r1", 500.0, label="before", commit="aaaaaaa"),
        _record("before_r2", 502.0, label="before", commit="aaaaaaa"),
        _record("after_r1", 400.0, label="after", commit="bbbbbbb"),
        _record("after_r2", 402.0, label="after", commit="bbbbbbb"),
    ]

    stats = group_runs(records)

    assert len(stats) == 2
    assert {stat.label for stat in stats} == {"before", "after"}
    assert {round(stat.mean_inference_fps) for stat in stats} == {501, 401}


def test_group_runs_survives_an_all_zero_group():
    """整組都是 0.0 張/秒時，離散度不能把整份 report 炸掉。

    0.0 是合法值不是異常輸入：`FpsMeter._safe_div` 在 `total_frames == 0`（那輪一格都
    沒處理完）時就回 0.0，而 `--repeat 2` 是預設值。離散度除以平均會 `ZeroDivisionError`，
    連帶其他組態的結果也一起印不出來。
    """
    stats = group_runs([_record("main_a", 0.0), _record("main_b", 0.0)])

    assert len(stats) == 1
    assert stats[0].runs == 2
    assert stats[0].mean_inference_fps == 0.0
    assert stats[0].spread_percent == 0.0


def test_engine_path_prefers_the_environment_override(tmp_path):
    """引擎身分要記**實際會載入的**那顆。

    `MODEL__MODEL_PATH` 會覆寫 `config.toml`；只讀設定檔的話 meta 記下的是一顆從未被
    量測過的引擎的 sha256——而事後判斷「這批數字是哪顆引擎跑的」全靠這個欄位。
    """
    config = tmp_path / "config.toml"
    config.write_text('[model]\nmodel_path = "from_config.engine"\n', encoding="utf-8")

    from_config, source_config = _engine_path(config, {})
    from_env, source_env = _engine_path(config, {"MODEL__MODEL_PATH": "from_env.engine"})

    assert (from_config, source_config) == ("from_config.engine", f"config:{config}")
    assert (from_env, source_env) == ("from_env.engine", "env:MODEL__MODEL_PATH")


def test_engine_path_rejects_a_blank_environment_override(tmp_path):
    """`MODEL__MODEL_PATH` 設了卻是空白＝誤用，要中止而不是悄悄回退到設定檔。

    回退的話 meta 記下的是一顆從未被量測過的引擎的 sha256——而事後判斷「這批數字是
    哪顆引擎跑的」全靠這個欄位；空字串更糟，`Path("").is_file()` 恆為 False，
    產物看起來完整，只是 sha256 欄位變成「找不到引擎檔」，掩蓋了實際載入的其實是
    設定檔那顆引擎。
    """
    config = tmp_path / "config.toml"
    config.write_text('[model]\nmodel_path = "from_config.engine"\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="MODEL__MODEL_PATH"):
        _engine_path(config, {"MODEL__MODEL_PATH": "   "})


def test_model_classes_prefers_the_environment_override(tmp_path):
    """偵測類別要記**實際生效的**那組，理由與引擎身分同型。

    `MODEL__CLASSES` 覆寫 `config.toml` 時只讀設定檔，meta 會記下一組沒被量到的類別；
    而少偵測一個類別本來就會比較快——事後就分不出「這層改動變快了」與「這層量的東西
    比較少」。
    """
    config = tmp_path / "config.toml"
    config.write_text("[model]\nclasses = [0, 2]\n", encoding="utf-8")

    assert _model_classes(config, {}) == "[0, 2]"
    assert _model_classes(config, {"MODEL__CLASSES": "[2]"}) == "[2]"


def test_model_classes_rejects_an_empty_classes_setting(tmp_path):
    """類別設定是空的就中止，不留白往下跑。

    記成空字串的話產物看起來完整、每個欄位都在，只有「這輪偵測了什麼」永遠查不回來，
    而這是判讀跨層 FPS 差異的前提。
    """
    config = tmp_path / "config.toml"
    config.write_text("[model]\nmodel_path = \"x.engine\"\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="classes"):
        _model_classes(config, {})


def test_model_classes_rejects_a_blank_environment_override(tmp_path):
    """`MODEL__CLASSES` 設了卻是空白＝誤用，要中止而不是悄悄回退到設定檔。

    回退的話 meta 記下的是一組沒被量到的類別——正是這個欄位要防的事；記成空字串更糟，
    產物看起來完整，只有這欄默默沒有內容。
    """
    config = tmp_path / "config.toml"
    config.write_text("[model]\nclasses = [0, 2]\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="MODEL__CLASSES"):
        _model_classes(config, {"MODEL__CLASSES": "   "})


def test_host_cpu_reads_the_first_model_name(tmp_path):
    """主機身分取 `/proc/cpuinfo` 的第一個 `model name`（多核心會有很多行相同的）。

    沒有這欄的代價已經發生過一次：2026-08-26 量到同組態跨天差 7.9%，最可能的解釋是
    VM 換了實體主機，但事後查不回來——機器一關就沒有任何地方記得那輪跑在哪裡。
    """
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.30GHz\n"
        "processor\t: 1\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.30GHz\n",
        encoding="utf-8",
    )

    assert _host_cpu(cpuinfo) == "Intel(R) Xeon(R) CPU @ 2.30GHz"


def test_host_cpu_falls_back_when_cpuinfo_is_unavailable(tmp_path):
    """讀不到就記 `<未知>`：這是給人判讀用的線索，不能反過來讓量測跑不起來。"""
    assert _host_cpu(tmp_path / "does-not-exist") == "<未知>"
    (tmp_path / "no-model-name").write_text("processor\t: 0\n", encoding="utf-8")
    assert _host_cpu(tmp_path / "no-model-name") == "<未知>"


# --------------------------------------------------------------------------------------
# 矩陣展開
# --------------------------------------------------------------------------------------


def test_expand_matrix_puts_repeat_at_the_outermost_level():
    """重複次數在最外層：同組態不連跑。

    連跑會把熱漂移集中在單一格——那一格的離散度變成「機器暖機曲線」而不是「量測雜訊」，
    而離散度正是判斷 FPS 差異有沒有意義的依據。
    """
    cases = expand_matrix(["bucket_a", "bucket_b"], [16, 8], "head", repeat=2, date="2026-08-01")

    assert [case.repeat_index for case in cases] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert [(case.bucket, case.batch) for case in cases[:4]] == [
        ("bucket_a", 16),
        ("bucket_a", 8),
        ("bucket_b", 16),
        ("bucket_b", 8),
    ]


def test_expand_matrix_names_are_unique_per_cell():
    """每格一個唯一名稱——名稱同時是四個產物檔的 stem，撞名等於覆蓋上一格的產物。"""
    cases = expand_matrix(["bucket_a", "bucket_b"], [16, 8], "head", repeat=3, date="2026-08-01")

    names = [case.name for case in cases]
    assert len(set(names)) == len(names) == 12


def test_expand_matrix_name_contains_foot_point_method():
    """落腳點方法要進名稱：對照組是**另一次執行、寫進同一個產物目錄**，靠名稱區分。

    少了它，`--foot-point bbox_bottom` 那次會逐檔覆蓋掉 head 那次的產物。
    """
    (case,) = expand_matrix(
        ["bucket_20260801_perf40"], [16], "bbox_bottom", repeat=1, date="2026-08-01"
    )

    assert case.name == "main_20260801_perf40_d20260801_b16_bbox_bottom_r1"


def test_expand_matrix_name_contains_the_analysis_date():
    """分析日期要進名稱，且與 bucket 名稱裡的日期是兩件事。

    `bucket` 名稱裡的 `20260801` 是資料集命名，`--date` 才是這輪分析的是哪一天。多日
    bucket 換 `--date` 重跑時，少了它兩輪名稱完全相同——不是被「產物已存在」擋下，
    就是配上 `--overwrite` 把前一天的產物毀掉。
    """
    (first,) = expand_matrix(["bucket_multi"], [16], "head", repeat=1, date="2026-08-01")
    (second,) = expand_matrix(["bucket_multi"], [16], "head", repeat=1, date="2026-08-02")

    assert first.name != second.name
    assert "d20260801" in first.name
    assert "d20260802" in second.name


# --------------------------------------------------------------------------------------
# 環境與安全
# --------------------------------------------------------------------------------------


def test_build_run_env_sets_four_namespaces_as_strings():
    """四個 pydantic-settings 命名空間都要設到，且值必須是 `str`。

    `os.environ` 只收字串：`MODEL__BATCH` 傳 int 會在 `Popen` 當場 `TypeError`；
    漏設任何一個則會靜默沿用 `config.toml` 的預設值，量到的不是指定的組態。
    """
    case = BenchCase(
        name="main_x", bucket="bucket_20260801_perf40", batch=16,
        foot_point_method="head", repeat_index=1,
    )

    env = build_run_env({"PATH": "/usr/bin"}, case, "2026-08-01")

    assert env["INPUT__BUCKET_DIR"] == "bucket_20260801_perf40"
    assert env["INPUT__DATE"] == "2026-08-01"
    assert env["MODEL__BATCH"] == "16"
    assert env["FOOT_POINT__METHOD"] == "head"
    assert all(isinstance(value, str) for value in env.values())
    assert env["PATH"] == "/usr/bin"


def test_extra_settings_env_is_case_insensitive():
    """pydantic-settings 預設不分大小寫，只比對大寫等於留一個繞過整套防護的後門。

    實測 `input__camera_ids` 小寫一樣會讓子進程只跑一路；漏掉它的話，那輪的子集
    工作量既不進 meta 也不進分組 key，會和全量那輪平均成一個看起來合理的數字。
    """
    extras = json.loads(
        extra_settings_env(
            {"input__camera_ids": '["camX"]', "Model__Batch": "4"}, MANAGED_SETTINGS_ENV
        )
    )

    assert extras == {"input__camera_ids": '["camX"]'}


def test_extra_settings_env_captures_runtime_env():
    """`CUDA_VISIBLE_DEVICES` 決定跑哪張卡，兩輪跑在不同卡上不可以被平均。

    它不是 settings 變數，但改變量測結果的程度不下於它們——連 GPU 取樣取的是不是
    同一張卡都取決於它。
    """
    extras = json.loads(
        extra_settings_env({"CUDA_VISIBLE_DEVICES": "1", "TERM": "xterm"}, MANAGED_SETTINGS_ENV)
    )

    assert extras == {"CUDA_VISIBLE_DEVICES": "1"}


def test_load_run_records_isolates_a_broken_artifact(tmp_path):
    """一份壞產物不可以讓整份報表印不出來，而且要說得出是哪一輪壞了。

    `parse_fps_log` 對重複的推論行刻意拋 `ValueError`，但訊息裡沒有檔名；不隔離的話
    使用者只看到一個沒有線索的例外，其他跑得好好的輪次也一起消失。
    """
    (tmp_path / "good.meta").write_text("name=good\n", encoding="utf-8")
    (tmp_path / "good.log").write_text(f"{_INFERENCE_LINE}\n", encoding="utf-8")
    (tmp_path / "bad.meta").write_text("name=bad\n", encoding="utf-8")
    (tmp_path / "bad.log").write_text(
        f"{_INFERENCE_LINE}\n{_INFERENCE_LINE}\n", encoding="utf-8"
    )

    records, broken = load_run_records(tmp_path)

    assert [record.name for record in records] == ["good"]
    assert len(broken) == 1
    assert broken[0].startswith("bad：")


@pytest.mark.parametrize(
    "label", ["../bucket_x/y", "a/b", "", " main", "main\nname=fake", "a..b"]
)
def test_check_name_component_rejects_unsafe_labels(label):
    """`--label` 直接是產物檔 stem 的一部分，而 bucket 的那兩道檢查看不到它。

    `--label '../<bucket>/x'` 會把產物寫進每輪開頭 `rmtree` 的目錄，矩陣跑完只剩最後
    一輪；帶換行則會把 meta 的逐行 `key=value` 格式撐破。
    """
    with pytest.raises(SystemExit):
        check_name_component(label, "--label")


@pytest.mark.parametrize("bucket", ["../etc", "a/b", "", " bucket_x", "bucket/../.."])
def test_bucket_output_dir_rejects_path_escape(bucket, tmp_path):
    """這個路徑會被 `shutil.rmtree`，而 bucket 是 CLI 字串。逃逸即中止。"""
    with pytest.raises(SystemExit):
        bucket_output_dir(bucket, tmp_path)


def test_bucket_output_dir_returns_dir_under_output_root(tmp_path):
    """正常的 bucket 名稱解析成 `<output_root>/<bucket>`。"""
    assert bucket_output_dir("bucket_20260801_perf40", tmp_path) == (
        tmp_path / "bucket_20260801_perf40"
    ).resolve()


def test_check_runs_dir_disjoint_rejects_runs_dir_inside_cleared_dir(tmp_path):
    """產物目錄落在每輪會被清空的目錄底下，產物會自己刪自己——第一輪之後全空。"""
    cleared = (tmp_path / "outputs" / "bucket_x").resolve()

    with pytest.raises(SystemExit, match="會被清空"):
        check_runs_dir_disjoint(cleared / "runs", [cleared])

    check_runs_dir_disjoint(tmp_path / "outputs" / "bench_e2e", [cleared])


# --------------------------------------------------------------------------------------
# 產物解析
# --------------------------------------------------------------------------------------


def test_meta_round_trips_value_containing_equals():
    """meta 的值可以帶 `=`（指令字串、環境變數），只能切第一個。"""
    pairs = {"command": "uv run --package video_analyze video_analyze", "note": "a=b=c"}

    assert parse_meta(format_meta(pairs)) == pairs


def test_parse_meta_skips_lines_without_equals():
    """舊產物的 meta 尾巴可能黏到別的輸出；沒有 `=` 的行略過而不是讓整份解析失敗。"""
    assert parse_meta("name=main_a\n矩陣完成\nexit_status=0\n") == {
        "name": "main_a",
        "exit_status": "0",
    }


def test_parse_gpu_log_finds_columns_by_header_not_fixed_index():
    """`dmon -s` 換指標集就換欄位順序：`um` 的 sm 在第 3 欄，`pucm` 在第 6 欄。

    寫死索引的話，舊產物（`-s um`）解出來會是對的、新產物（`-s pucm`）解出來是 gtemp，
    而 35「%」看起來完全像個合理的 SM 使用率。
    """
    old_format = (
        "#Time         gpu     sm    mem    enc    dec    jpg    ofa     fb   bar1   ccpm\n"
        "#HH:MM:SS     Idx      %      %      %      %      %      %     MB     MB     MB\n"
        " 21:54:15       0     17      7      0      0      0      0   2287     23      0\n"
        " 21:54:17       0     15      6      0      0      0      0   2287     23      0\n"
    )
    new_format = (
        "#Time         gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg"
        "    ofa   mclk   pclk     fb   bar1   ccpm\n"
        "#HH:MM:SS     Idx      W      C      C      %      %      %      %      %"
        "      %    MHz    MHz     MB     MB     MB\n"
        " 07:12:38       0     14     35      -     17      7      0      0      0"
        "      0    405    180   2287     21      0\n"
        " 07:12:39       0     14     35      -     15      6      0      0      0"
        "      0    405    180   2287     21      0\n"
    )

    for text in (old_format, new_format):
        usage = parse_gpu_log(text)
        assert usage.samples == 2
        assert usage.mean_sm_percent == pytest.approx(16.0)
        assert usage.max_sm_percent == 17.0
        assert usage.max_fb_mb == 2287.0


def test_parse_gpu_log_returns_none_when_there_is_no_data_row():
    """取樣一列都沒有時三個統計值為 None，`samples` 為 0。

    回 0.0 的話「取樣沒跑起來」與「GPU 真的閒置」在表上長得一模一樣。
    """
    usage = parse_gpu_log(
        "#Time         gpu     sm    mem\n#HH:MM:SS     Idx      %      %\n"
    )

    assert usage == GpuUsage(
        samples=0, mean_sm_percent=None, max_sm_percent=None, max_fb_mb=None
    )


@pytest.mark.parametrize(
    ("wait_status", "expected"),
    [(0, 0), (1 << 8, 1), (3 << 8, 3), (9, 137), (15, 143)],
)
def test_shell_exit_code_follows_shell_convention(wait_status, expected):
    """`os.wait4` 給的是原始 status word，不是 `$?`。

    直接寫進 meta 的話，exit code 3 會記成 768、被 SIGKILL 記成 9（看起來像 exit 9），
    而報表判「這輪成功了嗎」是比對 `exit_status == "0"`，舊產物的數字也就對不上了。
    """
    assert shell_exit_code(wait_status) == expected


def test_group_runs_keeps_a_zero_fps_run_instead_of_crashing():
    """`inference_fps == 0.0` 是合法的量測值（跑起來但一格都沒處理完）。

    取值與收成員兩處若用不同的判準（真值 vs `is not None`），這一筆會被收進成員卻
    濾出取值清單，該組只有它時 `statistics.mean([])` 直接拋 `StatisticsError`
    ——整份 report 因為一筆退化資料而印不出來。
    """
    stats = group_runs([_record("main_zero", 0.0)])

    assert stats[0].values == (0.0,)
    assert stats[0].mean_inference_fps == 0.0


def test_extra_settings_env_reports_only_the_keys_the_tool_did_not_set():
    """本工具只設四個變數，子進程卻繼承整份環境。

    環境裡先有一個 `INPUT__CAMERA_IDS`，那輪量的就是別的工作量，而 FPS 表上完全
    看不出來。這裡收的正是「會生效、但工具沒設」的那些。
    """
    env = {
        "PATH": "/usr/bin",
        "INPUT__BUCKET_DIR": "bucket_x",
        "INPUT__CAMERA_IDS": "cam1",
        "TRACKER__TRACK_BUFFER": "60",
        "UNRELATED__FOO": "1",
    }

    assert extra_settings_env(env, MANAGED_SETTINGS_ENV) == (
        '{"INPUT__CAMERA_IDS": "cam1", "TRACKER__TRACK_BUFFER": "60"}'
    )


def test_extra_settings_env_is_empty_json_when_environment_is_clean():
    """乾淨環境回 `"{}"`——舊產物沒有這個鍵時的預設值，分組行為要與它一致。"""
    assert extra_settings_env({"PATH": "/usr/bin"}, MANAGED_SETTINGS_ENV) == "{}"


def test_group_runs_does_not_average_across_different_inherited_env():
    """繼承到的 settings 環境不同的兩輪，結構上不可能被平均在一起。

    只把它記進 meta 是不夠的：`report` 照樣會把兩種工作量的 FPS 平均成一個看起來
    合理的數字。所以它進分組 key。
    """
    stats = group_runs(
        [
            _record("main_full", 500.0),
            _record("main_subset", 900.0, extra_settings_env='{"INPUT__CAMERA_IDS": "cam1"}'),
        ]
    )

    assert {stat.extra_settings_env: stat.values for stat in stats} == {
        "{}": (500.0,),
        '{"INPUT__CAMERA_IDS": "cam1"}': (900.0,),
    }


def test_build_run_env_covers_exactly_the_managed_keys():
    """`build_run_env` 設的與 `MANAGED_SETTINGS_ENV` 宣告的必須是同一組。

    兩邊漂移的話 `extra_settings_env` 會把工具自己設的變數當成「繼承到的」回報
    （每輪都跳警告、分組 key 也被撐開），或反過來漏掉一個真正的外來變數。
    """
    case = BenchCase(
        name="main_x", bucket="bucket_x", batch=16, foot_point_method="head", repeat_index=1,
    )

    env = build_run_env({"PATH": "/usr/bin"}, case, "2026-08-01")

    assert set(env) - {"PATH"} == set(MANAGED_SETTINGS_ENV)


def test_expand_matrix_rejects_colliding_run_names():
    """去前綴後同 tag 的兩個 bucket 會展開出同名的格子。

    開跑前的既存檔檢查抓不到（第一格的產物是它自己寫的），第二格會在矩陣**執行中途**
    覆蓋第一格，且沒有任何訊號。
    """
    with pytest.raises(SystemExit, match="撞名"):
        expand_matrix(["bucket_x", "x"], [16], "head", repeat=1, date="2026-08-01")

    with pytest.raises(SystemExit, match="撞名"):
        expand_matrix(["bucket_x", "bucket_x"], [16], "head", repeat=1, date="2026-08-01")


def test_clear_artifacts_removes_the_whole_set(tmp_path):
    """覆蓋前要整組刪，不能只靠寫入時截斷。

    `--no-gpu-log` 時 `.gpu.log` 根本不開檔、該輪沒寫出 parquet 時 `.parquet` 也不會
    被碰：留著就是上一輪的產物配這一輪的 meta，report 會印出屬於別輪的 sm%。
    """
    paths = artifact_paths(tmp_path, "main_x")
    assert set(paths) == {"meta", "log", "gpu", "parquet"}
    for path in paths.values():
        path.write_text("stale", encoding="utf-8")

    clear_artifacts(tmp_path, "main_x")

    assert not any(path.exists() for path in paths.values())
    clear_artifacts(tmp_path, "main_x")  # 再刪一次不應拋錯


def test_count_segments_only_counts_the_measured_date(tmp_path):
    """多日 bucket 只該數 `--date` 那一天——遞迴整個 bucket 會高估數倍。"""
    for day in ("01", "02"):
        segment_dir = tmp_path / "loc_cam001" / "2026" / "08" / day
        segment_dir.mkdir(parents=True)
        (segment_dir / "030000.000Z.mkv").touch()

    assert _count_segments(str(tmp_path), datetime.date(2026, 8, 1)) == 1


def test_count_segments_is_not_limited_to_mkv(tmp_path):
    """副檔名由 registry 的 `storage.file_ext` 決定；寫死 `.mkv` 會讓別的 bucket 記成 0。

    記成 0 不會報錯，只會讓事後追溯少掉「這輪到底有多少工作量」這個對照。
    """
    segment_dir = tmp_path / "loc_cam001" / "2026" / "08" / "01"
    segment_dir.mkdir(parents=True)
    (segment_dir / "030000.000Z.mp4").touch()
    (segment_dir / "031000.000Z.mp4").touch()

    assert _count_segments(str(tmp_path), datetime.date(2026, 8, 1)) == 2
