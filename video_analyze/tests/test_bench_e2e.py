"""`tools/bench_e2e.py` 的測試，全部針對純函式（不起子進程、不碰 GPU）。

分四組：FPS 口徑、矩陣展開、環境與安全、產物解析。每條的 docstring 寫「這條壞掉會
怎樣」——量測工具的錯誤幾乎都不會拋例外，只會產生看似合理的數字。

能 import `bench_e2e` 是靠 `conftest.py` 把 `tools/` 插進 `sys.path`（見該檔）。
"""

import json

import pytest
from bench_e2e import (
    BenchCase,
    FpsReading,
    GpuUsage,
    RunRecord,
    bucket_output_dir,
    build_run_env,
    check_runs_dir_disjoint,
    expand_matrix,
    format_meta,
    group_runs,
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
_TRACK_WORKER_LINE = json.dumps(
    {
        "component": "track_worker",
        "message": "追蹤進程結束",
        "frames": 20590,
        "elapsed_seconds": 40.0,
        "overall_fps": 514.34,
        "tracking_fps": 968.18,
        "severity": "INFO",
    },
    ensure_ascii=False,
)


def _record(name: str, fps: float | None, *, exit_status: str = "0", **meta: str) -> RunRecord:
    base = {"bucket": "bucket_20260801_perf40", "model_batch": "16", "foot_point_method": "head"}
    return RunRecord(
        name=name,
        meta={**base, **meta, "exit_status": exit_status},
        fps_reading=FpsReading(
            inference_fps=fps,
            inference_frames=20590 if fps else None,
            inference_elapsed_seconds=40.7 if fps else None,
            track_worker_fps=514.34,
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
    assert reading.track_worker_fps == 514.34
    assert not hasattr(reading, "fps")
    assert not hasattr(reading, "overall_fps")


def test_parse_fps_log_does_not_backfill_inference_from_track_worker():
    """推論那行缺席時不以追蹤值遞補，該輪標為未完成。

    遞補的方向剛好是系統性高估：崩在半路的那幾輪會混進統計，還把平均往上拉。
    """
    reading = parse_fps_log(f"{_TRACK_WORKER_LINE}\n")

    assert reading.inference_fps is None
    assert reading.track_worker_fps == 514.34
    assert reading.completed is False


def test_parse_fps_log_rejects_duplicate_inference_line():
    """同一份 log 出現兩行推論 `FPS 整體`，代表兩次執行寫進同一份檔案。

    靜默取其中一個就是「這輪的 meta 配上另一輪的 log」——產物看起來完好，數字對不上
    卻查不出原因。
    """
    with pytest.raises(ValueError, match="兩行推論進程"):
        parse_fps_log(f"{_INFERENCE_LINE}\n{_INFERENCE_LINE}\n")


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


# --------------------------------------------------------------------------------------
# 矩陣展開
# --------------------------------------------------------------------------------------


def test_expand_matrix_puts_repeat_at_the_outermost_level():
    """重複次數在最外層：同組態不連跑。

    連跑會把熱漂移集中在單一格——那一格的離散度變成「機器暖機曲線」而不是「量測雜訊」，
    而離散度正是判斷 FPS 差異有沒有意義的依據。
    """
    cases = expand_matrix(["bucket_a", "bucket_b"], [16, 8], "head", repeat=2)

    assert [case.repeat_index for case in cases] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert [(case.bucket, case.batch) for case in cases[:4]] == [
        ("bucket_a", 16),
        ("bucket_a", 8),
        ("bucket_b", 16),
        ("bucket_b", 8),
    ]


def test_expand_matrix_names_are_unique_per_cell():
    """每格一個唯一名稱——名稱同時是四個產物檔的 stem，撞名等於覆蓋上一格的產物。"""
    cases = expand_matrix(["bucket_a", "bucket_b"], [16, 8], "head", repeat=3)

    names = [case.name for case in cases]
    assert len(set(names)) == len(names) == 12


def test_expand_matrix_name_contains_foot_point_method():
    """落腳點方法要進名稱：對照組是**另一次執行、寫進同一個產物目錄**，靠名稱區分。

    少了它，`--foot-point bbox_bottom` 那次會逐檔覆蓋掉 head 那次的產物。
    """
    (case,) = expand_matrix(["bucket_20260801_perf40"], [16], "bbox_bottom", repeat=1)

    assert case.name == "main_20260801_perf40_b16_bbox_bottom_r1"


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
