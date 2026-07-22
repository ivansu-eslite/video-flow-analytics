"""`StructuredLogger` 的輸出契約。

`ensure_ascii=False` 與「ERROR 以上走 stderr」都是刻意的地端選擇（見 README），
外觀上很像可以順手「修掉」的東西，故用測試釘住。
"""

import json

from vfa_observability import StructuredLogger


def test_info_emits_one_json_line_to_stdout_with_readable_chinese(capsys):
    """中文不逸出成 \\uXXXX：地端直接看終端機是主要的維運手段。"""
    StructuredLogger(component="zone_map").info("區域統計已寫入", rows=6)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "區域統計已寫入" in captured.out
    assert "\\u" not in captured.out

    payload = json.loads(captured.out)
    assert payload == {
        "severity": "INFO",
        "message": "區域統計已寫入",
        "component": "zone_map",
        "rows": 6,
    }


def test_error_goes_to_stderr_not_stdout(capsys):
    """stdout 常被下游當資料管道收走，錯誤混進去會被一起消化掉。"""
    StructuredLogger(component="config").error("設定載入失敗")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["severity"] == "ERROR"


def test_exception_carries_type_message_and_stacktrace(capsys):
    logger = StructuredLogger(component="report_builder")
    try:
        raise ValueError("找不到設備登錄檔")
    except ValueError as exc:
        logger.exception("報表產生失敗", error=exc)

    error = json.loads(capsys.readouterr().err)["error"]
    assert error["type"] == "ValueError"
    assert error["message"] == "找不到設備登錄檔"
    assert "ValueError: 找不到設備登錄檔" in error["stacktrace"]
