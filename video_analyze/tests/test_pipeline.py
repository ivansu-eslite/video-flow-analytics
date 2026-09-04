"""子進程異常結束時的彙總訊息。

`_raise_if_abnormal` 原本是 `analyze_daily` 裡的 closure，那個形態下只有跑完整條
pipeline 才碰得到；提到模組層級之後可以用假的 process 物件直接驗訊息內容——彙總只讀
`name`／`pid`／`exitcode`／`is_alive()` 四樣，不需要真的起子進程。
"""

import json

import pytest

from video_analyze.services import pipeline


class _ProcStub:
    """假的子進程。"""

    def __init__(self, name: str, pid: int, exitcode: int | None, alive: bool = False):
        self.name = name
        self.pid = pid
        self.exitcode = exitcode
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _warning_records(captured_out: str) -> list[dict]:
    """從 stdout 挑出 `severity == "WARNING"` 的單行 JSON。"""
    records = []
    for line in captured_out.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        record = json.loads(stripped)
        if record.get("severity") == "WARNING":
            records.append(record)
    return records


def test_abnormal_exit_message_carries_the_role_and_its_camera():
    """`pid` 對不回「哪一路攝影機」，也對不回「讀取、推理還是追蹤」。

    真正的根因在子進程自己的 traceback 上，而那份輸出與另外十幾個進程交錯在同一份
    stderr；彙總訊息指不出角色的話，就得從時間順序猜是哪一個進程死掉。
    """
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._raise_if_abnormal([_ProcStub("reader[test_cam001]", 51234, 1)])

    message = str(excinfo.value)
    assert "reader[test_cam001]" in message
    assert "pid=51234" in message
    assert "exitcode=1" in message
    # 「自己拋例外」與「被訊號終止」在訊息上要分得開：exitcode=1 不掛任何訊號名
    assert "SIG" not in message


def test_abnormal_exit_message_names_the_signal_for_a_negative_exitcode():
    """`exitcode=-9` 要看得出是 SIGKILL——負值本身讀不出「誰殺的」。"""
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._raise_if_abnormal([_ProcStub("track[shard0]", 51230, -9)])

    message = str(excinfo.value)
    assert "track[shard0]" in message
    assert "exitcode=-9" in message
    assert "SIGKILL" in message


def test_abnormal_exit_survives_a_signal_value_it_cannot_name():
    """`signal.Signals(99)` 會拋 `ValueError`。

    彙總是失敗路徑上最後一道訊息，自己爆掉會把根因蓋掉——換成 `ValueError` 之後，
    呼叫端的 `except Exception` 照樣 `_terminate_all` 並拋出，但拋的是「訊號值查不到」
    而不是「子進程異常結束」。
    """
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._raise_if_abnormal([_ProcStub("inference", 51231, -99)])

    message = str(excinfo.value)
    assert "inference" in message
    assert "exitcode=-99" in message


def test_abnormal_exit_reports_every_abnormal_process():
    """一批同時死掉時逐個列出，不要只講第一個。"""
    with pytest.raises(RuntimeError) as excinfo:
        pipeline._raise_if_abnormal(
            [
                _ProcStub("reader[test_cam001]", 51234, 1),
                _ProcStub("track[shard0]", 51230, -9),
            ]
        )

    message = str(excinfo.value)
    assert "reader[test_cam001]" in message
    assert "track[shard0]" in message


def test_no_raise_when_every_process_ended_cleanly_or_is_still_running():
    """判定條件是 `not is_alive() and exitcode`：0 為 falsy，存活的一概不算。"""
    pipeline._raise_if_abnormal(
        [
            _ProcStub("reader[test_cam001]", 51234, 0),
            _ProcStub("inference", 51231, None, alive=True),
        ]
    )


def test_forced_kill_warning_carries_the_process_name(capsys):
    """同一條失敗路徑上、同樣只有 pid 的一筆 warning。"""

    class _StubbornProc(_ProcStub):
        """terminate 之後仍存活，逼出強制 kill 那條路徑。"""

        def __init__(self):
            super().__init__("reader[test_cam002]", 51235, None, alive=True)
            self.killed = False

        def terminate(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

        def kill(self) -> None:
            self.killed = True
            self._alive = False

    process = _StubbornProc()

    pipeline._terminate_all([process])

    assert process.killed
    (record,) = _warning_records(capsys.readouterr().out)
    assert record["component"] == "pipeline"
    assert record["name"] == "reader[test_cam002]"
    assert record["pid"] == 51235
