import logging

from video_analyze.detector import _basename, _log_model_metadata


class _FakeModel:
    """只帶 `_log_model_metadata` 會用到的屬性，不真載權重。"""

    def __init__(self, names=None, ckpt=None):
        self.names = names
        self.ckpt = ckpt


def test_basename_strips_absolute_path():
    assert _basename("/home/trainer/runs/yolo26m_baseline/weights/best.pt") == (
        "best.pt"
    )


def test_basename_returns_non_string_unchanged():
    assert _basename(None) is None
    assert _basename(123) == 123


def test_basename_returns_empty_string_unchanged():
    assert _basename("") == ""


def test_log_model_metadata_with_full_ckpt(caplog):
    model = _FakeModel(
        names={0: "head", 1: "vbody", 2: "fbody"},
        ckpt={
            "train_args": {
                "model": "/home/trainer/runs/yolo26m_baseline/weights/best.pt",
                "data": "/home/trainer/datasets/crowdhuman/data.yaml",
            },
            "version": "8.4.90",
            "date": "2026-07-14",
            "train_metrics": {"mAP50": 0.806, "recall": 0.72, "precision": 0.847},
        },
    )

    with caplog.at_level(logging.INFO):
        _log_model_metadata(model)

    text = caplog.text
    assert "fbody" in text
    assert "best.pt" in text
    assert "data.yaml" in text
    assert "8.4.90" in text
    assert "2026-07-14" in text
    assert "0.806" in text


def test_log_model_metadata_missing_train_args_logs_available_fields(caplog):
    model = _FakeModel(
        names={2: "fbody"},
        ckpt={"version": "8.4.90"},
    )

    with caplog.at_level(logging.INFO):
        _log_model_metadata(model)

    assert "8.4.90" in caplog.text


def test_log_model_metadata_non_dict_ckpt_warns_without_raising(caplog):
    model = _FakeModel(names={2: "fbody"}, ckpt=None)

    with caplog.at_level(logging.WARNING):
        _log_model_metadata(model)

    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_log_model_metadata_missing_names_attr_does_not_raise(caplog):
    class _NoNamesModel:
        ckpt = {"version": "8.4.90"}

    with caplog.at_level(logging.INFO):
        _log_model_metadata(_NoNamesModel())

    assert "8.4.90" in caplog.text
