import logging

import pytest

from video_analyze.config import settings
from video_analyze.detector import (
    YOLODetector,
    _basename,
    _log_model_metadata,
    _validate_classes,
)


class _FakeModel:
    """只帶 `_log_model_metadata`／`_validate_classes` 會用到的屬性，不真載權重。"""

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


def test_log_model_metadata_exception_during_read_warns_without_raising(caplog):
    class _RaisingNamesModel:
        ckpt = {"version": "8.4.90"}

        @property
        def names(self):
            raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        _log_model_metadata(_RaisingNamesModel())  # 不拋例外，只 warning

    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_validate_classes_passes_when_all_ids_present(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2])
    model = _FakeModel(names={0: "head", 1: "vbody", 2: "fbody"})

    _validate_classes(model)  # 不應拋例外


def test_validate_classes_raises_when_id_missing_from_model_names(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2, 5])
    model = _FakeModel(names={0: "head", 1: "vbody", 2: "fbody"})

    with pytest.raises(ValueError, match=r"\[5\]"):
        _validate_classes(model)


def test_validate_classes_skips_when_model_names_unavailable(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2])
    model = _FakeModel(names=None)

    _validate_classes(model)  # names 缺失時無法驗證，略過而非拋例外


def test_yolo_detector_raises_when_model_path_missing(monkeypatch, tmp_path):
    # 檔案不存在時必須直接 fail loud，不可讓 ultralytics 自行 fallback 下載到
    # 別的模型（那樣 _validate_classes 也擋不住，見同函式的說明）。
    monkeypatch.setattr(settings.model, "model_path", str(tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError):
        YOLODetector()
